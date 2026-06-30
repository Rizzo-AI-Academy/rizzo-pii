// App desktop Anonimizzatore PII (rizzo-pii).
//
// Architettura: il motore e' il backend Python/Flask (modello mmBERT) impacchettato con
// PyInstaller e incluso come risorsa ("backend/pii-backend/pii-backend.exe"). Tauri fa da
// finestra nativa: all'avvio mostra uno splash, lancia il backend come processo figlio,
// attende che il server locale risponda e poi apre la finestra principale puntata sull'UI
// Flask. Alla chiusura il processo figlio viene terminato.
// L'host e la porta sono configurabili via config.json (default 127.0.0.1:5005).
// NB: porta 5005 e non 5000 perche' su macOS la 5000 e' occupata da AirPlay Receiver.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::json;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

/// Logging diagnostico su file (~/rizzo-pii/tauri.log).
/// Usa un file nella stessa directory del backend.log.
fn tlog(msg: &str) {
    use std::io::Write;
    let dir = dirs::home_dir().unwrap_or_default().join("rizzo-pii");
    let _ = fs::create_dir_all(&dir);
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true).append(true)
        .open(dir.join("tauri.log"))
    {
        let _ = writeln!(f, "[{}] {}",
            chrono_timestamp(), msg);
    }
}

fn chrono_timestamp() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", d.as_secs(), d.subsec_millis())
}

/// Custodisce il processo del backend per poterlo terminare all'uscita.
struct BackendProcess(Mutex<Option<Child>>);

/// Stato condiviso: host e porta correnti letti dal config.json.
struct ServerAddr(Mutex<(String, u16)>);

// ---------------------------------------------------------------------------
// Configurazione persistente
// ---------------------------------------------------------------------------

/// Restituisce la directory di configurazione dell'app:
/// - Windows:  %LOCALAPPDATA%\rizzo-pii
/// - Linux:    ~/.local/share/rizzo-pii
/// - macOS:    ~/Library/Application Support/rizzo-pii
fn config_dir() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rizzo-pii")
}

/// Legge host e porta da config.json; se il file non esiste o e' malformato
/// restituisce i default (127.0.0.1, 5005).
fn load_config() -> (String, u16) {
    let path = config_dir().join("config.json");
    let default = ("127.0.0.1".to_string(), 5005u16);
    let Ok(data) = fs::read_to_string(&path) else {
        return default;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) else {
        return default;
    };
    let host = v
        .get("host")
        .and_then(|h| h.as_str())
        .unwrap_or("127.0.0.1")
        .to_string();
    let port = v
        .get("port")
        .and_then(|p| p.as_u64())
        .map(|p| p as u16)
        .unwrap_or(5005);
    (host, port)
}

/// Scrive (o sovrascrive) il config.json con host e porta indicati.
fn save_config_file(host: &str, port: u16) {
    let dir = config_dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("config.json");
    let content = json!({ "host": host, "port": port });
    let _ = fs::write(path, serde_json::to_string_pretty(&content).unwrap());
}

// ---------------------------------------------------------------------------
// Utilita'
// ---------------------------------------------------------------------------

/// True se sulla porta c'e' il NOSTRO Flask (non un servizio estraneo).
/// Esegue un HTTP GET /config e verifica che la risposta contenga "config_path"
/// (campo specifico del nostro endpoint). Se la porta e' libera, occupata da
/// un altro processo, o il server non risponde come previsto, restituisce false.
fn is_our_backend(host: &str, port: u16) -> bool {
    let addr_str = format!("{}:{}", host, port);
    let sock_addr: std::net::SocketAddr = match addr_str.parse() {
        Ok(a) => a,
        Err(_) => return false,
    };
    let mut stream = match TcpStream::connect_timeout(&sock_addr, Duration::from_millis(500)) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(2000)));
    let req = format!(
        "GET /config HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        addr_str
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    // leggi la risposta completa (headers + body) in un loop fino a EOF/timeout
    let mut response = Vec::with_capacity(4096);
    let mut buf = [0u8; 4096];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,                              // EOF: il server ha chiuso
            Ok(n) => response.extend_from_slice(&buf[..n]),
            Err(_) => break,                             // timeout o errore: basta
        }
    }
    let body = String::from_utf8_lossy(&response);
    body.contains("config_path")
}

/// Costruisce il path dell'eseguibile sidecar a partire dalla resource_dir.
fn sidecar_exe(resource_dir: &std::path::Path) -> PathBuf {
    let bin_name = if cfg!(windows) {
        "pii-backend.exe"
    } else {
        "pii-backend"
    };
    resource_dir
        .join("backend")
        .join("pii-backend")
        .join(bin_name)
}

/// Lancia il sidecar impostando le variabili PII_HOST e PII_PORT.
fn spawn_sidecar(exe: &std::path::Path, host: &str, port: u16) -> std::io::Result<Child> {
    Command::new(exe)
        .env("PII_HOST", host)
        .env("PII_PORT", port.to_string())
        .spawn()
}

// ---------------------------------------------------------------------------
// Polling del backend con rilevamento uscita anticipata
// ---------------------------------------------------------------------------

/// Codice d'uscita che il backend usa per segnalare un conflitto di porta.
const EXIT_PORT_CONFLICT: i32 = 76;

/// Ciclo di polling: attende che il backend risponda o che il processo figlio termini.
/// Restituisce Ok(()) se il backend e' pronto, Err(msg) se il processo e' uscito o timeout.
fn poll_backend(
    handle: &tauri::AppHandle,
    host: &str,
    port: u16,
) -> Result<(), String> {
    tlog(&format!("poll_backend: inizio polling su {}:{}", host, port));
    let bp = handle.state::<BackendProcess>();
    for i in 0..900 {
        // fino a ~180s (primo avvio: caricamento modello)
        // controlla PRIMA se il processo figlio e' uscito in anticipo
        // (cosi' un exit code 76 viene rilevato subito, senza che un servizio
        //  estraneo sulla stessa porta inganni il check di connettivita')
        {
            let mut guard = bp.0.lock().unwrap();
            if let Some(ref mut child) = *guard {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        // il processo e' terminato
                        let code = status.code().unwrap_or(-1);
                        tlog(&format!("poll_backend: child uscito con codice {} (iter {})", code, i));
                        // rimuovi il child morto dallo stato
                        *guard = None;
                        if code == EXIT_PORT_CONFLICT {
                            // conflitto di porta: mostra il form di configurazione nello splash
                            if let Some(splash) = handle.get_webview_window("splash") {
                                let js = format!(
                                    "if(typeof showConfigForm==='function'){{showConfigForm('{}',{},'Porta {} occupata. Scegli un\\'altra porta.')}}",
                                    host, port, port
                                );
                                let _ = splash.eval(&js);
                            }
                            return Err(format!("porta {} occupata (exit code 76)", port));
                        } else {
                            // errore generico
                            if let Some(splash) = handle.get_webview_window("splash") {
                                let _ = splash.eval(
                                    "var n=document.querySelector('.note');\
                                     if(n){n.textContent='Errore: il backend si è chiuso inaspettatamente.';}\
                                     var b=document.querySelector('.bar');if(b){b.style.display='none';}",
                                );
                            }
                            return Err(format!(
                                "backend uscito con codice {}",
                                code
                            ));
                        }
                    }
                    Ok(None) => {} // ancora in esecuzione, continua il polling
                    Err(_) => {}   // errore nel try_wait, ignora e continua
                }
            }
        }
        // verifica HTTP: il backend e' il NOSTRO Flask? (non un servizio estraneo)
        if is_our_backend(host, port) {
            tlog(&format!("poll_backend: backend pronto (iter {})", i));
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    // timeout raggiunto
    if let Some(splash) = handle.get_webview_window("splash") {
        let _ = splash.eval(
            "var n=document.querySelector('.note');\
             if(n){n.textContent='Errore: il backend non si è avviato. \
             Vedi il log in %LOCALAPPDATA%\\\\rizzo-pii\\\\backend.log';}\
             var b=document.querySelector('.bar');if(b){b.style.display='none';}",
        );
    }
    Err("timeout in attesa del backend".to_string())
}

// ---------------------------------------------------------------------------
// Comandi Tauri invocabili dal frontend (splash)
// ---------------------------------------------------------------------------

/// Salva host e porta nel config.json e aggiorna lo stato in-memory.
#[tauri::command]
fn save_config(host: String, port: u16, state: tauri::State<'_, ServerAddr>) {
    tlog(&format!("save_config: {}:{}", host, port));
    save_config_file(&host, port);
    let mut s = state.0.lock().unwrap();
    *s = (host, port);
}

/// Termina l'eventuale backend in esecuzione, rilegge la configurazione e rilancia il sidecar.
#[tauri::command]
fn retry_backend(app_handle: tauri::AppHandle) {
    tlog("retry_backend: chiamato");
    // termina il processo figlio precedente, se presente
    {
        let bp = app_handle.state::<BackendProcess>();
        if let Some(mut child) = bp.0.lock().unwrap().take() {
            tlog("retry_backend: killing old child");
            let _ = child.kill();
            let _ = child.wait();
        };
    }

    // leggi la configurazione aggiornata dallo stato in-memory
    let (host, port) = {
        let s = app_handle.state::<ServerAddr>();
        let guard = s.0.lock().unwrap();
        guard.clone()
    };
    tlog(&format!("retry_backend: config {}:{}", host, port));
    let url = format!("http://{}:{}", host, port);

    // spawn del sidecar
    if !is_our_backend(&host, port) {
        if let Ok(dir) = app_handle.path().resource_dir() {
            let exe = sidecar_exe(&dir);
            tlog(&format!("retry_backend: spawn {:?}", exe));
            match spawn_sidecar(&exe, &host, port) {
                Ok(child) => {
                    app_handle
                        .state::<BackendProcess>()
                        .0
                        .lock()
                        .unwrap()
                        .replace(child);
                }
                Err(e) => eprintln!("avvio backend fallito ({:?}): {}", exe, e),
            }
        }
    }

    // polling in un thread separato
    let handle = app_handle.clone();
    std::thread::spawn(move || {
        if poll_backend(&handle, &host, port).is_ok() {
            let built = WebviewWindowBuilder::new(
                &handle,
                "main",
                WebviewUrl::External(url.parse().unwrap()),
            )
            .title("Rizzo PII")
            .inner_size(1240.0, 840.0)
            .min_inner_size(900.0, 600.0)
            .center()
            .build();

            if built.is_ok() {
                if let Some(splash) = handle.get_webview_window("splash") {
                    let _ = splash.close();
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (host, port) = load_config();
    tlog(&format!("=== AVVIO === config: {}:{}", host, port));

    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .manage(ServerAddr(Mutex::new((host.clone(), port))))
        .invoke_handler(tauri::generate_handler![save_config, retry_backend])
        .setup(move |app| {
            let handle = app.handle().clone();
            let url = format!("http://{}:{}", host, port);

            // mostra la versione reale (da tauri.conf.json) nello splash
            let version = app.package_info().version.to_string();
            if let Some(splash) = app.get_webview_window("splash") {
                let _ = splash.eval(&format!(
                    "var v=document.getElementById('ver');if(v){{v.textContent='v{}';}};",
                    version
                ));
            }

            // 1) avvia il backend sidecar (salvo che un'istanza sia gia' in ascolto)
            let ours = is_our_backend(&host, port);
            tlog(&format!("is_our_backend({}:{}): {}", host, port, ours));
            if !ours {
                match app.path().resource_dir() {
                    Ok(dir) => {
                        let exe = sidecar_exe(&dir);
                        tlog(&format!("spawn sidecar: {:?}", exe));
                        match spawn_sidecar(&exe, &host, port) {
                            Ok(child) => {
                                tlog(&format!("sidecar avviato, pid={}", child.id()));
                                app.state::<BackendProcess>()
                                    .0
                                    .lock()
                                    .unwrap()
                                    .replace(child);
                            }
                            Err(e) => tlog(&format!("ERRORE spawn sidecar: {}", e)),
                        }
                    }
                    Err(e) => eprintln!("resource_dir non risolvibile: {}", e),
                }
            }

            // 2) attende il server, poi apre la finestra principale e chiude lo splash
            let host2 = host.clone();
            std::thread::spawn(move || {
                if poll_backend(&handle, &host2, port).is_ok() {
                    let built = WebviewWindowBuilder::new(
                        &handle,
                        "main",
                        WebviewUrl::External(url.parse().unwrap()),
                    )
                    .title("Rizzo PII")
                    .inner_size(1240.0, 840.0)
                    .min_inner_size(900.0, 600.0)
                    .center()
                    .build();

                    if built.is_ok() {
                        if let Some(splash) = handle.get_webview_window("splash") {
                            let _ = splash.close();
                        }
                    }
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("errore di avvio dell'applicazione Tauri")
        .run(|handle, event| {
            // alla chiusura dell'app: termina il backend
            if let tauri::RunEvent::Exit = event {
                if let Some(mut child) =
                    handle.state::<BackendProcess>().0.lock().unwrap().take()
                {
                    let _ = child.kill();
                }
            }
        });
}
