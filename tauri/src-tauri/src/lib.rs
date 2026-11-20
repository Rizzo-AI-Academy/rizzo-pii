// App desktop Anonimizzatore PII (rizzo-pii).
//
// Architettura: il motore e' il backend Python/Flask (modello mmBERT) impacchettato con
// PyInstaller e incluso come risorsa ("backend/pii-backend/pii-backend.exe"). Tauri fa da
// finestra nativa: all'avvio mostra uno splash, lancia il backend come processo figlio,
// attende che il server locale risponda e poi apre la finestra principale puntata sull'UI
// Flask. Alla chiusura il processo figlio viene terminato.
// L'host e la porta sono configurabili via config.json (default 127.0.0.1:5005).
// NB: porta 5005 e non 5000 perche' su macOS la 5000 e' occupata da AirPlay Receiver.
//
// Modello: l'installer "offline" lo ha dentro il sidecar; quello leggero lo scarica da
// Hugging Face al primo avvio (vedi model.rs) e lo passa al backend con PII_MODEL_DIR.

mod model;

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::json;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

/// Logging diagnostico su file (~/rizzo-pii/tauri.log).
/// Usa un file nella stessa directory del backend.log.
pub(crate) fn tlog(msg: &str) {
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

/// Cartella del modello scaricato da passare al backend (None = modello nel pacchetto).
struct ModelDir(Mutex<Option<PathBuf>>);

/// Evita due download in parallelo (doppio clic su "Riprova").
struct Downloading(Mutex<bool>);

/// Ultimo stato del download mostrato nello splash (JS). Un eval fatto prima che la
/// pagina dello splash sia caricata si perde: la pagina, appena pronta, chiede questo
/// stato con `splash_sync` e lo riapplica.
struct LastSplash(Mutex<Option<String>>);

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
    // il nome va risolto: "localhost:5005".parse::<SocketAddr>() e' un errore, il parse
    // accetta solo IP letterali. E si provano TUTTI gli indirizzi risolti, non il primo:
    // "localhost" da' prima ::1, mentre il backend puo' ascoltare solo su 127.0.0.1.
    let addrs: Vec<_> = match (host, port).to_socket_addrs() {
        Ok(it) => it.collect(),
        Err(_) => return false,
    };
    // i 500 ms sono il budget dell'intera chiamata, non di ogni indirizzo: poll_backend
    // la ripete 900 volte e la somma sarebbe l'attesa prima dell'errore.
    let attesa = Duration::from_millis(500) / addrs.len().max(1) as u32;
    let mut stream = match addrs
        .into_iter()
        .filter_map(|a| TcpStream::connect_timeout(&a, attesa).ok())
        .next()
    {
        Some(s) => s,
        None => return false,
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

/// Lancia il sidecar impostando PII_HOST e PII_PORT e, se il modello e' stato scaricato,
/// PII_MODEL_DIR (che app.py preferisce al modello impacchettato).
fn spawn_sidecar(
    exe: &std::path::Path,
    host: &str,
    port: u16,
    model_dir: Option<&std::path::Path>,
) -> std::io::Result<Child> {
    let mut cmd = Command::new(exe);
    cmd.env("PII_HOST", host).env("PII_PORT", port.to_string());
    if let Some(dir) = model_dir {
        cmd.env("PII_MODEL_DIR", dir);
    }
    cmd.spawn()
}

/// Esegue una funzione JS dello splash (se c'e' ancora) e la ricorda per `splash_sync`.
fn splash_js(handle: &tauri::AppHandle, js: &str) {
    *handle.state::<LastSplash>().0.lock().unwrap() = Some(js.to_string());
    if let Some(splash) = handle.get_webview_window("splash") {
        let _ = splash.eval(js);
    }
}

/// Stringa Rust -> letterale stringa JS (escape completo).
fn js_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| String::from("\"\""))
}

/// Avvia il sidecar (salvo che il nostro backend sia gia' in ascolto), attende che
/// risponda, apre la finestra principale e chiude lo splash. Non blocca.
fn launch_backend(handle: &tauri::AppHandle, host: String, port: u16) {
    let url = format!("http://{}:{}", host, port);
    let ours = is_our_backend(&host, port);
    tlog(&format!("is_our_backend({}:{}): {}", host, port, ours));
    if !ours {
        match handle.path().resource_dir() {
            Ok(dir) => {
                let exe = sidecar_exe(&dir);
                let model_dir = handle.state::<ModelDir>().0.lock().unwrap().clone();
                tlog(&format!("spawn sidecar: {:?} (modello: {:?})", exe, model_dir));
                match spawn_sidecar(&exe, &host, port, model_dir.as_deref()) {
                    Ok(child) => {
                        tlog(&format!("sidecar avviato, pid={}", child.id()));
                        handle.state::<BackendProcess>().0.lock().unwrap().replace(child);
                    }
                    Err(e) => tlog(&format!("ERRORE spawn sidecar: {}", e)),
                }
            }
            Err(e) => tlog(&format!("resource_dir non risolvibile: {}", e)),
        }
    }

    let handle = handle.clone();
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
            // il drop lo gestisce la pagina (HTML5): su Windows il gestore nativo di
            // Tauri si prende i file trascinati e l'evento non arriva mai (issue #96)
            .disable_drag_drop_handler()
            .build();

            if built.is_ok() {
                if let Some(splash) = handle.get_webview_window("splash") {
                    let _ = splash.close();
                }
            }
        }
    });
}

/// Si assicura che il modello ci sia (nel pacchetto, gia' scaricato, o scaricandolo ora
/// con l'avanzamento nello splash) e poi avvia il backend. Non blocca.
fn ensure_model_then_launch(handle: &tauri::AppHandle) {
    let (host, port) = handle.state::<ServerAddr>().0.lock().unwrap().clone();
    // backend gia' attivo (seconda istanza, o rilancio): niente da scaricare
    if is_our_backend(&host, port) {
        launch_backend(handle, host, port);
        return;
    }
    let bundled = handle
        .path()
        .resource_dir()
        .map(|d| model::bundled(&d))
        .unwrap_or(false);
    if bundled {
        tlog("modello nel pacchetto (installer offline)");
        *handle.state::<ModelDir>().0.lock().unwrap() = None;
        launch_backend(handle, host, port);
        return;
    }
    let dir = model::model_dir(&config_dir());
    if model::ready(&dir) {
        tlog(&format!("modello gia' scaricato in {:?}", dir));
        *handle.state::<ModelDir>().0.lock().unwrap() = Some(dir);
        launch_backend(handle, host, port);
        return;
    }

    {
        let state = handle.state::<Downloading>();
        let mut busy = state.0.lock().unwrap();
        if *busy {
            return;
        }
        *busy = true;
    }
    let handle = handle.clone();
    std::thread::spawn(move || {
        tlog(&format!("download del modello in {:?}", dir));
        splash_js(&handle, &format!("if(window.showDownload){{showDownload(0,{})}}", model::total_bytes()));
        let h2 = handle.clone();
        let res = model::download(&dir, |done, total| {
            splash_js(&h2, &format!("if(window.showDownload){{showDownload({},{})}}", done, total));
        });
        *handle.state::<Downloading>().0.lock().unwrap() = false;
        match res {
            Ok(()) => {
                tlog("download del modello completato e verificato");
                *handle.state::<ModelDir>().0.lock().unwrap() = Some(dir);
                splash_js(&handle, "if(window.showStarting){showStarting()}");
                launch_backend(&handle, host, port);
            }
            Err(e) => {
                tlog(&format!("ERRORE download modello: {}", e));
                splash_js(&handle, &format!(
                    "if(window.showDownloadError){{showDownloadError({},{})}}",
                    js_str(&e),
                    js_str(&dir.to_string_lossy())
                ));
            }
        }
    });
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
    launch_backend(&app_handle, host, port);
}

/// Chiamato dallo splash appena caricato: versione + ultimo stato del download.
#[tauri::command]
fn splash_sync(app_handle: tauri::AppHandle) -> serde_json::Value {
    let last = app_handle.state::<LastSplash>().0.lock().unwrap().clone();
    json!({
        "version": app_handle.package_info().version.to_string(),
        "js": last,
    })
}

/// "Riprova" dopo un download fallito: riparte da dove si era fermato.
#[tauri::command]
fn retry_download(app_handle: tauri::AppHandle) {
    tlog("retry_download: chiamato");
    ensure_model_then_launch(&app_handle);
}

/// Apre nel file manager la cartella del modello (per copiarci i file a mano).
#[tauri::command]
fn open_model_folder() {
    let dir = model::model_dir(&config_dir());
    let _ = fs::create_dir_all(&dir);
    let opener = if cfg!(windows) {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = Command::new(opener).arg(&dir).spawn();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (mut host, mut port) = load_config();

    // CLI args hanno precedenza sul config.json (stessa catena di serve.py)
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1; // salta argv[0]
    while i < args.len() {
        match args[i].as_str() {
            "--host" if i + 1 < args.len() => { host = args[i + 1].clone(); i += 2; }
            "--port" if i + 1 < args.len() => {
                if let Ok(p) = args[i + 1].parse::<u16>() { port = p; }
                i += 2;
            }
            _ => { i += 1; }
        }
    }

    tlog(&format!("=== AVVIO === config: {}:{}", host, port));

    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .manage(ServerAddr(Mutex::new((host.clone(), port))))
        .manage(ModelDir(Mutex::new(None)))
        .manage(Downloading(Mutex::new(false)))
        .manage(LastSplash(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            save_config,
            retry_backend,
            retry_download,
            open_model_folder,
            splash_sync
        ])
        .setup(move |app| {
            // mostra la versione reale (da tauri.conf.json) nello splash
            let version = app.package_info().version.to_string();
            if let Some(splash) = app.get_webview_window("splash") {
                let _ = splash.eval(&format!(
                    "var v=document.getElementById('ver');if(v){{v.textContent='v{}';}};",
                    version
                ));
            }

            // modello (nel pacchetto / gia' scaricato / da scaricare), poi il backend
            ensure_model_then_launch(app.handle());

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
