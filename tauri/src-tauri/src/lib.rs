// App desktop Anonimizzatore PII (rizzo-pii).
//
// Architettura: il motore e' il backend Python/Flask (modello mmBERT) impacchettato con
// PyInstaller e incluso come risorsa ("backend/pii-backend/pii-backend.exe"). Tauri fa da
// finestra nativa: all'avvio mostra uno splash, lancia il backend come processo figlio,
// attende che il server locale risponda e poi apre la finestra principale puntata sull'UI
// Flask (http://127.0.0.1:5005). Alla chiusura il processo figlio viene terminato.
//
// Porta 5005, non 5000: su macOS la 5000 e' occupata da AirPlay Receiver, che rispondeva al
// check di backend_ready() -> WebView aperta sul server sbagliato (schermata bianca).

use std::net::TcpStream;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

/// Custodisce il processo del backend per poterlo terminare all'uscita.
struct BackendProcess(Mutex<Option<Child>>);

// Passata al sidecar via PII_PORT (sotto) cosi' Flask e Tauri non possono divergere.
const PORT: &str = "5005";
const ADDR: &str = "127.0.0.1:5005";
const URL: &str = "http://127.0.0.1:5005";

/// True se il server locale accetta connessioni (= modello caricato, Flask in ascolto).
fn backend_ready() -> bool {
    match ADDR.parse() {
        Ok(addr) => TcpStream::connect_timeout(&addr, Duration::from_millis(400)).is_ok(),
        Err(_) => false,
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(BackendProcess(Mutex::new(None)))
        .setup(|app| {
            let handle = app.handle().clone();

            // mostra la versione reale (da tauri.conf.json) nello splash
            let version = app.package_info().version.to_string();
            if let Some(splash) = app.get_webview_window("splash") {
                let _ = splash.eval(&format!(
                    "var v=document.getElementById('ver');if(v){{v.textContent='v{}';}}",
                    version
                ));
            }

            // 1) avvia il backend sidecar (salvo che un'istanza sia gia' in ascolto)
            if !backend_ready() {
                match app.path().resource_dir() {
                    Ok(dir) => {
                        // su Windows il sidecar PyInstaller e' pii-backend.exe, su Linux pii-backend
                        let bin_name = if cfg!(windows) { "pii-backend.exe" } else { "pii-backend" };
                        let exe = dir
                            .join("backend")
                            .join("pii-backend")
                            .join(bin_name);
                        match Command::new(&exe).env("PII_PORT", PORT).spawn() {
                            Ok(child) => {
                                app.state::<BackendProcess>()
                                    .0
                                    .lock()
                                    .unwrap()
                                    .replace(child);
                            }
                            Err(e) => eprintln!("avvio backend fallito ({:?}): {}", exe, e),
                        }
                    }
                    Err(e) => eprintln!("resource_dir non risolvibile: {}", e),
                }
            }

            // 2) attende il server, poi apre la finestra principale e chiude lo splash
            std::thread::spawn(move || {
                let mut ready = false;
                for _ in 0..900 {
                    // fino a ~180s (primo avvio: caricamento modello)
                    if backend_ready() {
                        ready = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }

                if ready {
                    let built = WebviewWindowBuilder::new(
                        &handle,
                        "main",
                        WebviewUrl::External(URL.parse().unwrap()),
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
                } else if let Some(splash) = handle.get_webview_window("splash") {
                    // timeout: mostra un messaggio d'errore nello splash
                    let _ = splash.eval(
                        "var n=document.querySelector('.note');\
                         if(n){n.textContent='Errore: il backend non si è avviato. \
                         Vedi il log in %LOCALAPPDATA%\\\\rizzo-pii\\\\backend.log';}\
                         var b=document.querySelector('.bar');if(b){b.style.display='none';}",
                    );
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
