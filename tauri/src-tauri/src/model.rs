// Modello al primo avvio: scaricato da Hugging Face invece che dentro l'installer.
//
// L'installer "leggero" non contiene i pesi (1,2 GB): al primo avvio Tauri li scarica
// dal repo pubblico rizzoaiacademy/rizzo-pii-0.3B, li verifica e li tiene nella cartella
// dati dell'utente, cosi' gli avvii successivi (e gli aggiornamenti dell'app) non li
// riscaricano. L'installer "offline" invece li ha gia' dentro il sidecar: in quel caso
// questo modulo non fa nulla.
//
// Sicurezza / riproducibilita':
//   - la revisione e' un COMMIT preciso, non un ramo: se il ramo su HF si sposta, l'app
//     continua a scaricare esattamente i file con cui e' stata provata;
//   - ogni file e' verificato con dimensione + sha256 PRIMA di essere messo al suo posto:
//     un file troncato, corrotto o diverso non viene mai caricato dal backend;
//   - si scaricano solo i pesi: nessun dato dell'utente esce dal computer.
//
// Robustezza: download in "<file>.part" con ripresa (header Range) se la connessione
// cade o l'app viene chiusa a meta'; il proxy si prende da HTTPS_PROXY/HTTP_PROXY.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

pub const MODEL_REPO: &str = "rizzoaiacademy/rizzo-pii-0.3B";
pub const MODEL_VERSION: &str = "1.5.0";
/// Commit del ramo v1.5.0 su HF che contiene onnx/ (export fp32 verificato contro PyTorch).
pub const MODEL_COMMIT: &str = "8cc0815c52357cb9ff58cb08f139115a9d2e43c5";

/// (percorso nel repo, dimensione in byte, sha256)
pub const MODEL_FILES: &[(&str, u64, &str)] = &[
    ("config.json", 3137,
     "6b9d5bdd4a194d40fb40c53fdc302e225514cd7940d97f657c4082454ee7e91b"),
    ("special_tokens_map.json", 1051,
     "6e3204c5a4004719c185007a745d1940471a6fb02521cfa0a00306dbb6bc5903"),
    ("tokenizer_config.json", 46469,
     "3b66ca84514d9aacf4b90c65a61e561127c6c21d81bbf4e8ba9d46a4c15ac82d"),
    ("tokenizer.json", 34363286,
     "a0ab5772a8c5450f45a66c95985d6df2205bf09449ba7ff7bb46de034eaa118a"),
    ("onnx/model.onnx", 568606,
     "9d0fcfa86ec0751ecb737bb51d882be723c9998d4872137570bb4b5c5a02effe"),
    ("onnx/model.onnx_data", 1230259200,
     "dc7dbe401863292ebf1e7e396fd438424891b83fc6f2b877247b2550a7418b24"),
];

/// Totale da scaricare (per la barra di avanzamento).
pub fn total_bytes() -> u64 {
    MODEL_FILES.iter().map(|f| f.1).sum()
}

/// Cartella del modello scaricato: <config dir>/models/rizzo-pii-0.3B-v<versione>.
pub fn model_dir(config_dir: &Path) -> PathBuf {
    config_dir
        .join("models")
        .join(format!("rizzo-pii-0.3B-v{}", MODEL_VERSION))
}

/// True se l'installer ha gia' il modello dentro il sidecar (installer "offline").
pub fn bundled(resource_dir: &Path) -> bool {
    let base = resource_dir.join("backend").join("pii-backend");
    // layout PyInstaller onedir: _internal/pii_model (Windows/Linux e macOS non-bundle)
    base.join("_internal").join("pii_model").join("model.onnx_data").is_file()
}

/// File marcatore scritto dopo la verifica: evita di ricalcolare 1,2 GB di sha256 a
/// ogni avvio. Contiene il commit, cosi' un cambio di modello invalida la cache.
fn marker(dir: &Path) -> PathBuf {
    dir.join(".verified")
}

/// Il modello e' gia' scaricato e verificato?
pub fn ready(dir: &Path) -> bool {
    let ok_marker = fs::read_to_string(marker(dir))
        .map(|s| s.trim() == MODEL_COMMIT)
        .unwrap_or(false);
    ok_marker
        && MODEL_FILES.iter().all(|(rel, size, _)| {
            fs::metadata(dir.join(rel)).map(|m| m.len() == *size).unwrap_or(false)
        })
}

fn url(rel: &str) -> String {
    format!("https://huggingface.co/{}/resolve/{}/{}", MODEL_REPO, MODEL_COMMIT, rel)
}

fn hash_file(path: &Path) -> std::io::Result<String> {
    let mut h = Sha256::new();
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Messaggio d'errore breve per lo splash; il dettaglio tecnico finisce in tauri.log.
fn net_err(e: ureq::Error) -> String {
    crate::tlog(&format!("errore di rete: {}", e));
    match e {
        ureq::Error::Status(code, _) => format!("Hugging Face ha risposto con errore {}.", code),
        ureq::Error::Transport(t) => match t.kind() {
            ureq::ErrorKind::Dns => "huggingface.co non raggiungibile: sei connesso a internet?".into(),
            ureq::ErrorKind::ConnectionFailed => "Connessione a huggingface.co rifiutata o bloccata.".into(),
            ureq::ErrorKind::ProxyConnect | ureq::ErrorKind::ProxyUnauthorized | ureq::ErrorKind::InvalidProxyUrl =>
                "Il proxy non risponde o rifiuta la connessione.".into(),
            ureq::ErrorKind::Io => "Connessione interrotta.".into(),
            _ => "Connessione a huggingface.co non riuscita.".into(),
        },
    }
}

/// Scarica (o completa) il modello in `dir`. `progress(fatti, totale)` viene chiamata
/// spesso: chi la riceve decide quanto spesso aggiornare l'interfaccia.
pub fn download(dir: &Path, mut progress: impl FnMut(u64, u64)) -> Result<(), String> {
    fs::create_dir_all(dir.join("onnx")).map_err(|e| format!("cartella non creabile: {}", e))?;
    let _ = fs::remove_file(marker(dir));
    let total = total_bytes();
    let agent = ureq::AgentBuilder::new()
        .user_agent(&format!("rizzo-pii-desktop/{}", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        .try_proxy_from_env(true)
        .build();

    let mut done_before: u64 = 0;
    for (rel, size, sha) in MODEL_FILES {
        let dest = dir.join(rel);
        // file gia' completo e corretto (es. download interrotto sul file successivo)
        if fs::metadata(&dest).map(|m| m.len() == *size).unwrap_or(false)
            && hash_file(&dest).map(|h| h == *sha).unwrap_or(false)
        {
            done_before += size;
            progress(done_before, total);
            continue;
        }
        let part = dest.with_file_name(format!(
            "{}.part",
            dest.file_name().unwrap().to_string_lossy()
        ));
        let mut tentativi = 0;
        loop {
            tentativi += 1;
            match fetch(&agent, rel, &part, *size, done_before, total, &mut progress) {
                Ok(()) => break,
                Err(e) if tentativi < 4 => {
                    crate::tlog(&format!("download {}: tentativo {} fallito: {}", rel, tentativi, e));
                    std::thread::sleep(Duration::from_secs(2 * tentativi));
                }
                Err(e) => return Err(e),
            }
        }
        let got = hash_file(&part).map_err(|e| format!("lettura di {}: {}", rel, e))?;
        if got != *sha {
            let _ = fs::remove_file(&part);
            return Err(format!(
                "{} non corrisponde al file pubblicato (sha256 diverso): riprova",
                rel
            ));
        }
        let _ = fs::remove_file(&dest);
        fs::rename(&part, &dest).map_err(|e| format!("salvataggio di {}: {}", rel, e))?;
        done_before += size;
        progress(done_before, total);
    }
    fs::write(marker(dir), MODEL_COMMIT).map_err(|e| format!("marcatore: {}", e))?;
    Ok(())
}

/// Scarica un file in `part`, riprendendo da dove si era fermato.
fn fetch(
    agent: &ureq::Agent,
    rel: &str,
    part: &Path,
    size: u64,
    done_before: u64,
    total: u64,
    progress: &mut impl FnMut(u64, u64),
) -> Result<(), String> {
    let mut have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
    if have > size {
        let _ = fs::remove_file(part);
        have = 0;
    }
    if have == size {
        return Ok(());
    }
    let mut req = agent.get(&url(rel));
    if have > 0 {
        req = req.set("Range", &format!("bytes={}-", have));
    }
    let resp = req.call().map_err(net_err)?;
    // 206 = il server riprende; 200 = riparte da zero (Range ignorato): si riscrive tutto
    let append = have > 0 && resp.status() == 206;
    if !append {
        have = 0;
    }
    let mut out = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(part)
        .map_err(|e| format!("scrittura di {}: {}", rel, e))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut last = Instant::now();
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| { crate::tlog(&format!("lettura interrotta: {}", e)); "Connessione interrotta durante il download.".to_string() })?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| format!("scrittura su disco di {} ({}): spazio esaurito?", rel, e))?;
        have += n as u64;
        if last.elapsed() >= Duration::from_millis(200) {
            progress(done_before + have, total);
            last = Instant::now();
        }
    }
    out.flush().map_err(|e| format!("scrittura di {}: {}", rel, e))?;
    if have != size {
        return Err(format!("{} incompleto ({} di {} byte)", rel, have, size));
    }
    Ok(())
}
