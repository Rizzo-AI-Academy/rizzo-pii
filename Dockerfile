# ============================================================================
# Rizzo PII — webapp (server Flask + UI) in un container.
#
# L'immagine e' AUTOSUFFICIENTE: dentro ci sono le dipendenze CPU e il modello
# scaricato da Hugging Face in fase di build. A runtime NON esce nulla verso
# internet (HF_HUB_OFFLINE=1): e' lo stesso patto dell'app desktop, i documenti
# non lasciano la macchina.
#
#   docker build -t rizzo-pii .
#   docker run --rm -p 5005:5005 rizzo-pii        # -> http://127.0.0.1:5005
#
# NB: e' un'immagine CPU. Per la GPU servirebbe il torch cu128 + nvidia-runtime;
# per l'inferenza su un documento la CPU basta (vedi README, sezione Deployment).
# Per la build dei bundle Linux (.deb/.AppImage) c'e' Dockerfile.linux, altra cosa.
# ============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: runtime OpenMP richiesto dai kernel CPU di torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- dipendenze --------------------------------------------------------------
# torch dall'indice CPU: la wheel default tira dietro ~2.5 GB di CUDA che qui
# non servirebbero a niente. gunicorn perche' il server di sviluppo di Flask
# non e' un server (single-thread, nessun timeout).
#
# torch e transformers PINNATI alle versioni con cui l'immagine e' stata verificata:
# senza pin la stessa build fra sei mesi tira una major diversa e "funzionava ieri"
# diventa un bug da riprodurre. Per aggiornarli: alza i numeri e ricostruisci.
# (NB: transformers 5.x qui, mentre Dockerfile.linux resta sul 4.57.3 di PyInstaller.)
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" \
 && pip install "transformers==5.14.1" tokenizers safetensors flask pymupdf gunicorn huggingface_hub

# --- modello -----------------------------------------------------------------
# app.py con APP_MODEL_VERSION="main" cerca models/rizzo-pii-0.3B-main/ a partire
# dalla root della repo (parents[2] rispetto a src/app/app.py) -> /app/models/.
ARG MODEL_REPO=rizzoaiacademy/rizzo-pii-0.3B
ARG MODEL_REVISION=main
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', \
local_dir='/app/models/rizzo-pii-0.3B-main', \
allow_patterns=['*.json','*.safetensors','*.txt','*.model'])"

# --- applicazione ------------------------------------------------------------
# Solo src/app: il resto della repo (training, data pipeline, dataset) non serve
# a servire l'app e ogni file in piu' e' un layer che si invalida per niente.
WORKDIR /app
COPY src/app/ /app/src/app/

# utente non-root; la home deve essere scrivibile perche' server_config scrive
# config.json/prefs.json in ~/.local/share/rizzo-pii (preferenze dell'UI).
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
ENV HOME=/home/app

# In container si ascolta su tutte le interfacce: il confine di rete lo mette
# docker (-p 127.0.0.1:5005:5005 per non esporlo alla LAN), non il bind.
ENV PII_HOST=0.0.0.0 \
    PII_PORT=5005 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 5005

# /health e' readiness senza inferenza: 503 finche' il modello non e' caricato.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['PII_PORT']}/health\", timeout=4).status==200 else 1)"

# 1 worker: il modello sta in memoria una volta sola (ogni worker sarebbe una
# copia da ~1.2 GB). Il parallelismo lo danno i thread. timeout alto perche' un
# PDF lungo e' minuti di inferenza CPU, non secondi.
CMD ["sh", "-c", "exec gunicorn --chdir /app/src/app --bind ${PII_HOST}:${PII_PORT} --workers 1 --threads 4 --timeout 600 --graceful-timeout 30 --access-logfile - --error-logfile - app:app"]
