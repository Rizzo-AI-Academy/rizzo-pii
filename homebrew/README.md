# Homebrew (cask) — installazione macOS

La cask `rizzo-pii` è il file [`Casks/rizzo-pii.rb`](Casks/rizzo-pii.rb) di questa cartella:
**qui vive la versione da pubblicare**. Homebrew non legge le cask da questo repo: ci vuole un
**tap** dedicato (`Rizzo-AI-Academy/homebrew-rizzo-pii`), che contiene la stessa cask in
`Casks/rizzo-pii.rb`. La sincronizzazione è automatica via workflow (vedi sotto).

## Installazione (per gli utenti)

```bash
brew tap Rizzo-AI-Academy/rizzo-pii
brew install --cask rizzo-pii
```

Nota: la cask attuale è **solo per Apple Silicon** (arm64), come il DMG rilasciato.

## Setup del tap (una volta, admin dell'org)

1. Creare il repo **`Rizzo-AI-Academy/homebrew-rizzo-pii`** (il nome deve iniziare con
   `homebrew-`) con un README minimo.
2. Copiare la cask: `mkdir Casks && cp homebrew/Casks/rizzo-pii.rb Casks/rizzo-pii.rb` (la prima
   volta, finché il workflow non esiste, la sync è manuale).
3. Aggiungere il secret **`TAP_REPO_TOKEN`** alle Actions del repo `rizzo-pii`: un fine-grained
   PAT con permesso **Contents: Read and write** sul **solo** repo del tap (principio del minimo
   privilegio: il token non deve toccare altri repo).

## Pubblicazione automatica (`.github/workflows/publish-cask.yml`)

Quando viene **pubblicata** una release (niente prerelease), il workflow:

1. scarica l'asset `*-macOS-arm64.dmg` della release (senza DMG macOS → skip);
2. calcola `sha256` e ricava la versione dal tag (senza `v`);
3. aggiorna `version` e `sha256` in `homebrew/Casks/rizzo-pii.rb`;
4. committa e spinge su `main` di **questo** repo;
5. clona il tap, copia la cask in `Casks/` e spinge con `TAP_REPO_TOKEN`.

Limiti noti:

- se `main` ha la **branch protection** (PR obbligatoria), il push diretto fallisce → sync
  manuale del tap (copia del file) e aggiornamento del cask a mano;
- se manca `TAP_REPO_TOKEN`, il workflow si ferma al passo 4 e il tap va aggiornato a mano;
- se il nome del DMG cambia formato, va aggiornato il pattern `--pattern` nel workflow.

## Checklist di release (se il workflow non è intervenuto)

1. `shasum -a 256 Rizzo-PII-<v>-macOS-arm64.dmg`
2. aggiornare `version` e `sha256` in `Casks/rizzo-pii.rb` (qui in `homebrew/`)
3. copiare il file nel tap (`Rizzo-AI-Academy/homebrew-rizzo-pii`, `Casks/rizzo-pii.rb`)
4. commit + push in entrambi i repo.

Verifica di un aggiornamento della cask: `brew install --cask rizzo-pii` (dopo `brew tap`),
oppure localmente `brew install --cask ./Casks/rizzo-pii.rb`.
