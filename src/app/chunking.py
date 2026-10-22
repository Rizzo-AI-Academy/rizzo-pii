# -*- coding: utf-8 -*-
"""
Divisione del testo in chunk per il modello.

Come `detectors`, solo stringhe: nessun import di torch/transformers, quindi e'
testabile senza il modello. `app.py` ne ri-esporta i nomi.

Il chunk si misurava solo in parole separate da spazi. Un testo che di spazi non ne
ha (un JSON compatto, un CSV, un export) e' allora una parola sola e un chunk solo:
la pipeline lo tronca in silenzio a 8.192 token e quello che viene dopo non lo legge
nessuno. In un JSON compatto il taglio cade intorno al carattere 19.000; oltre, i
nomi restano in chiaro e l'output non lo segnala in nessun modo.
Due tetti in piu', che sulla prosa non scattano: su 481M caratteri di Wikipedia,
testo normativo, EUR-Lex e log i chunk restano identici a prima, uno per uno, salvo
in 15 documenti (tabelle di sostanze chimiche, marcatori di impaginazione, log):
  - un chunk non supera MAX_CHARS caratteri;
  - una "parola" che da sola li supera si spezza dopo i separatori di campo
    (`,` `;` `|`), e un campo ancora troppo lungo a MAX_PEZZO caratteri.
A 2.500 caratteri un JSON compatto sta in ~1.060 token e il testo piu' denso che il
tokenizer incontra (CSV senza spazi, base64: ~1,5 caratteri per token) in ~1.700:
sopra i 768 del training, lontano dagli 8.192 dove la pipeline taglia.
"""

import re

MAX_WORDS = 120      # parole per chunk (~180 subword, sotto il MAX_LEN del training)
OVERLAP = 20         # parole di sovrapposizione tra chunk consecutivi
MAX_CHARS = 2500     # caratteri per chunk, qualunque sia la lunghezza delle parole
MAX_PEZZO = 200      # lunghezza dei pezzi di un campo che da solo e' troppo lungo

# un campo e i separatori che lo chiudono: '{"id":0,' '"nome":"Giulia",'
_CAMPO = re.compile(r"[^,;|]*[,;|]+|[^,;|]+")


def _pezzi(text):
    """[(start, end), ...] delle parole; quelle piu' lunghe di un chunk, spezzate."""
    out = []
    for w in re.finditer(r"\S+", text):
        if w.end() - w.start() <= MAX_CHARS:
            out.append((w.start(), w.end()))
            continue
        for c in _CAMPO.finditer(text, w.start(), w.end()):
            for s in range(c.start(), c.end(), MAX_PEZZO):
                out.append((s, min(s + MAX_PEZZO, c.end())))
    return out


def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP, max_chars=MAX_CHARS):
    """Ritorna [(sottostringa, offset_char_globale), ...] senza tagliare parole
    (salvo quelle piu' lunghe di un chunk intero)."""
    words = _pezzi(text)
    if not words:
        return []
    chunks, i = [], 0
    while i < len(words):
        j = i + 1
        while (j < len(words) and j - i < max_words
               and words[j][1] - words[i][0] <= max_chars):
            j += 1
        start, end = words[i][0], words[j - 1][1]
        chunks.append((text[start:end], start))      # slice esatto -> offset diretti
        if j >= len(words):
            break
        # la stessa proporzione di sovrapposizione anche sui chunk corti: con 20 pezzi
        # fissi un chunk di 10 pezzi avanzerebbe di uno alla volta
        i = max(i + 1, j - min(overlap, (j - i) * overlap // max_words))
    return chunks
