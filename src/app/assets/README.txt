Asset grafici dell'app (mascotte rizzo-pii, il riccio).

File attesi (salva qui i PNG):
  mascot_idle.png   il riccio in idle      -> logo header + favicon
  mascot_doc.png    il riccio col documento -> empty state (Risultato / Ripristina)

Serviti da Flask su /assets/<file> (vedi route in app.py).
Se un file manca, l'app mostra un'emoji di fallback (nessun errore).
Inclusi nell'exe tramite build.spec: datas += [("src/app/assets", "assets")].
