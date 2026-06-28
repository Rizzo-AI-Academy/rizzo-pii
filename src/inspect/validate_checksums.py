import json
from pathlib import Path

SYNTH_SAMPLE = str(Path(__file__).resolve().parents[2] / "dataset" / "synthetic" / "synthetic_pii_it.jsonl")

def iban_ok(i):
    r = i[4:] + i[:4]
    n = int("".join(str(ord(c) - 55) if c.isalpha() else c for c in r))
    return n % 97 == 1

def piva_ok(p):
    if len(p) != 11 or not p.isdigit():
        return False
    t = 0
    for i, c in enumerate(map(int, p[:10])):
        if i % 2 == 0:
            t += c
        else:
            x = c * 2
            t += x - 9 if x > 9 else x
    return (10 - t % 10) % 10 == int(p[10])

ODD = {"0":1,"1":0,"2":5,"3":7,"4":9,"5":13,"6":15,"7":17,"8":19,"9":21,
       "A":1,"B":0,"C":5,"D":7,"E":9,"F":13,"G":15,"H":17,"I":19,"J":21,
       "K":2,"L":4,"M":18,"N":20,"O":11,"P":3,"Q":6,"R":8,"S":12,"T":14,
       "U":16,"V":10,"W":22,"X":25,"Y":24,"Z":23}

def cf_ok(c):
    if len(c) != 16:
        return False
    b = c[:15]
    t = sum((ODD[ch] if i % 2 == 0 else (int(ch) if ch.isdigit() else ord(ch) - 65))
            for i, ch in enumerate(b))
    return chr(65 + t % 26) == c[15]

ib = cf = pi = nib = ncf = npi = 0
for line in open(SYNTH_SAMPLE, encoding="utf-8"):
    for e in json.loads(line)["entities"]:
        if e["label"] == "IBAN":
            nib += 1; ib += iban_ok(e["value"])
        elif e["label"] == "CF":
            ncf += 1; cf += cf_ok(e["value"])
        elif e["label"] == "PIVA":
            npi += 1; pi += piva_ok(e["value"])

print(f"IBAN validi: {ib}/{nib}")
print(f"CF   validi: {cf}/{ncf}")
print(f"PIVA validi: {pi}/{npi}")
