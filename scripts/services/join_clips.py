"""Joins FLEURS clips with 0.5 s of silence until the recording reaches at least 311 s.

usage: python3 join_clips.py <clip directory> <dev.tsv> <joined.wav>
Writes <joined.wav> as 16-bit PCM and <joined.wav>.json with the clip names and total length.
"""
import os, sys, json, numpy as np, soundfile as sf
D, TSV, OUT = sys.argv[1:4]
ref = {c[1]: c[3] for c in (l.rstrip("\n").split("\t") for l in open(TSV, encoding="utf-8")) if len(c) > 3}
parts, names, rate, dur = [], [], None, 0.0
for f in sorted(os.listdir(D)):
    if f not in ref: continue
    a, r = sf.read(os.path.join(D, f), dtype="float32"); rate = rate or r
    parts += [a, np.zeros(int(r * 0.5), dtype="float32")]; names.append(f); dur += len(a) / r + 0.5
    if dur >= 311: break
sf.write(OUT, np.concatenate(parts), rate, subtype="PCM_16")
json.dump({"names": names, "audio_s": round(dur, 1)}, open(OUT + ".json", "w"))
print(len(names), "clips", round(dur, 1), "s")
