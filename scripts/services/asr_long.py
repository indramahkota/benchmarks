"""Qwen3-ASR on one long recording made by join_clips.py.

usage: python3 asr_long.py <clip directory> <dev.tsv> <joined.wav> <one short clip for warmup>
Reads <joined.wav>.json written by join_clips.py, loads the model with the short clip first, then
times the long recording and scores it against the joined reference texts. Server on :5005.
"""
import json, re, subprocess, sys, threading, time
import jiwer
D, TSV, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ref = {c[1]: c[3] for c in (l.rstrip("\n").split("\t") for l in open(TSV, encoding="utf-8")) if len(c) > 3}
meta = json.load(open(OUT + ".json")); texts = [ref[n] for n in meta["names"]]; dur = meta["audio_s"]
peak, stop = [0], threading.Event()
def sample():
    while not stop.is_set():
        peak[0] = max(peak[0], int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])); time.sleep(0.2)
threading.Thread(target=sample, daemon=True).start()
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()
subprocess.check_output(["curl", "-s", "-F", f"file=@{sys.argv[4]}", "http://127.0.0.1:5005/v1/audio/transcriptions"])  # warm: load model
t = time.time(); out = subprocess.check_output(["curl", "-s", "-F", f"file=@{OUT}", "http://127.0.0.1:5005/v1/audio/transcriptions"]); dt = time.time() - t
stop.set()
print(json.dumps({"clips": len(texts), "audio_s": round(dur, 1), "seconds": round(dt, 1), "wer": round(jiwer.wer(norm(" ".join(texts)), norm(json.loads(out)["text"])) * 100, 2), "peak_mib": peak[0]}))
