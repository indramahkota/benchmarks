"""Qwen3-ASR on FLEURS clips: WER, CER, latency per clip and peak card memory.

usage: python3 asr_bench.py <clip directory> <dev.tsv>
Sends every clip listed in dev.tsv to the server on :5005, one at a time. The first request
includes the model load. Text is lowercased and stripped of punctuation before scoring with jiwer.
"""
import json, os, re, subprocess, sys, threading, time
import jiwer
D, TSV, URL = sys.argv[1], sys.argv[2], "http://127.0.0.1:5005/v1/audio/transcriptions"
ref = {}
for line in open(TSV, encoding="utf-8"):
    c = line.rstrip("\n").split("\t")
    if len(c) > 3: ref[c[1]] = c[3]
peak, stop = [0], threading.Event()
def sample():
    while not stop.is_set():
        v = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
        peak[0] = max(peak[0], v); time.sleep(0.2)
threading.Thread(target=sample, daemon=True).start()
norm = lambda t: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()
hyps, refs, times = [], [], []
for f in sorted(os.listdir(D)):
    if f not in ref: continue
    t = time.time()
    out = subprocess.check_output(["curl", "-s", "-F", f"file=@{os.path.join(D, f)}", URL])
    times.append(time.time() - t)
    hyps.append(norm(json.loads(out)["text"])); refs.append(norm(ref[f]))
stop.set()
print(json.dumps({"clips": len(times), "wer": round(jiwer.wer(refs, hyps) * 100, 2), "cer": round(jiwer.cer(refs, hyps) * 100, 2),
  "first_s": round(times[0], 2), "median_after_first_s": round(sorted(times[1:])[len(times[1:]) // 2], 2),
  "mean_after_first_s": round(sum(times[1:]) / len(times[1:]), 2), "peak_mib": peak[0]}))
