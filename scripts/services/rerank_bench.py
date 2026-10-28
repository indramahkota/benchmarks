"""Reranker latency for 1, 5, 10, 20 and 40 candidates of 1,200 characters.

usage: python3 rerank_bench.py <English text file> <directory of Indonesian Markdown files>
Needs three llama-server instances of bge-reranker-v2-m3 Q8_0 started with
--reranking --pooling rank --ctx-size 8192 --batch-size 8192 --ubatch-size 8192 --parallel 1:
  :5002  -ngl 99                                   (GPU)
  :5012  -ngl 0 -t 16, with CUDA_VISIBLE_DEVICES=  (CPU, 16 threads)
  :5013  -ngl 0 -t 8,  with CUDA_VISIBLE_DEVICES=  (CPU, 8 threads)
A CUDA build started with -ngl 0 but with the GPU visible still offloads batched matrix work to the
GPU, so the CPU servers must not see the card. Median of three runs after one warmup.
"""
import json, re, statistics, time, urllib.request, sys, glob
def post(port, path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); out = urllib.request.urlopen(r, timeout=900).read(); return time.time() - t, json.loads(out)
def chunks(text, n):
    text = re.sub(r"\s+", " ", text); return [text[5000 + i*1200: 5000 + (i+1)*1200] for i in range(n)]
en = chunks(open(sys.argv[1], encoding="utf-8").read(), 40)
idt = chunks(" ".join(re.sub(r"^---.*?---", "", open(f, encoding="utf-8").read(), flags=re.S) for f in sorted(glob.glob(sys.argv[2] + "/*.md"))), 40)
sets = {"en": (en, "How does the card application authenticate a PIN?"),
        "id": (idt, "Bagaimana cara mengukur latensi layanan di GPU?"),
        "mix": ([c for p in zip(en[:20], idt[:20]) for c in p], "Bagaimana kartu memverifikasi PIN pengguna?")}
for name, (docs, q) in sets.items():
    toks = [len(post(5002, "/tokenize", {"content": d})[1]["tokens"]) for d in docs[:20]]
    print(f"[{name}] tokens per candidate: median {statistics.median(toks)}, min {min(toks)}, max {max(toks)}")
    for port, dev in ((5002, "gpu"), (5012, "cpu16"), (5013, "cpu8")):
        row = []
        for n in (1, 5, 10, 20, 40):
            if dev == "cpu8" and n != 1: continue
            if dev != "gpu" and n == 40: continue
            post(port, "/v1/rerank", {"query": q, "documents": docs[:n]})
            row.append(f"{n}:{round(statistics.median(post(port, '/v1/rerank', {'query': q, 'documents': docs[:n]})[0] for _ in range(3)) * 1000)}ms")
        print(f"  {dev}: " + "  ".join(row))
