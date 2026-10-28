"""Embedding latency of the Ollama tag bge-m3 configured with num_gpu 0.

usage: python3 embed_bench.py <text file>
Embeds a 40-character query and five 1,200-character chunks of the text file five times each.
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:11434"


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time(); out = urllib.request.urlopen(req, timeout=600).read(); return time.time() - t, json.loads(out)


text = open(sys.argv[1], encoding="utf-8").read()
chunks = [text[5000 + i * 1200:5000 + (i + 1) * 1200] for i in range(5)]
print("parameters:", post("/api/show", {"model": "bge-m3"})[1].get("parameters", "").replace("\n", "; "))
post("/api/embed", {"model": "bge-m3", "input": "warmup"})
query = "Bagaimana kartu memverifikasi PIN pengguna"[:40]
print("query 40 chars, s:", [round(post("/api/embed", {"model": "bge-m3", "input": query})[0], 3) for _ in range(5)])
print("chunk 1200 chars, s:", [round(post("/api/embed", {"model": "bge-m3", "input": c})[0], 3) for c in chunks])
ps = json.load(urllib.request.urlopen(URL + "/api/ps"))
print("size_vram:", [(m["name"], m.get("size_vram")) for m in ps["models"]])
