#!/usr/bin/env python3
"""Streaming benchmark for OpenAI-compatible /v1/completions endpoints.

Measures per request: TTFT, prefill tok/s, decode tok/s. Also aggregate
throughput when N requests run concurrently. Stdlib only.
"""
import argparse, json, statistics, sys, threading, time, urllib.request

FILLER = open(__file__.replace("bench.py", "prompt.txt")).read()


def build_prompt(target_words: int) -> str:
    words = FILLER.split()
    body = " ".join((words * (target_words // len(words) + 1))[:target_words])
    return (
        "<|im_start|>user\nBerikut adalah konteks kode:\n"
        + body
        + "\n\nTulis implementasi lengkap LRU cache thread-safe di Python "
        "beserta unit test yang sangat detail.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def one_request(url, model, prompt, max_tokens, out, cache=False):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": True}, "cache_prompt": cache,
    }).encode()
    req = urllib.request.Request(url + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and not d["choices"][0].get("finish_reason"):
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
                chunks += 1
    n_out = usage.get("completion_tokens") or chunks
    n_in = usage.get("prompt_tokens", 0)
    ttft = t_first - t0
    dec = (n_out - 1) / (t_last - t_first) if t_last > t_first else 0
    out.append({"ttft": ttft, "prefill_tps": n_in / ttft if ttft else 0,
                "decode_tps": dec, "n_in": n_in, "n_out": n_out,
                "total": t_last - t0})


def cooldown(limit=None):
    """Block until GPU temperature drops to `limit` C (laptop GPU throttles hard).

    Default 55 C; override with BENCH_COOL_LIMIT for servers that keep the GPU
    warm while idle (SGLang busy-polls, GPU stays in P0 at ~65 C).
    """
    import os, subprocess
    limit = limit or int(os.environ.get("BENCH_COOL_LIMIT", 55))
    while int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader"]).split()[0]) > limit:
        time.sleep(3)


def run(url, model, prompt, max_tokens, conc):
    cooldown()
    res, ths = [], []
    t0 = time.perf_counter()
    for _ in range(conc):
        th = threading.Thread(target=one_request,
                              args=(url, model, prompt, max_tokens, res))
        th.start(); ths.append(th)
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0
    return res, sum(r["n_out"] for r in res) / wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt-words", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conc", type=int, nargs="*", default=[1])
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()
    prompt = build_prompt(a.prompt_words)
    run(a.url, a.model, prompt[:2000], 16, 1)  # warmup (loads model)
    for c in a.conc:
        rows = []
        for i in range(a.runs):
            # vary prompt per run so no prefix cache is reused
            res, agg = run(a.url, a.model, f"[run {c}-{i}-{time.time()}] " + prompt,
                           a.max_tokens, c)
            rows.append((res, agg))
        flat = [r for res, _ in rows for r in res]
        s = {
            "label": a.label, "model": a.model, "conc": c,
            "n_in": flat[0]["n_in"], "n_out": flat[0]["n_out"],
            "ttft_s": statistics.median(r["ttft"] for r in flat),
            "prefill_tps": statistics.median(r["prefill_tps"] for r in flat),
            "decode_tps": statistics.median(r["decode_tps"] for r in flat),
            "agg_tps": statistics.median(agg for _, agg in rows),
        }
        print(json.dumps(s), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
