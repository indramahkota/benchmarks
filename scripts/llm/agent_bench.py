#!/usr/bin/env python3
"""Agentic-harness workload: multi-turn session with a long shared prefix,
plus a tool-calling validity check via /v1/chat/completions.
"""
import argparse, json, statistics, time, urllib.request
from bench import FILLER, cooldown, one_request

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Repo-relative path"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace old_string with new_string in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]}}},
]


def system_prompt(words):
    w = FILLER.split()
    ctx = " ".join((w * (words // len(w) + 1))[:words])
    return ("<|im_start|>system\nYou are a coding agent. Tools:\n"
            + json.dumps(TOOLS) + "\nRepository context:\n" + ctx + "<|im_end|>\n")


def session(url, model, turns, sys_words, max_tokens, tag):
    prompt = system_prompt(sys_words) + f"<|im_start|>user\n[{tag}] Fix the failing test.<|im_end|>\n"
    w = FILLER.split()
    ttfts = []
    for t in range(turns):
        full = prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        out = []
        cooldown()
        one_request(url, model, full, max_tokens, out, cache=True)
        ttfts.append(out[0]["ttft"])
        # append a synthetic assistant turn + tool result (~300 words)
        obs = " ".join(w[(t * 300) % len(w):][:300])
        prompt = (full + f"calling run_shell turn {t}<|im_end|>\n"
                  f"<|im_start|>user\n<tool_response>\n{obs}\n</tool_response><|im_end|>\n")
    return ttfts, out[0]["n_in"], out[0]["decode_tps"]


def tool_check(url, model, n=5):
    ok = 0
    for i in range(n):
        body = json.dumps({
            "model": model, "temperature": 0.0, "max_tokens": 512, "tools": TOOLS,
            "chat_template_kwargs": {"enable_thinking": False}, "think": False,
            "messages": [{"role": "user", "content": [
                "Show me the contents of src/main.py",
                "Run the test suite with pytest -q",
                "In app.py replace 'DEBUG = True' with 'DEBUG = False'",
                "List all files in the tests directory",
                "What does the file README.md say?"][i % 5]}],
        }).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        try:
            msg = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if calls and all(json.loads(c["function"]["arguments"]) is not None and
                             c["function"]["name"] in {t["function"]["name"] for t in TOOLS}
                             for c in calls):
                ok += 1
        except Exception as e:
            print("tool_check error:", e)
    return ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--sys-words", type=int, default=3500)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="results-agent.jsonl")
    a = ap.parse_args()
    ttfts, n_in, dec = session(a.url, a.model, a.turns, a.sys_words, a.max_tokens, time.time())
    ok, n = tool_check(a.url, a.model)
    s = {"label": a.label, "model": a.model, "turns": a.turns,
         "final_ctx": n_in, "ttft_first": ttfts[0],
         "ttft_later_median": statistics.median(ttfts[1:]),
         "session_ttft_total": sum(ttfts), "decode_tps_last": dec,
         "tool_calls_valid": f"{ok}/{n}"}
    print(json.dumps(s), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
