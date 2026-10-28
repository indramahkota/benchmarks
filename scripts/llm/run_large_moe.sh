#!/usr/bin/env bash
# Qwen3.6-35B-A3B and Qwen3-Coder-30B-A3B with experts on the CPU, on Ollama and llama.cpp.
# Before each run it stops llama-server on :8080 and every loaded Ollama model.
set -u
cd "${BENCH_WORKDIR:-.}"
mkdir -p logs
LOG=logs/run_large_moe.log
QC="${QWEN3_CODER_GGUF:?set QWEN3_CODER_GGUF to the Qwen3-Coder-30B-A3B Q4_K_M GGUF}"
Q36=models/Qwen3.6-35B-A3B-Q4_K_M.gguf
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
stop_own() {
  pkill -f 'llama-server.*--port 8080'
  ollama ps | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop
  sleep 10; nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | tee -a $LOG
}
bench_all() { # url model label
  python3 bench.py --url "$1" --model "$2" --label "$3" --runs 3 --conc 1 --out results.jsonl | tee -a $LOG
  python3 agent_bench.py --url "$1" --model "$2" --label "$3" | tee -a $LOG
}
ollama_run() { stop_own; say "ollama $1"; bench_all http://127.0.0.1:11434 "$1" "ollama"; }
llama_run() { # gguf ncmoe ub label
  stop_own; say "llama.cpp $4"
  llama.cpp/build/bin/llama-server -m "$1" --port 8080 --host 127.0.0.1 -ngl 99 --n-cpu-moe "$2" -fa on -c 32768 \
    -ctk q8_0 -ctv q8_0 -b 4096 -ub "$3" -t 7 --parallel 1 --jinja > server.log 2>&1 &
  for i in $(seq 240); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
  curl -sf localhost:8080/health >/dev/null || { say "FAILED start $4"; return 1; }
  bench_all http://127.0.0.1:8080 x "$4"
}
say "start"
ollama_run qwen3.6-32k
llama_run $Q36 36 4096 "llama.cpp ncmoe36 ub4096"
ollama_run qwen3-coder-32k
llama_run $QC 44 4096 "llama.cpp ncmoe44 ub4096"
stop_own; say "done"
