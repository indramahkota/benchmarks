#!/usr/bin/env bash
# Qwen3.5-4B on Ollama, llama.cpp, vLLM and SGLang, one server at a time.
# Before each runtime it stops llama-server on :8080, every vLLM and SGLang process, and every loaded Ollama model.
set -u
cd "${BENCH_WORKDIR:-.}"
mkdir -p logs
LOG=logs/run_qwen3.5-4b.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
stop_own() {
  pkill -f 'llama-server.*--port 8080'; pkill -f 'vllm serve'; pkill -f 'VLLM::'
  pkill -f 'sglang.launch_server'; pkill -f 'sglang::'
  ollama ps | tail -n +2 | awk '{print $1}' | xargs -r -n1 ollama stop
  sleep 8
  nvidia-smi --query-gpu=memory.used,temperature.gpu --format=csv,noheader | tee -a $LOG
}
bench_all() {
  python3 bench.py --url "$1" --model "$2" --label "$3" --runs 3 --conc 1 4 --out results-4b.jsonl | tee -a $LOG
  python3 agent_bench.py --url "$1" --model "$2" --label "$3" | tee -a $LOG
}
say "versions"; { ollama --version; llama.cpp/build/bin/llama-server --version 2>&1 | head -3; envs/vllm/bin/vllm --version; envs/sglang/bin/python -c 'import sglang;print(sglang.__version__)'; nvidia-smi --query-gpu=driver_version --format=csv,noheader; } 2>&1 | tee -a $LOG
stop_own; say "ollama"
bench_all http://127.0.0.1:11434 qwen3.5-4b-32k "ollama"
stop_own; say "llama.cpp ub2048"
llama.cpp/build/bin/llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --port 8080 --host 127.0.0.1 -ngl 99 --n-cpu-moe 0 -fa on -c 32768 \
  -ctk q8_0 -ctv q8_0 -b 4096 -ub 2048 -t 7 --parallel 4 --kv-unified --jinja > server.log 2>&1 &
for i in $(seq 180); do curl -sf localhost:8080/health >/dev/null && break; sleep 2; done
bench_all http://127.0.0.1:8080 x "llama.cpp ub2048 par4"
stop_own; say "vllm"
./vllm_try.sh --max-model-len 32768 --kv-cache-memory-bytes 1500000000 --max-num-seqs 16 \
  && bench_all http://127.0.0.1:8000 x "vllm kv1.5G seqs16"
stop_own; say "sglang"
./sglang_try.sh --mem-fraction-static 0.85 --max-running-requests 4 \
  --cuda-graph-max-bs-decode 4 --disable-prefill-cuda-graph --chunked-prefill-size 1024 \
  --mamba-radix-cache-strategy no_buffer --mamba-full-memory-ratio 1.0 --disable-overlap-schedule \
  && BENCH_COOL_LIMIT=68 bench_all http://127.0.0.1:30000 x "sglang mem0.85 mrr4 no_buffer, cool<=68C"
stop_own; say "done"
