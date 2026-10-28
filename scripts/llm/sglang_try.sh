#!/usr/bin/env bash
# usage: sglang_try.sh [extra args]: starts SGLang on :30000 and waits for health
cd "${BENCH_WORKDIR:-.}"
sp=$PWD/envs/sglang/lib/python3.12/site-packages/nvidia/cu13
export PATH=$PWD/envs/sglang/bin:$sp/bin:/usr/bin:/bin CUDA_HOME=$sp LIBRARY_PATH=$sp/lib
python -m sglang.launch_server --model-path models/Qwen3.5-4B-AWQ --served-model-name x \
  --port 30000 --context-length 32768 --tool-call-parser qwen3_coder "$@" > logs/sglang.log 2>&1 &
pid=$!
for i in $(seq 450); do
  curl -sf localhost:30000/health >/dev/null && { echo UP; exit 0; }
  kill -0 $pid 2>/dev/null || { echo DIED; exit 1; }
  sleep 2
done; echo TIMEOUT; exit 1
