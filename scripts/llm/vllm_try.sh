#!/usr/bin/env bash
# usage: vllm_try.sh [extra vllm args]: starts vLLM on :8000 and waits for health
cd "${BENCH_WORKDIR:-.}"
sp=$PWD/envs/vllm/lib/python3.12/site-packages/nvidia/cu13
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH=$PWD/envs/vllm/bin:$sp/bin:/usr/bin:/bin CUDA_HOME=$sp
vllm serve models/Qwen3.5-4B-AWQ --served-model-name x --port 8000 \
  --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder "$@" > logs/vllm.log 2>&1 &
pid=$!
for i in $(seq 450); do
  curl -sf localhost:8000/health >/dev/null && { echo UP; exit 0; }
  kill -0 $pid 2>/dev/null || { echo DIED; exit 1; }
  sleep 2
done; echo TIMEOUT; exit 1
