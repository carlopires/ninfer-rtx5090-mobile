#!/usr/bin/env bash
# Section 13 staged-context tester.
# Usage: ninfer-stage-test.sh <max-context> <kv-capacity> <spec|nomtp> [kv-dtype]
set -u

CTX="$1"
KV="$2"
SPEC="$3"
KVD="${4:-int8}"

NINFER_HOME="$HOME/code/ninfer"
MODEL="$HOME/models/ninfer/qwen3.8-27b-quasar/qwen3_8_27b_nvfp4.ninfer"
PORT=18080
LOG=/tmp/ninfer-stage-$(date +%s).log

ARGS=(
  "$MODEL" --host 127.0.0.1 --port "$PORT"
  --model-id qwen3.8-27b-quasar-nvfp4
  --max-concurrency 1
  --max-context "$CTX"
  --kv-capacity "$KV"
  --kv-dtype "$KVD"
  --device-state-slots 1
  --host-state-slots 2
  --host-kv-mib 2048
  --default-max-tokens 8192
)
if [[ "$SPEC" == "mtp" ]]; then
  ARGS+=(--spec mtp --draft-tokens 3 --lm-head-draft)
fi

nohup "$NINFER_HOME/build/apps/ninfer-serve" "${ARGS[@]}" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/ninfer-serve.pid
echo "launching ctx=$CTX kv=$KV spec=$SPEC kv-dtype=$KVD pid=$PID"

READY=0
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then READY=1; break; fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "RESULT: PROCESS_DIED"
    tail -25 "$LOG"
    exit 1
  fi
  sleep 2
done

if [[ "$READY" != 1 ]]; then
  echo "RESULT: TIMEOUT_WAITING_FOR_HEALTH"
  tail -25 "$LOG"
  kill "$PID" 2>/dev/null
  exit 1
fi

echo "READY"
echo "--- kv line from startup log ---"
grep 'KV capacity' "$LOG" | tail -1
echo "--- chat completion test ---"
RESP=$(curl -fsS --max-time 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-quasar-nvfp4","messages":[{"role":"user","content":"Reply with one short sentence confirming the API works."}],"max_tokens":128,"stream":false}')
if [[ $? -ne 0 ]]; then
  echo "RESULT: INIT_OK_CHAT_FAILED"
else
  echo "$RESP" | jq -r '"content:  " + .choices[0].message.content'
  echo "$RESP" | jq -r '"finish:   " + .choices[0].finish_reason'
  echo "$RESP" | jq -r '"usage:    prompt=\(.usage.prompt_tokens) completion=\(.usage.completion_tokens)"'
  echo "RESULT: OK"
fi

kill "$PID" 2>/dev/null
sleep 3
kill -0 "$PID" 2>/dev/null && { sleep 4; kill -9 "$PID" 2>/dev/null; }
wait "$PID" 2>/dev/null
echo "stopped"
