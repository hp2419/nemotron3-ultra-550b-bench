#!/usr/bin/env bash
# guidellm concurrency sweep against a running vLLM server.
#
#   ./bench.sh short  tuned      # 512/256,   c=8..300,  300s per level
#   ./bench.sh long   tuned      # 14000/5000, c=8..300, 900s per level
#
# Note: guidellm's new CLI uses `guidellm run` with kind=key,value arguments.
# `guidellm benchmark` does not exist; --constraint max_duration takes `seconds=`, not `value=`.
set -euo pipefail

SHAPE=${1:?usage: bench.sh <short|long> <label>}
LABEL=${2:?usage: bench.sh <short|long> <label>}
TARGET=${TARGET:-http://127.0.0.1:8000}
OUTDIR=${OUTDIR:-/workspace/results}

case "$SHAPE" in
  short) ISL=512;   OSL=256;  LEVELS="8,16,32,64,128,200,300"; DUR=300 ;;
  long)  ISL=14000; OSL=5000; LEVELS="8,32,128,300";           DUR=900 ;;
  *) echo "unknown shape: $SHAPE" >&2; exit 1 ;;
esac

mkdir -p "$OUTDIR"

guidellm run \
  --backend kind=openai_http,target="$TARGET" \
  --data kind=synthetic_text,prompt_tokens=$ISL,output_tokens=$OSL \
  --profile kind=concurrent,streams=8 \
  --override 'profile.streams' "$LEVELS" \
  --constraint kind=max_duration,seconds=$DUR \
  --output kind=json,path="$OUTDIR/${LABEL}_${ISL}_${OSL}.json"
