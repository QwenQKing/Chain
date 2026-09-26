
set -u
cd "$(dirname "$0")"
mkdir -p logs

export API_PROVIDER="${API:-default}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*"
export no_proxy="*"

export BUILD_LLM_MODEL="${BUILD_LLM_MODEL:-gpt-4o-mini}"
export EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
export EMBED_DIM="${EMBED_DIM:-1536}"

export MAX_OUTER_LOOPS="${MAX_OUTER_LOOPS:-1}"
export CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"
export CHAIN_SKIP_VIZ_EMBED="${CHAIN_SKIP_VIZ_EMBED:-true}"
export PYTHONUNBUFFERED=1

DEFAULT_MODELS="gpt-4o-mini deepseek-v3 gemini-2.0-flash"
read -r -a MODELS_ARR <<<"${MODELS:-$DEFAULT_MODELS}"

ID_DATASETS=(ai_futures polymarket metaculus_bin future_as_label)
OOD_DATASETS=(clinical_trial_ood forecast_ood golf_forecasting_ood kalshi_ood)

T_START=$(date +%s)

for M in "${MODELS_ARR[@]}"; do
  TAG=$(echo "$M" | tr '/' '_')
  echo "════════════════════════════════════════════════════════════"
  echo " Running CHAIN: $M"
  echo " Time: $(date)"
  echo "════════════════════════════════════════════════════════════"

  export LLM_MODEL="$M"

  for DS in "${ID_DATASETS[@]}"; do
    case $DS in
      future_as_label) W=32 ;;
      metaculus_bin)   W=64 ;;
      *)               W=128 ;;
    esac
    echo "  [$TAG] $DS (workers=$W)"
    CHAIN_EVAL_WORKERS=$W \
    HYPERGRAPH_DIR="datasets/KG/$DS" \
      python scripts/evaluate.py --test-file "datasets/eval/${DS}.jsonl"
  done

  for DS in "${OOD_DATASETS[@]}"; do
    echo "  [$TAG] $DS (workers=32, OOD)"
    CHAIN_EVAL_WORKERS=32 \
    HYPERGRAPH_DIR="datasets/KG/all_knowledge" \
      python scripts/evaluate.py --test-file "datasets/eval/${DS}.jsonl"
  done

  echo ""
done 2>&1 | tee "logs/chain_eval_$(date +%Y%m%d_%H%M%S).log"

T_END=$(date +%s)
ELAPSED=$(( (T_END - T_START) / 60 ))
echo ""
echo "════════════════════════════════════════════════════════════"
echo " CHAIN evaluation complete in ${ELAPSED} min."
echo "════════════════════════════════════════════════════════════"
