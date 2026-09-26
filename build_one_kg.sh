set -eu

cd "$(dirname "$0")"

DATASET="${1:-ai_futures}"
case "$DATASET" in
  ai_futures|future_as_label|metaculus_bin|polymarket) ;;
  *)
    echo "unsupported dataset: $DATASET" >&2
    exit 2
    ;;
esac

SOURCE="datasets/knowledge/${DATASET}.jsonl"
INPUT_DIR="datasets/knowledge_${DATASET}"
GRAPH_DIR="datasets/KG/${DATASET}"

if [ ! -f "$SOURCE" ]; then
  echo "knowledge file does not exist: $SOURCE" >&2
  exit 1
fi

mkdir -p "$INPUT_DIR"
rm -f "$INPUT_DIR"/*.jsonl
cp "$SOURCE" "$INPUT_DIR/"

export API_PROVIDER="${API_PROVIDER:-${API:-default}}"
export BUILD_LLM_MODEL="${BUILD_LLM_MODEL:-gpt-4o-mini}"
export EMBED_MODEL="${EMBED_MODEL:-text-embedding-3-small}"
export EMBED_DIM="${EMBED_DIM:-1536}"
export LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
export HYPERGRAPH_DIR="$GRAPH_DIR"

python scripts/build_knowledge.py \
  --knowledge-dir "$INPUT_DIR" \
  --force
