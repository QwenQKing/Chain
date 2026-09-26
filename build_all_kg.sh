
set -e

cd "$(dirname "$0")"

export API_PROVIDER="${API:-default}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY="*"
export no_proxy="*"

export BUILD_LLM_MODEL="gpt-4o-mini"
export EMBED_MODEL="text-embedding-3-small"
export EMBED_DIM="1536"
export LLM_MODEL="gpt-4o-mini"

mkdir -p datasets/knowledge_ai_futures
rm -f datasets/knowledge_ai_futures/*.jsonl
cp datasets/knowledge/ai_futures.jsonl datasets/knowledge_ai_futures/
HYPERGRAPH_DIR="datasets/KG/ai_futures" \
    python scripts/build_knowledge.py --knowledge-dir datasets/knowledge_ai_futures

mkdir -p datasets/knowledge_future_as_label
rm -f datasets/knowledge_future_as_label/*.jsonl
cp datasets/knowledge/future_as_label.jsonl datasets/knowledge_future_as_label/
HYPERGRAPH_DIR="datasets/KG/future_as_label" \
    python scripts/build_knowledge.py --knowledge-dir datasets/knowledge_future_as_label

mkdir -p datasets/knowledge_metaculus_bin
rm -f datasets/knowledge_metaculus_bin/*.jsonl
cp datasets/knowledge/metaculus_bin.jsonl datasets/knowledge_metaculus_bin/
HYPERGRAPH_DIR="datasets/KG/metaculus_bin" \
    python scripts/build_knowledge.py --knowledge-dir datasets/knowledge_metaculus_bin

mkdir -p datasets/knowledge_polymarket
rm -f datasets/knowledge_polymarket/*.jsonl
cp datasets/knowledge/polymarket.jsonl datasets/knowledge_polymarket/
HYPERGRAPH_DIR="datasets/KG/polymarket" \
    python scripts/build_knowledge.py --knowledge-dir datasets/knowledge_polymarket

python merge_knowledge.py

HYPERGRAPH_DIR="datasets/KG/all_knowledge" \
    python scripts/build_knowledge.py --knowledge-dir datasets/knowledge_all

echo ""
echo "═══════════════════════════════════════════════════════"
echo " CHAIN "
echo "═══════════════════════════════════════════════════════"
