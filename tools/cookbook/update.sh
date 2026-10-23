#!/usr/bin/env bash
# Обновляет profiles.default.json из SGLang cookbook (github.com/sgl-project/sglang, docs/src/snippets/configs).
# Использование: tools/cookbook/update.sh [путь к profiles.json]   (по умолчанию — profiles.default.json в корне)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$here/../../profiles.default.json}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/configs"
base="https://raw.githubusercontent.com/sgl-project/sglang/main/docs/src/snippets/configs"
for p in \
  ai-sage/gigachat3.5 ai-sage/gigachat3.5-benchmarks \
  zai-org/glm-5.2 zai-org/glm-5.2-benchmarks zai-org/glm-5.3 zai-org/glm-5.3-benchmarks \
  zai-org/glm-5.3-flash zai-org/glm-5.3-flash-benchmarks \
  Qwen/qwen3.8 Qwen/qwen3.8-benchmarks Qwen/qwen3.8-flash-next Qwen/qwen3.8-flash-next-benchmarks \
  Qwen/qwen3.8-27b Qwen/qwen3.8-27b-benchmarks \
  deepseek-ai/deepseek-v4 deepseek-ai/deepseek-v4-benchmarks deepseek-ai/deepseek-v4_1 \
  moonshotai/kimi-k3 moonshotai/kimi-k3-benchmarks \
  MiniMaxAI/minimax-m3 MiniMaxAI/minimax-m3-benchmarks MiniMaxAI/minimax-m3.1; do
  curl -fsSL "$base/$p.jsx" -o "$work/configs/$(echo "$p" | tr / _).jsx" || echo "нет $p — пропускаю" >&2
done
(cd "$work" && python3 -I "$here/extract.py" >/dev/null)
python3 -I "$here/build_profiles.py" "$work/cookbook.json" "$out"
echo "Готово: $out. Ручные профили (DeepSeek-V4.1-Flash из issue #41946, GigaChat 3.5, Mistral, Kimi-K2.6) добавляются скриптом build_profiles.py."
