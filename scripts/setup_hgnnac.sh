#!/usr/bin/env bash
# Clone the official HGNN-AC code (used unchanged; DGL is replaced by the small stand-in in dgl_shim/).
set -e
cd "$(dirname "$0")/.."
COMMIT=cca61fd237b55f14f2ce802680316a26719c1696
mkdir -p third_party
if [ ! -d third_party/HGNN-AC/model ]; then
  git clone https://github.com/liangchundong/HGNN-AC.git third_party/HGNN-AC
  git -C third_party/HGNN-AC checkout "$COMMIT"
fi
echo "HGNN-AC ready: third_party/HGNN-AC @ $COMMIT"
