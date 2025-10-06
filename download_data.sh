set -euo pipefail

OUT_DIR="${1:-multi30k_en-de_raw}"
BASE_URL="https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

FILES=(
  "train.en.gz"
  "train.de.gz"
  "val.en.gz"
  "val.de.gz"
  "test_2016_flickr.en.gz"
  "test_2016_flickr.de.gz"
)

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "Downloading Multi30k raw EN–DE files to $(pwd)"
for f in "${FILES[@]}"; do
  echo "  - $f"
  curl -L --fail -o "$f" "$BASE_URL/$f"
  gunzip -f "$f"
done

# Rename test files to a simpler name
mv test_2016_flickr.en test.en
mv test_2016_flickr.de test.de

echo "Done. Expected line counts (approx):"
# train: 29000, valid: 1014, test: 1000 (per Multi30k stats)
wc -l train.en train.de val.en val.de test.en test.de
