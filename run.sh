# Reorganization run script

# Usage (from repo root):
# 1. Install requirements: pip install -r requirements.txt
# 2. Unzip the data: unzip data/data.zip -d data/
# 3. Run the desired pipeline script via the src/ wrappers, for example:
#    python3 src/confidence_tune.py
#    python3 src/deploy_final.py
#    python3 src/make_figures.py

set -e
if [ "$1" = "all" ]; then
  echo "Running full pipeline (confidence_tune -> deploy_final -> metal_confidence -> classifier_confidence -> xgb_confidence -> tabpfn_in_stack -> build_workbook2 -> build_slides -> make_figures)"
  python3 src/confidence_tune.py
  python3 src/deploy_final.py
  python3 src/metal_confidence.py
  python3 src/classifier_confidence.py
  python3 src/xgb_confidence.py
  python3 src/tabpfn_in_stack.py || true
  python3 src/build_workbook2.py
  python3 src/build_slides.py
  python3 src/make_figures.py
else
  echo "Run specific script: python3 src/<script>.py or use 'all' to run the pipeline"
fi
