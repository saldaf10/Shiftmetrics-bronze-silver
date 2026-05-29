#!/bin/bash
# upload_ml.sh
# Sube el código del pipeline ML a GCS para que el Workbench lo pueda descargar.
# Ejecutar desde la raíz del repositorio en la máquina local.
#
# Uso:
#   bash infra/upload_ml.sh

set -euo pipefail

BUCKET="gs://shiftmetrics-bronze"
GCS_CODE_PATH="${BUCKET}/code/ml"
LOCAL_DIR="$(dirname "$0")/../ml"

echo "Subiendo ml/ → ${GCS_CODE_PATH}..."
gsutil -m cp -r "${LOCAL_DIR}/"* "${GCS_CODE_PATH}/"

echo ""
echo "✅ Código disponible en ${GCS_CODE_PATH}"
echo ""
echo "En el Workbench, ejecuta:"
echo "  gsutil -m cp -r ${GCS_CODE_PATH}/* ~/shiftmetrics/ml/"
echo "  cd ~/shiftmetrics/ml && python run_pipeline.py"
