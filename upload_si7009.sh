#!/bin/bash
# upload_si7009.sh
# Sube el código SI7009 a GCS para que el Workbench lo pueda descargar.
# Ejecutar desde la máquina local (donde está el código).
#
# Uso:
#   bash upload_si7009.sh

set -euo pipefail

BUCKET="gs://shiftmetrics-bronze"
GCS_CODE_PATH="${BUCKET}/code/SI7009"
LOCAL_DIR="$(dirname "$0")/SI7009"

echo "Subiendo SI7009 → ${GCS_CODE_PATH}..."
gsutil -m cp -r "${LOCAL_DIR}/"* "${GCS_CODE_PATH}/"

echo ""
echo "✅ Código disponible en ${GCS_CODE_PATH}"
echo ""
echo "En el Workbench, ejecuta:"
echo "  gsutil -m cp -r ${GCS_CODE_PATH}/* ~/shiftmetrics/SI7009/"
echo "  cd ~/shiftmetrics/SI7009 && python run_pipeline.py"
