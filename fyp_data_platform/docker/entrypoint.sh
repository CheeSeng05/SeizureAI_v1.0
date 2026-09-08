#!/bin/sh
set -eu

if [ "${FYP_REQUIRE_CUDA:-0}" = "1" ]; then
  python -c "import torch; assert torch.cuda.is_available(), 'CUDA was requested but PyTorch cannot access an NVIDIA GPU'; print('CUDA ready:', torch.cuda.get_device_name(0), '| PyTorch', torch.__version__, '| CUDA', torch.version.cuda)"
fi

python -m fyp_data_platform db-upgrade
exec gunicorn \
  --bind 0.0.0.0:8080 \
  --workers 1 \
  --threads 8 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  "fyp_data_platform.ui:create_app()"
