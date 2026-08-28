#!/bin/sh
set -eu

python -m fyp_data_platform db-upgrade
exec gunicorn \
  --bind 0.0.0.0:8080 \
  --workers 1 \
  --threads 8 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  "fyp_data_platform.ui:create_app()"
