#!/usr/bin/env bash
# Linux/macOS: ./run_night.sh [hours] [run-name]
cd "$(dirname "$0")" || exit 1
python -m astana_pilot.training.night --hours "${1:-8}" --run-name "${2:-night1}"
