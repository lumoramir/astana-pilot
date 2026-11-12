#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
python -m astana_pilot.training.night --smoke
