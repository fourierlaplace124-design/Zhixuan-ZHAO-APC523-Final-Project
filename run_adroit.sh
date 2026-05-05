#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p figures

mode="${1:-verify}"

usage() {
    cat <<'EOF'
Usage: bash run_adroit.sh <mode>

Modes:
  compile     Syntax and bytecode check only
  mms         Method of manufactured solutions verification
  stability   Stability-region figure generation
  smooth      Smooth-region convergence analysis
  reynolds    Reynolds-number error-localization analysis
  enrichment  Corner rational-enrichment analysis
  verify      compile + mms + stability
  full        compile + all analysis scripts
EOF
}

case "$mode" in
    compile)
        python -m compileall -q .
        ;;
    mms)
        python mms_verification.py
        ;;
    stability)
        python plot_stability_region.py
        ;;
    smooth)
        python run_smooth_region_convergence.py
        ;;
    reynolds)
        python run_reynolds_corner_density.py
        ;;
    enrichment)
        python demo_corner_rational_enrichment.py
        ;;
    verify)
        python -m compileall -q .
        python mms_verification.py
        python plot_stability_region.py
        ;;
    full)
        python -m compileall -q .
        python mms_verification.py
        python plot_stability_region.py
        python run_smooth_region_convergence.py
        python run_reynolds_corner_density.py
        python demo_corner_rational_enrichment.py
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage
        exit 2
        ;;
esac
