"""
core/config.py
--------------
Single source of truth for loading YAML configs and building SymbolSpec
objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from core.simulator import SymbolSpec

log = logging.getLogger(__name__)

# Fallback spec used when a symbol is not found in symbols.yaml
_EURUSD_DEFAULTS = {
    "pip_value":            0.0001,
    "pip_location":         4,
    "contract_size":        100_000,
    "typical_spread_pips":  1.0,
    "min_lot":              0.01,
    "max_lot":              100.0,
    "margin_requirement":   0.01,
}


def load_config(path: str = "config/config.yaml") -> dict:
    """Load and return the main YAML config as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_symbol_spec(symbols_config_path: str, symbol: str) -> SymbolSpec:
    """
    Build a SymbolSpec from symbols.yaml for the given symbol name.

    Strips timeframe suffixes (e.g. EURUSD_H1 → EURUSD) before lookup.
    Falls back to EURUSD defaults with a warning if the symbol is absent.
    """
    symbol_base = symbol.upper().split("_")[0]
    path        = Path(symbols_config_path)

    spec_raw = None
    if path.exists():
        with open(path) as f:
            cfg = yaml.safe_load(f)
        spec_raw = cfg.get("symbols", {}).get(symbol_base)

    if spec_raw is None:
        log.warning(
            "Symbol %s not found in %s — using EURUSD defaults.",
            symbol_base, symbols_config_path,
        )
        spec_raw = _EURUSD_DEFAULTS

    return SymbolSpec(
        name          = symbol_base,
        pip_value     = float(spec_raw["pip_value"]),
        pip_location  = int(spec_raw["pip_location"]),
        contract_size = int(spec_raw["contract_size"]),
        spread_pips   = float(spec_raw.get("typical_spread_pips", 1.0)),
        min_lot       = float(spec_raw["min_lot"]),
        max_lot       = float(spec_raw["max_lot"]),
        margin_rate   = float(spec_raw.get("margin_requirement", 0.01)),
    )
