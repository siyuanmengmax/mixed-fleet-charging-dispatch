"""
demand_model.py — historical zonal demand of the Chicago Loop (data/demand/historical_baselines.json,
built by scripts/prepare_demand.py from the City of Chicago Taxi Trips data) and the forecaster
used by the dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DEMAND_FILE = Path(__file__).parent.parent / "data" / "demand" / "historical_baselines.json"


class DemandModel:
    """Loads historical baselines and produces Poisson demand per 5-min step."""

    BINS_PER_DAY = 96          # 15-min bins
    CTRL_PER_BIN = 3           # 3 control steps per 15-min bin (5 min each)

    def __init__(self):
        with open(_DEMAND_FILE) as f:
            raw = json.load(f)
        self._means: dict[str, dict[str, list]] = raw["mean"]    # [day_type][zone][96]
        self._stds:  dict[str, dict[str, list]] = raw["std"]

    def demand_at_bin(
        self,
        zone_id: str,
        bin_idx: int,
        day_type: str = "weekday",
        scale_to_5min: bool = True,
    ) -> float:
        """Return mean demand trips per period (5-min if scale_to_5min else 15-min)."""
        v = self._means.get(day_type, {}).get(zone_id, [0.0] * 96)
        val = v[bin_idx % self.BINS_PER_DAY]
        return val / self.CTRL_PER_BIN if scale_to_5min else val

    def uncertainty_at_bin(
        self,
        zone_id: str,
        bin_idx: int,
        kappa: float = 1.5,
        day_type: str = "weekday",
        scale_to_5min: bool = True,
    ) -> float:
        """ξ_j(k) = κ × σ_j^hist(bin)."""
        v = self._stds.get(day_type, {}).get(zone_id, [0.0] * 96)
        std = v[bin_idx % self.BINS_PER_DAY]
        val = kappa * std
        return val / self.CTRL_PER_BIN if scale_to_5min else val

    def sample_demand(
        self,
        zone_id: str,
        bin_idx: int,
        rng: np.random.Generator,
        day_type: str = "weekday",
    ) -> int:
        """Poisson-sample demand for zone at given bin (5-min scaled)."""
        lam = self.demand_at_bin(zone_id, bin_idx, day_type)
        return int(rng.poisson(lam)) if lam > 0 else 0


class SimpleDemandForecaster:
    """Forecast of zonal demand: historical bin mean, blended with the requests observed now
    (weight `smoothing`), and the historical uncertainty xi = kappa * sigma."""

    def __init__(self, demand_model: DemandModel, zone_ids: list[str],
                 start_bin: int = 28, day_type: str = "weekday",
                 kappa: float = 1.5, smoothing: float = 0.2):
        self._dm = demand_model
        self._zone_ids = zone_ids
        self._start_bin = start_bin
        self._day_type = day_type
        self._kappa = kappa
        self._smoothing = smoothing
        self._last_obs: dict[str, float] = {z: 0.0 for z in zone_ids}

    def update(self, observed: dict[str, float]):
        self._last_obs = dict(observed)

    def forecast(self, step: int, start_bin: int) -> dict[str, float]:
        cur_bin = start_bin + step // 3
        return {
            z: (1 - self._smoothing) * self._dm.demand_at_bin(z, cur_bin, self._day_type)
               + self._smoothing * self._last_obs.get(z, 0.0)
            for z in self._zone_ids
        }

    def forecast_next(self, step: int, start_bin: int) -> dict[str, float]:
        nxt_bin = start_bin + (step + 1) // 3
        return {z: self._dm.demand_at_bin(z, nxt_bin, self._day_type) for z in self._zone_ids}

    def uncertainty(self, step: int, start_bin: int) -> dict[str, float]:
        cur_bin = start_bin + step // 3
        return {
            z: self._dm.uncertainty_at_bin(z, cur_bin, self._kappa, self._day_type)
            for z in self._zone_ids
        }
