"""
prepare_demand.py — Build demand baselines from Chicago taxi data.

Downloads trips from the Chicago Data Portal (Taxi Trips, dataset wrvz-psew),
filters to Chicago Loop pickup locations, assigns each trip to the nearest of
the 16 demand zones, and computes 15-min-bin demand statistics across weeks.

Output
------
chicago_loop/data/demand/historical_baselines.json
  {
    "mean":  {"weekday": {"Z01": [bins×96], ...}, "weekend": {...}},
    "std":   {"weekday": {"Z01": [bins×96], ...}, "weekend": {...}},
    "meta":  {...}
  }

Usage
-----
    python prepare_demand.py                   # download 12 weeks, 2023
    python prepare_demand.py --weeks 4         # smaller download for testing
    python prepare_demand.py --no-download     # generate realistic synthetic baseline
    python prepare_demand.py --csv my.csv      # use pre-downloaded CSV
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "demand"
NET_FILE  = SCRIPT_DIR.parent / "data" / "network" / "node_mappings.json"
OUT_FILE  = DATA_DIR / "historical_baselines.json"

# ---------------------------------------------------------------------------
# Chicago Loop bounding box (matches net.xml orig_boundary)
# ---------------------------------------------------------------------------

LOOP_BBOX = {
    "lat_min": 41.866010,
    "lat_max": 41.889789,
    "lon_min": -87.640499,
    "lon_max": -87.613343,
}

BINS_PER_DAY   = 96    # 24 h × 4 bins/h
BIN_MINUTES    = 15.0

# ---------------------------------------------------------------------------
# Chicago Data Portal — Taxi Trips dataset
# ---------------------------------------------------------------------------

SOCRATA_ENDPOINT = "https://data.cityofchicago.org/resource/wrvz-psew.json"
PAGE_LIMIT       = 10_000   # rows per Socrata request


def _build_soql(year: int, offset: int, limit: int) -> str:
    """Build SoQL query string for one page of Loop taxi pickups."""
    where = (
        f"trip_start_timestamp >= '{year}-01-01T00:00:00' "
        f"AND trip_start_timestamp < '{year+1}-01-01T00:00:00' "
        f"AND pickup_centroid_latitude  >= {LOOP_BBOX['lat_min']} "
        f"AND pickup_centroid_latitude  <= {LOOP_BBOX['lat_max']} "
        f"AND pickup_centroid_longitude >= {LOOP_BBOX['lon_min']} "
        f"AND pickup_centroid_longitude <= {LOOP_BBOX['lon_max']}"
    )
    select = "trip_start_timestamp,pickup_centroid_latitude,pickup_centroid_longitude"
    params = urllib.parse.urlencode({
        "$select": select,
        "$where": where,
        "$limit": limit,
        "$offset": offset,
        "$order": "trip_start_timestamp",
    })
    return f"{SOCRATA_ENDPOINT}?{params}"


def download_trips(year: int, max_rows: int = 200_000) -> list[dict]:
    """Fetch taxi pickups from Chicago Data Portal for the given year."""
    rows: list[dict] = []
    offset = 0
    logger.info(f"Downloading Chicago taxi data (year={year}, max_rows={max_rows}) …")
    while len(rows) < max_rows:
        url = _build_soql(year, offset, PAGE_LIMIT)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                page = json.loads(resp.read())
        except Exception as e:
            logger.error(f"Download failed at offset {offset}: {e}")
            break
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        logger.info(f"  fetched {len(rows):,} rows …")
        if len(page) < PAGE_LIMIT:
            break
    logger.info(f"Downloaded {len(rows):,} trips total.")
    return rows


def load_csv_trips(csv_path: Path) -> list[dict]:
    """
    Load trips from a pre-downloaded CSV.
    Expected columns (Chicago taxi export format):
      Trip Start Timestamp, Pickup Centroid Latitude, Pickup Centroid Longitude
    """
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts   = row.get("Trip Start Timestamp") or row.get("trip_start_timestamp", "")
            lat  = row.get("Pickup Centroid Latitude") or row.get("pickup_centroid_latitude", "")
            lon  = row.get("Pickup Centroid Longitude") or row.get("pickup_centroid_longitude", "")
            if ts and lat and lon:
                rows.append({
                    "trip_start_timestamp": ts,
                    "pickup_centroid_latitude": lat,
                    "pickup_centroid_longitude": lon,
                })
    logger.info(f"Loaded {len(rows):,} rows from {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Zone assignment — regular 4×4 grid tessellation
# Grid constants are read from node_mappings.json (written by network_builder.py)
# ---------------------------------------------------------------------------

_GRID: dict = {}   # populated by load_grid_params()


def load_grid_params() -> list[str]:
    """Load grid parameters and zone_ids from node_mappings.json."""
    if not NET_FILE.exists():
        raise FileNotFoundError(f"{NET_FILE} not found — run network_builder.py first.")
    with open(NET_FILE) as f:
        data = json.load(f)
    _GRID.update(data["grid"])
    zone_ids = [z["zone_id"] for z in data["demand_zones"]]
    return zone_ids


def grid_zone(lat: float, lon: float) -> str:
    """
    Assign a pickup (lat, lon) to a zone by its 4×4 grid cell.

    Grid: col increases west→east, row increases south→north.
    zone_id = Z{row*4+col+1:02d}
    """
    col = int((lon - _GRID["lon_min"]) / _GRID["lon_step"])
    row = int((lat - _GRID["lat_min"]) / _GRID["lat_step"])
    col = max(0, min(_GRID["n_cols"] - 1, col))
    row = max(0, min(_GRID["n_rows"] - 1, row))
    return f"Z{row * _GRID['n_cols'] + col + 1:02d}"


# ---------------------------------------------------------------------------
# Demand aggregation
# ---------------------------------------------------------------------------

def bin_index(ts: str) -> tuple[str, int] | None:
    """
    Parse ISO timestamp string → (day_type, bin_idx).
    day_type: 'weekday' or 'weekend'
    bin_idx : 0–95 (15-min bins from midnight)
    Returns None on parse failure.
    """
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(ts, fmt)
            break
        except ValueError:
            continue
    else:
        return None

    day_type = "weekend" if dt.weekday() >= 5 else "weekday"
    b = (dt.hour * 60 + dt.minute) // int(BIN_MINUTES)
    return day_type, b


def aggregate_trips(
    rows: list[dict],
    zones: list[dict],
    weeks: int,
) -> dict:
    """
    Aggregate raw trip rows into per-zone, per-bin weekly counts.

    Returns
    -------
    counts: {day_type: {zone_id: {bin_idx: [week_count, ...]}}}
    """
    logger.info("Assigning trips to demand zones …")
    # counts[day_type][zone_id][bin] -> list of counts per week
    # We index weeks by ISO week number derived from trip timestamp
    counts: dict[str, dict[str, dict[int, dict[int, float]]]] = {
        "weekday": defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
        "weekend": defaultdict(lambda: defaultdict(lambda: defaultdict(float))),
    }

    skipped = 0
    for row in rows:
        ts  = row.get("trip_start_timestamp", "")
        lat_s = row.get("pickup_centroid_latitude", "")
        lon_s = row.get("pickup_centroid_longitude", "")
        try:
            lat = float(lat_s)
            lon = float(lon_s)
        except (ValueError, TypeError):
            skipped += 1
            continue

        result = bin_index(ts)
        if result is None:
            skipped += 1
            continue
        day_type, b = result

        # Determine ISO week number for this trip
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y %I:%M:%S %p",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(ts, fmt)
                break
            except ValueError:
                continue
        else:
            skipped += 1
            continue

        week_num = dt.isocalendar()[1]

        zone_id = nearest_zone(lat, lon, zones)
        counts[day_type][zone_id][b][week_num] += 1.0

    if skipped:
        logger.warning(f"Skipped {skipped:,} rows due to missing/unparseable fields.")

    return counts


def compute_statistics(
    counts: dict,
    zone_ids: list[str],
    n_weeks_weekday: int,
    n_weeks_weekend: int,
) -> dict:
    """
    Compute per-bin mean and std from week-level counts.
    Units: trips per 15-min bin (per control step).
    """
    result = {
        "mean": {"weekday": {}, "weekend": {}},
        "std":  {"weekday": {}, "weekend": {}},
    }
    for day_type in ("weekday", "weekend"):
        n_w = n_weeks_weekday if day_type == "weekday" else n_weeks_weekend
        for z in zone_ids:
            mean_bins = []
            std_bins  = []
            for b in range(BINS_PER_DAY):
                week_counts = list(counts[day_type][z][b].values())
                # Pad with zeros for weeks where zone had no trips in this bin
                if len(week_counts) < n_w:
                    week_counts += [0.0] * (n_w - len(week_counts))
                arr = np.array(week_counts[:n_w], dtype=float)
                mean_bins.append(float(np.mean(arr)))
                std_bins.append(float(np.std(arr, ddof=1) if len(arr) > 1 else 0.5))
            result["mean"][day_type][z] = mean_bins
            result["std"][day_type][z]  = std_bins
    return result


# ---------------------------------------------------------------------------
# Synthetic realistic baseline (fallback / --no-download)
# ---------------------------------------------------------------------------

# Per-zone demand scale factors derived from Chicago Loop activity patterns:
#   Financial core (Z09-Z12) → high workday demand
#   Parks/cultural (Z01,Z07,Z08,Z13,Z14) → moderate, weekend-boosted
#   Transit hubs (Z15,Z16) → peaked at rush hours
#   Mixed (Z02-Z06) → intermediate
_ZONE_SCALE = {
    "Z01": 0.85,   # Millennium Park — tourist/leisure
    "Z02": 1.10,   # Theater District
    "Z03": 0.95,   # River North Gateway
    "Z04": 0.80,   # Merchandise Mart
    "Z05": 1.20,   # Daley Plaza — dense office
    "Z06": 1.05,   # Block 37
    "Z07": 0.75,   # Cultural Mile (N)
    "Z08": 0.70,   # Cultural Mile (S)
    "Z09": 1.30,   # LaSalle St (N) — financial core
    "Z10": 1.25,   # LaSalle St (S)
    "Z11": 1.00,   # Wacker Drive (E)
    "Z12": 1.00,   # Wacker Drive (W)
    "Z13": 0.80,   # Grant Park North
    "Z14": 0.65,   # Grant Park South
    "Z15": 1.40,   # Union Station — major transit hub
    "Z16": 1.35,   # Ogilvie / Metra
}

def _diurnal(hour: float, peaks: list[tuple[float, float]], base: float) -> float:
    """
    Multi-peak Gaussian diurnal profile.
    peaks: list of (peak_hour, amplitude)
    """
    val = base
    for ph, amp in peaks:
        val += amp * math.exp(-0.5 * ((hour - ph) / 1.2) ** 2)
    return max(val, base * 0.3)


def generate_synthetic_baseline(zone_ids: list[str], rng_seed: int = 0) -> dict:
    """
    Generate a realistic synthetic demand baseline for Chicago Loop.

    Weekday: morning peak 7-9am + afternoon peak 5-7pm
    Weekend: flatter profile with midday leisure peak
    """
    logger.info("Generating synthetic demand baseline …")
    rng = np.random.default_rng(rng_seed)
    BASE = 1.5   # baseline trips/15-min
    N_SYNTHETIC_WEEKS = 12

    result = {
        "mean": {"weekday": {}, "weekend": {}},
        "std":  {"weekday": {}, "weekend": {}},
    }

    for z in zone_ids:
        scale = _ZONE_SCALE.get(z, 1.0)
        for day_type in ("weekday", "weekend"):
            if day_type == "weekday":
                peaks = [(8.0, 4.0 * scale), (17.5, 3.5 * scale), (12.5, 1.5 * scale)]
            else:
                peaks = [(11.5, 2.5 * scale), (14.0, 2.0 * scale), (19.0, 2.0 * scale)]

            mean_bins, std_bins = [], []
            for b in range(BINS_PER_DAY):
                hour = b * BIN_MINUTES / 60.0
                mu = _diurnal(hour, peaks, BASE * scale)
                # Simulate N_SYNTHETIC_WEEKS of noisy weekly observations
                weekly = rng.poisson(mu, size=N_SYNTHETIC_WEEKS).astype(float)
                # Add small inter-week variation (±20%)
                weekly *= rng.uniform(0.8, 1.2, size=N_SYNTHETIC_WEEKS)
                mean_bins.append(float(np.mean(weekly)))
                std_bins.append(float(np.std(weekly, ddof=1)))

            result["mean"][day_type][z] = mean_bins
            result["std"][day_type][z]  = std_bins

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Chicago taxi demand baselines")
    parser.add_argument("--year",        type=int, default=2023)
    parser.add_argument("--weeks",       type=int, default=12,
                        help="Target number of weeks of data to pull (approx)")
    parser.add_argument("--max-rows",    type=int, default=300_000)
    parser.add_argument("--csv",         type=Path, default=None,
                        help="Path to pre-downloaded Chicago taxi CSV")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download; generate synthetic baseline instead")
    parser.add_argument("--output",      type=Path, default=OUT_FILE)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zones = load_zone_centroids()
    zone_ids = [z["zone_id"] for z in zones]

    # ------------------------------------------------------------------
    # Acquire trip data
    # ------------------------------------------------------------------
    if args.no_download:
        stats = generate_synthetic_baseline(zone_ids)
        meta = {"source": "synthetic", "zones": len(zone_ids), "bins_per_day": BINS_PER_DAY}

    else:
        if args.csv:
            rows = load_csv_trips(args.csv)
        else:
            rows = download_trips(args.year, max_rows=args.max_rows)

        if not rows:
            logger.warning("No trip data obtained — falling back to synthetic baseline.")
            stats = generate_synthetic_baseline(zone_ids)
            meta = {"source": "synthetic_fallback", "zones": len(zone_ids), "bins_per_day": BINS_PER_DAY}
        else:
            counts = aggregate_trips(rows, zones, weeks=args.weeks)
            # Estimate number of distinct weeks in data
            n_wday = max(1, args.weeks * 5 // 7)
            n_wend = max(1, args.weeks * 2 // 7)
            stats = compute_statistics(counts, zone_ids, n_wday, n_wend)
            meta = {
                "source": "chicago_data_portal",
                "year": args.year,
                "n_trips": len(rows),
                "target_weeks": args.weeks,
                "zones": len(zone_ids),
                "bins_per_day": BINS_PER_DAY,
                "bin_minutes": BIN_MINUTES,
            }

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    out = {**stats, "meta": meta}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved: {args.output}")

    # Quick sanity print
    wday_z01 = stats["mean"]["weekday"].get("Z01", [])
    if wday_z01:
        peak_bin = max(range(BINS_PER_DAY), key=lambda b: wday_z01[b])
        peak_hour = peak_bin * BIN_MINUTES / 60
        logger.info(f"Z01 weekday peak: bin {peak_bin} ({peak_hour:.1f}h), "
                    f"mean={wday_z01[peak_bin]:.2f} trips/15min")


if __name__ == "__main__":
    main()
