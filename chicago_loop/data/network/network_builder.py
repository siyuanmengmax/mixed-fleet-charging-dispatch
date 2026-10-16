"""
network_builder.py — Extract node mappings for demand zones and charging stations
from the Chicago Loop SUMO network.

Outputs:
    chicago_loop/data/network/node_mappings.json
    chicago_loop/data/network/network_summary.json

Usage:
    python network_builder.py [--net chicago_loop.net.xml] [--output node_mappings.json]
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Coordinate conversion: (lon, lat) → SUMO (x, y)
# Uses the UTM Zone 16N projection matching Chicago Loop net.xml
# ---------------------------------------------------------------------------

NET_OFFSET_X = -446851.06
NET_OFFSET_Y = -4635085.50

def lonlat_to_sumo(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 lon/lat to SUMO network x/y (metres)."""
    a = 6_378_137.0
    f = 1 / 298.257_223_563
    b = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    e = math.sqrt(e2)

    lon0 = math.radians(-87.0)   # UTM zone 16 central meridian
    lon_r = math.radians(lon)
    lat_r = math.radians(lat)

    N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = e2 / (1 - e2) * math.cos(lat_r) ** 2
    A = (lon_r - lon0) * math.cos(lat_r)

    M = a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64) * lat_r
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32) * math.sin(2 * lat_r)
        + (15 * e2 ** 2 / 256) * math.sin(4 * lat_r)
    )

    easting = 500_000 + 0.9996 * N * (
        A
        + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * e2 / (1 - e2)) * A ** 5 / 120
    )
    northing = 0.9996 * (
        M
        + N * math.tan(lat_r) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * e2 / (1 - e2)) * A ** 6 / 720
        )
    )

    sumo_x = easting + NET_OFFSET_X
    sumo_y = northing + NET_OFFSET_Y
    return sumo_x, sumo_y


# ---------------------------------------------------------------------------
# Demand zones — 4×4 regular grid tessellating the Chicago Loop
#
# Grid layout (col=0 west→east=3, row=0 south→north=3):
#   zone_id = Z{row*4+col+1:02d}
#
# Bbox: lon [-87.640499, -87.613343], lat [41.866010, 41.889789]
# Cell size: ~0.006789° lon × ~0.005945° lat  ≈  590 m × 662 m
# ---------------------------------------------------------------------------

GRID_LON_MIN  = -87.640499
GRID_LAT_MIN  =  41.866010
GRID_LON_STEP =  0.027156 / 4   # ≈ 0.006789°
GRID_LAT_STEP =  0.023779 / 4   # ≈ 0.005945°


@dataclass
class ZoneDef:
    zone_id: str
    name: str
    lon: float
    lat: float


def _grid_zone_defs() -> list[ZoneDef]:
    """Generate 16 zone centroids from a regular 4×4 grid over the Loop bbox."""
    names = {
        # row 0 (south): Z01-Z04
        (0, 0): "Union Station",
        (0, 1): "Printer's Row",
        (0, 2): "Museum Campus N",
        (0, 3): "Grant Park S",
        # row 1: Z05-Z08
        (1, 0): "Near West / Canal",
        (1, 1): "LaSalle / Adams",
        (1, 2): "State / Jackson",
        (1, 3): "Michigan Ave S",
        # row 2: Z09-Z12
        (2, 0): "Ogilvie-Metra",
        (2, 1): "Daley Plaza",
        (2, 2): "Block 37",
        (2, 3): "Millennium Park",
        # row 3 (north): Z13-Z16
        (3, 0): "River North W",
        (3, 1): "Merchandise Mart",
        (3, 2): "River North E",
        (3, 3): "Navy Pier Approach",
    }
    zones = []
    for row in range(4):
        for col in range(4):
            zone_id = f"Z{row * 4 + col + 1:02d}"
            lon = GRID_LON_MIN + (col + 0.5) * GRID_LON_STEP
            lat = GRID_LAT_MIN + (row + 0.5) * GRID_LAT_STEP
            zones.append(ZoneDef(zone_id, names[(row, col)], round(lon, 4), round(lat, 4)))
    return zones


DEMAND_ZONE_DEFS: list[ZoneDef] = _grid_zone_defs()


# ---------------------------------------------------------------------------
# Charging stations — 8 locations, distributed across the Loop
# Placed near parking garages / high-turnover areas
# ---------------------------------------------------------------------------

@dataclass
class StationDef:
    station_id: str
    name: str
    lon: float
    lat: float
    piles: int = 4
    power_kw: float = 50.0


STATION_DEFS: list[StationDef] = [
    StationDef("cs_1", "Millennium Park Garage",    -87.6215, 41.8820, piles=4),
    StationDef("cs_2", "Theater District Garage",   -87.6295, 41.8815, piles=4),
    StationDef("cs_3", "River North Parking",       -87.6355, 41.8872, piles=4),
    StationDef("cs_4", "Wacker-Lower Level",        -87.6275, 41.8862, piles=4),
    StationDef("cs_5", "Financial District (N)",    -87.6330, 41.8810, piles=4),
    StationDef("cs_6", "Financial District (S)",    -87.6325, 41.8760, piles=4),
    StationDef("cs_7", "Grant Park Garage",         -87.6195, 41.8760, piles=4),
    StationDef("cs_8", "Union Station Garage",      -87.6408, 41.8778, piles=4),
]


# ---------------------------------------------------------------------------
# Network loader
# ---------------------------------------------------------------------------

@dataclass
class Junction:
    node_id: str
    x: float
    y: float
    junction_type: str


def load_junctions(net_file: Path) -> list[Junction]:
    """Parse all non-internal junctions from SUMO net.xml."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    junctions = []
    for j in root.findall("junction"):
        if j.get("type") == "internal":
            continue
        junctions.append(Junction(
            node_id=j.get("id"),
            x=float(j.get("x")),
            y=float(j.get("y")),
            junction_type=j.get("type", ""),
        ))
    return junctions


def nearest_junction(
    target_x: float,
    target_y: float,
    junctions: list[Junction],
    exclude_types: set[str] = frozenset({"dead_end"}),
) -> Junction:
    """Return the closest junction (ignoring dead-ends by default)."""
    candidates = [j for j in junctions if j.junction_type not in exclude_types]
    if not candidates:
        candidates = junctions
    return min(
        candidates,
        key=lambda j: (j.x - target_x) ** 2 + (j.y - target_y) ** 2,
    )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(net_file: Path, output_file: Path):
    print(f"Loading network: {net_file}")
    junctions = load_junctions(net_file)
    print(f"  {len(junctions)} non-internal junctions loaded")

    zone_records = []
    station_records = []

    print("\n--- Demand Zones ---")
    used_nodes: set[str] = set()
    for zd in DEMAND_ZONE_DEFS:
        sx, sy = lonlat_to_sumo(zd.lon, zd.lat)
        # Exclude already-used nodes to avoid two zones mapping to same node
        candidates = [j for j in junctions
                      if j.junction_type not in {"dead_end"}
                      and j.node_id not in used_nodes]
        if not candidates:
            candidates = junctions
        jn = min(candidates, key=lambda j: (j.x - sx) ** 2 + (j.y - sy) ** 2)
        dist = math.sqrt((jn.x - sx) ** 2 + (jn.y - sy) ** 2)
        used_nodes.add(jn.node_id)
        print(f"  {zd.zone_id} {zd.name:<30} -> node {jn.node_id} "
              f"(dist={dist:.1f}m, type={jn.junction_type})")
        zone_records.append({
            "zone_id": zd.zone_id,
            "name": zd.name,
            "lon": zd.lon,
            "lat": zd.lat,
            "sumo_x": round(sx, 2),
            "sumo_y": round(sy, 2),
            "node_id": jn.node_id,
            "node_x": round(jn.x, 2),
            "node_y": round(jn.y, 2),
            "snap_dist_m": round(dist, 1),
        })

    print("\n--- Charging Stations ---")
    for sd in STATION_DEFS:
        sx, sy = lonlat_to_sumo(sd.lon, sd.lat)
        jn = nearest_junction(sx, sy, junctions)
        dist = math.sqrt((jn.x - sx) ** 2 + (jn.y - sy) ** 2)
        print(f"  {sd.station_id} {sd.name:<30} -> node {jn.node_id} "
              f"(dist={dist:.1f}m)")
        station_records.append({
            "station_id": sd.station_id,
            "name": sd.name,
            "lon": sd.lon,
            "lat": sd.lat,
            "sumo_x": round(sx, 2),
            "sumo_y": round(sy, 2),
            "node_id": jn.node_id,
            "node_x": round(jn.x, 2),
            "node_y": round(jn.y, 2),
            "snap_dist_m": round(dist, 1),
            "piles": sd.piles,
            "power_kw": sd.power_kw,
        })

    # ---------------------------------------------------------------------------
    # Quick sanity checks
    # ---------------------------------------------------------------------------
    print("\n--- Sanity checks ---")
    zone_nodes = [z["node_id"] for z in zone_records]
    station_nodes = [s["node_id"] for s in station_records]
    dup_zones = len(zone_nodes) - len(set(zone_nodes))
    dup_stations = len(station_nodes) - len(set(station_nodes))
    print(f"  Zone node duplicates   : {dup_zones}")
    print(f"  Station node duplicates: {dup_stations}")

    snap_dists = [z["snap_dist_m"] for z in zone_records + station_records]
    print(f"  Max snap distance      : {max(snap_dists):.1f}m")
    print(f"  Mean snap distance     : {sum(snap_dists)/len(snap_dists):.1f}m")

    # ---------------------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------------------
    out = {
        "network": {
            "file": str(net_file.name),
            "n_junctions": len(junctions),
            "bbox_sumo": {
                "x_min": round(min(j.x for j in junctions), 1),
                "x_max": round(max(j.x for j in junctions), 1),
                "y_min": round(min(j.y for j in junctions), 1),
                "y_max": round(max(j.y for j in junctions), 1),
            },
            "orig_boundary": "-87.640499,41.866010,-87.613343,41.889789",
        },
        "grid": {
            "lon_min": GRID_LON_MIN,
            "lat_min": GRID_LAT_MIN,
            "lon_step": round(GRID_LON_STEP, 6),
            "lat_step": round(GRID_LAT_STEP, 6),
            "n_cols": 4,
            "n_rows": 4,
        },
        "demand_zones": zone_records,
        "charging_stations": station_records,
        # Convenience dicts for direct use in the dispatch configuration
        "zone_nodes": {z["zone_id"]: z["node_id"] for z in zone_records},
        "station_nodes": {s["station_id"]: s["node_id"] for s in station_records},
        "zone_coords": {z["zone_id"]: {"x": z["node_x"], "y": z["node_y"]} for z in zone_records},
        "station_coords": {s["station_id"]: {"x": s["node_x"], "y": s["node_y"]} for s in station_records},
    }

    with open(output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {output_file}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        default=str(Path(__file__).parent / "chicago_loop.net.xml"),
        help="Path to SUMO .net.xml file",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "node_mappings.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()
    build(Path(args.net), Path(args.output))
