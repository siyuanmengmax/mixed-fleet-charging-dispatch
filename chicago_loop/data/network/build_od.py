"""
Network distances between all demand zones and charging stations (km), from the SUMO
network of the Chicago Loop.

The OSM extract is not fully connected (the largest strongly connected component of the
passenger network has 476 of 1,008 junctions). Locations outside it are snapped to the
nearest junction of that component, and the straight-line snap distance is added at both
ends of a trip.

  python -m chicago_loop.data.network.build_od      # writes od_km.json next to this file
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import sumolib

HERE = Path(__file__).parent


def build() -> dict:
    net = sumolib.net.readNet(str(HERE / "chicago_loop.net.xml"))
    nm = json.load(open(HERE / "node_mappings.json"))
    locs = ([(z["zone_id"], z["node_id"], z["node_x"], z["node_y"]) for z in nm["demand_zones"]]
            + [(s["station_id"], s["node_id"], s["node_x"], s["node_y"]) for s in nm["charging_stations"]])
    g = nx.DiGraph()
    for e in net.getEdges():
        if e.allows("passenger"):
            g.add_edge(e.getFromNode().getID(), e.getToNode().getID(), length=e.getLength())
    big = max(nx.strongly_connected_components(g), key=len)
    xy = {n: net.getNode(n).getCoord() for n in big}
    snap = {}
    for loc, node, x, y in locs:
        best = node if node in big else min(big, key=lambda m: math.dist(xy[m], (x, y)))
        snap[loc] = (best, math.dist(xy[best], (x, y)) if node not in big else 0.0)
    od = {}
    for a, (na, sa) in snap.items():
        dist = nx.single_source_dijkstra_path_length(g, na, weight="length")
        od[a] = {b: round((sa + dist[nb] + sb) / 1000.0, 4) if a != b else 0.0
                 for b, (nb, sb) in snap.items()}
    return {"od_km": od,
            "snapped": {a: {"node": n, "snap_m": round(d, 1)} for a, (n, d) in snap.items() if d > 0}}


if __name__ == "__main__":
    out = build()
    (HERE / "od_km.json").write_text(json.dumps(out, indent=1))
    d = [v for a in out["od_km"].values() for v in a.values() if v > 0]
    print(f"{len(out['od_km'])} locations, {len(d)} pairs, distance {min(d):.2f}-{max(d):.2f} km "
          f"(mean {sum(d) / len(d):.2f}); snapped: {out['snapped']}")
