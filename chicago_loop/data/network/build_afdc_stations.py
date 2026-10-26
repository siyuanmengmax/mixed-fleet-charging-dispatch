"""
build_afdc_stations.py — station layout from the public DC fast-charging sites around the Chicago Loop.

Source: U.S. DOE Alternative Fuels Data Center station API (developer.nlr.gov), public, available
DC fast-charging sites within 8 miles of the Loop centre, retrieved 2026-10-26 and stored in
afdc_dcfc_chicago_raw.json:

  curl "https://developer.nlr.gov/api/alt-fuel-stations/v1/nearest.json?api_key=DEMO_KEY&fuel_type=ELEC
        &ev_charging_level=dc_fast&status=E&access=public&latitude=41.8818&longitude=-87.6290&radius=8&limit=200"

Selection: sites with a CCS (J1772COMBO) connector, outside the bounding box of the study area, nearest
to the Loop centre first; N_SITES of them, with their reported number of DC fast ports. Tesla-only sites
are excluded. The road network covers the Loop only, so the distance from zone a to site s is the
network distance to the zone node b that minimizes it plus the rectilinear distance from b to the site
(the streets around the Loop form a north-south / east-west grid):  d(a, s) = min_b [od(a, b) + |dx| + |dy|].

  python -m chicago_loop.data.network.build_afdc_stations   # writes afdc_stations.json
"""

from __future__ import annotations

import json
from pathlib import Path

from chicago_loop.data.network.network_builder import lonlat_to_sumo

HERE = Path(__file__).parent
N_SITES = 8
BBOX = dict(lon=(-87.640499, -87.613343), lat=(41.866010, 41.889789))   # study area (network_builder.py)


def main():
    raw = json.load(open(HERE / "afdc_dcfc_chicago_raw.json"))["fuel_stations"]
    nm = json.load(open(HERE / "node_mappings.json"))
    od = json.load(open(HERE / "od_km.json"))["od_km"]
    inside = lambda s: (BBOX["lon"][0] <= s["longitude"] <= BBOX["lon"][1]
                        and BBOX["lat"][0] <= s["latitude"] <= BBOX["lat"][1])
    ccs = [s for s in raw if "J1772COMBO" in (s.get("ev_connector_types") or []) and s.get("ev_dc_fast_num")]
    sel = sorted((s for s in ccs if not inside(s)), key=lambda s: s["distance"])[:N_SITES]
    zones = {z["zone_id"]: (z["node_x"], z["node_y"]) for z in nm["demand_zones"]}
    sites = []
    for k, s in enumerate(sel, 1):
        x, y = lonlat_to_sumo(s["longitude"], s["latitude"])
        sites.append(dict(station_id=f"af_{k}", afdc_id=s["id"], name=s["station_name"], network=s["ev_network"],
                          lon=s["longitude"], lat=s["latitude"], x=x, y=y, piles=int(s["ev_dc_fast_num"]),
                          km_from_centre=round(s["distance"] * 1.609344, 2)))
    rect = lambda p, q: (abs(p[0] - q[0]) + abs(p[1] - q[1])) / 1000.0
    rows = {}
    for st in sites:
        p = (st["x"], st["y"])
        leg = {b: rect(zones[b], p) for b in zones}
        d = {a: min(od[a][b] + leg[b] for b in zones) for a in zones}
        st["access_km"] = round(min(leg.values()), 2)        # from the nearest zone node
        rows[st["station_id"]] = d
    for st in sites:                                       # station to station: rectilinear
        rows[st["station_id"]].update({o["station_id"]: rect((st["x"], st["y"]), (o["x"], o["y"])) for o in sites})
    json.dump(dict(source="AFDC station API (developer.nlr.gov), retrieved 2026-10-26", stations=sites, od_km=rows),
              open(HERE / "afdc_stations.json", "w"), indent=1)
    for st in sites:
        print(f"{st['station_id']} {st['name'][:34]:34s} {st['network'][:16]:16s} ports {st['piles']:2d} "
              f"{st['km_from_centre']:.2f} km from centre, access {st['access_km']:.2f} km, "
              f"zones {min(rows[st['station_id']][z] for z in zones):.2f}-{max(rows[st['station_id']][z] for z in zones):.2f} km")


if __name__ == "__main__":
    main()
