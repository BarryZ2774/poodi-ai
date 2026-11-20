#!/usr/bin/env python3 -I
"""Writes WorldTest/Resources/places.json: sample landmark buildings for "Colour my places".

Reads the bundled Overture buildings tiles (zoom 14) with tippecanoe-decode and,
for each landmark below, records the Overture building id, the ids of its
building parts and a centre point. Sample data only.
"""
import json, subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILES = os.path.join(ROOT, "WorldTest/Resources/sf-buildings.pmtiles")
OUT = os.path.join(ROOT, "WorldTest/Resources/places.json")

# (label, Overture building ids)
PLACES = [
    ("Transamerica Pyramid", ["f2170db5-bfaf-4473-8ff6-b7e858d2a1b5"]),
    ("Salesforce Tower", ["626a57f8-ccf3-497d-a764-266e91e9253d"]),
    ("Ferry Building", ["68b6339c-54c7-4736-a228-01345e78bcdd"]),
    ("Coit Tower", ["17e14c4b-63a1-4426-b667-76c9407d19cd"]),
    ("City Hall", ["7877a912-fb48-4329-baeb-e77d9ac9725d"]),
    ("Palace of Fine Arts", ["1a775ae1-a1c0-4f4c-b429-77b3e05d2306"]),
    ("Grace Cathedral", ["0770a15e-04b9-4b1b-8fec-955c2d885911"]),
    ("de Young Museum", ["17afd2d7-6b37-42b0-8515-e9c5e1732b02"]),
    ("SFMOMA", ["a7f4dc7e-2f13-4cc2-9251-d2afc04ca843"]),
    ("555 California Street", ["3225016c-52d4-4f8a-b306-f510f58d8dc5"]),
    ("Chase Center", ["f76afdc7-2884-4110-a565-c4090ba9b95f"]),
    ("Golden Gate Bridge towers", ["ae5b43e7-c025-4bbe-994c-1f018f56e141", "d79d767d-007f-45d8-8918-2cfe17ebbf64"]),
]

def coords(g):
    out = []
    def walk(c):
        if isinstance(c[0], (int, float)): out.append(c)
        else:
            for x in c: walk(x)
    walk(g["coordinates"])
    return out

def main():
    raw = subprocess.run(["tippecanoe-decode", "-z14", "-Z14", TILES], check=True, capture_output=True).stdout
    data = json.loads(raw)
    wanted = {bid: label for label, ids in PLACES for bid in ids}
    parts, pts, tile_ids = {}, {}, {}
    for tile in data["features"]:
        for layer in tile["features"]:
            name = layer["properties"]["layer"]
            for f in layer["features"]:
                p = f["properties"]
                bid = p.get("id") if name == "building" else p.get("building_id")
                if bid not in wanted: continue
                if name == "building_part": parts.setdefault(bid, set()).add(p["id"])
                # Numeric vector-tile feature id: what MapLibre's feature state is keyed on.
                if "id" in f: tile_ids.setdefault((name, bid), set()).add(str(f["id"]))
                pts.setdefault(bid, []).extend(coords(f["geometry"]))
    result = []
    for label, ids in PLACES:
        cs = [c for b in ids for c in pts.get(b, [])]
        if not cs:
            sys.exit(f"not found: {label}")
        result.append({
            "name": label,
            "buildingIds": ids,
            "partIds": sorted({x for b in ids for x in parts.get(b, ())}),
            "buildingTileIds": sorted({x for b in ids for x in tile_ids.get(("building", b), ())}),
            "partTileIds": sorted({x for b in ids for x in tile_ids.get(("building_part", b), ())}),
            "lon": round(sum(c[0] for c in cs) / len(cs), 6),
            "lat": round(sum(c[1] for c in cs) / len(cs), 6),
        })
    json.dump(result, open(OUT, "w"), indent=1)
    for r in result:
        print(f'{r["name"]:28} parts={len(r["partIds"]):3} tileIds={len(r["buildingTileIds"])}+{len(r["partTileIds"])} {r["lat"]},{r["lon"]}')

if __name__ == "__main__":
    main()
