#!/usr/bin/env python3 -I
"""Slims Overture building GeoJSON (one feature per line) for tippecanoe.

Usage: prepare_buildings.py <in.geojsonseq> <out.geojsonseq> building|building_part

Keeps only the properties the porcelain style and "Colour my places" read. The Overture
id (a 36-character string) is replaced by a numeric feature id, the feature's line number
in the input file: the strings made up more than half of the tiles. MapLibre feature state
uses the numeric id, and tools/make_places.py uses the same line-number rule.
Building parts keep `building_id` (few of them; the filtered fallback layer reads it).
Underground buildings are dropped: the style never draws them.
"""
import json, sys

KEEP = {
    "building": ["height", "min_height", "num_floors", "min_floor", "has_parts"],
    "building_part": ["building_id", "height", "min_height", "num_floors", "min_floor"],
}

def main():
    src, dst, kind = sys.argv[1:4]
    keep = KEEP[kind]
    kept = dropped = 0
    with open(src) as fin, open(dst, "w") as fout:
        for n, line in enumerate(fin, 1):
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            f = json.loads(line)
            p = f.get("properties") or {}
            if p.get("is_underground"):
                dropped += 1
                continue
            props = {k: p[k] for k in keep if p.get(k) is not None}
            if kind == "building" and not props.get("has_parts"):
                props.pop("has_parts", None)
            fout.write(json.dumps({"type": "Feature", "id": n, "geometry": f["geometry"], "properties": props},
                                  separators=(",", ":")) + "\n")
            kept += 1
    print(f"{kind}: kept {kept}, dropped {dropped} underground", file=sys.stderr)

if __name__ == "__main__":
    main()
