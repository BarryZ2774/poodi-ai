# Landmark models: the brief

Read this whole page before you write anything. It is the contract a model is checked against.

Everything here lives in the `landmarks/` folder of the app's repository, and every path below starts there. Change nothing outside `landmarks/`.

## What this is

A phone map app draws San Francisco as a pale 3D city from open map data. Most buildings come out fine as raised shapes. A few famous structures do not, and they get a model of their own. You write each model as code. Nothing is sculpted by hand and nothing is downloaded.

The app runs on phones and must stay smooth, so models are small and plain. The app colours them itself: pale by default, a warm colour once a person has been there.

## The order

Build **one** model first: `golden-gate-bridge`. It is judged by eye before anything else is ordered. Do not start the others.

Next in line, in this order, only when asked: `bay-bridge-west-span`, `alcatraz`. On hold until a legal check: `sfmoma`.

## What you hand in for a landmark

Everything for one landmark lives in `models/<slug>/`:

| File | What it is |
|---|---|
| `build.py` | The script that writes the model from numbers. Running it again gives the same file. |
| `<slug>.glb` | The model the script wrote, committed. |
| `facts.json` | Name, anchor, measurements with sources (see below). |
| `preview/front.png`, `side.png`, `three-quarter.png`, `top.png` | Pictures made by `tools/render.py`. Do not make your own renders by other means. |

Python 3.11 or newer. Keep dependencies few and list them in `requirements.txt` (numpy, trimesh and pygltflib are enough for most models).

## Rules for every model

1. **Format.** glTF 2.0 binary (`.glb`), one file, nothing external. No textures, no images, no animations, no cameras, no lights.
2. **Size and axes.** Real size, 1 unit = 1 metre. Y is up. +X is east and -Z is north, so the model already points the right way and the app never rotates it.
3. **Origin.** The origin sits at the anchor point given in `facts.json`, at ground level for a building and at mean sea level for a bridge. Nothing goes below zero except bridge piers, which may go 3 m under.
4. **Weight.** At most 6,000 triangles for a building and 25,000 for the Golden Gate Bridge. At most 400 KB per file. Fewer is better.
5. **Materials.** One mesh if you can, three at most. One material named `body`, and one optional material named `accent`. Both use base colour `#F4F1EC`, metallic 0, roughness 1. No vertex colours.
6. **Edges.** Hard edges where the real thing has edges, with normals exported. Only truly round things (cables, domes) are smooth.
7. **Clean geometry.** No holes you can see from outside, no faces inside the solid, no two faces on top of each other, normals pointing out.
8. **Nothing too thin.** No part thinner than 0.6 m, or it flickers and vanishes from far away. Draw cables and railings thicker than they are (main cables at least 1.2 m).
9. **The outline comes first.** What a person would sketch from memory must be there and in proportion. Detail smaller than about 2 m is left out. No cars, people, water, terrain, signs, text, logos or flags.
10. **True proportions.** Every main measurement comes from a published source and is listed in `facts.json` with its URL. The model must match each one within 2%.

## facts.json

```json
{
  "slug": "golden-gate-bridge",
  "name": "Golden Gate Bridge",
  "kind": "bridge",
  "anchor": {"lat": 0.0, "lon": 0.0, "what": "midpoint of the main span, at mean sea level"},
  "measurements_m": [
    {"what": "main span, tower centre to tower centre", "real": 0.0, "model": 0.0, "source": "https://..."}
  ],
  "checkpoints": [
    {"what": "San Francisco Tower centre", "lat": 0.0, "lon": 0.0, "min_height_m": 200, "within_m": 5}
  ],
  "simplified": ["what you left out or thickened, one line each"],
  "origin_of_work": "Written from published measurements for this project. No third-party mesh used."
}
```

`kind` is `bridge` or `building` and sets the triangle limit. `checkpoints` are real positions where the model must have geometry at least that tall. The check turns each one into model coordinates from the anchor and looks there.

## The Golden Gate Bridge

What must be recognisable:

- The two towers, with their stepped Art Deco shape: legs that narrow in steps, and the cross braces with open portals above the roadway.
- The roadway with its stiffening truss, as a simple long box. A hint of the truss pattern on its sides is welcome if the triangle limit allows.
- The two main cables with the correct sag, running over both tower tops down to the anchorages.
- The vertical ropes, hinted. Do not model every one. A rope every 15 to 20 m, drawn thick enough to see, reads well.
- The side spans and approach spans to both shores, the pylons, both anchorages, and the steel arch over Fort Point at the south end.

Measurements to confirm on the bridge district's own pages and cite. These are the values I expect, in metres: total length with approaches 2,737; main span 1,280; each side span 343; tower height above the water 227; tower height above the roadway 152; roadway width 27; clearance under the roadway 67; main cable diameter 0.92 (drawn thicker, see rule 8). If the official page says otherwise, the official page wins and you say so in the pull request.

Position. The model must sit on the real bridge. Take the two tower positions from OpenStreetMap, where they are mapped as buildings named "San Francisco Tower" and "Marin Tower", and make the anchor the midpoint between them. With the model placed at the anchor, each tower centre must land within 5 m of its OpenStreetMap footprint. For reference, open map data puts the South Anchorage near 37.80944, -122.47735 and the North Anchor near 37.82904, -122.47969.

The open map data already draws the two towers (22 stacked parts each, up to 225 m). The app hides those and shows your model in their place, so your towers must stand in exactly the same spots.

## How work is handed in

- One branch and one pull request per landmark, named `model/<slug>`.
- The pull request text gives: triangle count, file size, the table of real against model measurements, the list of what you simplified, and the four preview pictures.
- `tools/check.py models/<slug>` must pass before you ask for review. It runs on every push and tests rules 1 to 8, the 2% rule, and the tower positions.
- Passing the check is not approval. The pictures are judged by eye for one thing: would a person who knows the place name it at a glance? Expect two or three rounds.

## Not allowed

- Any mesh, outline or measurement taken from Google, Apple or Mapbox 3D data, or from a paid or downloaded model library.
- Meshes made by an AI image-to-3D or text-to-3D generator. The model must come from your own script.
- Changing `tools/` or this brief inside a model's pull request, or touching any file outside `landmarks/`. If a rule is wrong, say so in the pull request text.
