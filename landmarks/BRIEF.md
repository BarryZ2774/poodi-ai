# Landmark models: the brief

Read this whole page before you write anything. It is the contract a model is checked against.

Everything here lives in the `landmarks/` folder of the app's repository, and every path below starts there. Change nothing outside `landmarks/`.

## What this is

A phone map app draws San Francisco as a pale 3D city from open map data. Most buildings come out fine as raised shapes. A few famous structures do not, and they get a model of their own. You write each model as code. Nothing is sculpted by hand and nothing is downloaded.

The app runs on phones and must stay smooth, so models are small and plain. The app colours them itself: pale by default, a warm colour once a person has been there.

## The order

The list of models and their order comes with the ask. Each model is one pull request. `golden-gate-bridge` was the first.

On hold until a rights check, do not build: the Transamerica Pyramid, Salesforce Tower, SFMOMA, the de Young Museum, Oracle Park, Chase Center, and any building finished after 1990.

## What you hand in for a landmark

Everything for one landmark lives in `models/<slug>/`:

| File | What it is |
|---|---|
| `build.py` | The script that writes the model from numbers. Running it again gives the same file. |
| `<slug>.glb` | The model the script wrote, committed. |
| `facts.json` | Name, anchor, measurements with sources (see below). |
| `preview/front.png`, `side.png`, `three-quarter.png`, `top.png` | Pictures made by `tools/render.py`, 1600 x 1000 as it writes them. Do not make your own renders by other means. |

Python 3.11 or newer. Keep dependencies few and list them in `requirements.txt`, which stays the same on every model branch (numpy and one mesh library are enough for most models). What only one model needs goes into `models/<slug>/requirements.txt`. The tools have their own list in `tools/requirements.txt`.

## Rules for every model

1. **Format.** glTF 2.0 binary (`.glb`), one file, nothing external. No textures, no images, no animations, no cameras, no lights. No glTF extensions: positions and normals are plain 32-bit floats, so the simplest loader can read the file.
2. **Size and axes.** Real size, 1 unit = 1 metre. Y is up. +X is east and -Z is north, so the model already points the right way and the app never rotates it.
3. **Origin.** The origin sits at the anchor point given in `facts.json`, at ground level for a building and at mean sea level for a bridge. Nothing goes below zero except bridge piers, which may go 3 m under. The map is flat: ground and water are both at zero and there are no hills. Nothing may float: every support reaches zero, and a roadway that ends does so on an abutment that reaches zero.
4. **Weight.** At most 6,000 triangles for a building and 25,000 for a bridge. At most 400 KB per file for a building and 1,500 KB for a bridge (1 KB = 1,000 bytes). Fewer is better.
5. **Materials.** One mesh if you can, three at most. One material named `body`, and one optional material named `accent`. Both use base colour `#F4F1EC`, metallic 0, roughness 1. No vertex colours.
6. **Edges.** Hard edges where the real thing has edges, with normals exported. Only truly round things (cables, domes) are smooth.
7. **Clean geometry.** No holes you can see from outside, no faces inside the solid, no two faces on top of each other, normals pointing out.
8. **Nothing too thin.** No bar, wall, cable or fin thinner than 0.6 m along its length, or it flickers and vanishes from far away. Draw cables and railings thicker than they are (main cables at least 1.2 m). A spire or a roof may still end in a point.
9. **The outline comes first.** What a person would sketch from memory must be there and in proportion. Detail smaller than about 2 m is left out. No cars, people, water, terrain, signs, text, logos or flags. The one exception to "no terrain" is `alcatraz`, where the rock is the landmark.
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

Two optional marks on a measurement help the check. `"bbox": "height"`, `"length"` or `"width"` says the measurement is the model's own overall size, and the check then measures the model itself (height is the top above the origin; length and width are the sides of the smallest rectangle around the footprint). `"thickened": true` marks a part that rule 8 makes you draw thicker than it is, such as a main cable; the 2% rule does not apply to it, but the model value must not be smaller than the real one.

`kind` is `bridge` or `building` and sets the triangle and file size limits. `checkpoints` are real positions where the model must have geometry at least that tall. The check turns each one into model coordinates from the anchor and looks there. A bridge needs at least two checkpoints at least 100 m apart, so that both its position and its direction are tested.

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

## Buildings the open map data already draws

Most landmarks other than the bridges and Alcatraz already stand in the open map data as plain stacked boxes, and the model replaces them. For these: `kind` is `building`. Take the building's outline from OpenStreetMap with one small request, keep the model's ground outline within 3 m of it, and anchor on the outline's centre. Say in `facts.json` that the outline comes from OpenStreetMap (© OpenStreetMap contributors, ODbL). Add at least one checkpoint at the highest point. Cite a source for the height and the main sizes.

## How work is handed in

- One branch and one pull request per landmark, named `model/<slug>`.
- The pull request text gives: triangle count, file size, the table of real against model measurements, the list of what you simplified, and the four preview pictures.
- `tools/check.py models/<slug>` must pass before you ask for review. It runs on every push and tests rules 1 to 8, the 2% rule, the checkpoints, and that nothing hangs in the air.
- Passing the check is not approval. The pictures are judged by eye for one thing: would a person who knows the place name it at a glance? Expect two or three rounds.

## Not allowed

- Any mesh, outline or measurement taken from Google, Apple or Mapbox 3D data, or from a paid or downloaded model library.
- Meshes made by an AI image-to-3D or text-to-3D generator. The model must come from your own script.
- Changing `tools/` or this brief inside a model's pull request, or touching any file outside `landmarks/`. If a rule is wrong, say so in the pull request text.
