# Landmark models

The rules are in `BRIEF.md`. Each model lives in `models/<slug>/`; `models/_sample-tower/` is a worked example.

Set up once (Python 3.11 or newer): `python -m pip install -r tools/requirements.txt`. If your build script needs more packages, list them in `models/<slug>/requirements.txt`; CI installs that file before it checks the model (packages every model needs can go in `requirements.txt` next to this file).

Build a model: `python models/<slug>/build.py` (it must write `<slug>.glb` next to itself).

Check it: `python tools/check.py models/<slug>`. Add `--rebuild` to also confirm build.py rewrites the same file.

The check prints one line per test: PASS, FAIL, WARN (does not fail) or NOTE (what it cannot test). It exits 1 if anything fails.

Nothing may float (rule 3). The check looks at the model from below on a grid of about 0.25 m (coarser for big models): every underside more than 0.5 m up must have something beside it that reaches lower, all the way down to y = 0. Parts that do not are listed by height, size and x, z. A big block held along only a small part of its edge, such as an anchorage stuck to the side of a pier, gets a WARN. It cannot see parts that float above other parts, or a roadway that runs on past its last support.

Make the pictures: `python tools/render.py models/<slug>` writes `preview/front.png`, `side.png`, `three-quarter.png`, `top.png` (PNG, 1600 x 1000; the check fails pictures of any other size or format). Long models such as bridges are turned in the top view so they fill the picture; an arrow shows north.

Rendering is pure numpy on the CPU: no GPU or display needed, and the same pixels every run.

A close-up for a reviewer: `python tools/render.py models/<slug> --camera X Y Z --target X Y Z` writes one extra picture, `close-up.png` in the current folder (`--out` to choose another name, `--fov` for the width of the view in degrees, 50 by default). Positions are model metres. The camera may stand inside or right next to the model. Keep the picture out of the model folder.

Test the check itself: `python tools/tests/test_check.py` breaks the sample in many ways and expects each to fail for the right reason, and the good variants to pass; it also tests the close-up cutting and the caption characters in render.py.

Limits the check applies: a building 6,000 triangles and 400,000 bytes, a bridge 25,000 triangles and 1,500,000 bytes; no glTF extensions (plain 32-bit float positions and normals).

Two optional marks on a measurement in facts.json: `"bbox": "length"`, `"width"` or `"height"` makes the check measure the model itself (height = top above the origin; length and width = sides of the smallest rectangle around the footprint), and `"bbox": "height"` is also what the up-axis test compares against. `"thickened": true` marks a part drawn thicker on purpose (rule 8): the 2% rule is skipped, and the model value must not be smaller than the real one.

Checkpoints are placed from the anchor with a flat local approximation: +X east, -Z north, metres. A bridge needs at least two checkpoints at least 100 m apart, so its direction is tested as well as its position.

On every pull request that touches `landmarks/`, CI runs the tests, checks and renders each changed model, and attaches the pictures as the `landmark-previews` artifact.
