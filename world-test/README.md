# World Test

A small iOS test app (SwiftUI, Swift 6, iOS 26+) that shows San Francisco as a pale 3D city in two map engines, to compare them by eye and for smoothness on a real phone.

- **Ours:** MapLibre Native 6.31.0 with our own "porcelain" style (`WorldTest/Resources/porcelain-style.json`) over PMTiles bundled in the app: an OpenStreetMap base map (Protomaps daily build 2026-11-18) and Overture Maps buildings (release 2026-11-18.0, tiled by us so every building is there from zoom 11 to 14). No server needed.
- **Mapbox:** Mapbox Maps SDK 11.35.1 with the Standard style, theme `faded`, light preset `day`, 3D objects and landmarks on. Needs a public token in `Config/Secrets.xcconfig` (see `MORNING.md`).

To run it on a phone, see **MORNING.md**. Screenshots from the simulator are in `shots/`.

## Layout

| Path | What |
| --- | --- |
| `project.yml` | XcodeGen spec. `xcodegen generate` rebuilds `WorldTest.xcodeproj`. |
| `WorldTest/Sources` | SwiftUI app: shared camera model, the two map views, fps/memory monitor, and the landmark model layer (`GLBModel.swift`, `ModelLayer.swift`). |
| `WorldTest/Resources` | Style, PMTiles, glyphs (Noto Sans, SIL OFL), `places.json`. |
| `Config/` | `App.xcconfig`; `Secrets.xcconfig` (git-ignored) holds the Mapbox token. |
| `tools/make-data.sh` | Rebuilds all bundled data: extracts the base map, downloads Overture buildings, makes the building tiles with tippecanoe and rebuilds `places.json` (needs `brew install pmtiles tippecanoe uv`). |
| `tools/prepare_buildings.py` | Slims the Overture download for tippecanoe (called by `make-data.sh`). |
| `tools/make_places.py` | Rebuilds `places.json`, the sample landmarks for "Colour my places" (called by `make-data.sh`). |
| `tools/make_test_glb.py` | Writes `test-tower.glb`, the test landmark model (stepped tower, 60 m × 200 m). |
| `tools/shoot.sh` | Builds, installs and screenshots the Ours tab on a simulator. |

## Notes

- **Buildings:** fill-extrusion from Overture `height`, else `num_floors × 3.2 m`, else 6 m. Building parts are drawn with their own `height`/`min_height`, and the plain outline of any building with parts is hidden, so the Transamerica Pyramid, the Ferry Building clock tower and the Golden Gate Bridge towers show their real shapes.
- **Colour my places (Ours):** MapLibre feature state (`MLNVectorTileSource.setFeatureState`, new in 6.30) on the `building` and `building_part` layers. Feature state is keyed on the numeric vector-tile feature id: `promoteId` on the PMTiles source had no effect in MapLibre Native 6.31, so `places.json` stores those numeric ids. Our tiles give each feature the same id at every zoom (its line number in the Overture download), so colouring works zoomed out too. States are set only after the first frame, because states set earlier are dropped. If the API refuses, the app falls back to two filtered terracotta layers. Test the fallback with the launch argument `-forceFallback YES`.
- **Colour my places (Mapbox):** the Standard style's `select` state on the `standardBuildings` featureset, coloured with `colorBuildingSelect`. Mapbox building ids are not known ahead of time, so each landmark on screen is found with a rendered-feature query at its coordinate. This re-runs whenever the map comes to rest. Not tested yet: no token on the build Mac.
- **Building tiles** are made with tippecanoe from the Overture download for the box -122.54,37.69,-122.34,37.86: all 180,598 buildings and 1,666 building parts at zooms 11 to 14 (Overture's own tiles dropped most buildings below zoom 13.5, so the city looked patchy when zoomed out). Outlines are simplified to each zoom's resolution; heights are kept. The Overture id strings are left out of the tiles (they were more than half the size), so the file is 12.4 MB, smaller than Overture's extract (16.5 MB). The cost is memory: zoomed out to 12, with the whole city in view, the simulator used about 530 to 570 MB.
- **Landmark models (Ours):** `.glb` files drawn by our own Metal code through MapLibre's custom style layer (`MLNCustomStyleLayer`, available in 6.31; 7.0.0 adds no model layer, only an experimental plugin API in a separate package). The model is placed in true size from its anchor (1 unit = 1 m, Y up, +X east, -Z north), drawn with the same near-clipped projection matrix as the 3D buildings and depth-tested against them, so nearer buildings hide it. It is shaded with the buildings' fill-extrusion lighting formula, including one quirk: MapLibre 6.31 lights walls as if they faced the other way (its wall normals point inwards), and the model copies that so it matches its neighbours. "Colour my places" turns it terracotta. Materials and textures in the file are ignored; Draco- or meshopt-compressed files are refused with a console message. The test model stands on Alcatraz; **Model** flies there. `-testModelAt 37.7907,-122.3925` moves it (handy to see it hidden behind downtown towers).
- **Launch arguments** (Xcode → Scheme → Run → Arguments): `-preset downtown|bridge|top`, `-engine ours|mapbox`, `-colour YES`, `-autoTour YES` (loops through the presets and logs fps to the console), `-forceFallback YES`, `-camera "37.755,-122.49,12,30,50"` (exact lat, lon, zoom, bearing, pitch), `-orbit YES` (turns the camera around the centre forever and logs fps).

## Data and credits

© OpenStreetMap contributors (ODbL), via Protomaps. © Overture Maps Foundation (buildings: ODbL/CDLA, see docs.overturemaps.org/attribution). Noto Sans glyphs from protomaps/basemaps-assets, SIL Open Font License (`WorldTest/Resources/glyphs/OFL.txt`).
