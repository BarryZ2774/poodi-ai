@preconcurrency import MapLibre
import SwiftUI

/// "Ours": MapLibre Native with the bundled porcelain style over local PMTiles (OpenStreetMap + Overture buildings).
struct OursMapView: UIViewRepresentable {
    let model: AppModel
    let command: CameraCommand?
    let colourPlaces: Bool

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> MLNMapView {
        let mapView = MLNMapView(frame: UIScreen.main.bounds, styleURL: Self.styleURL(landmarks: model.landmarks))
        mapView.delegate = context.coordinator
        mapView.logoView.isHidden = true            // credit is shown by ContentView
        mapView.attributionButton.isHidden = true
        mapView.compassView.compassVisibility = .adaptive
        mapView.preferredFramesPerSecond = MLNMapViewPreferredFramesPerSecond(UIScreen.main.maximumFramesPerSecond)
        mapView.maximumPitch = 60
        context.coordinator.mapView = mapView
        context.coordinator.lastCommandID = command?.id
        context.coordinator.apply(model.camera, animated: false)
        return mapView
    }

    func updateUIView(_ mapView: MLNMapView, context: Context) {
        let coordinator = context.coordinator
        if let command, command.id != coordinator.lastCommandID {
            coordinator.lastCommandID = command.id
            coordinator.apply(command.camera, animated: command.animated)
        }
        coordinator.setColourPlaces(colourPlaces)
        coordinator.modelLayers.forEach { $0.highlighted = colourPlaces }
    }

    /// Fills the style template's placeholders with bundle file URLs and writes it to Caches.
    /// Where a landmark model stands, the map's own buildings and bridge roads are left out, so the map
    /// does not also draw its own version underneath. The filters use tile ids (the same at every zoom)
    /// and a fixed outline, so they hold while tiles reload.
    static func styleURL(landmarks: [Landmark]) -> URL {
        let bundle = Bundle.main
        guard let template = bundle.url(forResource: "porcelain-style", withExtension: "json"),
              var json = try? String(contentsOf: template, encoding: .utf8),
              let base = bundle.url(forResource: "sf-base", withExtension: "pmtiles"),
              let buildings = bundle.url(forResource: "sf-buildings", withExtension: "pmtiles"),
              let glyphs = bundle.url(forResource: "glyphs", withExtension: nil) else {
            fatalError("Bundled map data is missing")
        }
        let places = Place.loadAll()
        let idsJSON = String(data: try! JSONEncoder().encode(places.flatMap(\.buildingIds)), encoding: .utf8)!
        let tileIDsJSON = String(data: try! JSONEncoder().encode(places.flatMap(\.buildingTileIds).compactMap { Int($0) }),
                                 encoding: .utf8)!
        func encoded(_ value: some Encodable) -> String { String(data: try! JSONEncoder().encode(value), encoding: .utf8)! }
        let shown = landmarks.filter { $0.model != nil }
        let outlines = shown.compactMap(\.bridgeOutline).map { [$0] }      // MultiPolygon: polygons of rings
        let bridgeFilter = outlines.isEmpty ? "true"
            : #"["!", ["all", ["==", ["get", "is_bridge"], true], ["within", {"type": "MultiPolygon", "coordinates": \#(encoded(outlines))}]]]"#
        var glyphsURL = glyphs.absoluteString
        if glyphsURL.hasSuffix("/") { glyphsURL.removeLast() }
        json = json
            .replacingOccurrences(of: "__BASE__", with: base.absoluteString)
            .replacingOccurrences(of: "__BUILDINGS__", with: buildings.absoluteString)
            .replacingOccurrences(of: "__GLYPHS__", with: glyphsURL)
            .replacingOccurrences(of: "__VISITED_BUILDING_IDS__", with: idsJSON)
            .replacingOccurrences(of: "__VISITED_BUILDING_TILE_IDS__", with: tileIDsJSON)
            .replacingOccurrences(of: "__HIDDEN_BUILDING_TILE_IDS__", with: encoded(shown.flatMap(\.hiddenBuildings)))
            .replacingOccurrences(of: "__HIDDEN_PART_TILE_IDS__", with: encoded(shown.flatMap(\.hiddenParts)))
            .replacingOccurrences(of: "__BRIDGE_ROAD_FILTER__", with: bridgeFilter)
        let out = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("porcelain-style.json")
        try? json.write(to: out, atomically: true, encoding: .utf8)
        return out
    }

    final class Coordinator: NSObject, @preconcurrency MLNMapViewDelegate {
        let model: AppModel
        weak var mapView: MLNMapView?
        var lastCommandID: Int?
        private var styleLoaded = false
        private var appliedColour: Bool?
        private var usingFallback = false
        private var basePredicates: [String: NSPredicate] = [:]
        private(set) var modelLayers: [ModelLayer] = []

        init(model: AppModel) { self.model = model }

        func apply(_ camera: SharedCamera, animated: Bool) {
            guard let mapView else { return }
            let size = mapView.bounds.size == .zero ? UIScreen.main.bounds.size : mapView.bounds.size
            let altitude = MLNAltitudeForZoomLevel(camera.zoom, camera.pitch, camera.latitude, size)
            // MapLibre wants a heading in 0..<360; a negative one is treated as "no heading".
            let heading = (camera.bearing.truncatingRemainder(dividingBy: 360) + 360).truncatingRemainder(dividingBy: 360)
            let target = MLNMapCamera(lookingAtCenter: camera.center, altitude: altitude,
                                      pitch: camera.pitch, heading: heading)
            if animated {
                mapView.fly(to: target, withDuration: 2.2, completionHandler: nil)
            } else {
                mapView.setCamera(target, animated: false)
            }
        }

        // MARK: Colour my places

        func setColourPlaces(_ on: Bool) {
            guard styleLoaded, appliedColour != on, let style = mapView?.style,
                  let source = style.source(withIdentifier: "buildings") as? MLNVectorTileSource else { return }
            appliedColour = on

            if !model.forceFallback && !usingFallback {
                var dispatched = true
                for place in model.places {
                    for id in place.buildingTileIds {
                        dispatched = source.setFeatureState(sourceLayerID: "building", featureID: id, state: ["visited": on]) && dispatched
                    }
                    for id in place.partTileIds {
                        dispatched = source.setFeatureState(sourceLayerID: "building_part", featureID: id, state: ["visited": on]) && dispatched
                    }
                }
                if dispatched {
                    print("[WorldTest] Colour my places via feature state: \(on)")
                    // The state lands on the next frame; read one back a moment later, for the log only.
                    if let first = model.places.first?.buildingTileIds.first {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
                            let state = source.featureState(sourceLayerID: "building", featureID: first)
                            print("[WorldTest] Feature state read back: \(state ?? [:])")
                        }
                    }
                    return
                }
                print("[WorldTest] Feature state was not accepted, using the filtered fallback layers")
                usingFallback = true
            }
            setFallback(on, in: style)
        }

        /// Fallback: show the filtered terracotta layers and hide the same buildings from the pale layers,
        /// so the two never draw the same walls.
        private func setFallback(_ on: Bool, in style: MLNStyle) {
            // Buildings carry only a numeric feature id in the tiles; building parts carry their building's Overture id.
            let tileIDs = model.places.flatMap(\.buildingTileIds).compactMap { Int($0) }
            let ids = model.places.flatMap(\.buildingIds)
            let exclusions = [
                "buildings": NSPredicate(format: "NOT ($featureIdentifier IN %@)", tileIDs),
                "building-parts": NSPredicate(format: "NOT (building_id IN %@)", ids),
            ]
            for (layerID, exclusion) in exclusions {
                guard let layer = style.layer(withIdentifier: layerID) as? MLNVectorStyleLayer else { continue }
                if basePredicates[layerID] == nil { basePredicates[layerID] = layer.predicate ?? NSPredicate(value: true) }
                let base = basePredicates[layerID]!
                layer.predicate = on ? NSCompoundPredicate(andPredicateWithSubpredicates: [base, exclusion]) : base
            }
            for id in ["visited-fallback", "visited-parts-fallback"] {
                style.layer(withIdentifier: id)?.isVisible = on
            }
            print("[WorldTest] Colour my places via filtered layers: \(on)")
        }

        // MARK: MLNMapViewDelegate

        func mapView(_ mapView: MLNMapView, didFinishLoading style: MLNStyle) {
            styleLoaded = false
            appliedColour = nil
            addModels(to: style)
        }

        // MARK: Landmark models

        /// Landmark models drawn by our own Metal layer, above the buildings and below the labels.
        private func addModels(to style: MLNStyle) {
            modelLayers = []
            // Debug: "-noModels YES" leaves all models out, to compare with the map's own buildings.
            guard !UserDefaults.standard.bool(forKey: "noModels") else { return }
            // Debug: "-testModelAt 37.7936,-122.3966" moves the test model, e.g. next to tall buildings.
            var testAnchor = CameraPreset.testModelAnchor
            if let v = UserDefaults.standard.string(forKey: "testModelAt")?.split(separator: ",").compactMap({ Double($0) }), v.count == 2 {
                testAnchor = CLLocationCoordinate2D(latitude: v[0], longitude: v[1])
            }
            // Test model from tools/make_test_glb.py, standing in the water west of Alcatraz.
            var models: [(id: String, model: GLBModel, anchor: CLLocationCoordinate2D)] = []
            if let url = Bundle.main.url(forResource: "test-tower", withExtension: "glb") {
                do { models.append(("model-test-tower", try GLBModel(url: url), testAnchor)) } catch {
                    print("[WorldTest] Test model not shown: \(error)")
                }
            }
            // Landmark models from landmarks/index.json, each at its facts.json anchor, never turned.
            // Bridges have their origin at sea level and buildings at ground level; the map is flat at 0 m.
            models += self.model.landmarks.compactMap { l in l.model.map { ("model-\(l.slug)", $0, l.anchor) } }
            let below = style.layer(withIdentifier: "label-water")
            for m in models {
                let layer = ModelLayer(identifier: m.id, model: m.model, anchor: m.anchor)
                layer.highlighted = self.model.colourPlaces
                if let below { style.insertLayer(layer, below: below) } else { style.addLayer(layer) }
                modelLayers.append(layer)
                print("[WorldTest] Model \(m.id): \(m.model.triangleCount) triangles, \(m.model.maxY - m.model.minY) m tall")
            }
        }

        /// Feature state set before the first frame is dropped (the render source does not exist yet),
        /// so colouring starts once the map has rendered.
        func mapViewDidFinishRenderingMap(_ mapView: MLNMapView, fullyRendered: Bool) {
            guard !styleLoaded else { return }
            styleLoaded = true
            setColourPlaces(model.colourPlaces)
            startOrbitIfAsked()
        }

        // MARK: Orbit (launch argument "-orbit YES", for measuring fps)

        private var orbiting = false

        private func startOrbitIfAsked() {
            guard UserDefaults.standard.bool(forKey: "orbit"), !orbiting else { return }
            orbiting = true
            orbitStep()
        }

        /// Turns the camera a quarter turn around the map centre in four seconds, forever.
        private func orbitStep() {
            guard let mapView, let camera = mapView.camera.copy() as? MLNMapCamera else { return }
            camera.heading = (camera.heading + 90).truncatingRemainder(dividingBy: 360)
            mapView.setCamera(camera, withDuration: 4, animationTimingFunction: CAMediaTimingFunction(name: .linear)) { [weak self] in
                self?.orbitStep()
            }
        }

        func mapView(_ mapView: MLNMapView, regionIsChangingWith reason: MLNCameraChangeReason) {
            store(mapView)
        }

        func mapView(_ mapView: MLNMapView, regionDidChangeWith reason: MLNCameraChangeReason, animated: Bool) {
            store(mapView)
        }

        func mapViewDidFinishRenderingFrame(_ mapView: MLNMapView, fullyRendered: Bool) {
            model.perf.frameRendered()
        }

        private func store(_ mapView: MLNMapView) {
            model.perf.cameraMoved()
            let c = mapView.centerCoordinate
            model.camera = SharedCamera(latitude: c.latitude, longitude: c.longitude, zoom: mapView.zoomLevel,
                                        bearing: mapView.direction, pitch: mapView.camera.pitch)
        }
    }
}
