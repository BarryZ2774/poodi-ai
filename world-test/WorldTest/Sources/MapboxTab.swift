@preconcurrency import MapboxMaps
import SwiftUI

/// "Mapbox": Mapbox Standard, faded theme, day light, 3D objects and landmarks on.
/// Needs a public token (pk...) in Config/Secrets.xcconfig; without one it shows a single line.
struct MapboxTab: View {
    let model: AppModel

    static var token: String? {
        let value = (Bundle.main.object(forInfoDictionaryKey: "MBXAccessToken") as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.hasPrefix("pk.") ? value : nil
    }

    var body: some View {
        if let token = Self.token {
            MapboxMapView(model: model, token: token, command: model.command, colourPlaces: model.colourPlaces)
        } else {
            ZStack {
                Color(red: 0.91, green: 0.90, blue: 0.87)
                Text("Add a Mapbox token in Secrets.xcconfig")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

private struct MapboxMapView: UIViewRepresentable {
    let model: AppModel
    let token: String
    let command: CameraCommand?
    let colourPlaces: Bool

    static let terracotta = UIColor(red: 0xD9 / 255, green: 0x82 / 255, blue: 0x5B / 255, alpha: 1)

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }

    func makeUIView(context: Context) -> MapView {
        MapboxOptions.accessToken = token
        let style = MapStyle.standard(
            theme: .faded,
            lightPreset: .day,
            show3dObjects: true,
            colorBuildingSelect: StyleColor(Self.terracotta),
            show3dBuildings: true,
            show3dLandmarks: true
        )
        let options = MapInitOptions(mapStyle: style, cameraOptions: Coordinator.options(model.camera))
        let mapView = MapView(frame: UIScreen.main.bounds, mapInitOptions: options)
        let maxFPS = Float(UIScreen.main.maximumFramesPerSecond)
        mapView.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: maxFPS, preferred: maxFPS)
        context.coordinator.attach(mapView)
        context.coordinator.lastCommandID = command?.id
        return mapView
    }

    func updateUIView(_ mapView: MapView, context: Context) {
        let coordinator = context.coordinator
        if let command, command.id != coordinator.lastCommandID {
            coordinator.lastCommandID = command.id
            mapView.camera.fly(to: Coordinator.options(command.camera), duration: 2.2)
        }
        coordinator.setColourPlaces(colourPlaces)
    }

    final class Coordinator {
        let model: AppModel
        weak var mapView: MapView?
        var lastCommandID: Int?
        private var cancelables = Set<AnyCancelable>()
        private var colourOn = false
        private var styleLoaded = false

        init(model: AppModel) { self.model = model }

        static func options(_ c: SharedCamera) -> CameraOptions {
            CameraOptions(center: c.center, zoom: c.zoom, bearing: c.bearing, pitch: c.pitch)
        }

        func attach(_ mapView: MapView) {
            self.mapView = mapView
            let map = mapView.mapboxMap!
            map.onCameraChanged.observe { [weak self] event in
                guard let self else { return }
                self.model.perf.cameraMoved()
                let s = event.cameraState
                self.model.camera = SharedCamera(latitude: s.center.latitude, longitude: s.center.longitude,
                                                 zoom: s.zoom, bearing: s.bearing, pitch: s.pitch)
            }.store(in: &cancelables)
            map.onRenderFrameFinished.observe { [weak self] _ in
                self?.model.perf.frameRendered()
            }.store(in: &cancelables)
            map.onStyleLoaded.observe { [weak self] _ in
                self?.styleLoaded = true
                self?.selectVisiblePlaces()
            }.store(in: &cancelables)
            // Selection works on rendered features, so re-run it whenever the map settles.
            map.onMapIdle.observe { [weak self] _ in
                self?.selectVisiblePlaces()
            }.store(in: &cancelables)
        }

        func setColourPlaces(_ on: Bool) {
            guard on != colourOn else { return }
            colourOn = on
            if on {
                selectVisiblePlaces()
            } else {
                mapView?.mapboxMap.resetFeatureStates(featureset: .standardBuildings, callback: nil)
            }
        }

        /// Uses the Standard style's building "select" state (featureset `standardBuildings`).
        /// Mapbox building ids are not known ahead of time, so each sample place on screen is looked up
        /// with a rendered-feature query at its coordinate and the buildings found there are selected.
        private func selectVisiblePlaces() {
            guard colourOn, styleLoaded, let mapView, let map = mapView.mapboxMap else { return }
            let bounds = mapView.bounds
            for place in model.places {
                let point = map.point(for: place.coordinate)
                guard bounds.insetBy(dx: -20, dy: -20).contains(point) else { continue }
                let box = CGRect(x: point.x - 6, y: point.y - 6, width: 12, height: 12)
                map.queryRenderedFeatures(with: box, featureset: .standardBuildings) { [weak map] result in
                    guard let map, case .success(let features) = result else { return }
                    for feature in features {
                        map.setFeatureState(feature, state: .init(select: true))
                    }
                }
            }
        }
    }
}
