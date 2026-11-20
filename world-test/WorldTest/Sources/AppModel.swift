import CoreLocation
import Foundation
import Observation

/// One camera shared by both map engines, so switching tabs compares the same view.
struct SharedCamera: Equatable {
    var latitude: Double
    var longitude: Double
    var zoom: Double
    var bearing: Double
    var pitch: Double

    var center: CLLocationCoordinate2D { .init(latitude: latitude, longitude: longitude) }
}

enum CameraPreset: String, CaseIterable, Identifiable {
    case downtown, bridge, top
    var id: String { rawValue }

    var title: String {
        switch self {
        case .downtown: "Downtown"
        case .bridge: "Bridge"
        case .top: "Top"
        }
    }

    var camera: SharedCamera {
        switch self {
        // Transamerica Pyramid (left) and Salesforce Tower (right), looking north-east towards the Bay Bridge.
        case .downtown: SharedCamera(latitude: 37.7915, longitude: -122.3995, zoom: 14.6, bearing: 45, pitch: 60)
        // Golden Gate Bridge from the Presidio side, looking north-west across both towers.
        case .bridge: SharedCamera(latitude: 37.8170, longitude: -122.4775, zoom: 14.7, bearing: -40, pitch: 60)
        // Downtown and the north-east of the city from straight above.
        case .top: SharedCamera(latitude: 37.7880, longitude: -122.4130, zoom: 13.6, bearing: 0, pitch: 0)
        }
    }
}

enum Engine: String, CaseIterable, Identifiable {
    case ours, mapbox
    var id: String { rawValue }
    var title: String { self == .ours ? "Ours" : "Mapbox" }
}

/// A request to move the active map, applied once per `id`.
struct CameraCommand: Equatable {
    let id: Int
    let camera: SharedCamera
    let animated: Bool
}

@Observable
final class AppModel {
    var engine: Engine
    var colourPlaces: Bool
    private(set) var command: CameraCommand?

    /// Written by the map views on every camera change. Not observed, to avoid re-rendering SwiftUI per frame.
    @ObservationIgnored var camera: SharedCamera

    let perf = PerfMonitor()
    let places = Place.loadAll()
    /// Debug: skip feature state on the Ours tab and use the filtered fallback layers.
    let forceFallback: Bool

    private var nextCommandID = 1

    init() {
        // Launch arguments (for screenshots): -engine mapbox  -preset bridge  -colour YES  -forceFallback YES
        let defaults = UserDefaults.standard
        engine = Engine(rawValue: defaults.string(forKey: "engine") ?? "") ?? .ours
        colourPlaces = defaults.bool(forKey: "colour")
        forceFallback = defaults.bool(forKey: "forceFallback")
        let preset = CameraPreset(rawValue: defaults.string(forKey: "preset") ?? "") ?? .downtown
        camera = preset.camera
    }

    func go(to preset: CameraPreset) {
        command = CameraCommand(id: nextCommandID, camera: preset.camera, animated: true)
        nextCommandID += 1
    }
}

struct Place: Decodable {
    let name: String
    let buildingIds: [String]
    let partIds: [String]
    /// Numeric vector-tile feature ids (as strings) of the buildings and their parts, used for feature state.
    let buildingTileIds: [String]
    let partTileIds: [String]
    let lat: Double
    let lon: Double

    var coordinate: CLLocationCoordinate2D { .init(latitude: lat, longitude: lon) }

    static func loadAll() -> [Place] {
        guard let url = Bundle.main.url(forResource: "places", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let places = try? JSONDecoder().decode([Place].self, from: data) else { return [] }
        return places
    }
}
