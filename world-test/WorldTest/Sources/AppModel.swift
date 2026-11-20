import CoreLocation
import Foundation
import Observation
import UIKit

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
    case downtown, top, model
    var id: String { rawValue }

    var title: String {
        switch self {
        case .downtown: "Downtown"
        case .top: "Top"
        case .model: "Model"
        }
    }

    /// Where the test model stands: in the water west of Alcatraz (the Alcatraz landmark has the island).
    static let testModelAnchor = CLLocationCoordinate2D(latitude: 37.8270, longitude: -122.4300)

    var camera: SharedCamera {
        switch self {
        // Transamerica Pyramid (left) and Salesforce Tower (right), looking north-east towards the Bay Bridge.
        case .downtown: SharedCamera(latitude: 37.7915, longitude: -122.3995, zoom: 14.6, bearing: 45, pitch: 60)
        // Downtown and the north-east of the city from straight above.
        case .top: SharedCamera(latitude: 37.7880, longitude: -122.4130, zoom: 13.6, bearing: 0, pitch: 0)
        // The test model west of Alcatraz, seen from the south-west.
        case .model: SharedCamera(latitude: 37.8265, longitude: -122.4306, zoom: 15.9, bearing: 30, pitch: 60)
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
    /// Every landmark listed in Resources/landmarks/index.json, with its model (nil if it could not be read).
    let landmarks: [Landmark]
    /// Debug: skip feature state on the Ours tab and use the filtered fallback layers.
    let forceFallback: Bool

    private var nextCommandID = 1

    init() {
        // Launch arguments (for screenshots): -engine mapbox  -preset top  -colour YES  -forceFallback YES
        let defaults = UserDefaults.standard
        engine = Engine(rawValue: defaults.string(forKey: "engine") ?? "") ?? .ours
        colourPlaces = defaults.bool(forKey: "colour")
        forceFallback = defaults.bool(forKey: "forceFallback")
        landmarks = defaults.bool(forKey: "noModels") ? [] : Landmark.loadAll()
        let preset = CameraPreset(rawValue: defaults.string(forKey: "preset") ?? "") ?? .downtown
        camera = preset.camera
        // Exact camera for screenshots: -camera "37.755,-122.49,12,30,50" (lat, lon, zoom, bearing, pitch).
        // One string, because a separate negative value such as "-122.49" would be read as another flag.
        if let values = defaults.string(forKey: "camera")?.split(separator: ",").compactMap({ Double($0.trimmingCharacters(in: .whitespaces)) }),
           values.count == 5 {
            camera = SharedCamera(latitude: values[0], longitude: values[1], zoom: values[2], bearing: values[3], pitch: values[4])
        }
        // Landmark view for screenshots: -landmark coit-tower  (add -landmarkTop YES to look from straight above).
        if let slug = defaults.string(forKey: "landmark"), let landmark = landmarks.first(where: { $0.slug == slug }) {
            camera = landmark.camera(topDown: defaults.bool(forKey: "landmarkTop"))
        }
    }

    func go(to preset: CameraPreset) {
        go(to: preset.camera)
    }

    func go(to landmark: Landmark) {
        go(to: landmark.camera(topDown: false))
    }

    private func go(to camera: SharedCamera) {
        command = CameraCommand(id: nextCommandID, camera: camera, animated: true)
        nextCommandID += 1
    }
}

/// A landmark model from the landmark builders, copied by tools/pull-landmarks.sh.
struct Landmark: Identifiable {
    let slug: String
    let name: String
    let kind: String
    let commit: String
    /// Origin of the model: sea level for bridges, ground level for buildings (the map is flat at 0 m).
    let anchor: CLLocationCoordinate2D
    let model: GLBModel?
    let problem: String?
    /// The map's own buildings and building parts (tile ids) that stand where the model stands.
    let hiddenBuildings: [Int]
    let hiddenParts: [Int]
    /// For bridges: the outline seen from above ([lon, lat] ring) inside which the map's bridge roads are hidden.
    let bridgeOutline: [[Double]]?
    /// For buildings: the view direction with the fewest tall buildings in front (from tools/landmark_overlap.py).
    let viewBearing: Double?
    var id: String { slug }

    private struct Entry: Decodable {
        struct Anchor: Decodable { let lat: Double; let lon: Double }
        let slug: String; let name: String; let kind: String; let anchor: Anchor; let commit: String
    }

    private struct Overlap: Decodable {
        let buildings: [Int]; let parts: [Int]; let bridgeOutline: [[Double]]?; let viewBearing: Double?
    }

    static func loadAll() -> [Landmark] {
        guard let root = Bundle.main.url(forResource: "landmarks", withExtension: nil),
              let data = try? Data(contentsOf: root.appendingPathComponent("index.json")),
              let entries = try? JSONDecoder().decode([Entry].self, from: data) else {
            print("[WorldTest] No landmarks/index.json; run tools/pull-landmarks.sh")
            return []
        }
        let overlaps = (try? Data(contentsOf: root.appendingPathComponent("overlap.json")))
            .flatMap { try? JSONDecoder().decode([String: Overlap].self, from: $0) } ?? [:]
        return entries.map { entry in
            let url = root.appendingPathComponent(entry.slug).appendingPathComponent("\(entry.slug).glb")
            var model: GLBModel?
            var problem: String?
            do { model = try GLBModel(url: url) } catch {
                problem = "\(error)"
                print("[WorldTest] Landmark \(entry.slug) skipped: \(error)")
            }
            let overlap = model == nil ? nil : overlaps[entry.slug]
            return Landmark(slug: entry.slug, name: entry.name, kind: entry.kind, commit: entry.commit,
                            anchor: CLLocationCoordinate2D(latitude: entry.anchor.lat, longitude: entry.anchor.lon),
                            model: model, problem: problem,
                            hiddenBuildings: overlap?.buildings ?? [], hiddenParts: overlap?.parts ?? [],
                            bridgeOutline: overlap?.bridgeOutline, viewBearing: overlap?.viewBearing)
        }
    }

    /// The point `east` and `south` metres from the anchor.
    func coordinate(east: Double, south: Double) -> CLLocationCoordinate2D {
        let lat = anchor.latitude * .pi / 180
        let metresPerDegree = 40_075_016.685_578_49 / 360
        return CLLocationCoordinate2D(latitude: anchor.latitude - south / metresPerDegree,
                                      longitude: anchor.longitude + east / (metresPerDegree * cos(lat)))
    }

    /// A three-quarter view with the model filling about half the screen, or a view from straight above.
    func camera(topDown: Bool) -> SharedCamera {
        guard let m = model else {
            return SharedCamera(latitude: anchor.latitude, longitude: anchor.longitude, zoom: 16, bearing: 30, pitch: 55)
        }
        let (width, depth, height) = (Double(m.maxX - m.minX), Double(m.maxZ - m.minZ), Double(m.maxY - max(m.minY, 0)))
        let centre = coordinate(east: Double(m.minX + m.maxX) / 2, south: Double(m.minZ + m.maxZ) / 2)
        let span = max(width, depth)
        // Long models (bridges) are seen at an angle to their length; compact ones from the side with
        // the fewest tall buildings in front. Tall thin ones (Sutro Tower) are sized by their height.
        let long = kind == "bridge" || span > 2.5 * max(1, min(width, depth))
        let size = topDown ? span : max(span, height)
        let screen = Double(min(UIScreen.main.bounds.width, UIScreen.main.bounds.height))
        // At zoom z the world is 512 × 2^z points around; the model should cover about half the screen.
        let metresPerPointAtZoom0 = 40_075_016.685_578_49 * cos(anchor.latitude * .pi / 180) / 512
        let fill = topDown ? 0.75 : (long ? 1.5 : 0.6)
        let zoom = min(18, max(11, log2(metresPerPointAtZoom0 * fill * screen / size)))
        if topDown {
            return SharedCamera(latitude: centre.latitude, longitude: centre.longitude, zoom: zoom, bearing: 0, pitch: 0)
        }
        let bearing = long ? (m.longAxisBearing + 55).truncatingRemainder(dividingBy: 360) : (viewBearing ?? 30)
        return SharedCamera(latitude: centre.latitude, longitude: centre.longitude, zoom: zoom, bearing: bearing, pitch: 55)
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
