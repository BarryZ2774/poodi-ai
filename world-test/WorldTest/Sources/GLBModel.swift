import Foundation
import simd

/// The triangles of a glTF binary (.glb) model, flattened into one list in model space.
///
/// Model space as agreed for landmark models: 1 unit = 1 metre, Y up, +X east, -Z north.
/// Reads uncompressed triangle meshes with float positions (and normals if present) from the
/// default scene, applying node transforms. Materials and textures are ignored: the map draws
/// every model in its own porcelain colour. Compressed meshes (Draco, meshopt) are refused.
nonisolated struct GLBModel {
    /// Interleaved per-vertex data for Metal: position (3 floats) then normal (3 floats).
    struct Vertex {
        var px, py, pz: Float
        var nx, ny, nz: Float
    }

    var vertices: [Vertex] = []
    var minY: Float = .infinity
    var maxY: Float = -.infinity
    var triangleCount: Int { vertices.count / 3 }

    enum LoadError: Error, CustomStringConvertible {
        case notGLB, unsupported(String), broken(String)
        var description: String {
            switch self {
            case .notGLB: "not a glTF binary (.glb) file"
            case .unsupported(let what): "unsupported glTF feature: \(what)"
            case .broken(let what): "broken glTF: \(what)"
            }
        }
    }

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        func u32(_ offset: Int) -> UInt32 {
            data.subdata(in: offset..<offset + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        }
        guard data.count >= 20, u32(0) == 0x4654_6C67, u32(4) == 2 else { throw LoadError.notGLB }
        var offset = 12
        var json: [String: Any]?
        var bin = Data()
        while offset + 8 <= data.count {
            let length = Int(u32(offset)), type = u32(offset + 4)
            let chunk = data.subdata(in: offset + 8..<min(offset + 8 + length, data.count))
            if type == 0x4E4F_534A { json = try JSONSerialization.jsonObject(with: chunk) as? [String: Any] }
            if type == 0x004E_4942 { bin = chunk }
            offset += 8 + length
        }
        guard let json else { throw LoadError.broken("no JSON chunk") }
        let required = json["extensionsRequired"] as? [String] ?? []
        if let first = required.first { throw LoadError.unsupported(first) }

        let nodes = json["nodes"] as? [[String: Any]] ?? []
        let meshes = json["meshes"] as? [[String: Any]] ?? []
        let accessors = json["accessors"] as? [[String: Any]] ?? []
        let views = json["bufferViews"] as? [[String: Any]] ?? []
        if (json["buffers"] as? [[String: Any]])?.contains(where: { $0["uri"] != nil }) == true {
            throw LoadError.unsupported("external buffer files")
        }

        /// Reads an accessor as rows of `width` floats (positions, normals) or as integers (indices).
        func read(_ index: Int) throws -> (values: [Float], width: Int) {
            guard index < accessors.count else { throw LoadError.broken("accessor \(index)") }
            let a = accessors[index]
            if a["sparse"] != nil { throw LoadError.unsupported("sparse accessors") }
            let count = a["count"] as? Int ?? 0
            let width = ["SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4][a["type"] as? String ?? ""] ?? 1
            let component = a["componentType"] as? Int ?? 5126
            let size = [5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4][component] ?? 4
            guard let viewIndex = a["bufferView"] as? Int, viewIndex < views.count else {
                return (Array(repeating: 0, count: count * width), width)
            }
            let view = views[viewIndex]
            let start = (view["byteOffset"] as? Int ?? 0) + (a["byteOffset"] as? Int ?? 0)
            let stride = view["byteStride"] as? Int ?? width * size
            guard count == 0 || start + (count - 1) * stride + width * size <= bin.count else {
                throw LoadError.broken("accessor \(index) runs past the binary chunk")
            }
            var out = [Float](); out.reserveCapacity(count * width)
            bin.withUnsafeBytes { raw in
                for i in 0..<count {
                    for c in 0..<width {
                        let at = start + i * stride + c * size
                        switch component {
                        case 5126: out.append(raw.loadUnaligned(fromByteOffset: at, as: Float.self))
                        case 5125: out.append(Float(raw.loadUnaligned(fromByteOffset: at, as: UInt32.self)))
                        case 5123: out.append(Float(raw.loadUnaligned(fromByteOffset: at, as: UInt16.self)))
                        case 5121: out.append(Float(raw.load(fromByteOffset: at, as: UInt8.self)))
                        case 5122: out.append(Float(raw.loadUnaligned(fromByteOffset: at, as: Int16.self)))
                        default: out.append(Float(raw.load(fromByteOffset: at, as: Int8.self)))
                        }
                    }
                }
            }
            return (out, width)
        }

        func localMatrix(_ node: [String: Any]) -> simd_float4x4 {
            if let m = node["matrix"] as? [Double], m.count == 16 {
                let f = m.map(Float.init)
                return simd_float4x4(columns: (SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]),
                                               SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15])))
            }
            let t = (node["translation"] as? [Double] ?? [0, 0, 0]).map(Float.init)
            let r = (node["rotation"] as? [Double] ?? [0, 0, 0, 1]).map(Float.init)
            let s = (node["scale"] as? [Double] ?? [1, 1, 1]).map(Float.init)
            let rotation = simd_float4x4(simd_quatf(ix: r[0], iy: r[1], iz: r[2], r: r[3]))
            var m = rotation * simd_float4x4(diagonal: SIMD4(s[0], s[1], s[2], 1))
            m.columns.3 = SIMD4(t[0], t[1], t[2], 1)
            return m
        }

        var visited = Set<Int>()
        func visit(_ nodeIndex: Int, _ parent: simd_float4x4) throws {
            guard nodeIndex < nodes.count, visited.insert(nodeIndex).inserted else { return }
            let node = nodes[nodeIndex]
            let world = parent * localMatrix(node)
            if let meshIndex = node["mesh"] as? Int, meshIndex < meshes.count {
                let upper = simd_float3x3(SIMD3(world.columns.0.x, world.columns.0.y, world.columns.0.z),
                                          SIMD3(world.columns.1.x, world.columns.1.y, world.columns.1.z),
                                          SIMD3(world.columns.2.x, world.columns.2.y, world.columns.2.z))
                let normalMatrix = upper.inverse.transpose
                for primitive in meshes[meshIndex]["primitives"] as? [[String: Any]] ?? [] {
                    guard (primitive["mode"] as? Int ?? 4) == 4 else { continue }   // triangles only
                    if let ext = primitive["extensions"] as? [String: Any], !ext.isEmpty {
                        throw LoadError.unsupported(ext.keys.joined(separator: ", "))
                    }
                    let attributes = primitive["attributes"] as? [String: Int] ?? [:]
                    guard let positionIndex = attributes["POSITION"] else { continue }
                    let positions = try read(positionIndex).values
                    let normals = try attributes["NORMAL"].map { try read($0).values }
                    let vertexCount = positions.count / 3
                    let indices: [Int] = try primitive["indices"].flatMap { $0 as? Int }
                        .map { try read($0).values.map { Int($0) } } ?? Array(0..<vertexCount)
                    func position(_ i: Int) -> SIMD3<Float> {
                        let p = world * SIMD4(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2], 1)
                        return SIMD3(p.x, p.y, p.z)
                    }
                    var t = 0
                    while t + 2 < indices.count {
                        let ids = [indices[t], indices[t + 1], indices[t + 2]]
                        t += 3
                        guard ids.allSatisfy({ $0 < vertexCount }) else { continue }
                        let p = ids.map(position)
                        let face = simd_normalize(simd_cross(p[1] - p[0], p[2] - p[0]))
                        guard face.x.isFinite else { continue }       // degenerate triangle
                        for (k, i) in ids.enumerated() {
                            var n = face
                            if let normals, 3 * i + 2 < normals.count {
                                let given = normalMatrix * SIMD3(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2])
                                if simd_length(given) > 0 { n = simd_normalize(given) }
                            }
                            vertices.append(Vertex(px: p[k].x, py: p[k].y, pz: p[k].z, nx: n.x, ny: n.y, nz: n.z))
                            minY = min(minY, p[k].y); maxY = max(maxY, p[k].y)
                        }
                    }
                }
            }
            for child in node["children"] as? [Int] ?? [] { try visit(child, world) }
        }

        let scenes = json["scenes"] as? [[String: Any]] ?? []
        let sceneIndex = json["scene"] as? Int ?? 0
        let roots = sceneIndex < scenes.count ? scenes[sceneIndex]["nodes"] as? [Int] ?? [] : Array(nodes.indices)
        for root in roots { try visit(root, matrix_identity_float4x4) }
        guard !vertices.isEmpty else { throw LoadError.broken("no triangles") }
    }
}
