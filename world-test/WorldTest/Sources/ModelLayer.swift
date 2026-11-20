@preconcurrency import MapLibre
import Metal
import simd

/// Draws one landmark model (a .glb) on the Ours map with Metal, through MapLibre's custom style layer.
///
/// The model is placed in true size at `anchor`, turned by `heading` (degrees clockwise from north;
/// 0 means the model's -Z points north) and raised by `altitude` metres. It is drawn with the projection
/// matrix MapLibre uses for its 3D buildings and tested against their depth, so nearer buildings hide it.
/// It is lit with the same formula and light as the buildings' fill-extrusion shader, in their pale colour,
/// or terracotta when `highlighted`.
nonisolated final class ModelLayer: MLNCustomStyleLayer {
    let anchor: CLLocationCoordinate2D
    let heading: Double
    let altitude: Double
    var highlighted = false { didSet { if highlighted != oldValue { setNeedsDisplay() } } }

    private let model: GLBModel
    private var vertexBuffer: MTLBuffer?
    private var pipeline: MTLRenderPipelineState?
    private var pipelineFormats: [UInt] = []
    private var depthState: MTLDepthStencilState?
    private static let cacheLock = NSLock()
    nonisolated(unsafe) private static var pipelineCache: [[UInt]: MTLRenderPipelineState] = [:]

    // Same as the porcelain style: buildings #F4F1EC, visited #D9825B; light colour #FFFDF8,
    // intensity 0.18, position [1.15, 225, 45], anchored to the map.
    static let pale = SIMD4<Float>(0xF4, 0xF1, 0xEC, 255) / 255
    static let terracotta = SIMD4<Float>(0xD9, 0x82, 0x5B, 255) / 255
    static let lightColor = SIMD4<Float>(0xFF, 0xFD, 0xF8, 255) / 255
    static let lightIntensity: Float = 0.18
    static let lightPosition: SIMD3<Float> = {
        // MapLibre's spherical-to-cartesian rule (azimuth + 90°), in tile space: x east, y south, z up.
        let (r, azimuth, polar) = (1.15, (225.0 + 90) * .pi / 180, 45.0 * .pi / 180)
        return SIMD3(Float(r * cos(azimuth) * sin(polar)), Float(r * sin(azimuth) * sin(polar)), Float(r * cos(polar)))
    }()

    init(identifier: String, model: GLBModel, anchor: CLLocationCoordinate2D, heading: Double = 0, altitude: Double = 0) {
        self.model = model
        self.anchor = anchor
        self.heading = heading
        self.altitude = altitude
        super.init(identifier: identifier)
    }

    private struct Uniforms {
        var matrix: simd_float4x4
        var normalMatrix: simd_float4x4      // model axes -> tile axes (east, south, up)
        var color: SIMD4<Float>
        var light: SIMD4<Float>              // xyz position, w intensity
        var lightColor: SIMD4<Float>
        var heights: SIMD4<Float>            // x model height in metres
    }

    override func draw(in mapView: MLNMapView, with context: MLNStyleLayerDrawingContext) {
        guard let encoder = renderEncoder, let pass = renderPassDesc,
              let colorTexture = pass.colorAttachments[0].texture else { return }
        let device = encoder.device
        if vertexBuffer == nil {
            vertexBuffer = model.vertices.withUnsafeBytes {
                device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
        }
        let depthFormat = pass.depthAttachment.texture?.pixelFormat ?? .invalid
        let stencilFormat = pass.stencilAttachment.texture?.pixelFormat ?? .invalid
        let formats = [colorTexture.pixelFormat.rawValue, depthFormat.rawValue, stencilFormat.rawValue, UInt(colorTexture.sampleCount)]
        if pipeline == nil || formats != pipelineFormats {
            pipelineFormats = formats
            // One pipeline for all model layers: compiling the shader once per layer would stall the first frame.
            Self.cacheLock.lock()
            if let cached = Self.pipelineCache[formats] {
                pipeline = cached
            } else {
                pipeline = Self.makePipeline(device: device, color: colorTexture.pixelFormat, depth: depthFormat,
                                             stencil: stencilFormat, samples: colorTexture.sampleCount)
                Self.pipelineCache[formats] = pipeline
            }
            Self.cacheLock.unlock()
            let depth = MTLDepthStencilDescriptor()
            depth.depthCompareFunction = depthFormat == .invalid ? .always : .lessEqual
            depth.isDepthWriteEnabled = depthFormat != .invalid
            depthState = device.makeDepthStencilState(descriptor: depth)
        }
        guard let pipeline, let depthState, let vertexBuffer else { return }

        var uniforms = Uniforms(matrix: modelMatrix(context), normalMatrix: normalMatrix(),
                                color: highlighted ? Self.terracotta : Self.pale,
                                light: SIMD4(Self.lightPosition, Self.lightIntensity),
                                lightColor: Self.lightColor,
                                heights: SIMD4(max(model.maxY, 1), 0, 0, 0))
        encoder.pushDebugGroup("model \(identifier)")
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: model.vertices.count)
        encoder.popDebugGroup()
    }

    /// Model metres -> clip space, built in double precision: MapLibre's world space is in pixels of a
    /// 512 × 2^zoom world, far too large for floats at street zoom, so only the product is cast to float.
    private func modelMatrix(_ context: MLNStyleLayerDrawingContext) -> simd_float4x4 {
        let m = context.nearClippedProjectionMatrix
        let projection = simd_double4x4(columns: (SIMD4(m.m00, m.m01, m.m02, m.m03), SIMD4(m.m10, m.m11, m.m12, m.m13),
                                                  SIMD4(m.m20, m.m21, m.m22, m.m23), SIMD4(m.m30, m.m31, m.m32, m.m33)))
        let worldSize = 512 * pow(2, context.zoomLevel)
        let lat = anchor.latitude * .pi / 180
        let x0 = (anchor.longitude + 180) / 360 * worldSize
        let y0 = (1 - log(tan(.pi / 4 + lat / 2)) / .pi) / 2 * worldSize
        let k = worldSize / (40_075_016.685_578_49 * cos(lat))        // world pixels per metre at the anchor
        let h = heading * .pi / 180
        // Model X (east), Y (up), Z (south) -> world x (east, px), y (south, px), z (up, metres).
        let place = simd_double4x4(columns: (SIMD4(k * cos(h), k * sin(h), 0, 0),
                                             SIMD4(0, 0, 1, 0),
                                             SIMD4(-k * sin(h), k * cos(h), 0, 0),
                                             SIMD4(x0, y0, altitude, 1)))
        let product = projection * place
        return simd_float4x4(columns: (SIMD4<Float>(product.columns.0), SIMD4<Float>(product.columns.1),
                                       SIMD4<Float>(product.columns.2), SIMD4<Float>(product.columns.3)))
    }

    private func normalMatrix() -> simd_float4x4 {
        let h = Float(heading * .pi / 180)
        return simd_float4x4(columns: (SIMD4(cos(h), sin(h), 0, 0), SIMD4(0, 0, 1, 0),
                                       SIMD4(-sin(h), cos(h), 0, 0), SIMD4(0, 0, 0, 1)))
    }

    private static func makePipeline(device: MTLDevice, color: MTLPixelFormat, depth: MTLPixelFormat,
                                     stencil: MTLPixelFormat, samples: Int) -> MTLRenderPipelineState? {
        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.label = "World Test model"
            descriptor.vertexFunction = library.makeFunction(name: "modelVertex")
            descriptor.fragmentFunction = library.makeFunction(name: "modelFragment")
            descriptor.colorAttachments[0].pixelFormat = color
            descriptor.depthAttachmentPixelFormat = depth
            descriptor.stencilAttachmentPixelFormat = stencil
            descriptor.rasterSampleCount = samples
            return try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            print("[WorldTest] Model pipeline failed: \(error)")
            return nil
        }
    }

    /// Lighting copied from MapLibre Native's Metal fill-extrusion shader (6.31), so the model
    /// shades like the buildings around it.
    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex { packed_float3 position; packed_float3 normal; };
    struct Uniforms {
        float4x4 matrix;
        float4x4 normalMatrix;
        float4 color;
        float4 light;
        float4 lightColor;
        float4 heights;
    };
    struct Out { float4 position [[position]]; half4 color; };

    vertex Out modelVertex(uint vid [[vertex_id]],
                           const device Vertex* vertices [[buffer(0)]],
                           constant Uniforms& u [[buffer(1)]]) {
        const Vertex v = vertices[vid];
        const float3 p = float3(v.position);
        const float3 normal = normalize((u.normalMatrix * float4(float3(v.normal), 0.0)).xyz);

        float4 color = u.color;
        const float luminance = color.r * 0.2126 + color.g * 0.7152 + color.b * 0.0722;
        float4 vcolor = float4(0.0, 0.0, 0.0, 1.0);
        color += float4(0.03, 0.03, 0.03, 1.0);
        const float intensity = u.light.w;
        // MapLibre Native 6.31 lights building walls with normals that point into the building
        // (wall normal (-edge.y, edge.x) on clockwise vector-tile rings), so a wall is lit as if it
        // faced the other way. Mirror our walls the same way so the model matches its neighbours.
        const float3 litNormal = float3(-normal.xy, normal.z);
        const float directionalFraction = clamp(dot(litNormal, u.light.xyz), 0.0, 1.0);
        float directional = mix(1.0 - intensity, max(1.0 - luminance + intensity, 1.0), directionalFraction);
        if (abs(normal.z) < 0.5) {
            // Walls: darker towards the ground, as fill-extrusion-vertical-gradient does.
            const float height = u.heights.x;
            const float t = clamp(p.y / height, 0.0, 1.0);
            const float fMin = mix(0.7, 0.98, 1.0 - intensity);
            directional *= clamp(t * pow(height / 150.0, 0.5), fMin, 1.0);
        }
        const float3 minLight = mix(0.0, 0.3, 1.0 - u.lightColor.rgb);
        vcolor += float4(clamp(color.rgb * directional * u.lightColor.rgb, minLight, 1.0), 0.0);

        Out out;
        out.position = u.matrix * float4(p, 1.0);
        out.color = half4(vcolor);
        return out;
    }

    fragment half4 modelFragment(Out in [[stage_in]]) { return in.color; }
    """
}
