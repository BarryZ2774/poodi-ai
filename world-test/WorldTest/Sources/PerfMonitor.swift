import Darwin
import Observation
import QuartzCore
import UIKit

/// Frames per second while the map moves, measured with a display link, plus the app's memory footprint.
///
/// The map views call `frameRendered()` from their "frame finished" callbacks and `cameraMoved()` on camera
/// changes. A display link ticks every screen refresh; every half second it turns the rendered-frame count
/// into frames per second, but only while the camera has moved recently. When the map is still the
/// readout keeps the last moving value and says "idle".
@Observable
final class PerfMonitor {
    private(set) var fps: Int?
    private(set) var isMoving = false
    private(set) var memoryMB = 0
    let maxFPS = UIScreen.main.maximumFramesPerSecond
    @ObservationIgnored private let logToConsole = UserDefaults.standard.bool(forKey: "autoTour") || UserDefaults.standard.bool(forKey: "orbit")

    @ObservationIgnored private var link: CADisplayLink?
    @ObservationIgnored private var framesInWindow = 0
    @ObservationIgnored private var windowStart: CFTimeInterval = 0
    @ObservationIgnored private var lastMove: CFTimeInterval = 0
    @ObservationIgnored private var lastMemory: CFTimeInterval = 0

    func start() {
        guard link == nil else { return }
        let link = CADisplayLink(target: self, selector: #selector(tick(_:)))
        link.preferredFrameRateRange = CAFrameRateRange(minimum: 10, maximum: Float(maxFPS), preferred: Float(maxFPS))
        link.add(to: .main, forMode: .common)
        self.link = link
        windowStart = CACurrentMediaTime()
    }

    func frameRendered() { framesInWindow += 1 }
    func cameraMoved() { lastMove = CACurrentMediaTime() }

    func reset() {
        fps = nil
        framesInWindow = 0
        windowStart = CACurrentMediaTime()
    }

    @objc private func tick(_ link: CADisplayLink) {
        let now = link.timestamp
        let moving = now - lastMove < 0.3
        if moving != isMoving { isMoving = moving }
        let elapsed = now - windowStart
        if elapsed >= 0.5 {
            if moving {
                fps = Int((Double(framesInWindow) / elapsed).rounded())
                if logToConsole { print("[WorldTest] fps \(fps ?? 0) memory \(memoryMB) MB") }
            }
            framesInWindow = 0
            windowStart = now
        }
        if now - lastMemory >= 1 {
            lastMemory = now
            memoryMB = Self.footprintMB()
        }
    }

    static func footprintMB() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.phys_footprint / 1_048_576)
    }
}
