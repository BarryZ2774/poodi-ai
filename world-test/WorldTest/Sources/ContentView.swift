import SwiftUI

struct ContentView: View {
    @Bindable var model: AppModel

    var body: some View {
        ZStack {
            Group {
                switch model.engine {
                case .ours:
                    OursMapView(model: model, command: model.command, colourPlaces: model.colourPlaces)
                case .mapbox:
                    MapboxTab(model: model)
                }
            }
            .ignoresSafeArea()

            VStack(spacing: 10) {
                Picker("Map", selection: $model.engine) {
                    ForEach(Engine.allCases) { Text($0.title).tag($0) }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 260)

                HStack {
                    PerfReadout(perf: model.perf)
                    Spacer()
                }

                Spacer()

                VStack(spacing: 10) {
                    HStack(spacing: 6) {
                        presetButton(.downtown)
                        Menu {
                            ForEach(model.landmarks.filter { $0.model != nil }) { landmark in
                                Button(landmark.name) { model.go(to: landmark) }
                            }
                        } label: {
                            Text("Landmarks")
                        }
                        .menuOrder(.fixed)          // A to Z from the top, even when the menu opens upwards
                        .font(.subheadline)
                        .lineLimit(1)
                        .fixedSize()
                        .buttonStyle(.bordered)
                        .tint(.primary)
                        .disabled(model.landmarks.allSatisfy { $0.model == nil })
                        presetButton(.top)
                        presetButton(.model)
                    }
                    Toggle("Colour my places", isOn: $model.colourPlaces)
                        .font(.subheadline)
                        .tint(Color(red: 0.85, green: 0.51, blue: 0.36))
                        .frame(maxWidth: 260)
                }
                .padding(12)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))

                if model.engine == .ours {
                    HStack {
                        Text("© OpenStreetMap contributors, Overture Maps Foundation")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .onAppear { model.perf.start() }
        .task {
            // Launch argument "-autoTour YES": fly through the presets in a loop (for measuring fps).
            guard UserDefaults.standard.bool(forKey: "autoTour") else { return }
            var index = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(4))
                index = (index + 1) % CameraPreset.allCases.count
                model.go(to: CameraPreset.allCases[index])
            }
        }
        .onChange(of: model.engine) { model.perf.reset() }
    }
}

extension ContentView {
    private func presetButton(_ preset: CameraPreset) -> some View {
        Button(preset.title) { model.go(to: preset) }
            .font(.subheadline)
            .lineLimit(1)
            .fixedSize()
            .buttonStyle(.bordered)
            .tint(.primary)
    }
}

private struct PerfReadout: View {
    let perf: PerfMonitor

    var body: some View {
        let fpsText = perf.fps.map { "\($0)" } ?? "–"
        Text("\(fpsText) fps\(perf.isMoving ? "" : " (idle)") · max \(perf.maxFPS) · \(perf.memoryMB) MB")
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.thinMaterial, in: Capsule())
    }
}
