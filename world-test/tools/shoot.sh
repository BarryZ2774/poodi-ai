#!/bin/zsh
# Builds, installs and screenshots the Ours tab on a simulator.
# Usage: tools/shoot.sh <simulator-udid> [out-dir]
set -euo pipefail
cd "$(dirname "$0")/.."
SIM="$1"; OUT="${2:-shots}"; mkdir -p "$OUT"
xcodebuild -project WorldTest.xcodeproj -scheme WorldTest -destination "platform=iOS Simulator,id=$SIM" \
  -derivedDataPath build/dd -clonedSourcePackagesDirPath build/spm build -quiet
xcrun simctl install "$SIM" build/dd/Build/Products/Debug-iphonesimulator/WorldTest.app
shot() { # name, launch args...
  local name="$1"; shift
  xcrun simctl terminate "$SIM" dev.worldtest.app 2>/dev/null || true
  xcrun simctl launch "$SIM" dev.worldtest.app "$@" >/dev/null
  sleep 12
  # Display 1 is the outer (folded) screen on the iPhone Duo simulator; drop the flag on other phones.
  xcrun simctl io "$SIM" screenshot ${DISPLAY_FLAG:---display=1} "$OUT/$name.png" >/dev/null
  echo "$OUT/$name.png"
}
shot ours-downtown -preset downtown -colour NO
shot ours-bridge -preset bridge -colour NO
shot ours-top -preset top -colour NO
shot ours-downtown-coloured -preset downtown -colour YES
