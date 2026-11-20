# Running World Test on your iPhone

About five minutes the first time.

1. **Plug the phone into the Mac mini** with a cable and unlock it. If it asks "Trust This Computer?", tap **Trust** and enter the phone's passcode.
2. **Open the project:** double-click `WorldTest.xcodeproj` in `/Users/claudedev/Developer/poodi-ai/world-test`. Xcode opens; the first time it fetches its packages, so wait until "Resolving packages" (top of the window) finishes.
3. **Pick your team (only once):** in the left sidebar click the blue **WorldTest** icon → target **WorldTest** → tab **Signing & Capabilities** → **Team** → choose your Apple account. If the list is empty, choose **Add an Account…** and sign in with your Apple ID; a free account is fine.
   *If Xcode says the bundle identifier `dev.worldtest.app` is not available, change it to something unique such as `dev.worldtest.app2`.*
4. **Pick the phone** in the device menu at the top of the Xcode window (it shows a simulator name until you change it).
5. **Turn on Developer Mode if Xcode asks:** on the phone, Settings → Privacy & Security → **Developer Mode** → On. The phone restarts; unlock it and tap **Turn On**.
6. **Press Run** (the ▶ button, or ⌘R). The first build takes a minute or two.
7. **If the phone says "Untrusted Developer":** Settings → General → **VPN & Device Management** → your Apple ID → **Trust**. Then tap the World Test icon on the home screen.

## Mapbox tab

The Mapbox tab needs a Mapbox **public** token (it starts with `pk.`; the "Default public token" at account.mapbox.com → Tokens works).

1. In Xcode's left sidebar open **Config → Secrets.xcconfig**.
2. Paste the token after the equals sign: `MAPBOX_ACCESS_TOKEN = pk.your-token`
3. Press **Run** again.

Without a token the tab shows "Add a Mapbox token in Secrets.xcconfig" and everything else still works. The file is not in git.

## What to look at

**New tonight:**

- **Pinch out** to zoom 11 or 12: the whole city now stays built up, every building is there instead of patches.
- **Model** flies to a test tower (60 m wide, 200 m tall) on Alcatraz, drawn by our own code from a `.glb` file. Pan, tilt and rotate around it: it should stay put and match the buildings' light.
- **Bridge** now shows the Golden Gate Bridge model from the landmark builders. The map's own bridge towers are still underneath it.
- **Colour my places** also turns both models terracotta.

As before:

- **Ours / Mapbox** switch at the top. Both share one camera, so switching compares the same view.
- **Downtown, Bridge, Top, Model** fly to fixed views. Otherwise pan, pinch, rotate with two fingers, and tilt by dragging two fingers up or down.
- The small readout at top left: **frames per second while the map moves** (it says "idle" when still), the screen's maximum rate (120 on ProMotion), and the app's **memory** in MB.
- **Colour my places** turns about twelve sample landmarks terracotta.
- Unfold the phone to try the large screen; the app supports both screens and landscape.

With a free Apple account the app stops opening after 7 days; press Run again to renew it.
