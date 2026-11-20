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

- **Pinch out** to zoom 11 or 12: the whole city stays built up, every building is there instead of patches.
- **Landmarks** (the button next to Downtown) lists every landmark in the app. Pick one and the map flies to it, seen at an angle, sized to fit the screen. It replaces the old Bridge button.
- The app now has 17 landmarks:
  - Alcatraz Island
  - Coit Tower
  - Conservatory of Flowers
  - Dutch Windmill
  - Fort Point
  - Golden Gate Bridge
  - Grace Cathedral
  - Legion of Honor
  - Mission Dolores and Basilica
  - Painted Ladies
  - Palace of Fine Arts
  - Saints Peter and Paul Church
  - San Francisco City Hall
  - San Francisco Ferry Building
  - San Francisco–Oakland Bay Bridge, East Span
  - San Francisco–Oakland Bay Bridge, West Span
  - Sutro Tower
- These landmarks are **first drafts**. They are still being reviewed and improved, so some will look rough or slightly off; that is expected.
- Where a landmark stands, the map no longer draws its own plain boxes underneath it, and the white road line no longer runs across the water under the bridges. (One leftover: under part of the Bay Bridge east span, a short white road line still shows when zoomed out a little.)
- **Model** flies to a test tower (60 m wide, 200 m tall) in the water just west of Alcatraz.
- **Colour my places** also turns every landmark terracotta.

What to watch:

- The **memory** number at top left, especially zoomed out to 11 or 12 with the whole city in view. On the Mac's simulator it was about 450 to 530 MB there, and the landmarks add very little to it (at most about 15 MB).
- **Shimmer** while moving: pan, tilt and turn around a landmark (the bridges are the best test) and look for flickering stripes where it meets the map.

To get the newest landmarks later: on the Mac, open Terminal, run the two lines below, then press **Run** in Xcode again.

```
cd /Users/claudedev/Developer/poodi-ai/world-test
tools/pull-landmarks.sh
```

As before:

- **Ours / Mapbox** switch at the top. Both share one camera, so switching compares the same view.
- **Downtown, Top, Model** fly to fixed views. Otherwise pan, pinch, rotate with two fingers, and tilt by dragging two fingers up or down.
- The small readout at top left: **frames per second while the map moves** (it says "idle" when still), the screen's maximum rate (120 on ProMotion), and the app's **memory** in MB.
- **Colour my places** turns about twelve sample landmarks terracotta.
- Unfold the phone to try the large screen; the app supports both screens and landscape.

With a free Apple account the app stops opening after 7 days; press Run again to renew it.
