# Air Drawing Canvas

Draw in the air using your index finger — powered by OpenCV + MediaPipe.  
Includes a side slider for brush thickness, an eraser, background toggle (black/white), a configurable grid, and quick save/quit keys.

## Features

- Real-time index-finger tracking (MediaPipe Hands).
- Vertical thickness slider on the left (hover with index to adjust).
- Colors: Red / Green / Blue.
- Eraser mode (uses background color with a larger stroke).
- Clear All button and palm-open gesture to clear (hold briefly).
- Toggle background (black/white) without destroying the drawing.
- Grid overlay with adjustable spacing (G- / G+).
- Save your work with `s` (raw drawing + preview overlay).

## Requirements

- Python 3.10+ (tested on Windows; works on macOS/Linux with a webcam)

Install:

```bash
pip install -r requirements.txt


Run:
python air_drawing_canvas.py

Controls
-Draw: raise index finger only.
-Thickness: hover index over the left slider to change.
-Buttons (bottom bar):
    -Red / Green / Blue: change brush color
    -Eraser: switch to eraser mode
    -BG: toggle background (black/white)
    -G- / G+: decrease/increase grid spacing
    -Clear All: clear the canvas

Keyboard:
    - s — Save (drawing_YYYYMMDD-HHMMSS.png + preview_*.png)
    - q — Quit

Notes & Troubleshooting
    -MediaPipe / TFLite warnings on startup are common and safe to ignore.

    -If the webcam feed is dark, add more light or try a different camera.

    -If the color indicator ever gets hidden, make sure the info panel doesn’t overlap the top bar (this build already places the panel lower).

    -On Windows terminals with legacy encodings, avoid printing emojis to prevent UnicodeEncodeError.

Project Structure
air-drawing-canvas/
├─ air_drawing_canvas.py
├─ requirements.txt
├─ .gitignore
└─ README.md

```
