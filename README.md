
Interactive 3D particle scenes controlled by webcam hand gestures.

## Features

- Webcam-based hand tracking with MediaPipe
- Real-time particle rendering with Pygame + PyOpenGL
- Gesture-based mode switching
- Manual controls for rotating, moving, zooming, and mirroring the scene
- A custom love scene with a heart, `WILL U BE MINE?`, sparkle motion, and confetti

## Preview

![Nebula Love Pwn Camera Edition](/Screenshot%202026-06-17%20085223.png)

## Scene Map

| Gesture | Scene |
| --- | --- |
| No hand detected | Random particle idle mode |
| Open hand / five fingers | Solar system |
| Peace sign | `I LOVE PWN` |
| Thumbs up | `PLS IMPRUP` |
| `ILY` sign or finger heart | Love scene |

## Love Gesture

The love scene supports two hand shapes:

- `ILY sign`: thumb, index, and pinky open; middle and ring folded
- `Finger heart`: thumb and index pinched into a small heart gesture

## Love Scene Effects

- Pulsing main heart
- Gold outline/orbit around the heart
- Sparkle motion
- `WILL U BE MINE?` particle text
- Confetti / falling stars while love mode is active

## Requirements

- Windows 10/11 recommended
- Python `3.10` or `3.11`
- Webcam
- OpenGL-capable desktop environment

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
$env:CAMERA_INDEX="1,0,2"
python index.py
```

If the wrong camera opens, try a different index:

```powershell
$env:CAMERA_INDEX="0"
python index.py
```

## Controls

- `0` to `4`: force a mode manually
- `G`: re-enable gesture control
- `Arrow keys`: rotate the scene
- `Q` / `E`: roll
- `W` / `A` / `S` / `D`: move the scene
- `Z` / `X`: zoom in/out
- `M`: mirror text scenes
- `F`: face the scene forward again
- `R`: reset view and mirroring
- `Esc`: quit
- `q`: quit from the camera monitor window

## Notes

## WSL2

Running from Windows Python is strongly recommended.

If you run this from WSL2, webcam access may fail because Linux does not automatically see the Windows camera as `/dev/video*`. In that case:

1. Run the project with Windows Python instead of WSL Python, or
2. Attach the webcam to WSL manually via `usbipd`

Reference:
- https://learn.microsoft.com/en-us/windows/wsl/connect-usb
