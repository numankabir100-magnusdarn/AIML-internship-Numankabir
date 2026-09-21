# AIML-internship-Numankabir
Starting a new internship to excel my skills .All the work will be loaded here

Iam right now seconf semester Student of Cs and i Have chossen this path to lead myself ahead of all

---

# Predictive Spatial Intelligence System

Phase 1 provides live webcam detection and tracking with YOLOv8 and ByteTrack, plus a simple per-track velocity estimate.

## Setup

1. Run `setup.ps1` from the project folder to create `.venv` and install dependencies.
2. Start the app with `run.ps1`.
3. If you prefer manual commands, activate `.venv` and run `python main.py`.

## Controls

- `q` or `Esc` to exit.
- `c` to cycle the camera source (e.g. real webcam ↔ OBS virtual camera).
- `r` to reset the live session stats.
- The active source name is shown in the top-left corner, and a live session
  stats panel (uptime, active/total objects, alert/caution/monitor events,
  risk pairs, avg FPS) is overlaid in the top-right corner.
  Decision events are counted per episode: status changes from anything else
  into a given state count as one event, regardless of how many frames the
  state persists.

Configured camera sources live in `src/config.py` (`CAMERA_SOURCES`); the
default startup source is `AppConfig.camera_source_name`.