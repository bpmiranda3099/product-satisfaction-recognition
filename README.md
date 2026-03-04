# Product Satisfaction Recognition

Real-time computer-vision pipeline for detecting handled products and estimating user reaction while the product is being handled.

## Overview

This project combines object detection, pose estimation, hand landmark tracking, and face-based reaction inference into a single live recognition app.

## Features

- Live webcam or video-file processing.
- Product handling detection (single highest-confidence handled object per frame).
- Body and hand keypoint overlays.
- Face reaction inference only during product handling.

## Architecture

```text
product-satisfaction-recognition/
├── main.py
├── requirements.txt
├── LICENSE
├── .gitattributes
├── .gitignore
├── models/
│   ├── yolov8n.pt
│   └── yolo26n-pose.pt
└── src/
    └── app/
        ├── __init__.py
        ├── app.py
        ├── config.py
        ├── services/
        │   ├── capture.py
        │   ├── reaction.py
        │   └── vision.py
        └── utils/
            └── geometry.py
```

## Requirements

- Python 3.10+
- macOS/Linux/Windows with camera access
- Optional: `git-lfs` for model weight tracking

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run with webcam (default):

```bash
python main.py --source 0
```

Run with video file:

```bash
python main.py --source path/to/video.mp4
```

## Configuration

Default runtime settings are in:

- `src/app/config.py`

Key values include:

- object confidence threshold
- keypoint confidence threshold
- handling IoU threshold
- model filenames in `models/`

## Models

The app expects model files in `models/`:

- `yolov8n.pt`
- `yolo26n-pose.pt`

## Limitations

- Reaction readings are heuristic and may be inaccurate in some lighting, camera angles, face shapes, occlusions, and motion conditions.
- Output should be treated as experimental feedback, not a ground-truth emotional assessment.
- Further calibration, model tuning, and dataset-driven validation are required for production-grade reliability.

## Troubleshooting

- Camera open failure on macOS: enable camera permission for Terminal/iTerm/VS Code under Privacy & Security.
- Slow first run: initial model load/download may take time.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository and create a feature branch.
2. Keep changes focused and aligned with current architecture (`src/app/services`, `src/app/utils`).
3. Run a local sanity check before submitting:

```bash
python -m py_compile main.py src/app/*.py src/app/services/*.py src/app/utils/*.py
```

4. Open a pull request with:
   - what changed
   - why it changed
   - any known tradeoffs or follow-up work

## License

MIT. See [LICENSE](LICENSE).
