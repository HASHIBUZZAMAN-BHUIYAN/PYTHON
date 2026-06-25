# CNN Reference Folder

Architecture cheat-sheet and common patterns.

| File | Contents |
|------|----------|
| `cnn_architectures_cheatsheet.py` | LeNet, VGG-mini, ResNet-block, MobileNet-style, transfer learning skeleton |

## Related lessons
- ADVANCED/week2/day11 — CNN fundamentals
- ADVANCED/week2/day12 — CNN tuning
- ADVANCED/week2/day13 — Transfer learning (MobileNetV2)
- ADVANCED/week2/day14 — CNN capstone

## Runnable Projects

All commands are run from the `PYTHON/` folder using the venv Python.
Outputs (PNG plots) are saved to `CNN/outputs/`.

| File | Description | Run command |
|------|-------------|-------------|
| `road_scene_understanding.py` | Synthetic road-scene CNN 4-class classifier (car/sign detection) | `python CNN\road_scene_understanding.py` |
| `lane_and_edge_detection.py` | Classical CV lane detection: Gaussian blur -> Canny -> Hough lines | `python CNN\lane_and_edge_detection.py` |
| `human_pose_keypoints.py` | Stick-figure keypoint detector via thresholding + findContours | `python CNN\human_pose_keypoints.py` |
| `face_emotion_simple.py` | Haar cascade demo + tiny CNN for happy/sad synthetic face emotion | `python CNN\face_emotion_simple.py` |
| `medical_image_classifier.py` | Synthetic lesion vs. normal binary CNN with sensitivity/specificity | `python CNN\medical_image_classifier.py` |
| `cnn_feature_map_visualizer.py` | Forward-hook feature map visualization across 3 conv layers | `python CNN\cnn_feature_map_visualizer.py` |

> **Hardware note:** All scripts run on CPU only (no GPU required). Tested on
> Ryzen 7, 8 GB RAM. Peak RAM ~300 MB, longest run ~2 minutes.
> All datasets are synthetically generated — no file downloads needed.
