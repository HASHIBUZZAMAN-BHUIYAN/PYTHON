# CNN Reference Folder

Architecture cheat-sheet and common patterns.

| File | Contents |
|------|----------|
| `cnn_architectures_cheatsheet.py` | LeNet, VGG-mini, ResNet-block, MobileNet-style, transfer learning skeleton |

## Environment Setup

This folder has its **own dedicated virtual environment** (`CNN\.venv`) — separate from every other folder including BEGINNER and ADVANCED.

**From a fresh terminal:**
```
cd C:\Users\zen\Documents\GitHub\PYTHON
CNN\.venv\Scripts\activate
python CNN\road_scene_understanding.py
```

**Or:** double-click `CNN\activate.bat` — it activates the venv and sets the working directory automatically.

Installed packages (see `CNN\requirements.txt`): numpy, matplotlib, scikit-learn, Pillow, opencv-python, torch (CPU), torchvision (CPU)

---

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

---

## Round 2: Real-World Domains, Deeper Versions, and Creative Projects

All commands run from the `PYTHON/` folder. Outputs go to `CNN/outputs/`.

### New Real-World Domains

| File | Description | Run command |
|------|-------------|-------------|
| `satellite_land_classifier.py` | 4-class CNN (water/forest/urban/farmland) on synthetic satellite patches; outputs a land-use prediction map grid | `python CNN\satellite_land_classifier.py` |
| `crop_health_classifier.py` | Binary CNN (healthy vs diseased) on synthetic leaf images with brown/yellow spot patterns; mirrors precision-agriculture pipeline | `python CNN\crop_health_classifier.py` |
| `retail_product_classifier.py` | 3-class CNN (bottle/box/can) on OpenCV-drawn silhouettes; mirrors retail shelf vision systems | `python CNN\retail_product_classifier.py` |
| `wildlife_species_classifier.py` | 3-class CNN (bird/deer/rabbit) on geometric silhouettes; explicitly notes gap vs real camera-trap systems | `python CNN\wildlife_species_classifier.py` |

### Deeper Versions of Existing Projects

| File | Description | Run command |
|------|-------------|-------------|
| `object_detection_simple.py` | Sliding-window detector with NMS on synthetic scenes (1-3 shapes); explains YOLO/SSD as single-pass evolution | `python CNN\object_detection_simple.py` |
| `simple_object_tracker.py` | Centroid tracker across 10 synthetic frames; colour contour detection + greedy min-distance assignment; trajectory plot | `python CNN\simple_object_tracker.py` |
| `pose_based_activity_classifier.py` | Builds on pose keypoints: MLP classifier trained on 16-dim keypoint feature vectors (4 activities: standing/sitting/arms-raised/running) | `python CNN\pose_based_activity_classifier.py` |

### Creative / Fun Projects

| File | Description | Run command |
|------|-------------|-------------|
| `style_filter_artistic.py` | 4 classical OpenCV filters (cartoon, pencil sketch, emboss, oil painting); explicitly NOT neural style transfer; explains VGG-19 approach conceptually | `python CNN\style_filter_artistic.py` |
| `tiny_image_generator.py` | Conv autoencoder on 28x28 synthetic shapes; latent space interpolation (circle->square morphing); t-SNE latent projection; random samples | `python CNN\tiny_image_generator.py` |
| `image_filters_playground.py` | 7 classical convolution kernels (Gaussian/sharpen/Sobel-X/Sobel-Y/Laplacian/emboss/glow) with kernel matrices displayed; ties to CNN first-layer learning | `python CNN\image_filters_playground.py` |

> **Round 2 hardware note:** All run CPU-only. Slowest: `object_detection_simple.py`
> (~60s sliding-window over test scenes). All others under 10s. Zero downloads.
