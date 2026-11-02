# Vehicle Motion Tracker

Track moving vehicles in a video using **classical computer vision**. There are no neural networks, no training data and no GPU. Just OpenCV.

![Demo](demo.gif)

*Green boxes are tracked vehicles. Yellow lines are the recent paths of tracked points. Blue arrows show the direction and speed of motion. Orange boxes mean the vehicle was briefly lost and its position is being predicted.*

---

## What does it do?

The script takes a traffic video and, for every frame:

1. Finds the moving vehicles.
2. Gives each vehicle a unique ID.
3. Measures how far it moved since the last frame.
4. Draws a box, a motion arrow and a speed label (in pixels per second) on the video.
5. Saves the result as an animated GIF (`demo.gif`).

---

## How it works

The tracker is a four-step pipeline.

| Step | Technique | What it does |
|------|-----------|--------------|
| 1. Detect | **MOG2 background subtraction** | Learns what the empty road looks like. Anything that doesn't match is a moving object, so it becomes a bounding box. |
| 2. Pick points | **Shi-Tomasi corner detection** | Finds sharp, distinctive corners on the vehicles. Flat areas such as plain road can't be tracked reliably, but corners can. |
| 3. Track points | **Lucas-Kanade optical flow** | Follows each corner from one frame to the next. It assumes a point keeps the same brightness as it moves. |
| 4. Measure | **Median motion + smoothing** | Takes the median movement of the points inside each box, then smooths it over time so the speed label stays steady. |

### Keeping the tracking reliable

- **Forward-backward check:** each point is tracked forward and then backward. If it doesn't return to where it started, it's discarded as an outlier.
- **Persistent tracks:** each vehicle keeps its ID even if its box flickers. If it disappears for up to 5 frames, the tracker keeps predicting where it should be.
- **Fallback speed:** if too few points are inside a box, speed is estimated from the box's own movement. This is why the label doesn't disappear.

---

## Setup

**Requirements:** Python 3.8 or newer.

```bash
pip install opencv-python numpy pillow
```

You also need a traffic video named `traffic.mp4` placed next to the script. It works best with a **fixed camera** (highway or intersection footage). Free clips are available on Pexels and Pixabay.

---

## Run it

```bash
python main2.py
```

Or use a different video:

```bash
python main2.py path/to/your_video.mp4
```

- A live preview window opens. Press **Esc** to stop early.
- When it finishes (or you stop it), `demo.gif` is saved in the same folder.

---

## Project structure

```
WEEK3/
├── main2.py       # the tracker
├── traffic.mp4    # input video
├── demo.gif       # output animation
└── README.md
```

---

## Settings you can change

These are at the top of `main2.py`.

| Setting | Default | What it controls |
|---------|---------|------------------|
| `MAX_FRAMES` | 200 | How many frames to process. |
| `MIN_AREA` | 800 | Smallest object (in pixels) counted as a vehicle. Lower it to catch small or distant cars. |
| `WIDTH` | 640 | Video is resized to this width. Smaller means faster and a smaller GIF. |
| `REDETECT_EVERY` | 3 | How often new points are added. |
| `FB_THRESH` | 1.5 | Strictness of the forward-backward check. Lower is stricter. |
| `MATCH_DIST` | 80 | Max distance for matching a box to an existing vehicle. Lower it if IDs swap between close cars. |
| `MAX_MISSED` | 5 | Frames a vehicle can vanish before its track is dropped. |
| `SMOOTH` | 0.7 | Speed smoothing. Raise toward 0.85 for steadier labels. |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot open video` | Check that `traffic.mp4` is in the same folder as the script and plays in a normal video player. |
| `No module named ...` | Install packages with the same Python you run the script with: `python -m pip install opencv-python numpy pillow` |
| No cars detected at the start | The background model needs about 20 to 30 frames to learn. Early frames are weak. |
| Too many false boxes (shadows, trees) | Raise `MIN_AREA`. |
| GIF is too large | Lower `WIDTH` or `MAX_FRAMES`. |

---

## Limitations

- Needs a **static camera**. Background subtraction fails if the camera moves.
- Speed is in **image pixels per second**, not real-world km/h. Converting would need camera calibration.
- Vehicles that overlap heavily may merge into one box.
- Strong shadows and lighting changes can create false detections.

---

## A note on ORB and SIFT

This project tracks with Lucas-Kanade optical flow. ORB and SIFT are an alternative approach: they detect and describe keypoints in each frame, then **match** them between frames.

- **ORB:** fast binary descriptors, compared with Hamming distance.
- **SIFT:** robust to scale and rotation, compared with L2 distance.
- Wrong matches are filtered with **Lowe's ratio test** and **RANSAC**.

Lucas-Kanade is smoother and faster for small frame-to-frame motion, which is why it suits video tracking here.

---

## Built with

- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
