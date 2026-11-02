import os
import sys
import cv2
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "traffic.mp4")
OUT_GIF = os.path.join(BASE, "demo.gif")
MAX_FRAMES = 200
REDETECT_EVERY = 3
MAX_TRAIL = 20
MIN_AREA = 800
WIDTH = 640
FB_THRESH = 1.5
MATCH_DIST = 80
MAX_MISSED = 5
SMOOTH = 0.7

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 30.0

bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def get_foreground(frame):
    fg = bg.apply(frame)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg


def get_boxes(fg):
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > MIN_AREA]


def add_new_points(gray, fg, pts, trails):
    mask = cv2.dilate(fg, kernel, iterations=2)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=300, qualityLevel=0.01, minDistance=7, mask=mask
    )
    if corners is None:
        return pts, trails
    for c in corners:
        p = c.reshape(1, 1, 2)
        if len(pts) == 0 or np.min(np.linalg.norm(pts - p, axis=2)) > 7:
            pts = np.concatenate([pts, p], axis=0) if len(pts) else p
            trails.append([tuple(p.ravel())])
    return pts, trails


def box_flow(box, trails):
    x, y, bw, bh = box
    steps = []
    for t in trails:
        if len(t) < 2:
            continue
        px, py = t[-1]
        if x <= px <= x + bw and y <= py <= y + bh:
            steps.append((t[-1][0] - t[-2][0], t[-1][1] - t[-2][1]))
    if len(steps) >= 3:
        return np.median(steps, axis=0)
    return None


prev_gray = None
pts = np.empty((0, 1, 2), dtype=np.float32)
trails = []
tracks = {}
next_id = 0
frames = []
idx = 0

while idx < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (WIDTH, int(h * WIDTH / w)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg = get_foreground(frame)
    boxes = get_boxes(fg)

    if prev_gray is not None and len(pts) > 0:
        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk_params)
        back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, new_pts, None, **lk_params)
        fb_error = np.linalg.norm(pts - back_pts, axis=2).ravel()
        good = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb_error < FB_THRESH)
        pts = new_pts[good].reshape(-1, 1, 2)
        trails = [t + [tuple(p.ravel())] for t, p, g in zip(trails, new_pts, good) if g]
        trails = [t[-MAX_TRAIL:] for t in trails]

    if idx % REDETECT_EVERY == 0:
        pts, trails = add_new_points(gray, fg, pts, trails)

    used = set()
    for tid, t in tracks.items():
        pred = np.array(t["c"]) + t["v"]
        best, best_d = None, MATCH_DIST
        for i, (x, y, bw, bh) in enumerate(boxes):
            if i in used:
                continue
            d = np.linalg.norm(pred - np.array([x + bw / 2, y + bh / 2]))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            t["miss"] += 1
            t["box"] = (int(t["box"][0] + t["v"][0]), int(t["box"][1] + t["v"][1]),
                        t["box"][2], t["box"][3])
            t["c"] = (t["c"][0] + t["v"][0], t["c"][1] + t["v"][1])
            continue
        used.add(best)
        x, y, bw, bh = boxes[best]
        c_new = (x + bw / 2, y + bh / 2)
        flow = box_flow(boxes[best], trails)
        raw = flow if flow is not None else np.array([c_new[0] - t["c"][0], c_new[1] - t["c"][1]])
        t["v"] = SMOOTH * t["v"] + (1 - SMOOTH) * raw
        t["c"] = c_new
        t["box"] = boxes[best]
        t["miss"] = 0

    for i, b in enumerate(boxes):
        if i not in used:
            x, y, bw, bh = b
            tracks[next_id] = {"c": (x + bw / 2, y + bh / 2), "v": np.zeros(2), "box": b, "miss": 0}
            next_id += 1

    tracks = {k: v for k, v in tracks.items() if v["miss"] <= MAX_MISSED}

    vis = frame.copy()

    for t in trails:
        for a, b in zip(t[:-1], t[1:]):
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 255), 1)
        cv2.circle(vis, (int(t[-1][0]), int(t[-1][1])), 2, (0, 0, 255), -1)

    for tid, t in tracks.items():
        x, y, bw, bh = t["box"]
        color = (0, 255, 0) if t["miss"] == 0 else (0, 165, 255)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
        dx, dy = t["v"]
        cx, cy = int(x + bw / 2), int(y + bh / 2)
        cv2.arrowedLine(vis, (cx, cy), (int(cx + dx * 5), int(cy + dy * 5)), (255, 0, 0), 2, tipLength=0.4)
        speed = float(np.hypot(dx, dy)) * fps
        cv2.putText(vis, f"ID{tid} {speed:.0f}px/s", (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(vis, f"Vehicles: {len(tracks)}  Points: {len(pts)}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if idx % 2 == 0:
        frames.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    cv2.imshow("Vehicle Motion Tracker", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    prev_gray = gray
    idx += 1

cap.release()
cv2.destroyAllWindows()
if frames:
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(OUT_GIF, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"Saved {OUT_GIF} with {len(frames)} frames")
else:
    print("No frames captured. Check the video path and format.")