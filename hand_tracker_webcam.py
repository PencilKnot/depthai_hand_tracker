#!/usr/bin/env python3
"""
hand_tracker_webcam.py

Camera-free development equivalent of HandTracker.py + HandTrackerRenderer.py.
Uses the current MediaPipe Tasks API (mediapipe >= 0.10).
Produces the same HandRegion data structure as the OAK-D pipeline so your
XY coordinate logic transfers directly.

Model download (run once):
    python hand_tracker_webcam.py --download-models

Usage:
    python hand_tracker_webcam.py
    python hand_tracker_webcam.py --input path/to/video.mp4
    python hand_tracker_webcam.py --gesture          # enable gesture recognition
    python hand_tracker_webcam.py --solo             # detect one hand only
"""

import cv2
import numpy as np
import argparse
import urllib.request
import os
import time
from math import atan2, pi, floor, cos, sin
from collections import deque


# ---------------------------------------------------------------------------
# Model URLs — these are the current canonical MediaPipe model files.
# The hand_landmarker.task bundles palm detection + landmark model together.
# The gesture_recognizer.task bundles all three (palm + landmark + gesture).
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "hand_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "gesture_recognizer": (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = os.path.join(MODEL_DIR, f"{name}.task")
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")


# ---------------------------------------------------------------------------
# HandRegion — mirrors the HandRegion class in mediapipe_utils.py exactly.
# XY logic should read from this object the same way it will on OAK-D.
# ---------------------------------------------------------------------------
class HandRegion:
    """
    Mirrors mediapipe_utils.HandRegion.

    Key attributes (same names as in HandTracker.py):
      lm_score        : float, global landmark confidence
      handedness      : float, >0.5 = right hand, <0.5 = left hand
      label           : str, "right" or "left"
      landmarks       : np.array shape (21, 2), pixel XY in source image
      norm_landmarks  : np.array shape (21, 3), normalized [0-1] XY + Z depth
      world_landmarks : np.array shape (21, 3), metric-scale 3D (metres)
      gesture         : str or None, gesture name if recognised
      pd_score        : float, palm detection confidence
      pd_box          : [x, y, w, h] normalised, palm bounding box
    """

    def __init__(self):
        self.pd_score = None
        self.pd_box = None
        self.pd_kps = None
        self.lm_score = None
        self.handedness = None
        self.label = None
        self.norm_landmarks = None
        self.landmarks = None
        self.world_landmarks = None
        self.gesture = None
        # Drone-relevant extras
        self.thumb_state = -1
        self.index_state = -1
        self.middle_state = -1
        self.ring_state = -1
        self.little_state = -1

    def __repr__(self):
        return (
            f"HandRegion(label={self.label}, lm_score={self.lm_score:.2f}, "
            f"gesture={self.gesture})"
        )


# ---------------------------------------------------------------------------
# FPS utility
# ---------------------------------------------------------------------------
class FPS:
    def __init__(self, average_of=30):
        self.timestamps = deque(maxlen=average_of)
        self.nbf = -1

    def update(self):
        self.timestamps.append(time.monotonic())
        if len(self.timestamps) == 1:
            self.start = self.timestamps[0]
            self.fps = 0
        else:
            self.fps = (len(self.timestamps) - 1) / (
                self.timestamps[-1] - self.timestamps[0]
            )
        self.nbf += 1

    def get(self):
        return self.fps

    def draw(self, frame, orig=(10, 30), size=1, color=(0, 255, 0), thickness=2):
        cv2.putText(
            frame,
            f"FPS={self.get():.1f}",
            orig,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            color,
            thickness,
        )


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------
LINES_HAND = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
]

# MediaPipe gesture name -> display string mapping
# Matches the gesture names from the gesture recognizer model card
GESTURE_MAP = {
    "Closed_Fist": "FIST",
    "Open_Palm": "FIVE",
    "Pointing_Up": "ONE",
    "Thumb_Down": "THUMB_DOWN",
    "Thumb_Up": "THUMB_UP",
    "Victory": "PEACE",
    "ILoveYou": "ILY",
    "None": None,
}


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------
class HandTrackerWebcam:
    """
    Drop-in development replacement for HandTracker (without OAK-D).

    Provides the next_frame() interface:
        frame, hands, bag = tracker.next_frame()

    'hands' is a list of HandRegion objects with the same attributes
    as the OAK-D pipeline produces.
    """

    def __init__(
        self,
        input_src=None,
        use_gesture=False,
        solo=False,
        use_world_landmarks=True,
        lm_score_thresh=0.5,
    ):
        self.use_gesture = use_gesture
        self.solo = solo
        self.use_world_landmarks = use_world_landmarks
        self.lm_score_thresh = lm_score_thresh
        self.fps = FPS()

        # Import here so the file is still importable without mediapipe installed
        import mediapipe as mp

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        max_hands = 1 if solo else 2

        if use_gesture:
            model_path = os.path.join(MODEL_DIR, "gesture_recognizer.task")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    "Run:  python hand_tracker_webcam.py --download-models"
                )
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._recognizer = GestureRecognizer.create_from_options(options)
            self._mode = "gesture"
        else:
            model_path = os.path.join(MODEL_DIR, "hand_landmarker.task")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    "Run:  python hand_tracker_webcam.py --download-models"
                )
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._recognizer = HandLandmarker.create_from_options(options)
            self._mode = "landmark"

        self._mp_image_cls = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB

        # Open video source
        if input_src is None:
            self.cap = cv2.VideoCapture(0)
        elif isinstance(input_src, str):
            self.cap = cv2.VideoCapture(input_src)
        else:
            self.cap = cv2.VideoCapture(input_src)

        self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = max(self.img_w, self.img_h)
        self.pad_w = (self.frame_size - self.img_w) // 2
        self.pad_h = (self.frame_size - self.img_h) // 2
        self._frame_ts = 0  # millisecond timestamp for VIDEO mode

    def _build_hand_region(self, idx, result, img_w, img_h):
        """
        Convert one MediaPipe hand result into a HandRegion object.
        Mirrors the lm_postprocess() logic in HandTracker.py.
        """
        hand = HandRegion()

        # --- Handedness ---
        if result.handedness:
            hd = result.handedness[idx][0]
            # MediaPipe reports from mirrored camera: flip label
            hand.label = "right" if hd.category_name == "Right" else "left"
            hand.handedness = hd.score if hand.label == "left" else 1 - hd.score

        # --- Screen landmarks (normalised [0-1] in the source image) ---
        if result.hand_landmarks:
            lms = result.hand_landmarks[idx]
            hand.norm_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32
            )
            # Convert to pixel coordinates in source image
            hand.landmarks = np.array(
                [[int(lm.x * img_w), int(lm.y * img_h)] for lm in lms], dtype=np.int32
            )
            # for lm in lms[:3]:
            #     print(vars(lm))
            # lm_score: use median visibility as proxy for OAK-D's lm_score
            # visibilities = [lm.visibility for lm in lms]
            # hand.lm_score = float(np.median(visibilities))
            visibilities = [
                lm.visibility
                for lm in lms
                if hasattr(lm, "visibility") and lm.visibility is not None
            ]

            if visibilities:
                hand.lm_score = float(np.median(visibilities))
            else:
                hand.lm_score = 1.0

        # --- World landmarks (metric scale, metres, hand-centred origin) ---
        if self.use_world_landmarks and result.hand_world_landmarks:
            wlms = result.hand_world_landmarks[idx]
            hand.world_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in wlms], dtype=np.float32
            )

        # --- Gesture (only when using GestureRecognizer) ---
        if self.use_gesture and hasattr(result, "gestures") and result.gestures:
            raw = result.gestures[idx][0].category_name
            print(f"[DEBUG] Raw gesture: {raw}")
            hand.gesture = GESTURE_MAP.get(raw, raw)

        # --- Palm bounding box (approximate from landmarks) ---
        if hand.norm_landmarks is not None:
            xs = hand.norm_landmarks[:, 0]
            ys = hand.norm_landmarks[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            hand.pd_box = [x_min, y_min, x_max - x_min, y_max - y_min]
            hand.pd_score = hand.lm_score

        return hand

    def next_frame(self):
        """
        Mirrors HandTracker.next_frame().
        Returns: (frame, hands, bag)
          frame : BGR np.array or None at end of stream
          hands : list of HandRegion
          bag   : dict (empty here, used for body pre-focusing in OAK-D version)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, [], {}

        self.fps.update()
        self._frame_ts += 33  # ~30 fps timestamp in ms for VIDEO mode

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)

        result = self._recognizer.recognize_for_video(mp_image, self._frame_ts)

        hands = []
        n_hands = len(result.hand_landmarks) if result.hand_landmarks else 0
        for i in range(n_hands):
            hand = self._build_hand_region(i, result, self.img_w, self.img_h)
            if hand.lm_score is not None and hand.lm_score >= self.lm_score_thresh:
                hands.append(hand)

        return frame, hands, {}

    def exit(self):
        self.cap.release()


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
class HandTrackerRenderer:
    def __init__(self, tracker, output=None):
        self.tracker = tracker
        self.show_landmarks = True
        self.show_gesture = tracker.use_gesture
        self.show_fps = True
        self.show_world = False  # toggle with 'w' to print world coords

        if output:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(
                output, fourcc, 30, (tracker.img_w, tracker.img_h)
            )
        else:
            self.output = None

    def draw_hand(self, frame, hand):
        if hand.landmarks is None:
            return

        lms = hand.landmarks
        thick = max(1, int(hand.pd_box[2] * frame.shape[1] / 200)) if hand.pd_box else 2

        # Draw skeleton lines
        color = (0, 255, 0) if hand.label == "right" else (255, 100, 0)
        for a, b in LINES_HAND:
            cv2.line(frame, tuple(lms[a]), tuple(lms[b]), color, thick, cv2.LINE_AA)

        # Draw landmark dots
        for x, y in lms:
            cv2.circle(frame, (x, y), thick + 2, (0, 128, 255), -1)

        # Handedness label
        wrist_x, wrist_y = lms[0]
        cv2.putText(
            frame,
            f"{hand.label.upper()} {hand.handedness:.2f}",
            (wrist_x - 60, wrist_y + 30),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0) if hand.handedness > 0.5 else (0, 0, 255),
            2,
        )

        # Gesture label
        if self.show_gesture and hand.gesture:
            print("gesturing")
            tip_y = lms[:, 1].min()
            cx = int(lms[:, 0].mean())
            cv2.putText(
                frame,
                hand.gesture,
                (cx - 40, tip_y - 20),
                cv2.FONT_HERSHEY_PLAIN,
                2.5,
                (255, 255, 255),
                3,
            )

        # World landmark printout (toggled with 'w')
        if self.show_world and hand.world_landmarks is not None:
            wl = hand.world_landmarks
            # Index fingertip in metres from hand centre
            ix, iy, iz = wl[8]
            cv2.putText(
                frame,
                f"[8] x:{ix * 100:.1f}cm y:{iy * 100:.1f}cm z:{iz * 100:.1f}cm",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (200, 200, 0),
                2,
            )

    def draw(self, frame, hands, bag=None):
        for hand in hands:
            self.draw_hand(frame, hand)
        return frame

    def waitKey(self, delay=1):
        key = cv2.waitKey(delay)
        if key == ord("w"):
            self.show_world = not self.show_world
        elif key == ord("g") and self.tracker.use_gesture:
            # elif key == ord("g"):
            print("gesture toggled")
            self.show_gesture = not self.show_gesture
        elif key == ord("l"):
            self.show_landmarks = not self.show_landmarks
        elif key == ord("f"):
            self.show_fps = not self.show_fps
        elif key == 32:
            cv2.waitKey(0)
        return key

    def exit(self):
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-models", action="store_true", help="Download model files and exit"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Path to video/image (default: webcam)",
    )
    parser.add_argument(
        "-g", "--gesture", action="store_true", help="Enable gesture recognition"
    )
    parser.add_argument(
        "-s", "--solo", action="store_true", help="Solo mode: detect one hand only"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Path to output video file"
    )
    parser.add_argument(
        "--lm-thresh",
        type=float,
        default=0.5,
        help="Landmark confidence threshold (default=0.5)",
    )
    args = parser.parse_args()

    if args.download_models:
        print("Downloading models...")
        download_models()
        print("Done.")
        return

    tracker = HandTrackerWebcam(
        input_src=args.input,
        use_gesture=args.gesture,
        solo=args.solo,
        use_world_landmarks=True,
        lm_score_thresh=args.lm_thresh,
    )

    renderer = HandTrackerRenderer(tracker=tracker, output=args.output)

    print(
        "Controls:  q/ESC=quit  w=world coords  g=gesture  l=landmarks  f=fps  space=pause"
    )

    while True:
        frame, hands, bag = tracker.next_frame()
        if frame is None:
            break

        # ----------------------------------------------------------------
        # YOUR XY COORDINATE LOGIC GOES HERE
        # 'hands' is a list of HandRegion objects.
        # Access landmarks the same way HandTracker.py does:
        #
        #   for hand in hands:
        #       # Pixel XY of index fingertip (landmark 8)
        #       ix, iy = hand.landmarks[8]
        #
        #       # Normalised [0-1] XY of wrist (landmark 0)
        #       wx, wy = hand.norm_landmarks[0, :2]
        #
        #       # 3D world coords of all landmarks (metres, hand-centred)
        #       wl = hand.world_landmarks  # shape (21, 3)
        #
        #       # Gesture string (if --gesture flag used)
        #       print(hand.gesture)
        # ----------------------------------------------------------------

        frame = renderer.draw(frame, hands, bag)
        renderer._last_frame = frame

        # Draw FPS manually (renderer.waitKey reads _last_frame)
        if renderer.show_fps:
            tracker.fps.draw(
                frame, orig=(50, 50), size=1.5, color=(240, 180, 100), thickness=2
            )

        cv2.imshow("Hand tracking", frame)
        if renderer.output:
            renderer.output.write(frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break
        if key == ord("w"):
            renderer.show_world = not renderer.show_world
        if key == ord("g") and tracker.use_gesture:
            # elif key == ord("g"):
            print("gesture toggled")
            renderer.show_gesture = not renderer.show_gesture
        if key == ord("f"):
            renderer.show_fps = not renderer.show_fps
        if key == 32:
            cv2.waitKey(0)

    renderer.exit()
    tracker.exit()


if __name__ == "__main__":
    main()
