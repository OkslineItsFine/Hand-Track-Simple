import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "models/hand_landmarker.task"

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

MAX_HANDS = 2

GESTURE_COOLDOWN = 0.45

MODE_NAMES = [
    "CLEAR",
    "XRAY",
    "THERMAL",
    "PIXEL",
    "GLITCH",
    "NEGATIVE",
    "EDGE",
    "INVERT",
    "MONO",
    "NEON",
    "BLUR",
    "SCAN",
    "DATA",
    "VHS",
    "MATRIX",
    "SILHOUETTE",
    "ELECTRIC",
    "HORIZONTAL",
    "RAINBOW"
]


# =========================================================
# HAND TRACKER
# =========================================================

class HandTracker:

    def __init__(self, model_path):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,

            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        self.timestamp = 0

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp += 33

        result = self.landmarker.detect_for_video(
            image,
            self.timestamp
        )

        return result

    def close(self):
        self.landmarker.close()


# =========================================================
# UTILITY
# =========================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def landmark_to_point(landmark, width, height):

    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


# =========================================================
# PINCH DETECTION
# =========================================================

def get_pinch_ratio(hand):

    thumb = np.array([
        hand[4].x,
        hand[4].y
    ])

    index = np.array([
        hand[8].x,
        hand[8].y
    ])

    wrist = np.array([
        hand[0].x,
        hand[0].y
    ])

    middle_base = np.array([
        hand[9].x,
        hand[9].y
    ])

    distance = np.linalg.norm(
        thumb - index
    )

    hand_scale = np.linalg.norm(
        wrist - middle_base
    )

    if hand_scale <= 0:
        return 999

    return distance / hand_scale


def is_pinch(hand):

    ratio = get_pinch_ratio(hand)

    return ratio < 0.55


# =========================================================
# HAND DRAWING
# =========================================================

CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


def draw_hand(frame, hand):

    h, w = frame.shape[:2]

    points = []

    for p in hand:

        point = landmark_to_point(
            p,
            w,
            h
        )

        points.append(point)

    # skeleton
    for a, b in CONNECTIONS:

        cv2.line(
            frame,
            points[a],
            points[b],
            (80, 255, 100),
            2,
            cv2.LINE_AA
        )

    # joints
    for i, point in enumerate(points):

        radius = 5 if i in [4, 8] else 3

        cv2.circle(
            frame,
            point,
            radius,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    return points


# =========================================================
# PORTAL POINTS
# =========================================================

def get_portal_points(hands, width, height):
    """
    Ambil 4 titik portal berdasarkan:
    kiri  = index + thumb tangan kiri
    kanan = index + thumb tangan kanan
    """

    if len(hands) < 2:
        return None

    sorted_hands = sorted(
        hands,
        key=lambda hand: np.mean([p.x for p in hand])
    )

    left = sorted_hands[0]
    right = sorted_hands[1]

    points = np.array([
        landmark_to_point(left[8], width, height),   # left index
        landmark_to_point(right[8], width, height),  # right index
        landmark_to_point(right[4], width, height),  # right thumb
        landmark_to_point(left[4], width, height)    # left thumb
    ], dtype=np.float32)

    return points


def order_portal_points(points):
    """
    Mengembalikan 4 titik langsung agar garis portal diperbolehkan menyilang
    atau silang diagonal sesuai gerakan tangan pengguna.
    """

    if points is None or len(points) != 4:
        return None

    return np.asarray(points, dtype=np.int32)


# =========================================================
# PORTAL MASK
# =========================================================

def create_portal_mask(frame, points):

    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    if points is None:
        return mask

    cv2.fillPoly(
        mask,
        [points.astype(np.int32)],
        255
    )

    return mask


# =========================================================
# CLEAR
# =========================================================

def effect_clear(frame):
    return frame.copy()


# =========================================================
# XRAY
# =========================================================

def effect_xray(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    inverted = 255 - gray

    result = cv2.applyColorMap(
        inverted,
        cv2.COLORMAP_BONE
    )

    return result


# =========================================================
# THERMAL
# =========================================================

def effect_thermal(frame):

    small = cv2.resize(
        frame,
        (320, 180)
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2GRAY
    )

    result = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_JET
    )

    return cv2.resize(
        result,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )


# =========================================================
# PIXEL
# =========================================================

def effect_pixel(frame):

    small = cv2.resize(
        frame,
        (80, 45),
        interpolation=cv2.INTER_LINEAR
    )

    result = cv2.resize(
        small,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return result


# =========================================================
# GLITCH
# =========================================================

def effect_glitch(frame, t):

    result = frame.copy()

    h, w = frame.shape[:2]

    shift = int(
        math.sin(t * 8) * 12
    )

    if shift > 0:

        result[:, shift:] = frame[:, :-shift]

    elif shift < 0:

        result[:, :shift] = frame[:, -shift:]

    # horizontal glitch blocks
    for _ in range(7):

        y = random.randint(
            0,
            h - 20
        )

        height = random.randint(
            2,
            10
        )

        x_shift = random.randint(
            -25,
            25
        )

        if x_shift > 0:

            result[
                y:y + height,
                x_shift:
            ] = frame[
                y:y + height,
                :-x_shift
            ]

        elif x_shift < 0:

            result[
                y:y + height,
                :x_shift
            ] = frame[
                y:y + height,
                -x_shift:
            ]

    return result


# =========================================================
# NEGATIVE
# =========================================================

def effect_negative(frame):

    return cv2.bitwise_not(frame)


# =========================================================
# EDGE
# =========================================================

def effect_edge(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        70,
        150
    )

    result = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    return result


# =========================================================
# INVERT
# =========================================================

def effect_invert(frame):

    return 255 - frame


# =========================================================
# MONO
# =========================================================

def effect_mono(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )


# =========================================================
# NEON
# =========================================================

def effect_neon(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        130
    )

    edges = cv2.GaussianBlur(
        edges,
        (5, 5),
        0
    )

    neon = np.zeros_like(frame)

    neon[:, :, 1] = edges
    neon[:, :, 2] = edges // 2

    return cv2.addWeighted(
        frame // 3,
        0.4,
        neon,
        1.5,
        0
    )


# =========================================================
# BLUR
# =========================================================

def effect_blur(frame):

    return cv2.GaussianBlur(
        frame,
        (21, 21),
        0
    )


# =========================================================
# SCAN
# =========================================================

def effect_scan(frame, t):

    result = frame.copy()

    h, w = frame.shape[:2]

    # darken slightly
    result = cv2.convertScaleAbs(
        result,
        alpha=0.82,
        beta=0
    )

    # moving scan line
    y = int(
        ((t * 180) % (h + 200)) - 100
    )

    overlay = result.copy()

    cv2.rectangle(
        overlay,
        (0, y),
        (w, y + 45),
        (255, 255, 255),
        -1
    )

    result = cv2.addWeighted(
        result,
        0.82,
        overlay,
        0.18,
        0
    )

    # scanlines
    for yy in range(0, h, 5):

        cv2.line(
            result,
            (0, yy),
            (w, yy),
            (30, 30, 30),
            1
        )

    return result


# =========================================================
# DATA
# =========================================================

def effect_data(frame, t):

    result = frame.copy()

    h, w = frame.shape[:2]

    # digital horizontal displacement
    for _ in range(12):

        y = random.randint(
            0,
            h - 3
        )

        height = random.randint(
            1,
            4
        )

        shift = int(
            math.sin(
                t * 5 + y
            ) * 15
        )

        if shift > 0:

            result[
                y:y + height,
                shift:
            ] = frame[
                y:y + height,
                :-shift
            ]

        elif shift < 0:

            result[
                y:y + height,
                :shift
            ] = frame[
                y:y + height,
                -shift:
            ]

    # digital bars
    for i in range(5):

        yy = int(
            (i / 5) * h +
            math.sin(t * 2 + i) * 30
        )

        cv2.line(
            result,
            (0, yy),
            (w, yy),
            (100, 100, 100),
            1
        )

    return result


# =========================================================
# VHS
# =========================================================

def effect_vhs(frame, t):

    result = frame.copy()

    h, w = frame.shape[:2]

    # RGB separation
    b, g, r = cv2.split(result)

    shift = int(
        math.sin(t * 5) * 4
    )

    r = np.roll(
        r,
        shift,
        axis=1
    )

    b = np.roll(
        b,
        -shift,
        axis=1
    )

    result = cv2.merge([
        b,
        g,
        r
    ])

    # scanlines
    for y in range(
        0,
        h,
        4
    ):

        result[y:y + 1] = (
            result[y:y + 1] * 0.65
        ).astype(np.uint8)

    # horizontal tracking line
    line_y = int(
        ((t * 100) % (h + 100)) - 50
    )

    cv2.line(
        result,
        (0, line_y),
        (w, line_y),
        (220, 220, 220),
        2
    )

    return result


# =========================================================
# MATRIX
# =========================================================

def effect_matrix(frame, t):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(
        gray
    )

    result = np.zeros_like(frame)

    result[:, :, 1] = gray

    h, w = frame.shape[:2]

    # digital vertical streaks
    for x in range(
        0,
        w,
        18
    ):

        speed = (
            x * 0.013 + t * 2
        )

        y = int(
            (speed * 100) %
            (h + 100)
        )

        length = random.randint(
            10,
            50
        )

        cv2.line(
            result,
            (x, y),
            (
                x,
                min(
                    h - 1,
                    y + length
                )
            ),
            (100, 255, 120),
            1
        )

    return result


# =========================================================
# SILHOUETTE
# =========================================================

def effect_silhouette(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    edges = cv2.Canny(
        blur,
        40,
        110
    )

    result = np.zeros_like(frame)

    result[:, :, 0] = edges
    result[:, :, 1] = edges
    result[:, :, 2] = edges

    return result


# =========================================================
# ELECTRIC
# =========================================================

def effect_electric(frame, t):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        60,
        140
    )

    edges = cv2.GaussianBlur(
        edges,
        (3, 3),
        0
    )

    result = np.zeros_like(frame)

    pulse = (
        math.sin(t * 5) + 1
    ) * 0.5

    result[:, :, 0] = (
        edges * 0.8
    ).astype(np.uint8)

    result[:, :, 1] = (
        edges * pulse
    ).astype(np.uint8)

    result[:, :, 2] = (
        edges * 0.2
    ).astype(np.uint8)

    return result


# =========================================================
# HORIZONTAL
# =========================================================

def effect_horizontal(frame, t):

    result = frame.copy()

    h, w = frame.shape[:2]

    # darker background
    result = cv2.convertScaleAbs(
        result,
        alpha=0.7,
        beta=0
    )

    # moving horizontal bands
    for i in range(6):

        y = int(
            (
                t * 130 +
                i * h / 6
            ) % (h + 80)
        )

        y2 = min(
            h,
            y + 25
        )

        overlay = result[
            y:y2
        ].copy()

        if overlay.size:

            overlay = cv2.convertScaleAbs(
                overlay,
                alpha=1.5,
                beta=20
            )

            result[
                y:y2
            ] = overlay

    # fine lines
    for y in range(
        0,
        h,
        8
    ):

        cv2.line(
            result,
            (0, y),
            (w, y),
            (80, 80, 80),
            1
        )

    return result


# =========================================================
# RAINBOW
# =========================================================

def effect_rainbow(frame, t):

    h, w = frame.shape[:2]

    hsv = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    x = np.arange(w)

    hue = (
        x * 180 / w +
        t * 40
    ) % 180

    hsv[:, :, 0] = np.tile(
        hue.astype(np.uint8),
        (h, 1)
    )

    hsv[:, :, 1] = 210
    hsv[:, :, 2] = 255

    rainbow = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    factor = gray.astype(np.float32) / 255.0

    rainbow = (
        rainbow.astype(np.float32)
        * factor[:, :, None]
    )

    rainbow = np.clip(
        rainbow,
        0,
        255
    ).astype(np.uint8)

    return rainbow


# =========================================================
# EFFECT SELECTOR
# =========================================================

def apply_effect(frame, mode, t):

    if mode == 0:
        return effect_clear(frame)

    elif mode == 1:
        return effect_xray(frame)

    elif mode == 2:
        return effect_thermal(frame)

    elif mode == 3:
        return effect_pixel(frame)

    elif mode == 4:
        return effect_glitch(frame, t)

    elif mode == 5:
        return effect_negative(frame)

    elif mode == 6:
        return effect_edge(frame)

    elif mode == 7:
        return effect_invert(frame)

    elif mode == 8:
        return effect_mono(frame)

    elif mode == 9:
        return effect_neon(frame)

    elif mode == 10:
        return effect_blur(frame)

    elif mode == 11:
        return effect_scan(frame, t)

    elif mode == 12:
        return effect_data(frame, t)

    elif mode == 13:
        return effect_vhs(frame, t)

    elif mode == 14:
        return effect_matrix(frame, t)

    elif mode == 15:
        return effect_silhouette(frame)

    elif mode == 16:
        return effect_electric(frame, t)

    elif mode == 17:
        return effect_horizontal(frame, t)

    elif mode == 18:
        return effect_rainbow(frame, t)

    return frame


# =========================================================
# PORTAL BORDER
# =========================================================

def draw_portal_border(frame, points, t):

    if points is None:
        return

    overlay = frame.copy()

    # glow
    glow = np.zeros_like(frame)

    cv2.polylines(
        glow,
        [points],
        True,
        (255, 255, 255),
        8,
        cv2.LINE_AA
    )

    glow = cv2.GaussianBlur(
        glow,
        (21, 21),
        0
    )

    frame[:] = cv2.addWeighted(
        frame,
        1.0,
        glow,
        0.35,
        0
    )

    # main border
    cv2.polylines(
        frame,
        [points],
        True,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # animated small highlights
    num_pts = len(points)
    for i in range(num_pts):

        p1 = points[i]
        p2 = points[(i + 1) % num_pts]

        ratio = (
            math.sin(t * 2 + i)
            + 1
        ) / 2

        x = int(
            p1[0] +
            (p2[0] - p1[0]) *
            ratio
        )

        y = int(
            p1[1] +
            (p2[1] - p1[1]) *
            ratio
        )

        cv2.circle(
            frame,
            (x, y),
            4,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )


# =========================================================
# CORNER MARKERS
# =========================================================

def draw_corners(frame, points):

    if points is None:
        return

    for x, y in points:

        size = 13

        cv2.line(
            frame,
            (x - size, y),
            (x + size, y),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.line(
            frame,
            (x, y - size),
            (x, y + size),
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


# =========================================================
# PORTAL EFFECT
# =========================================================

def apply_portal_effect(
    frame,
    effect,
    points
):

    if points is None:
        return frame

    mask = create_portal_mask(
        frame,
        points
    )

    result = frame.copy()

    inside = cv2.bitwise_and(
        effect,
        effect,
        mask=mask
    )

    outside_mask = cv2.bitwise_not(
        mask
    )

    outside = cv2.bitwise_and(
        frame,
        frame,
        mask=outside_mask
    )

    result = cv2.add(
        outside,
        inside
    )

    return result


# =========================================================
# UI
# =========================================================

def draw_ui(
    frame,
    mode,
    hands_count,
    left_pinch,
    right_pinch
):

    # title
    cv2.putText(
        frame,
        "Hand-Track",
        (25, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # mode
    cv2.putText(
        frame,
        "MODE: " + MODE_NAMES[mode],
        (25, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # hands
    cv2.putText(
        frame,
        "HANDS: " + str(hands_count),
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # gesture status
    left_text = (
        "LEFT PINCH < PREV"
        if left_pinch
        else "LEFT < PREV"
    )

    right_text = (
        "RIGHT PINCH > NEXT"
        if right_pinch
        else "RIGHT > NEXT"
    )

    cv2.putText(
        frame,
        left_text,
        (25, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 255, 200),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        right_text,
        (25, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 255, 200),
        1,
        cv2.LINE_AA
    )

smoothed_points = None


def smooth_portal_points(new_points, smoothing=0.55):
    """
    Smoothing per titik berdasarkan pasangan tangan yang stabil.
    Nilai 0.55 = cukup halus tetapi tetap responsif.
    """

    global smoothed_points

    if new_points is None:
        smoothed_points = None
        return None

    new_points = np.asarray(
        new_points,
        dtype=np.float32
    )

    if smoothed_points is None:
        smoothed_points = new_points.copy()
    else:
        smoothed_points = (
            smoothed_points * (1.0 - smoothing)
            + new_points * smoothing
        )

    return smoothed_points.copy()


# =========================================================
# MAIN
# =========================================================

def main():

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Camera tidak bisa dibuka.")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    tracker = HandTracker(
        MODEL_PATH
    )

    cv2.namedWindow(
        "Hand-Track",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Hand-Track",
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    mode = 0

    # gesture state
    left_pinched_previous = False
    right_pinched_previous = False

    last_gesture_time = 0

    print("")
    print("================================")
    print("       HAND-TRACK STARTED")
    print("================================")
    print("")
    print("Left hand pinch  = Previous")
    print("Right hand pinch = Next")
    print("Q = Quit")
    print("")

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            h, w = frame.shape[:2]

            current_time = time.time()

            # ---------------------------------------------
            # DETECT HANDS
            # ---------------------------------------------

            result = tracker.detect(
                frame
            )

            hands = result.hand_landmarks

            # ---------------------------------------------
            # DRAW HANDS
            # ---------------------------------------------

            for hand in hands:

                draw_hand(
                    frame,
                    hand
                )

            # ---------------------------------------------
            # PINCH DETECTION
            # ---------------------------------------------

            left_pinch = False
            right_pinch = False

            if len(hands) > 0:

                sorted_hands = sorted(
                    hands,
                    key=lambda hand:
                    np.mean(
                        [p.x for p in hand]
                    )
                )

                if len(sorted_hands) == 1:

                    hand = sorted_hands[0]

                    center_x = np.mean(
                        [p.x for p in hand]
                    )

                    pinch = is_pinch(
                        hand
                    )

                    if center_x < 0.5:
                        left_pinch = pinch

                    else:
                        right_pinch = pinch

                else:

                    left_hand = sorted_hands[0]
                    right_hand = sorted_hands[1]

                    left_pinch = is_pinch(
                        left_hand
                    )

                    right_pinch = is_pinch(
                        right_hand
                    )

            # ---------------------------------------------
            # GESTURE CHANGE
            # ---------------------------------------------

            left_new_pinch = (
                left_pinch
                and not left_pinched_previous
            )

            right_new_pinch = (
                right_pinch
                and not right_pinched_previous
            )

            if (
                current_time -
                last_gesture_time
                >
                GESTURE_COOLDOWN
            ):

                # if only left pinches
                if (
                    left_new_pinch
                    and not right_new_pinch
                ):

                    mode = (
                        mode - 1
                    ) % len(MODE_NAMES)

                    last_gesture_time = (
                        current_time
                    )

                # if only right pinches
                elif (
                    right_new_pinch
                    and not left_new_pinch
                ):

                    mode = (
                        mode + 1
                    ) % len(MODE_NAMES)

                    last_gesture_time = (
                        current_time
                    )

            left_pinched_previous = (
                left_pinch
            )

            right_pinched_previous = (
                right_pinch
            )

            # ---------------------------------------------
            # PORTAL
            # ---------------------------------------------

            portal_points = get_portal_points(
                hands,
                w,
                h
            )

            # Smooth titik berdasarkan tangan terlebih dahulu.
            portal_points = smooth_portal_points(
                portal_points,
                0.55
            )

            # Kembalikan 4 titik secara langsung tanpa di-convex/sort ulang.
            portal_points = order_portal_points(
                portal_points
            )

            # ---------------------------------------------
            # EFFECT
            # ---------------------------------------------

            effect = apply_effect(
                frame,
                mode,
                current_time
            )

            # ---------------------------------------------
            # APPLY EFFECT ONLY INSIDE PORTAL
            # ---------------------------------------------

            frame = apply_portal_effect(
                frame,
                effect,
                portal_points
            )

            # ---------------------------------------------
            # PORTAL BORDER
            # ---------------------------------------------

            draw_portal_border(
                frame,
                portal_points,
                current_time
            )

            draw_corners(
                frame,
                portal_points
            )

            # ---------------------------------------------
            # UI
            # ---------------------------------------------

            draw_ui(
                frame,
                mode,
                len(hands),
                left_pinch,
                right_pinch
            )

            # ---------------------------------------------
            # SHOW
            # ---------------------------------------------

            cv2.imshow(
                "Hand-Track",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        tracker.close()

        cap.release()

        cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    main()