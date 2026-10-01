
import cv2
from pathlib import Path

from detector import VehicleDetector


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "traffic_test.jpg"
)

MODEL_PATH = PROJECT_DIR / "yolo26s.pt"

OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = OUTPUT_DIR / "detection_debug.jpg"


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print("Image loaded successfully.")
print("Image dimensions:", image.shape)


# =========================================================
# INITIALIZE DETECTOR
# =========================================================

detector = VehicleDetector(
    model_path=str(MODEL_PATH)
)


# =========================================================
# RUN DETECTION
# =========================================================

results = detector.detect(image)

cars = results["cars"]
people = results["people"]

print("\n" + "=" * 55)
print("DETECTION TEST RESULTS")
print("=" * 55)

print(f"Cars detected: {len(cars)}")
print(f"People detected: {len(people)}")


# =========================================================
# DRAW DETECTIONS
# =========================================================

debug_image = image.copy()


def draw_box(
    image,
    box,
    label,
    color,
    thickness=3
):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    text_thickness = 2

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            label,
            font,
            font_scale,
            text_thickness
        )
    )

    label_y = max(
        y1,
        text_height + 10
    )

    cv2.rectangle(
        image,
        (
            x1,
            label_y - text_height - 10
        ),
        (
            x1 + text_width + 8,
            label_y + baseline
        ),
        color,
        -1
    )

    cv2.putText(
        image,
        label,
        (x1 + 4, label_y - 4),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA
    )


# ---------------------------------------------------------
# DRAW CARS
# ---------------------------------------------------------

for index, car in enumerate(cars, 1):

    confidence = car["confidence"] * 100

    label = (
        f"Car {index} "
        f"{confidence:.1f}%"
    )

    # Blue in OpenCV BGR
    draw_box(
        debug_image,
        car["box"],
        label,
        (255, 0, 0),
        thickness=4
    )


# ---------------------------------------------------------
# DRAW PEOPLE
# ---------------------------------------------------------

for index, person in enumerate(people, 1):

    confidence = person["confidence"] * 100

    label = (
        f"Person {index} "
        f"{confidence:.1f}%"
    )

    # Green in OpenCV BGR
    draw_box(
        debug_image,
        person["box"],
        label,
        (0, 255, 0),
        thickness=2
    )


# =========================================================
# SUMMARY PANEL
# =========================================================

summary = (
    f"Cars: {len(cars)} | "
    f"People: {len(people)}"
)

cv2.rectangle(
    debug_image,
    (15, 15),
    (650, 75),
    (30, 30, 30),
    -1
)

cv2.putText(
    debug_image,
    summary,
    (30, 55),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# =========================================================
# SAVE DEBUG IMAGE
# =========================================================

saved = cv2.imwrite(
    str(OUTPUT_PATH),
    debug_image
)

if not saved:
    raise RuntimeError(
        "Failed to save detection debug image."
    )

print("\nDebug image saved successfully:")
print(OUTPUT_PATH)

print("\nTest completed.")
