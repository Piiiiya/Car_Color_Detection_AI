
import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = PROJECT_ROOT / "yolo26s.pt"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEBUG_IMAGE_PATH = (
    OUTPUT_DIR / "debug_detection.jpg"
)

# ============================================================
# IMPORT DETECTOR
# ============================================================

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detector import VehicleDetector


# ============================================================
# FIND INPUT IMAGE
# ============================================================

def find_test_image():

    """
    Search common project folders for an image.

    If a specific image is preferred, replace this function
    with a direct path to that image.
    """

    search_directories = [
        PROJECT_ROOT / "data" / "test",
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "images",
        PROJECT_ROOT / "test_images",
    ]

    supported_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp",
        "*.bmp",
    ]

    for directory in search_directories:

        if not directory.exists():
            continue

        for extension in supported_extensions:

            images = sorted(
                directory.rglob(extension)
            )

            if images:
                return images[0]

    raise FileNotFoundError(
        "No test image found. Please place a traffic image "
        "inside data/test or update the image path."
    )


# ============================================================
# DRAW DETECTION BOXES
# ============================================================

def draw_detections(
    image,
    detections,
    color,
    label,
):

    """
    Draw bounding boxes and confidence labels.
    """

    output = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        confidence = detection["confidence"]

        text = (
            f"{label} {confidence:.2f}"
        )

        # Bounding box
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        # Text background
        (text_width, text_height), baseline = (
            cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2,
            )
        )

        text_y = max(
            y1,
            text_height + 10,
        )

        cv2.rectangle(
            output,
            (
                x1,
                text_y - text_height - 8,
            ),
            (
                x1 + text_width + 8,
                text_y + baseline,
            ),
            color,
            -1,
        )

        # Label
        cv2.putText(
            output,
            text,
            (x1 + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("CAR COLOR DETECTION AI - DETECTOR TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_path = find_test_image()

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Unable to load image: {image_path}"
        )

    height, width = image.shape[:2]

    print("\nImage loaded successfully.")
    print(f"Image path: {image_path}")
    print(
        f"Image dimensions: "
        f"({height}, {width}, 3)"
    )

    # --------------------------------------------------------
    # Initialize detector
    # --------------------------------------------------------

    detector = VehicleDetector(
        model_path=str(MODEL_PATH),
        device="cpu",
    )

    # --------------------------------------------------------
    # Run detection
    # --------------------------------------------------------

    print("\nRunning detection...")

    results = detector.detect_all(image)

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    cars = results["cars"]

    people = results["people"]

    traffic_lights = results["traffic_lights"]

    car_count = results["car_count"]

    people_count = results["people_count"]

    traffic_light_count = (
        results["traffic_light_count"]
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DETECTION TEST RESULTS")
    print("=" * 60)

    print(f"Cars detected: {car_count}")

    print(f"People detected: {people_count}")

    print(
        f"Traffic lights detected: "
        f"{traffic_light_count}"
    )

    # --------------------------------------------------------
    # Car details
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CAR DETAILS")
    print("-" * 60)

    if cars:

        for index, car in enumerate(
            cars,
            start=1,
        ):

            print(
                f"Car {index}: "
                f"BBox={car['bbox']}, "
                f"Confidence={car['confidence']:.2f}"
            )

    else:
        print("No cars detected.")

    # --------------------------------------------------------
    # Person details
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("PERSON DETAILS")
    print("-" * 60)

    if people:

        for index, person in enumerate(
            people,
            start=1,
        ):

            print(
                f"Person {index}: "
                f"BBox={person['bbox']}, "
                f"Confidence={person['confidence']:.2f}"
            )

    else:
        print("No people detected.")

    # --------------------------------------------------------
    # Traffic light details
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("TRAFFIC LIGHT DETAILS")
    print("-" * 60)

    if traffic_lights:

        for index, light in enumerate(
            traffic_lights,
            start=1,
        ):

            print(
                f"Traffic light {index}: "
                f"BBox={light['bbox']}, "
                f"Confidence={light['confidence']:.2f}"
            )

    else:
        print("No traffic lights detected.")

    # --------------------------------------------------------
    # Draw boxes
    # --------------------------------------------------------

    output_image = image.copy()

    # Cars: blue boxes
    output_image = draw_detections(
        output_image,
        cars,
        color=(255, 0, 0),
        label="Car",
    )

    # People: green boxes
    output_image = draw_detections(
        output_image,
        people,
        color=(0, 255, 0),
        label="Person",
    )

    # Traffic lights: yellow boxes
    output_image = draw_detections(
        output_image,
        traffic_lights,
        color=(0, 255, 255),
        label="Traffic Light",
    )

    # --------------------------------------------------------
    # Add summary to image
    # --------------------------------------------------------

    summary = (
        f"Cars: {car_count} | "
        f"People: {people_count} | "
        f"Traffic Lights: {traffic_light_count}"
    )

    cv2.rectangle(
        output_image,
        (10, 10),
        (min(width - 10, 900), 65),
        (30, 30, 30),
        -1,
    )

    cv2.putText(
        output_image,
        summary,
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Save debug image
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(DEBUG_IMAGE_PATH),
        output_image,
    )

    if success:

        print("\nDebug image saved successfully:")
        print(DEBUG_IMAGE_PATH)

    else:

        print("\nWARNING: Could not save debug image.")

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nTEST FAILED")
        print(f"Error: {error}")

        raise
