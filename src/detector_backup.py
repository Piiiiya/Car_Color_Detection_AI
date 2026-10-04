
import os
from pathlib import Path

# ================================================================
# CPU SETTINGS
# ================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    Car, person, and traffic-light detector.

    Features:
        - Full-image inference.
        - Overlapping tiled inference.
        - Separate confidence thresholds.
        - High-resolution traffic-light detection.
        - Class-aware duplicate removal.
        - CPU inference.
        - Backward compatibility with existing project files.
    """

    PERSON_CLASS = 0
    CAR_CLASS = 2
    TRAFFIC_LIGHT_CLASS = 9

    def __init__(
        self,
        model_path="yolo26s.pt",
        confidence=0.20,
        iou=0.45,
        max_detections=500,
        device="cpu",
        tile_size=960,
        tile_overlap=0.25
    ):

        self.model_path = str(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.device = device

        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)

        # Separate confidence thresholds.
        self.person_confidence = 0.20
        self.car_confidence = 0.20
        self.traffic_light_confidence = 0.08

        # Higher resolution for small traffic lights.
        self.traffic_light_imgsz = 1536

        if not 0 <= self.tile_overlap < 1:
            raise ValueError(
                "tile_overlap must be between 0 and 1."
            )

        if self.tile_size <= 0:
            raise ValueError(
                "tile_size must be greater than zero."
            )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        print("Loading YOLO model...")

        self.model = YOLO(self.model_path)

        print("VehicleDetector initialized successfully.")
        print(
            f"YOLO model: {Path(self.model_path).resolve()}"
        )
        print(f"Inference device: {self.device}")
        print(f"Tile size: {self.tile_size}")
        print(f"Tile overlap: {self.tile_overlap}")

    # ============================================================
    # YOLO PREDICTION
    # ============================================================

    def _predict(
        self,
        image,
        confidence=None,
        classes=None,
        imgsz=1280
    ):

        if confidence is None:
            confidence = self.confidence

        if classes is None:
            classes = [
                self.PERSON_CLASS,
                self.CAR_CLASS,
                self.TRAFFIC_LIGHT_CLASS
            ]

        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=self.iou,
            imgsz=imgsz,
            device=self.device,
            classes=classes,
            max_det=self.max_detections,
            verbose=False
        )

        return results[0]

    # ============================================================
    # EXTRACT DETECTIONS
    # ============================================================

    @staticmethod
    def _extract_detections(
        result,
        offset_x=0,
        offset_y=0,
        image_width=None,
        image_height=None
    ):

        detections = []

        if result is None or result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(round(x1 + offset_x))
            y1 = int(round(y1 + offset_y))
            x2 = int(round(x2 + offset_x))
            y2 = int(round(y2 + offset_y))

            if image_width is not None:
                x1 = max(0, min(x1, image_width - 1))
                x2 = max(0, min(x2, image_width - 1))

            if image_height is not None:
                y1 = max(0, min(y1, image_height - 1))
                y2 = max(0, min(y2, image_height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2, y2)

            detections.append({
                "bbox": bbox,
                "box": bbox,
                "confidence": confidence,
                "class_id": class_id
            })

        return detections

    # ============================================================
    # GENERATE OVERLAPPING TILES
    # ============================================================

    def _generate_tiles(
        self,
        image_width,
        image_height,
        tile_size=None
    ):

        if tile_size is None:
            tile_size = self.tile_size

        step = max(
            1,
            int(tile_size * (1 - self.tile_overlap))
        )

        x_positions = list(
            range(
                0,
                max(1, image_width - tile_size + 1),
                step
            )
        )

        y_positions = list(
            range(
                0,
                max(1, image_height - tile_size + 1),
                step
            )
        )

        final_x = max(0, image_width - tile_size)
        final_y = max(0, image_height - tile_size)

        if not x_positions or x_positions[-1] != final_x:
            x_positions.append(final_x)

        if not y_positions or y_positions[-1] != final_y:
            y_positions.append(final_y)

        tiles = []

        for y in sorted(set(y_positions)):
            for x in sorted(set(x_positions)):

                x2 = min(x + tile_size, image_width)
                y2 = min(y + tile_size, image_height)

                if x2 <= x or y2 <= y:
                    continue

                tiles.append((x, y, x2, y2))

        return tiles

    # ============================================================
    # INTERSECTION OVER UNION
    # ============================================================

    @staticmethod
    def _calculate_iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0, ix2 - ix1)
        intersection_height = max(0, iy2 - iy1)

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = (
            max(0, ax2 - ax1)
            * max(0, ay2 - ay1)
        )

        area_b = (
            max(0, bx2 - bx1)
            * max(0, by2 - by1)
        )

        union_area = (
            area_a + area_b - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # ============================================================
    # CONTAINMENT OVERLAP
    # ============================================================

    @staticmethod
    def _containment_overlap(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(0, ix2 - ix1)
        intersection_height = max(0, iy2 - iy1)

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = (
            max(0, ax2 - ax1)
            * max(0, ay2 - ay1)
        )

        area_b = (
            max(0, bx2 - bx1)
            * max(0, by2 - by1)
        )

        smaller_area = min(area_a, area_b)

        if smaller_area <= 0:
            return 0.0

        return intersection_area / smaller_area

    # ============================================================
    # DUPLICATE REMOVAL
    # ============================================================

    def _deduplicate(
        self,
        detections,
        iou_threshold=0.45,
        containment_threshold=0.85
    ):

        if not detections:
            return []

        final_detections = []

        class_ids = sorted({
            item["class_id"]
            for item in detections
        })

        for class_id in class_ids:

            class_detections = [
                item
                for item in detections
                if item["class_id"] == class_id
            ]

            class_detections.sort(
                key=lambda item: item["confidence"],
                reverse=True
            )

            kept = []

            for candidate in class_detections:

                candidate_box = candidate["bbox"]
                duplicate = False

                for existing in kept:

                    existing_box = existing["bbox"]

                    iou_score = self._calculate_iou(
                        candidate_box,
                        existing_box
                    )

                    containment_score = (
                        self._containment_overlap(
                            candidate_box,
                            existing_box
                        )
                    )

                    if (
                        iou_score >= iou_threshold
                        or containment_score >= containment_threshold
                    ):
                        duplicate = True
                        break

                if not duplicate:
                    kept.append(candidate)

            final_detections.extend(kept)

        return final_detections

    # ============================================================
    # CLASS-SPECIFIC DETECTION
    # ============================================================

    def _detect_class(
        self,
        image,
        class_id,
        confidence,
        imgsz,
        tile_size=None,
        label="Object"
    ):

        image_height, image_width = image.shape[:2]

        all_detections = []

        print(f"\nDetecting {label.lower()}...")

        # Full-image inference.
        full_result = self._predict(
            image,
            confidence=confidence,
            classes=[class_id],
            imgsz=imgsz
        )

        full_detections = self._extract_detections(
            full_result,
            image_width=image_width,
            image_height=image_height
        )

        all_detections.extend(full_detections)

        print(
            f"{label} full-image detections: "
            f"{len(full_detections)}"
        )

        # Tiled inference.
        tiles = self._generate_tiles(
            image_width,
            image_height,
            tile_size=tile_size
        )

        print(
            f"{label} tiled inference: {len(tiles)} tiles"
        )

        for tile_number, (x1, y1, x2, y2) in enumerate(
            tiles,
            start=1
        ):

            tile = image[y1:y2, x1:x2]

            if tile.size == 0:
                continue

            result = self._predict(
                tile,
                confidence=confidence,
                classes=[class_id],
                imgsz=imgsz
            )

            tile_detections = self._extract_detections(
                result,
                offset_x=x1,
                offset_y=y1,
                image_width=image_width,
                image_height=image_height
            )

            all_detections.extend(tile_detections)

        print(
            f"{label} raw detections: "
            f"{len(all_detections)}"
        )

        # Remove duplicate detections.
        final_detections = self._deduplicate(
            all_detections,
            iou_threshold=0.45,
            containment_threshold=0.85
        )

        print(
            f"{label} after duplicate removal: "
            f"{len(final_detections)}"
        )

        return final_detections

    # ============================================================
    # MAIN DETECTION METHOD
    # ============================================================

    def detect(self, image):

        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input image must be a 3-channel BGR image."
            )

        print("\nStarting vehicle detection...")

        # Cars.
        cars = self._detect_class(
            image=image,
            class_id=self.CAR_CLASS,
            confidence=self.car_confidence,
            imgsz=1280,
            tile_size=self.tile_size,
            label="Cars"
        )

        # People.
        people = self._detect_class(
            image=image,
            class_id=self.PERSON_CLASS,
            confidence=self.person_confidence,
            imgsz=1280,
            tile_size=self.tile_size,
            label="People"
        )

        # Traffic lights.
        traffic_lights = self._detect_class(
            image=image,
            class_id=self.TRAFFIC_LIGHT_CLASS,
            confidence=self.traffic_light_confidence,
            imgsz=self.traffic_light_imgsz,
            tile_size=self.tile_size,
            label="Traffic lights"
        )

        detections = (
            cars
            + people
            + traffic_lights
        )

        car_count = len(cars)
        people_count = len(people)
        traffic_lights_count = len(traffic_lights)

        print("\n" + "=" * 55)
        print("FINAL DETECTION RESULTS")
        print("=" * 55)

        print(f"Cars detected: {car_count}")
        print(f"People detected: {people_count}")
        print(f"Traffic lights detected: {traffic_lights_count}")
        print(f"Total objects: {len(detections)}")

        # Include all common key names for compatibility.
        return {
            "main_result": None,
            "detections": detections,

            "cars": cars,
            "people": people,
            "traffic_lights": traffic_lights,

            "cars_count": car_count,
            "people_count": people_count,
            "traffic_lights_count": traffic_lights_count,

            "car_count": car_count,
            "person_count": people_count,
            "traffic_light_count": traffic_lights_count,

            "car_details": cars,
            "people_details": people,
            "traffic_light_details": traffic_lights
        }

    # ============================================================
    # BACKWARD COMPATIBILITY
    # ============================================================

    def detect_all(self, image):
        """
        Compatibility with the existing test_detector.py.
        """

        return self.detect(image)
