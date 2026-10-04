
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class VehicleDetector:
    """
    YOLO vehicle detector with full-image and tiled inference.

    Supports:
    - Car detection
    - Person detection
    - Traffic-light detection
    - Duplicate removal
    - detect() and detect_all() compatibility
    - Count and detection-list outputs
    """

    PERSON_CLASS = 0
    CAR_CLASS = 2
    TRAFFIC_LIGHT_CLASS = 9

    def __init__(
        self,
        model_path="yolo26s.pt",
        conf=0.25,
        iou=0.45,
        max_det=300,
        device="cpu",
        tile_size=960,
        tile_overlap=0.25,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_absolute():
            self.model_path = (
                Path(__file__).resolve().parent.parent
                / self.model_path
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.device = device
        self.tile_size = int(tile_size)
        self.tile_overlap = float(tile_overlap)

        self.model = YOLO(str(self.model_path))

        print("VehicleDetector initialized successfully.")
        print(f"YOLO model: {self.model_path}")
        print(f"Inference device: {self.device}")
        print(f"Tile size: {self.tile_size}")
        print(f"Tile overlap: {self.tile_overlap}")

    # =========================================================
    # IMAGE VALIDATION
    # =========================================================

    @staticmethod
    def _validate_image(image):
        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input image must have shape (height, width, 3)."
            )

        if image.size == 0:
            raise ValueError("Input image is empty.")

    # =========================================================
    # TILE GENERATION
    # =========================================================

    def _generate_tiles(self, image):
        height, width = image.shape[:2]

        tile_size = min(
            self.tile_size,
            height,
            width
        )

        if height <= tile_size and width <= tile_size:
            return [(0, 0, width, height)]

        stride = max(
            1,
            int(tile_size * (1.0 - self.tile_overlap))
        )

        def positions(length):
            if length <= tile_size:
                return [0]

            result = list(
                range(0, length - tile_size + 1, stride)
            )

            final_position = length - tile_size

            if result[-1] != final_position:
                result.append(final_position)

            return result

        x_positions = positions(width)
        y_positions = positions(height)

        tiles = []

        for y1 in y_positions:
            for x1 in x_positions:
                x2 = min(x1 + tile_size, width)
                y2 = min(y1 + tile_size, height)

                tiles.append((x1, y1, x2, y2))

        return tiles

    # =========================================================
    # YOLO INFERENCE
    # =========================================================

    def _predict(
        self,
        image,
        confidence=None,
        imgsz=1280,
    ):
        confidence = (
            self.conf
            if confidence is None
            else float(confidence)
        )

        return self.model.predict(
            source=image,
            conf=confidence,
            iou=self.iou,
            imgsz=imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )

    # =========================================================
    # EXTRACT DETECTIONS
    # =========================================================

    @staticmethod
    def _extract_detections(
        results,
        offset_x=0,
        offset_y=0,
        allowed_classes=None,
    ):
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                if (
                    allowed_classes is not None
                    and class_id not in allowed_classes
                ):
                    continue

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy().tolist()
                )

                coordinates = (
                    int(round(x1 + offset_x)),
                    int(round(y1 + offset_y)),
                    int(round(x2 + offset_x)),
                    int(round(y2 + offset_y)),
                )

                detections.append({
                    "box": coordinates,
                    "bbox": coordinates,
                    "confidence": confidence,
                    "class_id": class_id,
                })

        return detections

    # =========================================================
    # BOX GEOMETRY
    # =========================================================

    @staticmethod
    def _box_area(box):
        x1, y1, x2, y2 = box

        return (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = (
            max(0, ix2 - ix1)
            * max(0, iy2 - iy1)
        )

        area_a = VehicleDetector._box_area(box_a)
        area_b = VehicleDetector._box_area(box_b)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _containment_ratio(small_box, large_box):
        sx1, sy1, sx2, sy2 = small_box
        lx1, ly1, lx2, ly2 = large_box

        ix1 = max(sx1, lx1)
        iy1 = max(sy1, ly1)
        ix2 = min(sx2, lx2)
        iy2 = min(sy2, ly2)

        intersection = (
            max(0, ix2 - ix1)
            * max(0, iy2 - iy1)
        )

        small_area = VehicleDetector._box_area(
            small_box
        )

        if small_area <= 0:
            return 0.0

        return intersection / small_area

    # =========================================================
    # DUPLICATE REMOVAL
    # =========================================================

    def _remove_duplicates(
        self,
        detections,
        iou_threshold=0.50,
        containment_threshold=0.85,
        suppress_contained=True,
    ):
        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        kept = []

        for candidate in detections:
            candidate_box = candidate["box"]
            candidate_area = self._box_area(
                candidate_box
            )

            if candidate_area <= 0:
                continue

            duplicate = False

            for accepted in kept:
                accepted_box = accepted["box"]

                overlap = self._iou(
                    candidate_box,
                    accepted_box
                )

                if overlap >= iou_threshold:
                    duplicate = True
                    break

                if not suppress_contained:
                    continue

                accepted_area = self._box_area(
                    accepted_box
                )

                if accepted_area <= 0:
                    continue

                smaller_area = min(
                    candidate_area,
                    accepted_area
                )

                larger_area = max(
                    candidate_area,
                    accepted_area
                )

                area_ratio = (
                    smaller_area / larger_area
                )

                if area_ratio > 0.65:
                    continue

                containment = self._containment_ratio(
                    candidate_box,
                    accepted_box
                )

                reverse_containment = (
                    self._containment_ratio(
                        accepted_box,
                        candidate_box
                    )
                )

                if (
                    containment >= containment_threshold
                    or reverse_containment
                    >= containment_threshold
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(candidate)

        return kept

    # =========================================================
    # FULL IMAGE + TILED DETECTION
    # =========================================================

    def _detect_class(
        self,
        image,
        class_id,
        confidence,
        full_imgsz=1536,
        tile_imgsz=1280,
        suppress_contained=True,
    ):
        height, width = image.shape[:2]

        print("Full-image inference...")

        full_results = self._predict(
            image,
            confidence=confidence,
            imgsz=full_imgsz,
        )

        full_detections = self._extract_detections(
            full_results,
            allowed_classes={class_id},
        )

        print(
            f"Full-image detections: "
            f"{len(full_detections)}"
        )

        tiles = self._generate_tiles(image)

        print(f"Tiled inference: {len(tiles)} tiles")

        tile_detections = []

        for x1, y1, x2, y2 in tiles:
            tile = image[y1:y2, x1:x2]

            if tile.size == 0:
                continue

            tile_results = self._predict(
                tile,
                confidence=confidence,
                imgsz=tile_imgsz,
            )

            detections = self._extract_detections(
                tile_results,
                offset_x=x1,
                offset_y=y1,
                allowed_classes={class_id},
            )

            tile_detections.extend(detections)

        print(
            f"Tile detections: {len(tile_detections)}"
        )

        all_detections = (
            full_detections + tile_detections
        )

        print(
            f"Combined raw detections: "
            f"{len(all_detections)}"
        )

        # Clip all coordinates to image boundaries.
        for detection in all_detections:
            x1, y1, x2, y2 = detection["box"]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            coordinates = (x1, y1, x2, y2)

            detection["box"] = coordinates
            detection["bbox"] = coordinates

        all_detections = [
            detection
            for detection in all_detections
            if (
                detection["box"][2]
                > detection["box"][0]
                and detection["box"][3]
                > detection["box"][1]
            )
        ]

        final_detections = self._remove_duplicates(
            all_detections,
            iou_threshold=0.50,
            containment_threshold=0.85,
            suppress_contained=suppress_contained,
        )

        print(
            f"Detections after duplicate removal: "
            f"{len(final_detections)}"
        )

        return final_detections

    # =========================================================
    # MAIN DETECTION
    # =========================================================

    def detect(self, image):
        self._validate_image(image)

        print("\nStarting vehicle detection...\n")

        print("Detecting cars...")

        cars = self._detect_class(
            image=image,
            class_id=self.CAR_CLASS,
            confidence=0.30,
            full_imgsz=1536,
            tile_imgsz=1280,
            suppress_contained=True,
        )

        print("\nDetecting people...")

        people = self._detect_class(
            image=image,
            class_id=self.PERSON_CLASS,
            confidence=0.25,
            full_imgsz=1536,
            tile_imgsz=1280,
            suppress_contained=False,
        )

        print("\nDetecting traffic lights...")

        traffic_lights = self._detect_class(
            image=image,
            class_id=self.TRAFFIC_LIGHT_CLASS,
            confidence=0.20,
            full_imgsz=1536,
            tile_imgsz=1280,
            suppress_contained=False,
        )

        car_count = len(cars)
        person_count = len(people)
        traffic_light_count = len(traffic_lights)

        total_objects = (
            car_count
            + person_count
            + traffic_light_count
        )

        print("\nDetection completed.")

        return {
            # Detection lists
            "cars": cars,
            "people": people,
            "traffic_lights": traffic_lights,
            "main_result": traffic_lights,

            # Counts expected by test_detector.py
            "car_count": car_count,
            "person_count": person_count,
            "people_count": person_count,
            "traffic_light_count": traffic_light_count,
            "total_objects": total_objects,
        }

    # =========================================================
    # BACKWARD COMPATIBILITY
    # =========================================================

    def detect_all(self, image):
        return self.detect(image)
