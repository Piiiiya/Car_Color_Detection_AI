
import cv2
import numpy as np

from ultralytics import YOLO


class VehicleDetector:

    # COCO class IDs
    PERSON_CLASS = 0
    CAR_CLASS = 2
    TRAFFIC_LIGHT_CLASS = 9

    # COCO class names
    CLASS_NAMES = {
        0: "person",
        2: "car",
        9: "traffic light"
    }

    def __init__(
        self,
        model_path="yolo26s.pt",
        confidence=0.15,
        image_size=1280,
        iou=0.45,
        max_det=300
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.image_size = image_size
        self.iou = iou
        self.max_det = max_det

        print("VehicleDetector initialized successfully.")

    # =========================================================
    # YOLO PREDICTION
    # =========================================================

    def _predict(
        self,
        image,
        confidence=None,
        image_size=None,
        classes=None
    ):

        if confidence is None:
            confidence = self.confidence

        if image_size is None:
            image_size = self.image_size

        results = self.model.predict(
            source=image,
            conf=confidence,
            imgsz=image_size,
            iou=self.iou,
            max_det=self.max_det,
            classes=classes,
            verbose=False
        )

        return results

    # =========================================================
    # EXTRACT DETECTIONS
    # =========================================================

    @staticmethod
    def _extract_detections(
        results,
        image_width,
        image_height,
        target_class
    ):

        detections = []

        if results is None:
            return detections

        # Handle a single result or a list of results
        if isinstance(results, (list, tuple)):
            result_list = results
        else:
            result_list = [results]

        for result in result_list:

            if result is None:
                continue

            boxes = getattr(
                result,
                "boxes",
                None
            )

            if boxes is None:
                continue

            for box in boxes:

                class_id = int(
                    box.cls[0].item()
                )

                if class_id != target_class:
                    continue

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    coordinates
                )

                # Keep coordinates inside image boundaries
                x1 = max(
                    0,
                    min(x1, image_width - 1)
                )

                y1 = max(
                    0,
                    min(y1, image_height - 1)
                )

                x2 = max(
                    0,
                    min(x2, image_width)
                )

                y2 = max(
                    0,
                    min(y2, image_height)
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # FIX: Include class_name
                class_name = VehicleDetector.CLASS_NAMES.get(
                    class_id,
                    str(class_id)
                )

                detections.append({

                    "box": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "confidence": confidence,

                    "class_id": class_id,

                    "class_name": class_name

                })

        return detections

    # =========================================================
    # INTERSECTION OVER UNION
    # =========================================================

    @staticmethod
    def _iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0,
            ax2 - ax1
        ) * max(
            0,
            ay2 - ay1
        )

        area_b = max(
            0,
            bx2 - bx1
        ) * max(
            0,
            by2 - by1
        )

        union_area = (
            area_a
            + area_b
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return (
            intersection_area
            / union_area
        )

    # =========================================================
    # CONTAINMENT RATIO
    # =========================================================

    @staticmethod
    def _containment_ratio(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0,
            ax2 - ax1
        ) * max(
            0,
            ay2 - ay1
        )

        area_b = max(
            0,
            bx2 - bx1
        ) * max(
            0,
            by2 - by1
        )

        smaller_area = min(
            area_a,
            area_b
        )

        if smaller_area <= 0:
            return 0.0

        return (
            intersection_area
            / smaller_area
        )

    # =========================================================
    # DUPLICATE REMOVAL
    # =========================================================

    @classmethod
    def _remove_duplicates(
        cls,
        detections,
        iou_threshold=0.45,
        containment_threshold=0.75
    ):

        if not detections:
            return []

        ordered = sorted(
            detections,
            key=lambda item: item["confidence"],
            reverse=True
        )

        kept = []

        for candidate in ordered:

            candidate_box = candidate["box"]

            duplicate = False

            for existing in kept:

                existing_box = existing["box"]

                overlap = cls._iou(
                    candidate_box,
                    existing_box
                )

                containment = cls._containment_ratio(
                    candidate_box,
                    existing_box
                )

                if (
                    overlap >= iou_threshold
                    or containment >= containment_threshold
                ):

                    duplicate = True
                    break

            if not duplicate:
                kept.append(candidate)

        return kept

    # =========================================================
    # IMAGE TILING
    # =========================================================

    @staticmethod
    def _generate_tiles(
        image,
        rows=3,
        columns=4,
        overlap=0.25
    ):

        height, width = image.shape[:2]

        tiles = []

        tile_height = int(
            height / (
                rows
                - overlap * (rows - 1)
            )
        )

        tile_width = int(
            width / (
                columns
                - overlap * (columns - 1)
            )
        )

        step_y = int(
            tile_height * (1 - overlap)
        )

        step_x = int(
            tile_width * (1 - overlap)
        )

        for row in range(rows):

            for column in range(columns):

                x1 = column * step_x
                y1 = row * step_y

                x2 = min(
                    x1 + tile_width,
                    width
                )

                y2 = min(
                    y1 + tile_height,
                    height
                )

                # Ensure final column reaches the right edge
                if column == columns - 1:

                    x2 = width

                    x1 = max(
                        0,
                        x2 - tile_width
                    )

                # Ensure final row reaches the bottom edge
                if row == rows - 1:

                    y2 = height

                    y1 = max(
                        0,
                        y2 - tile_height
                    )

                tile = image[
                    y1:y2,
                    x1:x2
                ]

                if tile.size == 0:
                    continue

                tiles.append({

                    "image": tile,

                    "offset_x": x1,

                    "offset_y": y1

                })

        return tiles

    # =========================================================
    # CAR DETECTION
    # =========================================================

    def _detect_cars(
        self,
        image,
        image_width,
        image_height
    ):

        results = self._predict(
            image,
            confidence=0.30,
            image_size=1536,
            classes=[
                self.CAR_CLASS
            ]
        )

        detections = self._extract_detections(
            results,
            image_width,
            image_height,
            self.CAR_CLASS
        )

        detections = self._remove_duplicates(
            detections,
            iou_threshold=0.45,
            containment_threshold=0.75
        )

        filtered = []

        for detection in detections:

            x1, y1, x2, y2 = detection["box"]

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 20 or box_height < 15:
                continue

            filtered.append(detection)

        print(
            f"Car detections: {len(filtered)}"
        )

        return filtered

    # =========================================================
    # PERSON DETECTION
    # =========================================================

    def _detect_people(
        self,
        image,
        image_width,
        image_height
    ):

        # -----------------------------------------------------
        # 1. FULL-IMAGE DETECTION
        # -----------------------------------------------------

        full_results = self._predict(
            image,
            confidence=0.25,
            image_size=1536,
            classes=[
                self.PERSON_CLASS
            ]
        )

        full_detections = self._extract_detections(
            full_results,
            image_width,
            image_height,
            self.PERSON_CLASS
        )

        full_detections = self._remove_duplicates(
            full_detections,
            iou_threshold=0.45,
            containment_threshold=0.75
        )

        # -----------------------------------------------------
        # 2. HIGH-RESOLUTION TILED DETECTION
        # -----------------------------------------------------

        tile_detections = []

        tiles = self._generate_tiles(
            image,
            rows=3,
            columns=4,
            overlap=0.25
        )

        for tile_data in tiles:

            tile = tile_data["image"]

            offset_x = tile_data["offset_x"]
            offset_y = tile_data["offset_y"]

            tile_results = self._predict(
                tile,
                confidence=0.25,
                image_size=1280,
                classes=[
                    self.PERSON_CLASS
                ]
            )

            detections = self._extract_detections(
                tile_results,
                tile.shape[1],
                tile.shape[0],
                self.PERSON_CLASS
            )

            for detection in detections:

                x1, y1, x2, y2 = detection["box"]

                detection["box"] = [
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y
                ]

                tile_detections.append(
                    detection
                )

        # -----------------------------------------------------
        # 3. REMOVE DUPLICATES BETWEEN TILES
        # -----------------------------------------------------

        tile_detections = self._remove_duplicates(
            tile_detections,
            iou_threshold=0.40,
            containment_threshold=0.70
        )

        # -----------------------------------------------------
        # 4. MERGE FULL-IMAGE AND TILE DETECTIONS
        # -----------------------------------------------------

        merged = list(full_detections)

        for tile_detection in tile_detections:

            tile_box = tile_detection["box"]

            duplicate = False

            for full_detection in full_detections:

                full_box = full_detection["box"]

                overlap = self._iou(
                    tile_box,
                    full_box
                )

                containment = self._containment_ratio(
                    tile_box,
                    full_box
                )

                if (
                    overlap >= 0.35
                    or containment >= 0.70
                ):

                    duplicate = True
                    break

            if not duplicate:
                merged.append(
                    tile_detection
                )

        # -----------------------------------------------------
        # 5. FINAL DUPLICATE FILTER
        # -----------------------------------------------------

        merged = self._remove_duplicates(
            merged,
            iou_threshold=0.45,
            containment_threshold=0.75
        )

        # -----------------------------------------------------
        # 6. FILTER INVALID BOXES
        # -----------------------------------------------------

        filtered = []

        for detection in merged:

            x1, y1, x2, y2 = detection["box"]

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 8 or box_height < 12:
                continue

            aspect_ratio = (
                box_width / max(box_height, 1)
            )

            if aspect_ratio > 2.5:
                continue

            filtered.append(detection)

        print(
            f"Person detections: {len(filtered)}"
        )

        return filtered

    # =========================================================
    # TRAFFIC-LIGHT DETECTION
    # =========================================================

    def _detect_traffic_lights(
        self,
        image
    ):

        results = self._predict(
            image,
            confidence=self.confidence,
            image_size=self.image_size,
            classes=[
                self.TRAFFIC_LIGHT_CLASS
            ]
        )

        return results

    # =========================================================
    # MAIN DETECTION
    # =========================================================

    def detect(self, image):

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input must be a NumPy image."
            )

        if image.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected a three-channel BGR image."
            )

        image_height, image_width = image.shape[:2]

        # Detect cars
        cars = self._detect_cars(
            image,
            image_width,
            image_height
        )

        # Detect people
        people = self._detect_people(
            image,
            image_width,
            image_height
        )

        # Detect traffic lights
        main_result = self._detect_traffic_lights(
            image
        )

        return {

            "main_result": main_result,

            "cars": cars,

            "people": people

        }
