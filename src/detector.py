
import cv2
import numpy as np

from ultralytics import YOLO


class VehicleDetector:
    """
    Detects cars, people, and traffic lights in traffic images.

    Features:
    - Car detection
    - Person detection
    - Traffic light detection
    - Tiled person detection for large images
    - Duplicate detection filtering
    - CPU-only inference
    - Dictionary output compatible with CarColorPipeline
    """

    # COCO class IDs
    PERSON_CLASS_ID = 0
    CAR_CLASS_ID = 2
    TRAFFIC_LIGHT_CLASS_ID = 9

    def __init__(
        self,
        model_path="yolo26s.pt",
        confidence=0.25,
        iou_threshold=0.45,
        max_detections=300,
        device="cpu",
    ):

        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.device = device

        self.model = YOLO(self.model_path)

        print("VehicleDetector initialized successfully.")
        print(f"YOLO model: {self.model_path}")
        print(f"Inference device: {self.device}")

    # ========================================================
    # GENERAL YOLO PREDICTION
    # ========================================================

    def _predict(
        self,
        image,
        conf=0.25,
        imgsz=1280,
        iou=0.45,
        max_det=300,
        classes=None,
    ):

        if image is None or not isinstance(image, np.ndarray):
            return []

        if image.size == 0:
            return []

        try:

            results = self.model.predict(
                source=image,
                conf=conf,
                imgsz=imgsz,
                iou=iou,
                max_det=max_det,
                classes=classes,
                device=self.device,
                verbose=False,
            )

            return results

        except Exception as error:

            print(f"YOLO prediction error: {error}")
            return []

    # ========================================================
    # EXTRACT DETECTION BOXES
    # ========================================================

    @staticmethod
    def _extract_detections(
        results,
        image_width,
        image_height,
    ):

        detections = []

        if results is None:
            return detections

        if not isinstance(results, (list, tuple)):
            results = [results]

        for result in results:

            if result is None or result.boxes is None:
                continue

            boxes = result.boxes

            if len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(
                xyxy,
                confidences,
                class_ids,
            ):

                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, min(x1, image_width - 1))
                y1 = max(0, min(y1, image_height - 1))
                x2 = max(0, min(x2, image_width))
                y2 = max(0, min(y2, image_height))

                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = result.names.get(
                    int(class_id),
                    str(class_id),
                )

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float(confidence),
                        "class_id": int(class_id),
                        "class_name": str(class_name).lower(),
                    }
                )

        return detections

    # ========================================================
    # DETECT CARS
    # ========================================================

    def _detect_cars(self, image):

        height, width = image.shape[:2]

        results = self._predict(
            image=image,
            conf=0.30,
            imgsz=1536,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=[self.CAR_CLASS_ID],
        )

        detections = self._extract_detections(
            results,
            width,
            height,
        )

        cars = [
            detection
            for detection in detections
            if detection["class_id"] == self.CAR_CLASS_ID
        ]

        return cars

    # ========================================================
    # DETECT PEOPLE
    # ========================================================

    def _detect_people(self, image):

        height, width = image.shape[:2]

        # Full-image detection
        full_results = self._predict(
            image=image,
            conf=0.25,
            imgsz=1536,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=[self.PERSON_CLASS_ID],
        )

        full_detections = self._extract_detections(
            full_results,
            width,
            height,
        )

        people = [
            detection
            for detection in full_detections
            if detection["class_id"] == self.PERSON_CLASS_ID
        ]

        # Tiled detection for smaller people in large images
        rows = 3
        cols = 4
        overlap = 0.25

        tile_height = int(
            height / (rows - (rows - 1) * overlap)
        )

        tile_width = int(
            width / (cols - (cols - 1) * overlap)
        )

        step_y = max(
            1,
            int(tile_height * (1 - overlap)),
        )

        step_x = max(
            1,
            int(tile_width * (1 - overlap)),
        )

        for row in range(rows):

            for col in range(cols):

                x1 = col * step_x
                y1 = row * step_y

                x2 = min(
                    x1 + tile_width,
                    width,
                )

                y2 = min(
                    y1 + tile_height,
                    height,
                )

                # Ensure the last tiles reach image boundaries
                if col == cols - 1:
                    x2 = width

                if row == rows - 1:
                    y2 = height

                if x2 <= x1 or y2 <= y1:
                    continue

                tile = image[y1:y2, x1:x2]

                if tile.size == 0:
                    continue

                tile_results = self._predict(
                    image=tile,
                    conf=0.25,
                    imgsz=1280,
                    iou=self.iou_threshold,
                    max_det=self.max_detections,
                    classes=[self.PERSON_CLASS_ID],
                )

                tile_detections = self._extract_detections(
                    tile_results,
                    tile.shape[1],
                    tile.shape[0],
                )

                for detection in tile_detections:

                    if (
                        detection["class_id"]
                        != self.PERSON_CLASS_ID
                    ):
                        continue

                    bx1, by1, bx2, by2 = detection["bbox"]

                    # Convert tile coordinates to full-image coordinates
                    detection["bbox"] = (
                        bx1 + x1,
                        by1 + y1,
                        bx2 + x1,
                        by2 + y1,
                    )

                    people.append(detection)

        # Remove duplicate detections
        people = self._remove_duplicates(
            people,
            iou_threshold=0.45,
        )

        return people

    # ========================================================
    # DETECT TRAFFIC LIGHTS
    # ========================================================

    def _detect_traffic_lights(self, image):

        height, width = image.shape[:2]

        results = self._predict(
            image=image,
            conf=0.20,
            imgsz=1280,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            classes=[self.TRAFFIC_LIGHT_CLASS_ID],
        )

        detections = self._extract_detections(
            results,
            width,
            height,
        )

        traffic_lights = [
            detection
            for detection in detections
            if (
                detection["class_id"]
                == self.TRAFFIC_LIGHT_CLASS_ID
            )
        ]

        return traffic_lights

    # ========================================================
    # CALCULATE INTERSECTION OVER UNION
    # ========================================================

    @staticmethod
    def _calculate_iou(box1, box2):

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])

        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(
            0,
            x2 - x1,
        )

        intersection_height = max(
            0,
            y2 - y1,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area1 = (
            max(0, box1[2] - box1[0])
            * max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0])
            * max(0, box2[3] - box2[1])
        )

        union_area = (
            area1 + area2 - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # ========================================================
    # REMOVE DUPLICATE DETECTIONS
    # ========================================================

    @classmethod
    def _remove_duplicates(
        cls,
        detections,
        iou_threshold=0.45,
    ):

        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda item: item["confidence"],
            reverse=True,
        )

        kept = []

        for detection in detections:

            is_duplicate = False

            for existing in kept:

                overlap = cls._calculate_iou(
                    detection["bbox"],
                    existing["bbox"],
                )

                if overlap >= iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(detection)

        return kept

    # ========================================================
    # MAIN DETECTION FUNCTION
    # ========================================================

    def detect(self, image):

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

        if image.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        print("Starting vehicle detection...")

        # Detect cars
        cars = self._detect_cars(image)

        print(f"Cars detected: {len(cars)}")

        # Detect people
        people = self._detect_people(image)

        print(f"People detected: {len(people)}")

        # Detect traffic lights
        traffic_lights = self._detect_traffic_lights(
            image
        )

        print(
            f"Traffic lights detected: "
            f"{len(traffic_lights)}"
        )

        # Full-image YOLO results
        main_result = self._predict(
            image=image,
            conf=self.confidence,
            imgsz=1536,
            iou=self.iou_threshold,
            max_det=self.max_detections,
        )

        # IMPORTANT:
        # Return a dictionary because pipeline.py expects one.
        return {
            "main_result": main_result,
            "cars": cars,
            "people": people,
            "traffic_lights": traffic_lights,
            "car_count": len(cars),
            "people_count": len(people),
            "traffic_light_count": len(traffic_lights),
        }

    # ========================================================
    # DETECT ALL OBJECTS
    # ========================================================

    def detect_all(self, image):

        return self.detect(image)
