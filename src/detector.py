from ultralytics import YOLO
import cv2
import numpy as np


class VehicleDetector:

    def __init__(
        self,
        model_path="yolo26s.pt",
        confidence=0.15,
        imgsz=1280,
        iou=0.50,
        max_det=500
    ):
        self.model = YOLO(model_path)

        self.confidence = confidence
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det

        # COCO classes
        self.PERSON_CLASS = 0
        self.CAR_CLASS = 2
        self.TRAFFIC_LIGHT_CLASS = 9

    # ---------------------------------------------------------
    # NORMAL FULL-IMAGE DETECTION
    # ---------------------------------------------------------
    def _predict(
        self,
        image,
        conf=None,
        imgsz=None,
        classes=None
    ):

        if conf is None:
            conf = self.confidence

        if imgsz is None:
            imgsz = self.imgsz

        results = self.model.predict(
            source=image,
            conf=conf,
            iou=self.iou,
            imgsz=imgsz,
            max_det=self.max_det,
            classes=classes,
            verbose=False
        )

        return results[0]

    # ---------------------------------------------------------
    # IOU
    # ---------------------------------------------------------
    @staticmethod
    def _iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)

        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        intersection = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    # ---------------------------------------------------------
    # MERGE DUPLICATE PERSON DETECTIONS
    # ---------------------------------------------------------
    def _merge_person_detections(
        self,
        detections,
        iou_threshold=0.45
    ):

        if not detections:
            return []

        # Highest confidence first
        detections = sorted(
            detections,
            key=lambda x: x["confidence"],
            reverse=True
        )

        kept = []

        for candidate in detections:

            duplicate = False

            for existing in kept:

                iou = self._iou(
                    candidate["box"],
                    existing["box"]
                )

                if iou >= iou_threshold:
                    duplicate = True

                    # Keep the higher confidence box
                    if candidate["confidence"] > existing["confidence"]:
                        existing.update(candidate)

                    break

            if not duplicate:
                kept.append(candidate)

        return kept

    # ---------------------------------------------------------
    # DETECT PEOPLE ON FULL IMAGE
    # ---------------------------------------------------------
    def _detect_people_full_image(self, image):

        result = self._predict(
            image,
            conf=0.12,
            imgsz=1536,
            classes=[self.PERSON_CLASS]
        )

        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for box, confidence in zip(boxes, confs):

            x1, y1, x2, y2 = box.astype(int)

            w = x2 - x1
            h = y2 - y1

            if w < 8 or h < 12:
                continue

            detections.append(
                {
                    "box": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "confidence": float(confidence)
                }
            )

        return detections

    # ---------------------------------------------------------
    # TILE GENERATOR
    # ---------------------------------------------------------
    def _generate_tiles(
        self,
        image,
        rows=2,
        cols=3,
        overlap=0.20
    ):

        height, width = image.shape[:2]

        tile_w = int(
            width / (
                cols - overlap * (cols - 1)
            )
        )

        tile_h = int(
            height / (
                rows - overlap * (rows - 1)
            )
        )

        step_x = int(
            tile_w * (1.0 - overlap)
        )

        step_y = int(
            tile_h * (1.0 - overlap)
        )

        tiles = []

        y_positions = []
        x_positions = []

        y = 0

        while True:

            y_positions.append(y)

            if y + tile_h >= height:
                break

            y += step_y

            if y + tile_h > height:
                y = height - tile_h

        x = 0

        while True:

            x_positions.append(x)

            if x + tile_w >= width:
                break

            x += step_x

            if x + tile_w > width:
                x = width - tile_w

        for y1 in y_positions:

            for x1 in x_positions:

                x2 = min(
                    width,
                    x1 + tile_w
                )

                y2 = min(
                    height,
                    y1 + tile_h
                )

                tile = image[
                    y1:y2,
                    x1:x2
                ]

                if tile.size == 0:
                    continue

                tiles.append(
                    {
                        "image": tile,
                        "offset_x": x1,
                        "offset_y": y1
                    }
                )

        return tiles

    # ---------------------------------------------------------
    # DETECT PEOPLE USING TILES
    # ---------------------------------------------------------
    def _detect_people_tiled(self, image):

        tiles = self._generate_tiles(
            image,
            rows=2,
            cols=3,
            overlap=0.20
        )

        all_detections = []

        for tile_data in tiles:

            tile = tile_data["image"]

            offset_x = tile_data["offset_x"]
            offset_y = tile_data["offset_y"]

            result = self._predict(
                tile,
                conf=0.10,
                imgsz=960,
                classes=[self.PERSON_CLASS]
            )

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, confidence in zip(boxes, confs):

                x1, y1, x2, y2 = box.astype(int)

                # Convert tile coordinates
                x1 += offset_x
                x2 += offset_x

                y1 += offset_y
                y2 += offset_y

                w = x2 - x1
                h = y2 - y1

                if w < 8 or h < 12:
                    continue

                all_detections.append(
                    {
                        "box": (
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ),
                        "confidence": float(confidence)
                    }
                )

        return all_detections

    # ---------------------------------------------------------
    # FINAL PEOPLE DETECTION
    # ---------------------------------------------------------
    def _detect_people(self, image):

        full_detections = (
            self._detect_people_full_image(image)
        )

        tiled_detections = (
            self._detect_people_tiled(image)
        )

        combined = (
            full_detections +
            tiled_detections
        )

        final_people = (
            self._merge_person_detections(
                combined,
                iou_threshold=0.45
            )
        )

        return final_people

    # ---------------------------------------------------------
    # MAIN DETECTION
    # ---------------------------------------------------------
    def detect(self, image):

        # -----------------------------------------------------
        # 1. Detect cars + traffic lights normally
        # -----------------------------------------------------

        main_result = self._predict(
            image,
            conf=self.confidence,
            imgsz=self.imgsz,
            classes=[
                self.CAR_CLASS,
                self.TRAFFIC_LIGHT_CLASS
            ]
        )

        # -----------------------------------------------------
        # 2. Detect people separately using full image + tiles
        # -----------------------------------------------------

        people = self._detect_people(image)

        return {
            "main_result": main_result,
            "people": people
        }