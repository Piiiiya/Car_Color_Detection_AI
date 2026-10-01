
import os
import cv2
import numpy as np


class CarColorPipeline:

    def __init__(self, detector, classifier):
        self.detector = detector
        self.classifier = classifier

        # Debug crop directory
        self.debug_dir = os.path.join(
            "outputs",
            "debug_crops"
        )

        os.makedirs(
            self.debug_dir,
            exist_ok=True
        )

    # =========================================================
    # DRAW TEXT WITH BACKGROUND
    # =========================================================

    @staticmethod
    def _draw_label(
        image,
        text,
        x,
        y,
        color,
        font_scale=0.55,
        thickness=2
    ):

        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        y = max(y, text_h + baseline + 4)

        cv2.rectangle(
            image,
            (
                x,
                y - text_h - baseline - 4
            ),
            (
                x + text_w + 6,
                y
            ),
            color,
            -1
        )

        cv2.putText(
            image,
            text,
            (
                x + 3,
                y - baseline - 2
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    # =========================================================
    # DRAW PERSON
    # =========================================================

    def _draw_person(self, image, person):

        x1, y1, x2, y2 = person["box"]
        confidence = person["confidence"]

        color = (0, 180, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = f"person {confidence * 100:.1f}%"

        self._draw_label(
            image,
            label,
            x1,
            y1,
            color
        )

    # =========================================================
    # DRAW TRAFFIC LIGHT
    # =========================================================

    def _draw_traffic_light(
        self,
        image,
        box,
        confidence
    ):

        x1, y1, x2, y2 = box

        color = (0, 255, 255)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = (
            f"traffic light "
            f"{confidence * 100:.1f}%"
        )

        self._draw_label(
            image,
            label,
            x1,
            y1,
            color
        )

    # =========================================================
    # DRAW CAR
    # =========================================================

    def _draw_car(
        self,
        image,
        box,
        color_name,
        confidence
    ):

        x1, y1, x2, y2 = box

        # Blue car -> red rectangle
        # Other car -> blue rectangle

        if color_name.lower() == "blue":
            box_color = (0, 0, 255)
        else:
            box_color = (255, 0, 0)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )

        label = (
            f"car ({color_name}) "
            f"{confidence:.1f}%"
        )

        self._draw_label(
            image,
            label,
            x1,
            y1,
            box_color
        )

    # =========================================================
    # DRAW SUMMARY PANEL
    # =========================================================

    def _draw_summary(
        self,
        image,
        cars,
        blue_cars,
        other_cars,
        people,
        traffic_lights
    ):

        height, width = image.shape[:2]

        panel_width = min(
            360,
            max(280, width // 4)
        )

        panel_height = 185

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (
                10 + panel_width,
                10 + panel_height
            ),
            (25, 25, 25),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.78,
            image,
            0.22,
            0,
            image
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        lines = [
            f"Cars: {cars}",
            f"Blue Cars: {blue_cars}",
            f"Other Cars: {other_cars}",
            f"People: {people}",
            f"Traffic Lights: {traffic_lights}"
        ]

        y = 42

        for line in lines:

            cv2.putText(
                image,
                line,
                (25, y),
                font,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            y += 30

    # =========================================================
    # EXTRACT MAIN YOLO DETECTIONS
    # =========================================================

    def _extract_main_detections(self, result):

        detections = []

        if result is None or result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        names = result.names

        for box, confidence, class_id in zip(
            boxes,
            confs,
            classes
        ):

            x1, y1, x2, y2 = box.astype(int)

            if isinstance(names, dict):
                class_name = names.get(
                    int(class_id),
                    str(class_id)
                )
            else:
                class_name = names[int(class_id)]

            detections.append({
                "box": (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ),
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": str(class_name).lower()
            })

        return detections

    # =========================================================
    # SAVE DEBUG CROP
    # =========================================================

    def _save_debug_crop(
        self,
        crop,
        car_number
    ):

        if crop is None or crop.size == 0:
            return None

        os.makedirs(
            self.debug_dir,
            exist_ok=True
        )

        filename = f"car_{car_number:03d}.jpg"

        filepath = os.path.join(
            self.debug_dir,
            filename
        )

        success = cv2.imwrite(
            filepath,
            crop
        )

        if success:
            print(
                f"Debug crop saved: {filepath}"
            )
            return filepath

        print(
            f"Could not save debug crop: {filepath}"
        )

        return None

    # =========================================================
    # SAFE FLOAT CONVERSION
    # =========================================================

    @staticmethod
    def _safe_float(value, default=0.0):

        try:
            value = float(value)

            if not np.isfinite(value):
                return default

            return value

        except (TypeError, ValueError):
            return default

    # =========================================================
    # MAIN PROCESS
    # =========================================================

    def process(self, image):

        if image is None:
            raise ValueError(
                "Input image is None."
            )

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Input image must be a NumPy array."
            )

        if image.size == 0:
            raise ValueError(
                "Input image is empty."
            )

        output = image.copy()

        # -----------------------------------------------------
        # RUN DETECTOR
        # -----------------------------------------------------

        detection_result = self.detector.detect(image)

        if not isinstance(detection_result, dict):
            raise TypeError(
                "VehicleDetector.detect() "
                "must return a dictionary."
            )

        main_result = detection_result.get(
            "main_result"
        )

        car_detections = detection_result.get(
            "cars",
            []
        )

        people = detection_result.get(
            "people",
            []
        )

        # -----------------------------------------------------
        # EXTRACT TRAFFIC LIGHT DETECTIONS
        # -----------------------------------------------------

        main_detections = (
            self._extract_main_detections(
                main_result
            )
        )

        # Add full-image and tiled car detections
        main_detections.extend(
            car_detections
        )

        # -----------------------------------------------------
        # COUNTERS
        # -----------------------------------------------------

        cars = 0
        blue_cars = 0
        other_cars = 0
        traffic_lights = 0

        car_details = []

        # -----------------------------------------------------
        # PROCESS CARS AND TRAFFIC LIGHTS
        # -----------------------------------------------------

        for detection in main_detections:

            class_name = str(
                detection.get(
                    "class_name",
                    ""
                )
            ).lower()

            confidence = self._safe_float(
                detection.get(
                    "confidence",
                    0.0
                )
            )

            x1, y1, x2, y2 = detection["box"]

            # ---------------------------------------------
            # TRAFFIC LIGHT
            # ---------------------------------------------

            if class_name in (
                "traffic light",
                "traffic_light",
                "trafficlight"
            ):

                traffic_lights += 1

                self._draw_traffic_light(
                    output,
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    confidence
                )

                continue

            # ---------------------------------------------
            # CAR
            # ---------------------------------------------

            if class_name not in (
                "car",
                "vehicle"
            ):
                continue

            if confidence < 0.10:
                continue

            # ---------------------------------------------
            # VALIDATE BOX
            # ---------------------------------------------

            x1 = max(
                0,
                min(
                    int(x1),
                    image.shape[1] - 1
                )
            )

            y1 = max(
                0,
                min(
                    int(y1),
                    image.shape[0] - 1
                )
            )

            x2 = max(
                0,
                min(
                    int(x2),
                    image.shape[1]
                )
            )

            y2 = max(
                0,
                min(
                    int(y2),
                    image.shape[0]
                )
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 15 or box_height < 12:
                continue

            # ---------------------------------------------
            # CROP CAR
            # ---------------------------------------------

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            # ---------------------------------------------
            # SAVE CROP FOR DEBUGGING
            # ---------------------------------------------

            debug_path = self._save_debug_crop(
                crop,
                cars + 1
            )

            # ---------------------------------------------
            # COLOR PREDICTION
            # ---------------------------------------------

            try:

                color_result = self.classifier.predict(
                    crop
                )

                if not isinstance(color_result, dict):
                    raise TypeError(
                        "Classifier must return a dictionary."
                    )

            except Exception as exc:

                print(
                    "Color prediction failed:",
                    exc
                )

                color_result = {
                    "color": "unknown",
                    "confidence": 0.0,
                    "pixel_color": "unknown",
                    "pixel_confidence": 0.0,
                    "cnn_color": "unknown",
                    "cnn_confidence": 0.0,
                    "reason": "Color prediction failed",
                    "blue_pixel_percentage": 0.0,
                    "black_pixel_percentage": 0.0,
                    "blue_region_percentage": 0.0,
                    "black_region_percentage": 0.0
                }

            # ---------------------------------------------
            # READ CLASSIFIER OUTPUT
            # ---------------------------------------------

            color_name = str(
                color_result.get(
                    "color",
                    "unknown"
                )
            ).lower()

            color_confidence = self._safe_float(
                color_result.get(
                    "confidence",
                    0.0
                )
            )

            pixel_color = str(
                color_result.get(
                    "pixel_color",
                    "unknown"
                )
            )

            pixel_confidence = self._safe_float(
                color_result.get(
                    "pixel_confidence",
                    0.0
                )
            )

            cnn_color = str(
                color_result.get(
                    "cnn_color",
                    "unknown"
                )
            )

            cnn_confidence = self._safe_float(
                color_result.get(
                    "cnn_confidence",
                    0.0
                )
            )

            reason = str(
                color_result.get(
                    "reason",
                    ""
                )
            )

            blue_pixel_percentage = self._safe_float(
                color_result.get(
                    "blue_pixel_percentage",
                    0.0
                )
            )

            black_pixel_percentage = self._safe_float(
                color_result.get(
                    "black_pixel_percentage",
                    0.0
                )
            )

            blue_region_percentage = self._safe_float(
                color_result.get(
                    "blue_region_percentage",
                    0.0
                )
            )

            black_region_percentage = self._safe_float(
                color_result.get(
                    "black_region_percentage",
                    0.0
                )
            )

            # ---------------------------------------------
            # COUNT CAR
            # ---------------------------------------------

            cars += 1

            if color_name == "blue":
                blue_cars += 1
            else:
                other_cars += 1

            # ---------------------------------------------
            # DRAW CAR
            # ---------------------------------------------

            self._draw_car(
                output,
                (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                color_name,
                color_confidence
            )

            # ---------------------------------------------
            # SAVE DETAILS
            # ---------------------------------------------

            car_details.append({
                "Car": cars,
                "Color": color_name,
                "Confidence": round(
                    color_confidence,
                    2
                ),
                "Pixel Color": pixel_color,
                "Pixel Confidence": round(
                    pixel_confidence,
                    2
                ),
                "CNN Color": cnn_color,
                "CNN Confidence": round(
                    cnn_confidence,
                    2
                ),
                "Blue Pixels %": round(
                    blue_pixel_percentage,
                    2
                ),
                "Black Pixels %": round(
                    black_pixel_percentage,
                    2
                ),
                "Blue Region %": round(
                    blue_region_percentage,
                    2
                ),
                "Black Region %": round(
                    black_region_percentage,
                    2
                ),
                "Reason": reason,
                "Debug Crop": debug_path
            })

        # =====================================================
        # PROCESS PEOPLE
        # =====================================================

        final_people = []

        for person in people:

            if not isinstance(person, dict):
                continue

            if (
                "box" not in person
                or "confidence" not in person
            ):
                continue

            confidence = self._safe_float(
                person["confidence"]
            )

            if confidence < 0.10:
                continue

            x1, y1, x2, y2 = person["box"]

            x1 = max(
                0,
                min(
                    int(x1),
                    image.shape[1] - 1
                )
            )

            y1 = max(
                0,
                min(
                    int(y1),
                    image.shape[0] - 1
                )
            )

            x2 = max(
                0,
                min(
                    int(x2),
                    image.shape[1]
                )
            )

            y2 = max(
                0,
                min(
                    int(y2),
                    image.shape[0]
                )
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if box_width < 8 or box_height < 12:
                continue

            person_data = {
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
                "confidence": confidence
            }

            final_people.append(
                person_data
            )

            self._draw_person(
                output,
                person_data
            )

        people_count = len(final_people)

        # =====================================================
        # DRAW SUMMARY
        # =====================================================

        self._draw_summary(
            output,
            cars=cars,
            blue_cars=blue_cars,
            other_cars=other_cars,
            people=people_count,
            traffic_lights=traffic_lights
        )

        # =====================================================
        # RETURN RESULTS
        # =====================================================

        return {
            "image": output,
            "cars": cars,
            "blue_cars": blue_cars,
            "other_cars": other_cars,
            "people": people_count,
            "traffic_lights": traffic_lights,
            "car_details": car_details,
            "people_details": final_people
        }
