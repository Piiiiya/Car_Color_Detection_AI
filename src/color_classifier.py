
import cv2
import numpy as np


class CarColorClassifier:

    def __init__(
        self,
        model,
        img_size=(224, 224),
        id_to_label=None
    ):

        self.model = model
        self.img_size = img_size

        self.id_to_label = id_to_label or {
            0: "beige",
            1: "black",
            2: "blue",
            3: "brown",
            4: "gold",
            5: "green",
            6: "grey",
            7: "orange",
            8: "pink",
            9: "purple",
            10: "red",
            11: "silver",
            12: "tan",
            13: "white",
            14: "yellow"
        }

        self.neutral_colors = {
            "black", "grey", "silver", "white"
        }

        self.chromatic_colors = {
            "red", "orange", "yellow", "green",
            "blue", "purple", "pink"
        }

        self.earth_colors = {
            "beige", "brown", "gold", "tan"
        }

        self.all_colors = set(self.id_to_label.values())

    # ============================================================
    # SAFE IMAGE HANDLING
    # ============================================================

    @staticmethod
    def _valid_image(image):

        return (
            image is not None
            and isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim == 3
            and image.shape[2] >= 3
        )

    # ============================================================
    # BODY REGION
    # ============================================================

    def _body_crop(self, image):

        if not self._valid_image(image):
            return image

        h, w = image.shape[:2]

        if h < 20 or w < 20:
            return image

        # Focus on the central vehicle body.
        # Avoid extreme top, bottom, and side edges where
        # windows, road, and background are more likely.

        x1 = int(w * 0.10)
        x2 = int(w * 0.90)

        y1 = int(h * 0.22)
        y2 = int(h * 0.82)

        body = image[y1:y2, x1:x2]

        if body.size == 0:
            return image

        return body

    # ============================================================
    # CNN PREDICTION
    # ============================================================

    def _cnn_predict(self, image):

        resized = cv2.resize(
            image,
            self.img_size,
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        x = rgb.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        prediction = self.model.predict(
            x,
            verbose=0
        )

        probabilities = np.asarray(
            prediction,
            dtype=np.float32
        ).reshape(-1)

        if probabilities.size == 0:
            return "unknown", 0.0, probabilities

        class_id = int(np.argmax(probabilities))

        color = self.id_to_label.get(
            class_id,
            "unknown"
        )

        confidence = float(
            probabilities[class_id] * 100.0
        )

        return color, confidence, probabilities

    # ============================================================
    # COLOUR MASKS
    # ============================================================

    def _build_color_masks(self, hsv):

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        masks = {

            "red": (
                ((H <= 10) | (H >= 170))
                & (S >= 65)
                & (V >= 45)
            ),

            "orange": (
                (H > 10)
                & (H <= 22)
                & (S >= 65)
                & (V >= 50)
            ),

            "yellow": (
                (H > 22)
                & (H <= 38)
                & (S >= 55)
                & (V >= 65)
            ),

            "green": (
                (H > 38)
                & (H < 85)
                & (S >= 55)
                & (V >= 40)
            ),

            "blue": (
                (H >= 85)
                & (H <= 135)
                & (S >= 50)
                & (V >= 35)
            ),

            "purple": (
                (H > 135)
                & (H < 170)
                & (S >= 50)
                & (V >= 35)
            ),

            "black": (
                (V <= 65)
                & (S <= 150)
            ),

            "white": (
                (V >= 190)
                & (S <= 65)
            ),

            "grey": (
                (V > 65)
                & (V < 190)
                & (S <= 65)
            ),

            "brown": (
                (H >= 5)
                & (H < 25)
                & (S >= 35)
                & (S < 190)
                & (V >= 35)
                & (V < 170)
            )
        }

        return masks

    # ============================================================
    # MASK CLEANING
    # ============================================================

    @staticmethod
    def _clean_mask(mask):

        mask_uint8 = (
            mask.astype(np.uint8) * 255
        )

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        mask_uint8 = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_OPEN,
            kernel
        )

        mask_uint8 = cv2.morphologyEx(
            mask_uint8,
            cv2.MORPH_CLOSE,
            kernel
        )

        return mask_uint8

    # ============================================================
    # CONNECTED REGION ANALYSIS
    # ============================================================

    @staticmethod
    def _largest_region(mask_uint8):

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask_uint8,
                connectivity=8
            )
        )

        largest_area = 0

        for i in range(1, num_labels):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            largest_area = max(
                largest_area,
                int(area)
            )

        return largest_area

    # ============================================================
    # PIXEL ANALYSIS
    # ============================================================

    def _pixel_analysis(self, image):

        body = self._body_crop(image)

        if body is None or body.size == 0:
            return self._empty_pixel_result()

        body = cv2.resize(
            body,
            (300, 150),
            interpolation=cv2.INTER_AREA
        )

        hsv = cv2.cvtColor(
            body,
            cv2.COLOR_BGR2HSV
        )

        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return self._empty_pixel_result()

        masks = self._build_color_masks(hsv)

        percentages = {}
        largest_regions = {}
        color_scores = {}

        for color, mask in masks.items():

            cleaned = self._clean_mask(mask)

            cleaned_bool = cleaned > 0

            pixel_count = int(
                np.count_nonzero(cleaned_bool)
            )

            percentage = (
                pixel_count / total_pixels
            ) * 100.0

            largest_area = self._largest_region(
                cleaned
            )

            region_percentage = (
                largest_area / total_pixels
            ) * 100.0

            percentages[color] = float(
                percentage
            )

            largest_regions[color] = float(
                region_percentage
            )

            # Give more importance to total coverage,
            # while still rewarding connected body-colour areas.

            color_scores[color] = float(
                percentage * 0.65
                +
                region_percentage * 0.35
            )

        ranked = sorted(
            color_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        if ranked:

            pixel_color = ranked[0][0]
            pixel_confidence = ranked[0][1]

        else:

            pixel_color = "unknown"
            pixel_confidence = 0.0

        secondary_color = "unknown"
        secondary_score = 0.0

        if len(ranked) > 1:

            for color, score in ranked[1:]:

                # Require meaningful evidence for a secondary colour.
                if (
                    percentages.get(color, 0.0) >= 8.0
                    and largest_regions.get(color, 0.0) >= 3.0
                ):

                    secondary_color = color
                    secondary_score = score
                    break

        return {

            "pixel_color": pixel_color,

            "pixel_confidence": float(
                pixel_confidence
            ),

            "secondary_color": secondary_color,

            "secondary_confidence": float(
                secondary_score
            ),

            "blue_pixel_percentage": float(
                percentages.get("blue", 0.0)
            ),

            "black_pixel_percentage": float(
                percentages.get("black", 0.0)
            ),

            "blue_region_percentage": float(
                largest_regions.get("blue", 0.0)
            ),

            "black_region_percentage": float(
                largest_regions.get("black", 0.0)
            ),

            "largest_color_region": float(
                largest_regions.get(
                    pixel_color,
                    0.0
                )
            ),

            "color_scores": color_scores,

            "percentages": percentages,

            "largest_regions": largest_regions
        }

    # ============================================================
    # EMPTY RESULT
    # ============================================================

    @staticmethod
    def _empty_pixel_result():

        return {

            "pixel_color": "unknown",
            "pixel_confidence": 0.0,

            "secondary_color": "unknown",
            "secondary_confidence": 0.0,

            "blue_pixel_percentage": 0.0,
            "black_pixel_percentage": 0.0,

            "blue_region_percentage": 0.0,
            "black_region_percentage": 0.0,

            "largest_color_region": 0.0,

            "color_scores": {},
            "percentages": {},
            "largest_regions": {}
        }

    # ============================================================
    # FINAL PREDICTION
    # ============================================================

    def predict(self, image):

        if not self._valid_image(image):

            return {

                "color": "unknown",
                "confidence": 0.0,

                "pixel_color": "unknown",
                "pixel_confidence": 0.0,

                "cnn_color": "unknown",
                "cnn_confidence": 0.0,

                "secondary_color": "unknown",
                "secondary_confidence": 0.0,

                "blue_pixel_percentage": 0.0,
                "black_pixel_percentage": 0.0,

                "blue_region_percentage": 0.0,
                "black_region_percentage": 0.0,

                "largest_color_region": 0.0,

                "reason": "No valid image",

                "color_scores": {}
            }

        # --------------------------------------------------------
        # 1. CNN PREDICTION
        # --------------------------------------------------------

        (
            cnn_color,
            cnn_confidence,
            cnn_probabilities
        ) = self._cnn_predict(image)

        # --------------------------------------------------------
        # 2. PIXEL ANALYSIS
        # --------------------------------------------------------

        pixel_result = self._pixel_analysis(
            image
        )

        pixel_color = pixel_result[
            "pixel_color"
        ]

        pixel_confidence = pixel_result[
            "pixel_confidence"
        ]

        secondary_color = pixel_result[
            "secondary_color"
        ]

        secondary_confidence = pixel_result[
            "secondary_confidence"
        ]

        percentages = pixel_result[
            "percentages"
        ]

        largest_regions = pixel_result[
            "largest_regions"
        ]

        color_scores = pixel_result[
            "color_scores"
        ]

        # --------------------------------------------------------
        # 3. INITIAL DECISION
        # --------------------------------------------------------

        final_color = cnn_color
        final_confidence = cnn_confidence

        reason = "CNN primary prediction"

        # --------------------------------------------------------
        # 4. RANK PIXEL EVIDENCE
        # --------------------------------------------------------

        ranked_colors = sorted(
            color_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        best_pixel_color = "unknown"
        best_pixel_score = 0.0
        second_pixel_score = 0.0

        if ranked_colors:

            best_pixel_color = ranked_colors[0][0]
            best_pixel_score = ranked_colors[0][1]

            if len(ranked_colors) > 1:
                second_pixel_score = ranked_colors[1][1]

        score_margin = (
            best_pixel_score - second_pixel_score
        )

        best_pixel_percentage = percentages.get(
            best_pixel_color,
            0.0
        )

        best_pixel_region = largest_regions.get(
            best_pixel_color,
            0.0
        )

        # --------------------------------------------------------
        # 5. STRONG PIXEL EVIDENCE
        # --------------------------------------------------------

        strong_pixel_evidence = (

            best_pixel_percentage >= 40.0

            and best_pixel_region >= 15.0

            and score_margin >= 10.0

        )

        # --------------------------------------------------------
        # 6. CNN AND PIXEL AGREEMENT
        # --------------------------------------------------------

        if (
            cnn_color == pixel_color
            and pixel_confidence >= 10.0
        ):

            final_color = cnn_color
            final_confidence = cnn_confidence

            reason = "CNN and pixel analysis agree"

        # --------------------------------------------------------
        # 7. STRONG PIXEL OVERRIDE
        # --------------------------------------------------------

        elif strong_pixel_evidence:

            if cnn_confidence < 45.0:

                final_color = best_pixel_color

                final_confidence = min(
                    85.0,
                    max(
                        45.0,
                        best_pixel_score
                    )
                )

                reason = (
                    "Strong dominant pixel evidence "
                    "with low CNN confidence"
                )

            else:

                reason = (
                    "CNN retained because pixel evidence "
                    "does not justify overriding it"
                )

        # --------------------------------------------------------
        # 8. NEUTRAL COLOUR PROTECTION
        # --------------------------------------------------------

        elif (
            cnn_color in self.neutral_colors
            and cnn_confidence >= 30.0
        ):

            final_color = cnn_color
            final_confidence = cnn_confidence

            reason = (
                "CNN neutral prediction retained; "
                "pixel evidence is inconclusive"
            )

        # --------------------------------------------------------
        # 9. FALLBACK
        # --------------------------------------------------------

        if final_color == "unknown":

            final_color = cnn_color
            final_confidence = cnn_confidence

            reason = "CNN fallback"

        # --------------------------------------------------------
        # 10. SECONDARY COLOUR VALIDATION
        # --------------------------------------------------------

        if secondary_color == final_color:

            secondary_color = "unknown"
            secondary_confidence = 0.0

        # Only report a secondary colour when there is
        # sufficient pixel coverage and a connected region.

        if secondary_color != "unknown":

            secondary_coverage = percentages.get(
                secondary_color,
                0.0
            )

            secondary_region = largest_regions.get(
                secondary_color,
                0.0
            )

            if (
                secondary_coverage < 8.0
                or secondary_region < 3.0
            ):

                secondary_color = "unknown"
                secondary_confidence = 0.0

        # --------------------------------------------------------
        # 11. RETURN
        # --------------------------------------------------------

        return {

            "color": final_color,

            "confidence": float(
                final_confidence
            ),

            "pixel_color": pixel_color,

            "pixel_confidence": float(
                pixel_confidence
            ),

            "cnn_color": cnn_color,

            "cnn_confidence": float(
                cnn_confidence
            ),

            "secondary_color": secondary_color,

            "secondary_confidence": float(
                secondary_confidence
            ),

            "blue_pixel_percentage": float(
                pixel_result[
                    "blue_pixel_percentage"
                ]
            ),

            "black_pixel_percentage": float(
                pixel_result[
                    "black_pixel_percentage"
                ]
            ),

            "blue_region_percentage": float(
                pixel_result[
                    "blue_region_percentage"
                ]
            ),

            "black_region_percentage": float(
                pixel_result[
                    "black_region_percentage"
                ]
            ),

            "largest_color_region": float(
                pixel_result[
                    "largest_color_region"
                ]
            ),

            "reason": reason,

            "color_scores": color_scores
        }
