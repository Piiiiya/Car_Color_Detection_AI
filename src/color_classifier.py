
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
            "black",
            "grey",
            "silver",
            "white"
        }

        self.chromatic_colors = {
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "purple",
            "pink"
        }

        self.earth_colors = {
            "beige",
            "brown",
            "gold",
            "tan"
        }

    # ============================================================
    # BODY REGION
    # ============================================================

    def _body_crop(self, image):

        h, w = image.shape[:2]

        if h < 20 or w < 20:
            return image

        x1 = int(w * 0.08)
        x2 = int(w * 0.92)

        y1 = int(h * 0.20)
        y2 = int(h * 0.85)

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

        x = np.expand_dims(
            x,
            axis=0
        )

        prediction = self.model.predict(
            x,
            verbose=0
        )

        probabilities = np.asarray(
            prediction
        )[0]

        class_id = int(
            np.argmax(probabilities)
        )

        color = self.id_to_label.get(
            class_id,
            "unknown"
        )

        confidence = float(
            probabilities[class_id]
        ) * 100.0

        return (
            color,
            confidence,
            probabilities
        )

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

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        total_pixels = H.size

        if total_pixels == 0:
            return self._empty_pixel_result()

        # --------------------------------------------------------
        # COLOUR MASKS
        # --------------------------------------------------------

        masks = {

            "red": (
                ((H <= 10) | (H >= 170))
                & (S >= 55)
                & (V >= 40)
            ),

            "orange": (
                (H > 10)
                & (H <= 22)
                & (S >= 55)
                & (V >= 40)
            ),

            "yellow": (
                (H > 22)
                & (H <= 38)
                & (S >= 50)
                & (V >= 55)
            ),

            "green": (
                (H > 38)
                & (H < 85)
                & (S >= 50)
                & (V >= 40)
            ),

            "blue": (
                (H >= 85)
                & (H <= 135)
                & (S >= 45)
                & (V >= 35)
            ),

            "purple": (
                (H > 135)
                & (H < 170)
                & (S >= 45)
                & (V >= 35)
            ),

            "black": (
                (V <= 65)
                & (S <= 130)
            ),

            "white": (
                (V >= 190)
                & (S <= 55)
            ),

            "grey": (
                (V > 65)
                & (V < 190)
                & (S <= 55)
            ),

            "brown": (
                (H >= 5)
                & (H < 25)
                & (S >= 35)
                & (S < 180)
                & (V >= 35)
                & (V < 175)
            )
        }

        # --------------------------------------------------------
        # CLEAN MASKS
        # --------------------------------------------------------

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8
        )

        percentages = {}
        largest_regions = {}

        for color, mask in masks.items():

            mask_uint8 = (
                mask.astype(np.uint8) * 255
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

            cleaned = mask_uint8 > 0

            percentages[color] = (
                np.count_nonzero(cleaned)
                / total_pixels
            ) * 100.0

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
                    area
                )

            largest_regions[color] = (
                largest_area
                / total_pixels
            ) * 100.0

        # --------------------------------------------------------
        # PIXEL SCORES
        # --------------------------------------------------------

        color_scores = {}

        for color in masks:

            percentage = percentages[color]

            region = largest_regions[color]

            color_scores[color] = (
                percentage * 0.55
                +
                region * 0.45
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

        return {

            "pixel_color": pixel_color,

            "pixel_confidence": float(
                pixel_confidence
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

        if image is None or image.size == 0:

            return {

                "color": "unknown",
                "confidence": 0.0,

                "pixel_color": "unknown",
                "pixel_confidence": 0.0,

                "cnn_color": "unknown",
                "cnn_confidence": 0.0,

                "blue_pixel_percentage": 0.0,
                "black_pixel_percentage": 0.0,

                "blue_region_percentage": 0.0,
                "black_region_percentage": 0.0,

                "largest_color_region": 0.0,

                "reason": "No image",

                "color_scores": {}
            }

        # --------------------------------------------------------
        # 1. CNN
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

        if ranked_colors:

            best_pixel_color = ranked_colors[0][0]

            best_pixel_score = ranked_colors[0][1]

            second_pixel_score = (
                ranked_colors[1][1]
                if len(ranked_colors) > 1
                else 0.0
            )

        else:

            best_pixel_color = "unknown"

            best_pixel_score = 0.0

            second_pixel_score = 0.0

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
        # 5. CHECK STRONG PIXEL EVIDENCE
        # --------------------------------------------------------

        strong_pixel_evidence = (

            best_pixel_percentage >= 45.0

            and best_pixel_region >= 18.0

            and score_margin >= 12.0

        )

        # --------------------------------------------------------
        # 6. CNN AND PIXEL AGREEMENT
        # --------------------------------------------------------

        if (
            cnn_color == pixel_color
            and pixel_confidence >= 12.0
        ):

            final_color = cnn_color

            # Keep CNN probability separate from pixel score.
            final_confidence = cnn_confidence

            reason = "CNN and pixel analysis agree"

        # --------------------------------------------------------
        # 7. PIXEL OVERRIDE
        # --------------------------------------------------------

        elif strong_pixel_evidence:

            # Pixel analysis may override only when CNN
            # confidence is relatively low.
            #
            # This prevents weak pixel evidence from
            # changing a reasonably confident CNN prediction.

            if cnn_confidence < 40.0:

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
        # 10. RETURN
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
