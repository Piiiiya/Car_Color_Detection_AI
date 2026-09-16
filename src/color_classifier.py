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

        x = rgb.astype(
            np.float32
        ) / 255.0

        x = np.expand_dims(
            x,
            axis=0
        )

        prediction = self.model.predict(
            x,
            verbose=0
        )

        prediction = np.asarray(
            prediction
        )[0]

        class_id = int(
            np.argmax(prediction)
        )

        confidence = float(
            prediction[class_id]
        ) * 100.0

        color = self.id_to_label.get(
            class_id,
            "unknown"
        )

        return color, confidence

    # ============================================================
    # PERCENTAGE
    # ============================================================

    def _percentage(self, mask):

        if mask is None:
            return 0.0

        total = mask.size

        if total == 0:
            return 0.0

        return (
            np.count_nonzero(mask)
            / total
        ) * 100.0

    # ============================================================
    # LARGEST CONNECTED REGION
    # ============================================================

    def _largest_component_percentage(
        self,
        mask
    ):

        if mask is None:
            return 0.0

        if np.count_nonzero(mask) == 0:
            return 0.0

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return 0.0

        largest_area = 0

        for i in range(1, num_labels):

            area = stats[
                i,
                cv2.CC_STAT_AREA
            ]

            if area > largest_area:
                largest_area = area

        total_area = (
            mask.shape[0] *
            mask.shape[1]
        )

        if total_area == 0:
            return 0.0

        return (
            largest_area /
            total_area
        ) * 100.0

    # ============================================================
    # PIXEL ANALYSIS
    # ============================================================

    def _pixel_analysis(self, image):

        h, w = image.shape[:2]

        if h < 20 or w < 20:

            return {
                "pixel_color": "unknown",
                "pixel_confidence": 0.0,
                "blue_pixel_percentage": 0.0,
                "largest_color_region": 0.0,
                "color_scores": {},
                "percentages": {},
                "largest_regions": {}
            }

        # --------------------------------------------------------
        # CENTER REGION
        # --------------------------------------------------------

        x1 = int(w * 0.08)
        x2 = int(w * 0.92)

        y1 = int(h * 0.12)
        y2 = int(h * 0.92)

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:
            crop = image

        hsv = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2HSV
        )

        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # ========================================================
        # COLOR MASKS
        # ========================================================

        masks = {}

        # --------------------------------------------------------
        # RED
        # --------------------------------------------------------

        red_mask = (
            (
                (H <= 10) |
                (H >= 170)
            )
            &
            (S >= 65)
            &
            (V >= 55)
        )

        masks["red"] = (
            red_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # ORANGE
        # --------------------------------------------------------

        orange_mask = (
            (H >= 10)
            &
            (H < 22)
            &
            (S >= 65)
            &
            (V >= 55)
        )

        masks["orange"] = (
            orange_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # YELLOW
        # --------------------------------------------------------

        yellow_mask = (
            (H >= 22)
            &
            (H < 38)
            &
            (S >= 55)
            &
            (V >= 65)
        )

        masks["yellow"] = (
            yellow_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # GREEN
        # --------------------------------------------------------

        green_mask = (
            (H >= 38)
            &
            (H < 85)
            &
            (S >= 45)
            &
            (V >= 45)
        )

        masks["green"] = (
            green_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # BLUE
        #
        # Blue/cyan/turquoise.
        # --------------------------------------------------------

        blue_mask = (
            (H >= 85)
            &
            (H <= 135)
            &
            (S >= 35)
            &
            (V >= 55)
        )

        masks["blue"] = (
            blue_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # PURPLE
        # --------------------------------------------------------

        purple_mask = (
            (H > 135)
            &
            (H < 170)
            &
            (S >= 50)
            &
            (V >= 50)
        )

        masks["purple"] = (
            purple_mask.astype(np.uint8) * 255
        )

        # ========================================================
        # NEUTRAL COLORS
        # ========================================================

        # --------------------------------------------------------
        # BLACK
        # --------------------------------------------------------

        black_mask = (
            (V <= 70)
            &
            (S <= 120)
        )

        masks["black"] = (
            black_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # WHITE
        # --------------------------------------------------------

        white_mask = (
            (V >= 185)
            &
            (S <= 45)
        )

        masks["white"] = (
            white_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # GREY
        # --------------------------------------------------------

        grey_mask = (
            (V > 70)
            &
            (V < 190)
            &
            (S <= 55)
        )

        masks["grey"] = (
            grey_mask.astype(np.uint8) * 255
        )

        # --------------------------------------------------------
        # BROWN
        # --------------------------------------------------------

        brown_mask = (
            (
                (H >= 5)
                &
                (H < 25)
            )
            &
            (S >= 35)
            &
            (S < 180)
            &
            (V >= 40)
            &
            (V < 190)
        )

        masks["brown"] = (
            brown_mask.astype(np.uint8) * 255
        )

        # ========================================================
        # CLEAN MASKS
        # ========================================================

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        for color in masks:

            masks[color] = cv2.morphologyEx(
                masks[color],
                cv2.MORPH_OPEN,
                kernel
            )

            masks[color] = cv2.morphologyEx(
                masks[color],
                cv2.MORPH_CLOSE,
                kernel
            )

        # ========================================================
        # PERCENTAGES
        # ========================================================

        percentages = {}

        largest_regions = {}

        for color, mask in masks.items():

            percentages[color] = (
                self._percentage(mask)
            )

            largest_regions[color] = (
                self._largest_component_percentage(
                    mask
                )
            )

        # ========================================================
        # IMPORTANT SCORES
        # ========================================================

        black_percentage = percentages.get(
            "black",
            0.0
        )

        black_region = largest_regions.get(
            "black",
            0.0
        )

        blue_percentage = percentages.get(
            "blue",
            0.0
        )

        blue_region = largest_regions.get(
            "blue",
            0.0
        )

        grey_percentage = percentages.get(
            "grey",
            0.0
        )

        grey_region = largest_regions.get(
            "grey",
            0.0
        )

        white_percentage = percentages.get(
            "white",
            0.0
        )

        white_region = largest_regions.get(
            "white",
            0.0
        )

        # ========================================================
        # SCORES
        # ========================================================

        black_score = (
            black_percentage * 0.55
            +
            black_region * 0.45
        )

        blue_score = (
            blue_percentage * 0.40
            +
            blue_region * 0.60
        )

        grey_score = (
            grey_percentage * 0.55
            +
            grey_region * 0.45
        )

        white_score = (
            white_percentage * 0.55
            +
            white_region * 0.45
        )

        # --------------------------------------------------------
        # CHROMATIC SCORES
        # --------------------------------------------------------

        color_scores = {}

        for color in [
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "purple",
            "brown"
        ]:

            color_scores[color] = (
                percentages[color] * 0.40
                +
                largest_regions[color] * 0.60
            )

        color_scores["black"] = black_score
        color_scores["grey"] = grey_score
        color_scores["white"] = white_score

        # ========================================================
        # PIXEL DECISION
        # ========================================================

        # --------------------------------------------------------
        # BLACK
        # --------------------------------------------------------

        if (
            black_percentage >= 25
            and
            black_percentage >= (
                blue_percentage * 1.20
            )
        ):

            pixel_color = "black"

            pixel_confidence = min(
                99.0,
                max(
                    60.0,
                    black_score
                )
            )

            largest_region = black_region

        # --------------------------------------------------------
        # BLUE
        # --------------------------------------------------------

        elif (
            blue_percentage >= 20
            and
            blue_region >= 5
            and
            blue_score > black_score
            and
            blue_score > grey_score
        ):

            pixel_color = "blue"

            pixel_confidence = min(
                99.0,
                max(
                    60.0,
                    blue_score
                )
            )

            largest_region = blue_region

        # --------------------------------------------------------
        # WHITE
        # --------------------------------------------------------

        elif (
            white_percentage >= 30
            and
            white_score >= blue_score
        ):

            pixel_color = "white"

            pixel_confidence = min(
                99.0,
                max(
                    60.0,
                    white_score
                )
            )

            largest_region = white_region

        # --------------------------------------------------------
        # GREY
        # --------------------------------------------------------

        elif (
            grey_percentage >= 25
            and
            grey_score >= blue_score
        ):

            pixel_color = "grey"

            pixel_confidence = min(
                99.0,
                max(
                    55.0,
                    grey_score
                )
            )

            largest_region = grey_region

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        else:

            pixel_color = max(
                color_scores,
                key=color_scores.get
            )

            pixel_confidence = min(
                99.0,
                max(
                    0.0,
                    color_scores[pixel_color]
                )
            )

            largest_region = largest_regions.get(
                pixel_color,
                0.0
            )

        # ========================================================
        # RETURN PIXEL ANALYSIS
        # ========================================================

        return {
            "pixel_color": pixel_color,
            "pixel_confidence": float(
                pixel_confidence
            ),
            "blue_pixel_percentage": float(
                blue_percentage
            ),
            "largest_color_region": float(
                largest_region
            ),
            "color_scores": color_scores,
            "percentages": percentages,
            "largest_regions": largest_regions
        }

    # ============================================================
    # MAIN PREDICT
    # ============================================================

    def predict(self, image):

        if image is None:

            return {
                "color": "unknown",
                "confidence": 0.0,
                "pixel_color": "unknown",
                "pixel_confidence": 0.0,
                "cnn_color": "unknown",
                "cnn_confidence": 0.0,
                "blue_pixel_percentage": 0.0,
                "largest_color_region": 0.0,
                "reason": "No image"
            }

        # ========================================================
        # CNN
        # ========================================================

        cnn_color, cnn_confidence = (
            self._cnn_predict(image)
        )

        # ========================================================
        # PIXEL ANALYSIS
        # ========================================================

        pixel_result = self._pixel_analysis(
            image
        )

        pixel_color = pixel_result[
            "pixel_color"
        ]

        pixel_confidence = pixel_result[
            "pixel_confidence"
        ]

        blue_percentage = pixel_result[
            "blue_pixel_percentage"
        ]

        largest_region = pixel_result[
            "largest_color_region"
        ]

        percentages = pixel_result[
            "percentages"
        ]

        largest_regions = pixel_result[
            "largest_regions"
        ]

        # ========================================================
        # IMPORTANT VALUES
        # ========================================================

        black_pct = percentages.get(
            "black",
            0.0
        )

        black_region = largest_regions.get(
            "black",
            0.0
        )

        blue_pct = percentages.get(
            "blue",
            0.0
        )

        blue_region = largest_regions.get(
            "blue",
            0.0
        )

        # ========================================================
        # FINAL DECISION
        # ========================================================

        final_color = cnn_color

        final_confidence = (
            cnn_confidence
        )

        reason = "CNN prediction"

        # ========================================================
        # 1. BLACK OVERRIDE
        # ========================================================

        if (
            black_pct >= 25
            and
            black_region >= 5
            and
            black_pct >= (
                blue_pct * 1.20
            )
        ):

            final_color = "black"

            final_confidence = min(
                99.0,
                max(
                    60.0,
                    (
                        black_pct * 0.55
                        +
                        black_region * 0.45
                    )
                )
            )

            reason = (
                "Strong dark/black body evidence"
            )

        # ========================================================
        # 2. BLUE OVERRIDE
        # ========================================================

        elif (
            blue_pct >= 20
            and
            blue_region >= 5
            and
            blue_pct > black_pct
            and
            blue_region > black_region
        ):

            final_color = "blue"

            final_confidence = min(
                99.0,
                max(
                    60.0,
                    (
                        blue_pct * 0.40
                        +
                        blue_region * 0.60
                    )
                )
            )

            reason = (
                "Strong continuous blue/cyan body region"
            )

        # ========================================================
        # 3. STRONG PIXEL COLOR
        # ========================================================

        elif pixel_confidence >= 65:

            final_color = pixel_color

            final_confidence = min(
                99.0,
                max(
                    55.0,
                    pixel_confidence
                )
            )

            reason = (
                "Strong pixel color evidence"
            )

        # ========================================================
        # 4. CNN FALLBACK
        # ========================================================

        else:

            final_color = cnn_color

            final_confidence = (
                cnn_confidence
            )

            reason = (
                "CNN prediction used as fallback"
            )

        # ========================================================
        # FINAL BLACK SAFETY CHECK
        # ========================================================

        if (
            final_color == "blue"
            and
            black_pct >= 35
            and
            black_region >= 8
            and
            black_pct > blue_pct
        ):

            final_color = "black"

            final_confidence = min(
                99.0,
                max(
                    65.0,
                    black_pct
                )
            )

            reason = (
                "Dark body evidence overrode blue reflection"
            )

        # ========================================================
        # RETURN
        # ========================================================

        return {
            "color": final_color,

            "confidence": float(
                final_confidence
            ),

            "pixel_color": pixel_color,

            "pixel_confidence": float(
                pixel_confidence
            ),

            "blue_pixel_percentage": float(
                blue_percentage
            ),

            "largest_color_region": float(
                largest_region
            ),

            "cnn_color": cnn_color,

            "cnn_confidence": float(
                cnn_confidence
            ),

            "reason": reason,

            "color_scores": pixel_result[
                "color_scores"
            ]
        }