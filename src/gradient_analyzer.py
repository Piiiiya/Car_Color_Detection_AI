
import cv2
import numpy as np


class CarGradientAnalyzer:

    def __init__(self, number_of_colors=4):
        self.number_of_colors = number_of_colors

    @staticmethod
    def _rgb_to_hex(rgb):
        r, g, b = [int(np.clip(value, 0, 255)) for value in rgb]
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _lab_distance(color1, color2):
        return float(np.linalg.norm(color1 - color2))

    def analyze(self, image):

        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image provided.")

        if image.size == 0:
            raise ValueError("The image is empty.")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image.")

        # Resize for faster processing
        resized = cv2.resize(
            image,
            (150, 150),
            interpolation=cv2.INTER_AREA
        )

        # Convert BGR to LAB
        lab_image = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2LAB
        )

        pixels = lab_image.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        k = min(self.number_of_colors, len(pixels))

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.2
        )

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )

        # Calculate percentage of each cluster
        counts = np.bincount(
            labels.flatten(),
            minlength=k
        )

        percentages = counts / counts.sum() * 100

        # Convert cluster centers back to BGR
        centers_uint8 = np.uint8(
            np.clip(centers, 0, 255)
        ).reshape(1, k, 3)

        centers_bgr = cv2.cvtColor(
            centers_uint8,
            cv2.COLOR_LAB2BGR
        )[0]

        shades = []

        for index in range(k):

            b, g, r = [
                int(value)
                for value in centers_bgr[index]
            ]

            rgb = [r, g, b]

            shades.append({
                "rgb": rgb,
                "hex": self._rgb_to_hex(rgb),
                "percentage": round(
                    float(percentages[index]),
                    2
                ),
                "lab": centers[index].tolist()
            })

        # Sort shades by percentage
        shades.sort(
            key=lambda item: item["percentage"],
            reverse=True
        )

        dominant_shade = shades[0]

        # Measure color variation between cluster centers
        variation = 0.0

        for i in range(len(centers)):

            for j in range(i + 1, len(centers)):

                distance = self._lab_distance(
                    centers[i],
                    centers[j]
                )

                variation = max(
                    variation,
                    distance
                )

        # Preliminary multi-shade / gradient indicator
        significant_shades = [
            shade
            for shade in shades
            if shade["percentage"] >= 10
        ]

        possible_gradient = (
            len(significant_shades) >= 2
            and variation >= 20
        )

        if possible_gradient:
            status = "Possible multiple shades or gradient"
        elif variation >= 12:
            status = "Moderate color variation"
        else:
            status = "Mostly uniform color"

        return {
            "dominant_rgb": dominant_shade["rgb"],
            "dominant_hex": dominant_shade["hex"],
            "dominant_percentage": dominant_shade["percentage"],
            "shades": shades,
            "color_variation": round(variation, 2),
            "possible_gradient": possible_gradient,
            "status": status
        }
