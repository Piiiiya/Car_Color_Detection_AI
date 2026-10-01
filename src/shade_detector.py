
import cv2
import numpy as np


class CarShadeDetector:

    def __init__(self):

        # Named shade palette
        self.shade_palette = {
            "Black": "#000000",
            "Charcoal": "#36454F",
            "Dark Grey": "#555555",
            "Grey": "#808080",
            "Light Grey": "#D3D3D3",
            "Silver": "#C0C0C0",
            "White": "#FFFFFF",
            "Ivory": "#FFFFF0",
            "Beige": "#F5F5DC",
            "Cream": "#FFFDD0",

            "Dark Red": "#8B0000",
            "Maroon": "#800000",
            "Crimson": "#DC143C",
            "Firebrick": "#B22222",
            "Red": "#FF0000",
            "Tomato": "#FF6347",
            "Light Coral": "#F08080",
            "Salmon": "#FA8072",
            "Pink": "#FFC0CB",
            "Hot Pink": "#FF69B4",

            "Dark Orange": "#FF8C00",
            "Orange": "#FFA500",
            "Coral": "#FF7F50",
            "Gold": "#FFD700",
            "Yellow": "#FFFF00",
            "Khaki": "#F0E68C",

            "Dark Green": "#006400",
            "Forest Green": "#228B22",
            "Green": "#008000",
            "Lime Green": "#32CD32",
            "Light Green": "#90EE90",
            "Olive": "#808000",

            "Dark Blue": "#00008B",
            "Navy": "#000080",
            "Blue": "#0000FF",
            "Royal Blue": "#4169E1",
            "Dodger Blue": "#1E90FF",
            "Sky Blue": "#87CEEB",
            "Light Blue": "#ADD8E6",
            "Steel Blue": "#4682B4",
            "Teal": "#008080",
            "Cyan": "#00FFFF",

            "Dark Purple": "#301934",
            "Purple": "#800080",
            "Indigo": "#4B0082",
            "Violet": "#EE82EE",
            "Lavender": "#E6E6FA",

            "Brown": "#A52A2A",
            "Dark Brown": "#654321",
            "Chocolate": "#D2691E",
            "Tan": "#D2B48C",
            "Saddle Brown": "#8B4513",

            "Gold Metallic": "#D4AF37",
            "Bronze": "#CD7F32",
            "Copper": "#B87333"
        }

        # Convert HEX palette into RGB arrays
        self.palette_rgb = {}

        for name, hex_code in self.shade_palette.items():

            hex_code = hex_code.lstrip("#")

            rgb = tuple(
                int(hex_code[i:i + 2], 16)
                for i in (0, 2, 4)
            )

            self.palette_rgb[name] = np.array(
                rgb,
                dtype=np.float32
            )

    # --------------------------------------------------
    # EXTRACT DOMINANT COLOR USING K-MEANS
    # --------------------------------------------------

    def extract_dominant_color(self, image, k=3):

        if image is None:
            raise ValueError("Image is empty or invalid.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image.")

        # Resize for faster processing
        resized = cv2.resize(
            image,
            (150, 150),
            interpolation=cv2.INTER_AREA
        )

        # Convert BGR to RGB
        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        # Convert to pixel array
        pixels = rgb.reshape((-1, 3)).astype(
            np.float32
        )

        # K-Means criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.2
        )

        # Cluster pixels
        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_PP_CENTERS
        )

        # Find most frequent cluster
        counts = np.bincount(
            labels.flatten(),
            minlength=k
        )

        dominant_index = int(
            np.argmax(counts)
        )

        dominant_rgb = centers[
            dominant_index
        ]

        return dominant_rgb, counts, centers

    # --------------------------------------------------
    # FIND CLOSEST NAMED SHADE
    # --------------------------------------------------

    def identify_shade(self, rgb_color):

        best_match = None
        best_distance = float("inf")

        for shade_name, palette_rgb in self.palette_rgb.items():

            distance = np.linalg.norm(
                rgb_color - palette_rgb
            )

            if distance < best_distance:

                best_distance = distance
                best_match = shade_name

        # Convert RGB values to HEX
        rgb_int = np.clip(
            np.round(rgb_color),
            0,
            255
        ).astype(int)

        hex_code = "#{:02X}{:02X}{:02X}".format(
            *rgb_int
        )

        # Similarity score, not a calibrated probability
        similarity = max(
            0.0,
            100.0 * (
                1.0 - best_distance / 441.67
            )
        )

        return {
            "shade_name": best_match,
            "rgb": tuple(int(x) for x in rgb_int),
            "hex": hex_code,
            "similarity": round(similarity, 2)
        }

    # --------------------------------------------------
    # MAIN PREDICTION
    # --------------------------------------------------

    def predict(self, image):

        dominant_rgb, counts, centers = (
            self.extract_dominant_color(image)
        )

        shade_result = self.identify_shade(
            dominant_rgb
        )

        return {
            "shade": shade_result["shade_name"],
            "hex": shade_result["hex"],
            "rgb": shade_result["rgb"],
            "similarity": shade_result["similarity"],
            "dominant_rgb": tuple(
                int(x) for x in np.round(dominant_rgb)
            )
        }
