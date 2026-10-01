# Car Color Detection AI

An AI-powered computer vision application that detects cars, identifies their colours, and counts vehicles and pedestrians in traffic images.

## Project Overview

Car Color Detection AI combines object detection and deep learning to analyze traffic scenes. The application identifies vehicles, classifies their colours, detects pedestrians, and displays the results through an interactive Streamlit interface.

## Key Features

- Car detection in traffic images.
- Classification of 15 car colours.
- Vehicle counting.
- Pedestrian detection and counting.
- Traffic light detection.
- Colour-coded bounding boxes:
  - Red rectangles for blue cars.
  - Blue rectangles for cars of other colours.
- Image upload and preview.
- Annotated output image.
- Interactive Streamlit GUI.
- CPU-based inference support.

## Technologies Used

- Python
- TensorFlow / Keras
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Streamlit
- Pillow

## Model Architecture

The project uses two main components:

1. **YOLO Object Detection:** Detects cars, pedestrians, and traffic lights in images.
2. **CNN-Based Colour Classification:** Predicts the colour of detected cars using a trained deep learning model.

The detection and classification results are combined through a processing pipeline to generate the final annotated image.

## Supported Car Colours

Beige, Black, Blue, Brown, Gold, Green, Grey, Orange, Pink, Purple, Red, Silver, Tan, White, and Yellow.

## Project Structure

```text
Car_Color_Detection_AI/
│
├── app/
│   └── app.py
│
├── src/
│   ├── color_classifier.py
│   ├── config.py
│   ├── detector.py
│   ├── gradient_analyzer.py
│   ├── pipeline.py
│   ├── shade_detector.py
│   └── test_detector.py
│
├── models/
│   └── car_color_best.keras
│
├── outputs/
│   └── debug_detection.jpg
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── packages.txt
└── README.md
```

*Note: The structure above is a guide. Update it if your final repository contains additional files or folders.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Piiiiya/Car_Color_Detection_AI.git
cd Car_Color_Detection_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app/app.py
```

The application will open in your browser.

## How to Use

1. Launch the Streamlit application.
2. Upload a traffic image.
3. Preview the input image.
4. Click **Detect Cars & Colours**.
5. View the detected vehicles, colour classifications, and pedestrian counts.
6. Review the annotated output image.

## Limitations

- Detection accuracy depends on image quality, lighting, occlusion, and object size.
- Small or partially hidden objects may not be detected.
- Similar car colours can sometimes be confused.
- Processing time depends on image resolution and available hardware.
- Results may vary across different traffic scenes.

## Future Improvements

- Real-time traffic video analysis.
- Improved detection in low-light conditions.
- Enhanced colour classification accuracy.
- Vehicle tracking across video frames.
- Traffic analytics dashboard.
- Optimized inference for faster processing.

## Project Status

Developed as part of an AI/ML internship project.

## Author

**Piya Shaikh**

GitHub: [Piiiiya](https://github.com/Piiiiya)
