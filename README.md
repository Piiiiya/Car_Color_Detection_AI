# 🚗 Car Color Detection AI

### Machine Learning and Deep Learning Based Vehicle Detection and Color Classification

## 📌 Project Overview

Car Color Detection AI is a computer vision project developed using Machine Learning and Deep Learning techniques to detect vehicles, classify their colors, and count cars and people in traffic images.

The project combines YOLO-based object detection, a CNN-based car color classification model, OpenCV image processing, and a Streamlit graphical user interface.

It is designed to demonstrate how AI and computer vision can be applied to traffic monitoring and vehicle analysis.

## 🎯 Problem Statement

Identifying vehicles and their colors in traffic images manually can be time-consuming. This project aims to automate vehicle detection and color classification while providing a visual representation of the results.

The system processes an input image, detects vehicles and people, predicts vehicle colors, and displays the results through an interactive interface.

## ✨ Key Features

- **Vehicle Detection:** Detects cars in traffic images using YOLO.
- **Car Color Classification:** Predicts vehicle colors using a trained deep learning model.
- **People Counting:** Detects and counts people present in the input image.
- **Color-Based Bounding Boxes:** Displays red rectangles for blue cars and blue rectangles for cars of other colors.
- **Image Preview:** Allows users to preview the input image through the GUI.
- **Visual Results:** Displays detected objects, predicted colors, and counts on the processed image.
- **Streamlit Interface:** Provides a user-friendly interface for interacting with the system.
- **Image Processing:** Uses OpenCV for image manipulation and visualization.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep learning and car color classification |
| YOLO | Object detection |
| OpenCV | Image processing and visualization |
| NumPy | Numerical operations |
| Streamlit | Graphical user interface |
| Jupyter Notebook | Data exploration and experiments |
| Git and GitHub | Version control and project hosting |

## 🧠 Project Workflow

1. Upload an input traffic image.
2. Process the image using the object detection model.
3. Identify cars and people in the image.
4. Extract detected vehicle regions.
5. Predict the color of each detected car using the trained classification model.
6. Draw bounding boxes and labels around detected objects.
7. Count detected cars and people.
8. Display the processed image and results in the Streamlit application.

## 📂 Project Structure

```text
Car_Color_Detection_AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── car_color_best.keras
│   └── car_color_model.keras
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_vehicle_detection.ipynb
│
├── outputs/
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
├── .gitignore
└── requirements.txt
```

**Note:** Large trained model files and datasets are not included in the public GitHub repository. They must be placed in their expected project directories before running the application.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Piiiiya/Car_Color_Detection_AI.git
```

### 2. Navigate to the project directory

```bash
cd Car_Color_Detection_AI
```

### 3. Create a virtual environment

For Windows:

```bash
python -m venv .venv
```

### 4. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add the required trained models

Place the trained Keras model files and required YOLO weights in the locations expected by the application configuration.

The model files are stored separately from this repository because of their size.

## ▶️ How to Run the Application

From the project root directory, run:

```bash
streamlit run app/app.py
```

The application will open in your web browser.

Upload an image to view the vehicle detection, color classification, and counting results.

## 📊 Results and Demonstration

The application provides visual results including:

- Detected cars with predicted color labels.
- Bounding boxes following the project's color-display rules.
- Total detected car count.
- Total detected people count.
- Processed image preview.

Screenshots and sample outputs can be added to this section after testing the final application.

## 🔬 Experiments and Development

The project includes Jupyter notebooks for dataset exploration and vehicle detection experiments.

Additional Python modules support color classification, detection, image analysis, shade detection, and the overall processing pipeline.

## 🚀 Future Improvements

- Improve detection of partially visible and distant vehicles.
- Enhance classification of similar colors and shades.
- Reduce duplicate detections and false positives.
- Improve performance under different lighting conditions.
- Optimize inference speed.
- Add support for real-time video processing.
- Expand evaluation using additional test images.

## 👩‍💻 Project Information

**Project:** Car Color Detection AI  
**Domain:** Artificial Intelligence and Machine Learning  
**Application:** Computer Vision and Traffic Image Analysis  
**Interface:** Streamlit  
**Purpose:** Internship project and practical application of machine learning and deep learning techniques.

---

*Developed as part of the Elevance Skills internship program.*![Uploading Screenshot (664).png…]()
