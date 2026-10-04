# 🚘 Car Colour Detection AI

An AI-powered computer vision application that detects vehicles, classifies car colours, counts people, and identifies traffic lights in traffic images.

The project combines **YOLO object detection**, a **custom Convolutional Neural Network (CNN)**, and **Streamlit** to provide an interactive image-based detection system.

## 🌐 Live Demo

**[Click here to try Car Colour Detection AI](https://carcolordetectionai-bxzkyma9kdj4cghut9txu2.streamlit.app/)**

The application is deployed on Streamlit Community Cloud.

## 📂 GitHub Repository

[View Source Code on GitHub](https://github.com/Piiiiya/Car_Color_Detection_AI)

## ✨ Features

- 🚗 Detects cars in uploaded traffic images.
- 🎨 Classifies vehicles into 15 colour categories.
- 🔵 Identifies blue cars separately from other colours.
- 👤 Detects and counts people.
- 🚦 Detects traffic lights.
- 🟥 Draws red bounding boxes around blue cars.
- 🟦 Draws blue bounding boxes around other-colour cars.
- 🟩 Uses green bounding boxes for people.
- 🟨 Uses yellow bounding boxes for traffic lights.
- 📊 Displays detection counts and car classification details.
- 🖼️ Provides an annotated image preview through a Streamlit GUI.
- ⚡ Uses CPU-based inference for deployment compatibility.

## 🧠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO | Object detection |
| TensorFlow / Keras | Custom CNN colour classification |
| OpenCV | Image processing and bounding boxes |
| NumPy | Numerical operations |
| Pandas | Detection result tables |
| Streamlit | Interactive web interface |
| PyTorch | YOLO inference backend |

## 🎨 Colour Classes

The custom CNN classifies cars into 15 colour categories:

`Beige`, `Black`, `Blue`, `Brown`, `Gold`, `Green`, `Grey`, `Orange`, `Pink`, `Purple`, `Red`, `Silver`, `Tan`, `White`, `Yellow`

## 🏗️ Project Structure

```text
Car_Color_Detection_AI/
│
├── app/
│   └── app.py
│
├── src/
│   ├── detector.py
│   ├── color_classifier.py
│   ├── pipeline.py
│   ├── shade_detector.py
│   └── gradient_analyzer.py
│
├── models/
│   └── car_color_best.keras
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── data/
│
├── outputs/
│
├── yolo26s.pt
├── requirements.txt
├── README.md
└── .gitignore
```

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

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## 🔍 How It Works

1. The user uploads a traffic image.
2. YOLO detects cars, people, and traffic lights.
3. Detected car regions are passed to the custom CNN.
4. The CNN predicts the colour of each detected vehicle.
5. The pipeline assigns bounding box colours based on vehicle classification.
6. The application displays the annotated image, detection counts, and available car details.

## 📊 Model Performance

The custom CNN achieved approximately **78.98% test accuracy** and a **0.74 macro F1-score** on the evaluated 15-class dataset.

Performance may vary depending on lighting conditions, vehicle size, occlusion, reflections, and image quality.

## 🚀 Deployment

The application is hosted on **Streamlit Community Cloud**.

🔗 **[Launch Live Application](https://carcolordetectionai-bxzkyma9kdj4cghut9txu2.streamlit.app/)**

## 👩‍💻 Author

**Piya Shaikh**

- GitHub: [Piiiiya](https://github.com/Piiiiya)
- Project Repository: [Car Colour Detection AI](https://github.com/Piiiiya/Car_Color_Detection_AI)

---

*Developed as a Machine Learning and Deep Learning project to demonstrate object detection, image classification, computer vision, and web application deployment.*
