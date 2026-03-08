# Traffic Object Detection API

End-to-end computer vision service for detecting traffic objects in aerial images using **YOLOv8** and **FastAPI**.

The project includes the full pipeline:

* dataset preparation from **VisDrone**
* model fine-tuning
* inference pipeline
* REST API for image upload and detection
* serving annotated prediction images

---

# Features

* Object detection using YOLOv8
* Custom dataset preparation from VisDrone
* Fine-tuned model for traffic objects
* REST API built with FastAPI
* Image upload endpoint
* JSON detection results
* Annotated prediction image generation
* Prediction image serving via HTTP

---

# Detected Classes

The model detects the following traffic-related classes:

* person
* car
* bus
* truck
* bike

---

# Project Structure

```
traffic-object-detection-api
│
├── app
│   └── main.py                 # FastAPI application
│
├── src
│   ├── data
│   │   └── convert_visdrone_to_yolo.py
│   │
│   ├── inference
│   │   └── predict.py
│   │
│   └── training
│       └── train_yolo.py
│
├── models
│   └── best.pt                 # trained YOLO model
│
├── tests
│   ├── sample_1.jpg
│   └── sample_2.jpg
│
├── runs                       # prediction outputs (ignored by git)
├── datasets                   # dataset files (ignored by git)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository

```
git clone https://github.com/Shershl/traffic-object-detection-api.git
cd traffic-object-detection-api
```

Create virtual environment

```
python -m venv .venv
```

Activate environment

Windows

```
.venv\Scripts\activate
```

Linux / Mac

```
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Running the API

Start the FastAPI server:

```
uvicorn app.main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

### Health check

```
GET /health
```

Response

```json
{
  "status": "ok"
}
```

---

### Detect objects

```
POST /detect
```

Upload an image and receive detection results.

Example response:

```json
{
  "filename": "image.jpg",
  "saved_file_path": "runs/uploads/abc123.jpg",
  "prediction_image_url": "/prediction/abc123_prediction.jpg",
  "detections": [
    {
      "class_id": 1,
      "class_name": "car",
      "confidence": 0.91,
      "bbox": [1423, 802, 1622, 972]
    }
  ],
  "detections_count": 5
}
```

---

### Get prediction image

```
GET /prediction/{filename}
```

Returns annotated detection image.

Example:

```
http://127.0.0.1:8000/prediction/example_prediction.jpg
```

---

# Training the Model

Training is performed using YOLOv8.

Run:

```
python src/training/train_yolo.py
```

Training configuration:

* epochs: 80
* image size: 640
* batch size: 16
* dataset: VisDrone converted to YOLO format

---

# Dataset Preparation

The dataset conversion script transforms VisDrone annotations into YOLO format.

Run:

```
python src/data/convert_visdrone_to_yolo.py
```

Steps performed:

* parse VisDrone annotations
* filter ignored objects
* map classes to custom categories
* convert bounding boxes to YOLO format
* generate labels and dataset splits

---

# Example Output

The API returns:

* detected objects
* class names
* confidence scores
* bounding boxes
* annotated image

Example prediction:

```
runs/predict/example_prediction.jpg
```

---

# Technologies Used

* Python
* FastAPI
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Pillow

---

# Future Improvements

Possible improvements:

* batch inference endpoint
* video detection
* Docker support
* async processing
* model benchmarking
* frontend interface
* traffic analytics (vehicle counting, density estimation)

---

# License

This project is intended for educational and portfolio purposes.
