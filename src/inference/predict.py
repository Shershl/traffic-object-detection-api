from pathlib import Path

import cv2
from ultralytics import YOLO


MODEL_PATH = "runs/detect/train2/weights/best.pt"
CLASS_NAMES = ["person", "car", "bus", "truck", "bike"]

model = YOLO(MODEL_PATH)


def save_prediction_image(
    result,
    original_image_path: str,
    output_dir: str = "runs/predict",
) -> str:
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    plotted_image = result.plot()

    original_name = Path(original_image_path).stem
    output_path = output_dir_path / f"{original_name}_prediction.jpg"

    cv2.imwrite(str(output_path), plotted_image)

    return str(output_path)


def predict_image(
    image_path: str,
    conf_threshold: float = 0.5,
    save_annotated: bool = True,
    output_dir: str = "runs/predict",
) -> dict:
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=False,
        verbose=False
    )

    detections = []
    annotated_image_path = None

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "confidence": round(conf, 3),
                "bbox": [round(x) for x in xyxy]
            })

        if save_annotated:
            annotated_image_path = save_prediction_image(
                result=result,
                original_image_path=image_path,
                output_dir=output_dir
            )

    return {
        "detections": detections,
        "annotated_image_path": annotated_image_path
    }


if __name__ == "__main__":
    image_path = "tests/0000001_02999_d_000005.jpg"

    result = predict_image(
        image_path=image_path,
        conf_threshold=0.5,
        save_annotated=True,
        output_dir="runs/predict"
    )

    print(result)