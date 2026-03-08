from ultralytics import YOLO

def main():

    model = YOLO("runs/detect/train2/weights/best.pt")

    model.train(
        data="datasets/visdrone_yolo/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    main()