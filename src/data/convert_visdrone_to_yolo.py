from pathlib import Path
import shutil

# Final classes for the project
TARGET_CLASSES = {
    0: "person",
    1: "car",
    2: "bus",
    3: "truck",
    4: "bike",
}


VISDRONE_TO_TARGET = {
    1: 0,   # pedestrian -> person
    2: 0,   # people -> person
    3: 4,   # bicycle -> bike
    4: 1,   # car -> car
    5: 1,   # van -> car
    6: 3,   # truck -> truck
    9: 2,   # bus -> bus
    10: 4,  # motor -> bike
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def convert_bbox_to_yolo(
    img_width: int,
    img_height: int,
    x: float,
    y: float,
    w: float,
    h: float,
) -> tuple[float, float, float, float]:
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def process_split(
    source_images_dir: Path,
    source_annotations_dir: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
) -> None:
    ensure_dir(target_images_dir)
    ensure_dir(target_labels_dir)

    annotation_files = sorted(source_annotations_dir.glob("*.txt"))
    processed = 0
    skipped = 0

    for ann_file in annotation_files:
        image_file = source_images_dir / f"{ann_file.stem}.jpg"
        if not image_file.exists():
            skipped += 1
            continue

        try:
            import cv2

            image = cv2.imread(str(image_file))
            if image is None:
                skipped += 1
                continue

            img_height, img_width = image.shape[:2]
        except Exception:
            skipped += 1
            continue

        yolo_lines: list[str] = []

        with ann_file.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) < 8:
                    continue

                try:
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    score = int(parts[4])
                    category_id = int(parts[5])
                    truncation = int(parts[6])
                    occlusion = int(parts[7])
                except ValueError:
                    continue

                # ignored regions
                if category_id not in VISDRONE_TO_TARGET:
                    continue


                if score == 0:
                    continue


                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                class_id = VISDRONE_TO_TARGET[category_id]
                x_c, y_c, w, h = convert_bbox_to_yolo(
                    img_width=img_width,
                    img_height=img_height,
                    x=bbox_left,
                    y=bbox_top,
                    w=bbox_width,
                    h=bbox_height,
                )

                # additional boundary check
                if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue

                yolo_lines.append(
                    f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                )


        shutil.copy2(image_file, target_images_dir / image_file.name)

        label_output_path = target_labels_dir / f"{ann_file.stem}.txt"
        with label_output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        processed += 1

    print(f"Done: {source_images_dir.name}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    visdrone_root = project_root / "datasets" / "visdrone"
    output_root = project_root / "datasets" / "visdrone_yolo"

    train_images = visdrone_root / "VisDrone2019-DET-train" / "images"
    train_annotations = visdrone_root / "VisDrone2019-DET-train" / "annotations"

    val_images = visdrone_root / "VisDrone2019-DET-val" / "images"
    val_annotations = visdrone_root / "VisDrone2019-DET-val" / "annotations"

    process_split(
        source_images_dir=train_images,
        source_annotations_dir=train_annotations,
        target_images_dir=output_root / "images" / "train",
        target_labels_dir=output_root / "labels" / "train",
    )

    process_split(
        source_images_dir=val_images,
        source_annotations_dir=val_annotations,
        target_images_dir=output_root / "images" / "val",
        target_labels_dir=output_root / "labels" / "val",
    )

    print("\nConversion finished.")
    print(f"YOLO dataset saved to: {output_root}")


if __name__ == "__main__":
    main()