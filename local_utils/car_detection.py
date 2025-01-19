import cv2
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5x", pretrained=True)


def detect_cars(frame):
    """
    Detect cars in the provided video frame using YOLOv5.
    """
    results = model(frame)

    detections = results.pandas().xyxy[0]
    cars = []

    for _, row in detections.iterrows():
        if row["name"] == "car":
            x1, y1, x2, y2 = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )
            car_crop = frame[y1:y2, x1:x2]
            cars.append((car_crop, (x1, y1, x2, y2)))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return frame, cars
