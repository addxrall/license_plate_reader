from ultralytics import YOLO


def detect_license_plate(car_image):
    license_plate_detector = YOLO("models/license_plate_detector.pt")

    results = license_plate_detector(car_image)[0]

    plates = []

    for detection in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        if score > 0.5:
            plates.append({"bbox": [x1, y1, x2, y2], "score": score})

    return plates
