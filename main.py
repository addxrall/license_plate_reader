import cv2

from local_utils.car_detection import detect_cars
from local_utils.ocr import extract_plate_text
from local_utils.plate_detection import detect_license_plate


def process_frame(frame):
    """
    Process a single frame of the video, detect cars, detect license plates,
    and annotate the frame with bounding boxes and text.
    """

    img, cars = detect_cars(frame)

    for i, (car, bbox) in enumerate(cars):
        print(f"\nProcessing Car {i + 1}...")
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        plates = detect_license_plate(car)

        if plates:
            for plate in plates:
                plate_bbox = plate["bbox"]
                plate_crop = car[
                    int(plate_bbox[1]) : int(plate_bbox[3]),
                    int(plate_bbox[0]) : int(plate_bbox[2]),
                ]

                plate_text = extract_plate_text(plate_crop)
                plate_text = str(plate_text)
                print(f"License Plate Detected: {plate_text}")

                if plate_text:
                    padding = 10
                    text_size = cv2.getTextSize(
                        plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2
                    )[0]
                    text_width, text_height = text_size

                    cv2.rectangle(
                        img,
                        (x1, y1 - text_height - padding),
                        (x1 + text_width + 2 * padding, y1),
                        (255, 255, 255),
                        -1,
                    )

                    cv2.putText(
                        img,
                        plate_text,
                        (
                            x1 + padding,
                            y1 - padding + 5,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 0),
                        1,
                    )
                else:
                    print("OCR did not detect any text.")
        else:
            print("No license plate detected.")

    return img


def process_video(input_video_path, output_video_path):
    """
    Process a video, detect cars and license plates, and save the output to a new video.
    """
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        out.write(processed_frame)

        cv2.imshow("Processed Video", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video_path = "test_vid/testvid.mp4"
    output_video_path = "output/output_video.mp4"

    process_video(input_video_path, output_video_path)
