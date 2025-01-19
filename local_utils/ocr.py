import easyocr


def extract_plate_text(plate_image):
    reader = easyocr.Reader(["en"])

    result = reader.readtext(plate_image)

    if result:
        plate_text = result[0][1]
        return plate_text.strip()

    return None
