import os
import cv2
from PIL import Image
import pytesseract
import numpy as np


# Function to read bounding box with 8 coordinates and convert to rectangular box (x, y, w, h)
def parse_bounding_box(line):
    coords = list(map(int, line.strip().split(",")))  # Split by commas
    x_values = [coords[i] for i in range(0, len(coords), 2)]  # Extract all x values
    y_values = [coords[i + 1] for i in range(0, len(coords), 2)]  # Extract all y values
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    return x_min, y_min, x_max - x_min, y_max - y_min  # Convert to (x, y, w, h)


# Function to crop the image into 49 pieces (7x7 grid)
def crop_image_into_grid(image, grid_size=7):
    img_h, img_w, _ = image.shape  # Get image dimensions
    piece_h = img_h // grid_size  # Height of each piece
    piece_w = img_w // grid_size  # Width of each piece

    cropped_images = []

    # Loop through the grid and crop each piece
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the start and end coordinates of each piece
            y_start = i * piece_h
            y_end = (
                (i + 1) * piece_h if i < grid_size - 1 else img_h
            )  # Handle bottom edge
            x_start = j * piece_w
            x_end = (
                (j + 1) * piece_w if j < grid_size - 1 else img_w
            )  # Handle right edge

            # Crop the piece from the image
            cropped_piece = image[y_start:y_end, x_start:x_end]
            cropped_images.append(cropped_piece)

            # Optionally, save each cropped piece as a separate file (for verification)
            cv2.imwrite(f"crop_{i}_{j}.png", cropped_piece)

    return cropped_images, piece_h, piece_w


def create_position_encoding(bounding_box, image_shape, grid_h=7, grid_w=7):
    x, y, w, h = bounding_box
    img_h, img_w, _ = image_shape

    # Split the image into a grid of size grid_h * grid_w
    piece_h = img_h / grid_h  # Height of one piece
    piece_w = img_w / grid_w  # Width of one piece

    # Calculate the center of the bounding box
    center_x = x + w / 2
    center_y = y + h / 2

    # Determine the (h_i, w_i) position by finding which grid piece the center falls into
    h_i = int(center_y // piece_h)
    w_i = int(center_x // piece_w)

    # Return the position in the grid (h_i, w_i)
    return h_i, w_i


def preprocess_image(image):
    # Check if the input is a PIL image, if so, convert it to a NumPy array
    if isinstance(image, Image.Image):  # Check if it's a PIL image
        image = np.array(image)  # Convert to NumPy array

    # Ensure the image is in grayscale
    if (
        len(image.shape) == 3
    ):  # Check if the image has multiple channels (i.e., color image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image  # It's already grayscale

    # Resize the image to improve OCR accuracy (optional)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Apply Otsu's thresholding to binarize the image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the colors (optional, depends on image characteristics)
    img = cv2.bitwise_not(img)

    # Convert the processed image back to PIL format for pytesseract
    pil_img = Image.fromarray(img)

    return pil_img


# Function to perform OCR on cropped image
def perform_ocr(image):
    pil_image = preprocess_image(Image.fromarray(image))
    # custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist="0123456789.,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"'
    custom_config = r"--oem 1 --psm 6"
    ocr_text = pytesseract.image_to_string(pil_image, lang="swe", config=custom_config)
    return ocr_text


def combine_ocr_and_position(ocr_results, position_encodings):
    combined = []
    for i, ocr in enumerate(ocr_results):
        combined.append(
            {
                "segment": i,
                "ocr_result": ocr.strip(),
                "position_encoding": position_encodings[i],
            }
        )
    return combined


# Directory containing the image and bounding box files
directory_path = "craft_sample"

# Store images and bounding boxes
images = []
bounding_boxes = {}

# Iterate over files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png"):
        # Read and store the image
        image_path = os.path.join(directory_path, filename)
        image = cv2.imread(image_path)
        images.append((filename, image))
    elif filename.endswith(".txt") and filename.startswith("res_"):
        # Read and store the bounding boxes
        bounding_box_path = os.path.join(directory_path, filename)
        with open(bounding_box_path, "r") as f:
            # Parse bounding boxes and store
            boxes = [parse_bounding_box(line) for line in f.readlines()]
            bounding_boxes[filename] = boxes

# Iterate over images and their corresponding bounding boxes
for image_filename, image in images:
    # Extract the image number from the image filename (e.g., '74.png' -> '74')
    image_number = image_filename.replace(".png", "")

    # Construct the corresponding bounding box file name (e.g., 'res_74.txt')
    corresponding_txt_filename = f"res_{image_number}.txt"

    if corresponding_txt_filename in bounding_boxes:
        boxes = bounding_boxes[corresponding_txt_filename]

        ocr_results = []
        position_encodings = []

        for i, box in enumerate(boxes):
            # Crop the image using the bounding box
            x, y, w, h = box
            cropped_img = image[y : y + h, x : x + w]

            # Perform OCR on the cropped image
            ocr_text = perform_ocr(cropped_img)
            ocr_results.append(ocr_text)

            # # Save the cropped image segments for verification (optional)
            # cv2.imwrite(f"segment_{image_filename}_{i}.png", cropped_img)

            # Add position encoding for the current bounding box
            position_encodings.append(create_position_encoding(box, image.shape))

        # # Output OCR results
        # print(f"OCR Results for {image_filename}:")
        # for idx, text in enumerate(ocr_results):
        #     print(f"Segment {idx}: {text}")

        # # Output position encodings
        # print(f"Position Encodings for {image_filename}:")
        # for idx, encoding in enumerate(position_encodings):
        #     print(f"Encoding {idx}: {encoding}")
        combined_context = combine_ocr_and_position(ocr_results, position_encodings)

        for context in combined_context:
            print(
                f"Segment {context['segment']} | OCR: {context['ocr_result']} | Position Encoding: {context['position_encoding']}"
            )
    else:
        print(f"No bounding box file found for {image_filename}")
