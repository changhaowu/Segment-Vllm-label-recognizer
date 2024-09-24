import os
import cv2
from PIL import Image
import pytesseract
import numpy as np


class ImageProcessor:
    def __init__(
        self, directory_path, grid_size=7, lang="swe", output_file="output.txt"
    ):
        self.directory_path = directory_path
        self.grid_size = grid_size
        self.lang = lang
        self.output_file = output_file
        self.images = []
        self.bounding_boxes = {}

    def parse_bounding_box(self, line):
        coords = list(map(int, line.strip().split(",")))
        x_values = [coords[i] for i in range(0, len(coords), 2)]
        y_values = [coords[i + 1] for i in range(0, len(coords), 2)]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        return x_min, y_min, x_max - x_min, y_max - y_min

    def crop_image_into_grid(self, image):
        img_h, img_w, _ = image.shape
        piece_h = img_h // self.grid_size
        piece_w = img_w // self.grid_size

        cropped_images = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_start = i * piece_h
                y_end = (i + 1) * piece_h if i < self.grid_size - 1 else img_h
                x_start = j * piece_w
                x_end = (j + 1) * piece_w if j < self.grid_size - 1 else img_w

                cropped_piece = image[y_start:y_end, x_start:x_end]
                cropped_images.append(cropped_piece)

        return cropped_images, piece_h, piece_w

    def create_position_encoding(self, bounding_box, image_shape):
        x, y, w, h = bounding_box
        img_h, img_w, _ = image_shape

        piece_h = img_h / self.grid_size
        piece_w = img_w / self.grid_size

        center_x = x + w / 2
        center_y = y + h / 2

        h_i = int(center_y // piece_h)
        w_i = int(center_x // piece_w)

        return h_i, w_i

    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        if len(image.shape) == 3:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img = image

        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        pil_img = Image.fromarray(img)

        return pil_img

    def perform_ocr(self, image):
        pil_image = self.preprocess_image(Image.fromarray(image))
        custom_config = r"--oem 1 --psm 6"
        ocr_text = pytesseract.image_to_string(
            pil_image, lang=self.lang, config=custom_config
        )
        return ocr_text.strip()

    def combine_ocr_and_position(self, ocr_results, position_encodings):
        combined = []
        for i, ocr in enumerate(ocr_results):
            combined.append(
                {
                    "segment": i,
                    "ocr_result": ocr,
                    "position_encoding": position_encodings[i],
                }
            )
        return combined

    def save_combined_to_file(self, combined_context, image_filename):
        with open(self.output_file, "a") as f:
            f.write(f"Results for {image_filename}:\n")
            for context in combined_context:
                f.write(
                    f"Segment {context['segment']} | OCR: {context['ocr_result']} | Position Encoding: {context['position_encoding']}\n"
                )
            f.write("\n")

    def process_directory(self):
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".png"):
                image_path = os.path.join(self.directory_path, filename)
                image = cv2.imread(image_path)
                self.images.append((filename, image))
            elif filename.endswith(".txt") and filename.startswith("res_"):
                bounding_box_path = os.path.join(self.directory_path, filename)
                with open(bounding_box_path, "r") as f:
                    boxes = [self.parse_bounding_box(line) for line in f.readlines()]
                    self.bounding_boxes[filename] = boxes

        for image_filename, image in self.images:
            image_number = image_filename.replace(".png", "")
            corresponding_txt_filename = f"res_{image_number}.txt"

            if corresponding_txt_filename in self.bounding_boxes:
                boxes = self.bounding_boxes[corresponding_txt_filename]

                ocr_results = []
                position_encodings = []

                for i, box in enumerate(boxes):
                    x, y, w, h = box
                    cropped_img = image[y : y + h, x : x + w]
                    ocr_text = self.perform_ocr(cropped_img)
                    ocr_results.append(ocr_text)
                    position_encodings.append(
                        self.create_position_encoding(box, image.shape)
                    )

                combined_context = self.combine_ocr_and_position(
                    ocr_results, position_encodings
                )
                self.save_combined_to_file(combined_context, image_filename)
            else:
                print(f"No bounding box file found for {image_filename}")


if __name__ == "__main__":
    processor = ImageProcessor(
        directory_path="craft_sample",
        grid_size=7,
        lang="swe",
        output_file="combined_results.txt",
    )
    processor.process_directory()
