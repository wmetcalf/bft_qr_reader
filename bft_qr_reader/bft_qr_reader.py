import cv2
import numpy as np
import argparse
import os
import uuid
import zxingcpp
from qreader import QReader
import gc
import json
import traceback
import logging

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import shutil
import os
import magic
import tempfile
import subprocess
import multiprocessing
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()

# Set up logging

logger = logging.getLogger(__name__)

app = FastAPI()

# Global variable to store the BFTQRCodeReader instance
qr_code_reader = None
we_chat_model_dir = None

def get_model_dir(args_model_dir=None):
    if args_model_dir:
        return args_model_dir
    else:
        package_dir = os.path.dirname(__file__)
        return os.path.join(package_dir, 'models')

# app = FastAPI()
@asynccontextmanager
async def lifespan(app: FastAPI):
    global we_chat_model_dir
    logger.info("Lifespan event started.")
    # You can replace 'path_to_wechat_models' with the actual path to your models
    app.state.qr_code_reader = BFTQRCodeReader(wechat_model_dir=get_model_dir())
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not hasattr(app.state, "qr_code_reader"):
        logger.error("QR code reader is not initialized in app.state!")
        raise HTTPException(status_code=500, detail="QR code reader is not initialized.")

    qr_code_reader = app.state.qr_code_reader

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Function to run the QR code reader in a separate thread
        def run_qr_code_reader():
            return qr_code_reader.enhance_and_decode(temp_file_path, temp_dir, False, False)

        try:
            # Use asyncio.wait_for to apply a timeout for the synchronous task
            success = await asyncio.wait_for(asyncio.get_running_loop().run_in_executor(executor, run_qr_code_reader), timeout=45.0)
        except asyncio.TimeoutError:
            logger.error("QR code detection timed out.")
            raise HTTPException(status_code=408, detail="QR code detection timed out.")

    # Return the result of the QR code detection
    if success:
        return JSONResponse({"status": "QR code detected", "message": qr_code_reader.results})
    else:
        return JSONResponse({"status": "No QR code detected"})


@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <html>
        <head>
            <title>Nom Nom Feed Me</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/upload/" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit">
            </form>
        </body>
    </html>
    """


class BFTQRCodeReader:
    def __init__(self, wechat_model_dir="./models/", methods_to_try="zxing,opencv,wechat,qreader"):
        self.qreader = QReader(model_size="s")
        detect_prototxt = os.path.join(wechat_model_dir, "detect.prototxt")
        detect_caffemodel = os.path.join(wechat_model_dir, "detect.caffemodel")
        sr_prototxt = os.path.join(wechat_model_dir, "sr.prototxt")
        sr_caffemodel = os.path.join(wechat_model_dir, "sr.caffemodel")
        self.we_chat_detector = cv2.wechat_qrcode_WeChatQRCode(detect_prototxt, detect_caffemodel, sr_prototxt, sr_caffemodel)
        self.ocv_qr_code_detector = cv2.QRCodeDetector()
        self.save_matched = False
        self.results = []

        # Split the methods_to_try string and trim whitespace
        methods_list = [method.strip() for method in methods_to_try.split(",")]

        # Initialize decode methods based on the provided list
        self.decode_methods = {}
        if "zxing" in methods_list:
            self.decode_methods["zxing"] = self.try_decode_zxingcpp
        if "opencv" in methods_list:
            self.decode_methods["opencv"] = self.try_decode_opencv
        if "wechat" in methods_list:
            self.decode_methods["wechat"] = self.try_decode_wechat
        if "qreader" in methods_list:
            self.decode_methods["qreader"] = self.try_decode_qreader

        # If no methods are specified, default to all methods
        if not self.decode_methods:
            self.decode_methods = {
                "zxing": self.try_decode_zxingcpp,
                "opencv": self.try_decode_opencv,
                "wechat": self.try_decode_wechat,
                "qreader": self.try_decode_qreader,
            }

    def try_decode_qreader(self, image, method_name, output_dir, image_path):
        # Use the detect_and_decode function to get the decoded QR data
        decoded_texts = self.qreader.detect_and_decode(image=image)
        success = False

        if decoded_texts:
            if len(decoded_texts) == 1 and not decoded_texts[0]:
                return success
            else:
                decoded_text = " ".join([text for text in decoded_texts if text is not None])
                # print(f"Decoded text (qrreader): {decoded_text}")
                success = True

        if success:
            tmp_dict = {
                "method": method_name,
                "decoded_text": decoded_text,
                "image_path": image_path,
                "model": "qrreader",
                "output_path": "",
            }
            if self.save_matched:
                unique_filename = f"{method_name.replace(' ', '_')}_{uuid.uuid4()}_qrreader.png"
                output_path = os.path.join(output_dir, unique_filename)
                cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # print(f"QR Code found and saved as: {output_path}")
                tmp_dict["output_path"] = output_path
            if tmp_dict not in self.results:
                self.results.append(tmp_dict)
        return success

    def try_decode_zxingcpp(self, image, method_name, output_dir, image_path):
        decoded_objects = zxingcpp.read_barcodes(image)
        success = False

        for obj in decoded_objects:
            # print(f"Decoded text (zxingcpp): {obj.text}")
            success = True

        if success:
            tmp_dict = {"method": method_name, "decoded_text": obj.text, "image_path": image_path, "model": "zxingcpp", "output_path": ""}
            if self.save_matched:
                unique_filename = f"{method_name.replace(' ', '_')}_{uuid.uuid4()}_zxingcpp.png"
                output_path = os.path.join(output_dir, unique_filename)
                cv2.imwrite(output_path, image)
                # print(f"QR Code found and saved as: {output_path}")
                tmp_dict["output_path"] = output_path
            if tmp_dict not in self.results:
                self.results.append(tmp_dict)

        return success

    def try_decode_opencv(self, image, method_name, output_dir, image_path):
        data, _, _ = self.ocv_qr_code_detector.detectAndDecode(image)
        success = False
        if data:
            success = True
            # print(f"Decoded text (opencv): {data}")
            tmp_dict = {"method": method_name, "decoded_text": data, "image_path": image_path, "model": "opencv", "output_path": ""}
            if self.save_matched:
                unique_filename = f"{method_name.replace(' ', '_')}_{uuid.uuid4()}_opencv.png"
                output_path = os.path.join(output_dir, unique_filename)
                cv2.imwrite(output_path, image)
                # print(f"QR Code found and saved as: {output_path}")
                tmp_dict["output_path"] = output_path
            if tmp_dict not in self.results:
                self.results.append(tmp_dict)
        return success

    def try_decode_wechat(self, image, method_name, output_dir, image_path):
        res, _ = self.we_chat_detector.detectAndDecode(image)
        success = False

        if len(res) > 0:
            # for text in res:
            # print(f"Decoded text (wechat): {text}")
            success = True

        if success:
            tmp_dict = {"method": method_name, "decoded_text": "".join(res), "image_path": image_path, "model": "wechat", "output_path": ""}
            if self.save_matched:
                unique_filename = f"{method_name.replace(' ', '_')}_{uuid.uuid4()}_wechat.png"
                output_path = os.path.join(output_dir, unique_filename)
                cv2.imwrite(output_path, image)
                # print(f"QR Code found and saved as: {output_path}")
                tmp_dict["output_path"] = output_path
            if tmp_dict not in self.results:
                self.results.append(tmp_dict)
        return success

    def crop_solid_color_bottom(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the last row that isn't a solid color
        bottom = gray.shape[0] - 1
        while bottom > 0:
            if not np.all(gray[bottom] == gray[bottom][0]):
                break
            bottom -= 1

        cropped_image = image[: bottom + 1, :]

        return cropped_image

    def crop_and_maintain_aspect_ratio(self, image):
        cropped_image = self.crop_solid_color_bottom(image)
        aspect_ratio = cropped_image.shape[1] / cropped_image.shape[0]
        return cropped_image

    def crop_and_resize(self, image):
        cropped_image = self.crop_solid_color_bottom(image)
        resized_image = cv2.resize(cropped_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return resized_image

    def selective_blur(self, image):
        if image is None or image.size == 0:
            raise ValueError("The image is empty or invalid.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        blurred = cv2.GaussianBlur(bw, (31, 31), 0)

        mask = cv2.inRange(bw, 1, 255)
        mask_inv = cv2.bitwise_not(mask)

        selective_blur = cv2.bitwise_and(blurred, blurred, mask=mask)
        whites = cv2.bitwise_and(gray, gray, mask=mask_inv)

        result = cv2.add(selective_blur, whites)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return adaptive_thresh

    def detect_and_remove_logo_by_grid(self, img, qr_detections=None):
        def find_alignment_squares(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            alignment_squares = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

            return alignment_squares

        def get_grid_from_alignment_dyn(img, alignment_squares):
            centers = [cv2.moments(cnt) for cnt in alignment_squares]
            centroids = [(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) for M in centers]

            centroids.sort(key=lambda x: x[1])

            top_left = centroids[0]
            top_right = centroids[1]

            bottom_left = centroids[2]

            block_size_x = abs(top_right[0] - top_left[0]) // len(alignment_squares)
            block_size_y = abs(bottom_left[1] - top_left[1]) // len(alignment_squares)

            return block_size_x, block_size_y, top_left

        if not qr_detections:
            logger.debug("No QR code detected")
            return None

        qr_info = qr_detections[0]
        x_min, y_min, x_max, y_max = qr_info["bbox_xyxy"]

        qr_img = img[int(y_min) : int(y_max), int(x_min) : int(x_max)]

        alignment_squares = find_alignment_squares(qr_img)
        if len(alignment_squares) < 3:
            logger.debug("Could not detect all alignment squares")
            return None

        block_size_x, block_size_y, top_left = get_grid_from_alignment_dyn(qr_img, alignment_squares)
        h, w = qr_img.shape[:2]

        central_x_start = int(w * 0.30)
        central_x_end = int(w * 0.70)
        central_y_start = int(h * 0.40)
        central_y_end = int(h * 0.60)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (central_x_start, central_y_start), (central_x_end, central_y_end), 255, thickness=-1)

        inpainted_img = cv2.inpaint(qr_img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        img[int(y_min) : int(y_max), int(x_min) : int(x_max)] = inpainted_img
        return img

    def scale_image_with_aspect_ratio(self, image, scale_factor=None, width=None, height=None):
        if scale_factor is None and width is None and height is None:
            raise ValueError("You must provide either a scale factor, a target width, or a target height.")

        original_height, original_width = image.shape[:2]

        if scale_factor is not None:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
        elif width is not None:
            aspect_ratio = original_height / original_width
            new_width = width
            new_height = int(width * aspect_ratio)
        elif height is not None:
            aspect_ratio = original_width / original_height
            new_height = height
            new_width = int(height * aspect_ratio)

        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return scaled_image

    def enhance_and_decode(self, path, output_dir, bft, save_matched=False):
        self.results = []
        if save_matched:
            self.save_matched = save_matched
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if os.path.isfile(path):
            image_paths = [path]
        elif os.path.isdir(path):
            image_paths = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        else:
            logger.warning(f"The provided path '{path}' is neither a file nor a directory.")
            self.results = []

        for image_path in image_paths:
            try:
                image = None
                roid_image = None
                have_roids = False
                mime_type = magic.from_file(image_path, mime=True)
                temp_png_path = None
                roid_methods = []
                qr_detections = None
                image_copy = None
                grey = None
                if mime_type.startswith("image/"):
                    if mime_type == "image/wmf":
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
                                temp_png_path = temp_png.name
                            subprocess.run(
                                ["inkscape", image_path, "-d", "300", "--export-type=png", "--export-filename", temp_png_path], check=True
                            )
                            logger.debug(f"converted {image_path} to {temp_png_path}")
                            image = cv2.imread(temp_png_path)
                            image_path = temp_png_path
                        except Exception as e:
                            logger.debug(f"failed to convert wmf to PNG {e}")
                    elif mime_type == "image/svg+xml":
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_png:
                            temp_png_path = temp_png.name
                        subprocess.run(["rsvg-convert", image_path, "-o", temp_png_path], check=True)
                        logger.debug(f"converted {image_path} to {temp_png_path}")
                        image = cv2.imread(temp_png_path)
                        image_path = temp_png_path
                    else:
                        image = cv2.imread(image_path)
                else:
                    logger.debug(f"File is not an image {mime_type}")
                    if temp_png_path and os.path.exists(temp_png_path):
                        os.remove(temp_png_path)
                    continue
                if image is None or image.size == 0:
                    logger.warning("Failed to convert image: doesn't exist or has size of 0")
                    if temp_png_path and os.path.exists(temp_png_path):
                        os.remove(temp_png_path)
                    continue

                # Check if the image is a solid color
                if np.all(image == image[0, 0]):
                    logger.warning("Image is a solid color, skipping")
                    if temp_png_path and os.path.exists(temp_png_path):
                        os.remove(temp_png_path)
                    continue
                image.setflags(write=False)

                try:
                    qr_detections = self.qreader.detect(image, is_bgr=True)
                except Exception as e:
                    logger.debug(f"failed to detect qrcode {e} Will try again with scaling for very small images")
                    logger.debug(traceback.format_exc())
                if qr_detections is not None and image.shape[0] < 100 or image.shape[1] < 100:
                    try:
                        qr_detections = self.qreader.detect(self.scale_image_with_aspect_ratio(image, scale_factor=4.0), is_bgr=True)
                        if qr_detections is not None:
                            image_copy = self.scale_image_with_aspect_ratio(image, scale_factor=4.0)
                    except:
                        logger.debug(f"failed to detect qrcode {e} with scale for small image")
                        logger.debug(traceback.format_exc())
                try:
                    if qr_detections is not None:
                        if not image_copy:
                            image_copy = image.copy()
                        image.setflags(write=False)
                        roid_image = self.detect_and_remove_logo_by_grid(image_copy, qr_detections)
                        have_roids = True
                except Exception as e:
                    logger.warning(f"Failed to generate ROID image {e}")
                successful_methods = []
                if image is None:
                    logger.warning(f"Error: Cannot read image from path {image_path}")
                    if temp_png_path and os.path.exists(temp_png_path):
                        os.remove(temp_png_path)
                    continue
                logger.debug(f"Working on Image: {image_path}")
                methods = [
                    ("Original", image),
                    ("Adaptive Threshold", self.preprocess_image(image)),
                    ("Scale Image 0.5", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=0.5)),
                    ("Scale Image 1.5", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=1.5)),
                    ("Scale Image 2.0", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=2.0)),
                    ("Cropped and Maintained Aspect Ratio", self.crop_and_maintain_aspect_ratio(image)),
                    (
                        "Cropped and Maintained Aspect Ratio Grey",
                        cv2.cvtColor(self.crop_and_maintain_aspect_ratio(image), cv2.COLOR_BGR2GRAY),
                    ),
                    ("Crop and Resize Grey", cv2.cvtColor(self.crop_and_resize(image), cv2.COLOR_BGR2GRAY)),
                    ("DarkPixelBlur", self.selective_blur(image)),
                    ("Histogram Equalization", cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))),
                    (
                        "Morphological Transformations",
                        cv2.morphologyEx(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
                    ),
                    ("Bilateral Filtering", cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 9, 75, 75)),
                    (
                        "Sharpening",
                        cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
                    ),
                    # ("Rotate 90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                    # ("Rotate 180", cv2.rotate(image, cv2.ROTATE_180)),
                    # ("Rotate 270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ]
                if image.shape[0] < 100 or image.shape[1] < 100:
                    methods.extend(
                        [
                            ("Scale Image 4.0", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=4.0)),
                            ("Scale Image 6.0", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=6.0)),
                        ]
                    )
                if roid_image is not None and roid_image.size > 0:
                    roid_methods = [
                        ("Original", roid_image),
                        ("DarkPixelBlur", self.selective_blur(roid_image)),
                        ("Adaptive Threshold", self.preprocess_image(roid_image)),
                        (
                            "Sharpening",
                            cv2.filter2D(cv2.cvtColor(roid_image, cv2.COLOR_BGR2GRAY), -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
                        ),
                        ("Scale Image 0.5", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=0.5)),
                        ("Scale Image 1.5", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=1.5)),
                        ("Scale Image 2.0", self.scale_image_with_aspect_ratio(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), scale_factor=2.0)),
                        ("Histogram Equalization", cv2.equalizeHist(cv2.cvtColor(roid_image, cv2.COLOR_BGR2GRAY))),
                        (
                            "Morphological Transformations",
                            cv2.morphologyEx(cv2.cvtColor(roid_image, cv2.COLOR_BGR2GRAY), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)),
                        ),
                        ("Bilateral Filtering", cv2.bilateralFilter(cv2.cvtColor(roid_image, cv2.COLOR_BGR2GRAY), 9, 75, 75)),
                    ]
                    # Check if the roid_image is smaller than 100x100 pixels
                    if roid_image.shape[0] < 100 or roid_image.shape[1] < 100:
                        roid_methods.extend(
                            [
                                (
                                    "Scale Image 4.0",
                                    self.scale_image_with_aspect_ratio(cv2.cvtColor(roid_image, cv2.COLOR_BGR2GRAY), scale_factor=4.0),
                                ),
                                (
                                    "Scale Image 6.0",
                                    self.scale_image_with_aspect_ratio(cv2.cvtColor(roid_image, cv2.COLOR_BGR2GRAY), scale_factor=6.0),
                                ),
                            ]
                        )

                for method_name, processed_image in methods:
                    if not bft and successful_methods:
                        break
                    for decode_method in self.decode_methods:
                        logger.debug(f"Trying {decode_method} with method {method_name}")
                        if self.decode_methods[decode_method](processed_image, method_name, output_dir, image_path):
                            successful_methods.append(method_name)
                            if bft:
                                continue
                            else:
                                break

                # for qr_info in qr_detections:
                #    x_min, y_min, x_max, y_max = qr_info["bbox_xyxy"]
                #    qr_img = image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
                #    for method_name, processed_image in methods:
                #        if not bft and successful_methods:
                #            break
                #        for decode_method in self.decode_methods:
                #            logger.debug(f"Trying extracted {decode_method} with method {method_name}")
                #           if self.decode_methods[decode_method](processed_image, method_name, output_dir, image_path):
                #                successful_methods.append(method_name)
                #                if bft:
                #                    continue
                #                else:
                #                    break

                if not successful_methods and have_roids:
                    for method_name, processed_image in roid_methods:
                        if not bft and successful_methods:
                            break
                        for decode_method in self.decode_methods:
                            logger.debug(f"Trying ROID {decode_method} with method {method_name}")
                            if self.decode_methods[decode_method](processed_image, method_name, output_dir, image_path):
                                successful_methods.append(method_name)
                                if bft:
                                    continue
                                else:
                                    break

                if not successful_methods:
                    logger.debug("No QR codes found using the available methods.")
                else:
                    logger.debug(f"QR codes were successfully detected using the following methods: {', '.join(successful_methods)}")
                try:
                    gc.collect()
                    if temp_png_path:
                        os.remove(temp_png_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"An error occurred while processing the image: {e}")
                logger.debug(traceback.format_exc())
        return self.results


def main():
    global we_chat_model_dir
    parser = argparse.ArgumentParser(description="Enhance and decode QR codes from an image using multiple detectors.")
    parser.add_argument("-i", "--input", required=False, help="Path to the input image")
    parser.add_argument("-o", "--output", required=False, default="/tmp", help="Directory to save the output images. Default is /tmp/ ")
    parser.add_argument("--model_dir", type=str, help="Directory containing the WeChat QR code model files")
    parser.add_argument("--methods", required=False, default="zxing,opencv,wechat,qreader")
    parser.add_argument(
        "-b",
        "--bft",
        required=False,
        action="store_true",
        default=False,
        help="Blunt Force Trauma Keep Detecting even after first success https://www.youtube.com/watch?v=dtjGvBnAxVE",
    )
    parser.add_argument(
        "-s",
        "--save_matched",
        required=False,
        action="store_true",
        default=False,
        help="Save the image when a QR code is detected",
    )
    parser.add_argument("-j", "--json_dump", required=False, action="store_true", default=False, help="Dump bft_dump.json to output directory"),
    parser.add_argument(
        "-l",
        "--log_level",
        required=False,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default is INFO.",
    )
    parser.add_argument("--webserver", action="store_true", help="Run as webserver")
    parser.add_argument("--port", type=int, default=1111, help="Port for the webserver")
    parser.add_argument("--recycle_workers", type=int, default=20, help="Recycle workers after this many requests default is 20")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="how many workers should we use default is 1")
    args = parser.parse_args()
    we_chat_model_dir = get_model_dir(args.model_dir)    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.webserver:
        uvicorn.run("bft_qr_reader.bft_qr_reader:app", host="0.0.0.0", port=args.port, limit_max_requests=args.recycle_workers, workers=args.workers)
    else:
        if args.input:
            qr_code_reader = BFTQRCodeReader(we_chat_model_dir, args.methods)
            results = qr_code_reader.enhance_and_decode(args.input, args.output, args.bft, args.save_matched)
            new_results = {"message": results}
            if args.json_dump:
                output = os.path.join(args.output, "bft_dump.json")
                with open(output, "w") as f:
                    f.write(json.dumps(new_results, indent=4))
            logger.debug(new_results)
        else:
            logger.debug("You must specify an input argument")


if __name__ == "__main__":
    main()
