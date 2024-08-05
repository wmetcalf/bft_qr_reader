import cv2
import numpy as np
import argparse
import os
import uuid
import zxingcpp
from qreader import QReader
import gc
import json


class BFTQRCodeReader:
    def __init__(self, wechat_model_dir):
        self.qreader = QReader(model_size="s", weights_folder=wechat_model_dir)
        detect_prototxt = os.path.join(wechat_model_dir, "detect.prototxt")
        detect_caffemodel = os.path.join(wechat_model_dir, "detect.caffemodel")
        sr_prototxt = os.path.join(wechat_model_dir, "sr.prototxt")
        sr_caffemodel = os.path.join(wechat_model_dir, "sr.caffemodel")
        self.we_chat_detector = cv2.wechat_qrcode_WeChatQRCode(detect_prototxt, detect_caffemodel, sr_prototxt, sr_caffemodel)
        self.ocv_qr_code_detector = cv2.QRCodeDetector()
        self.save_matched = False
        self.results = []

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

    def enhance_and_decode(self, path, output_dir, bft, save_matched=False):
        self.results = []
        if save_matched:
            self.save_matched = save_matched
        # Create the output directory if it doesn't exist
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        if os.path.isfile(path):
            image_paths = [path]
        elif os.path.isdir(path):
            image_paths = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
        else:
            print(f"The provided path '{image_path}' is neither a file nor a directory.")
            self.results = []

        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                successful_methods = []
                if image is None:
                    print(f"Error: Cannot read image from path {image_path}")
                    continue
                print("Working on Image: ", image_path)
                methods = [
                    ("Original", image),
                    ("Cropped and Maintained Aspect Ratio", self.crop_and_maintain_aspect_ratio(image)),
                    (
                        "Cropped and Maintained Aspect Ratio Grey",
                        cv2.cvtColor(self.crop_and_maintain_aspect_ratio(image), cv2.COLOR_BGR2GRAY),
                    ),
                    ("Crop and Resize Grey", cv2.cvtColor(self.crop_and_resize(image), cv2.COLOR_BGR2GRAY)),
                    ("DarkPixelBlur", self.selective_blur(image)),
                    ("Adaptive Threshold", self.preprocess_image(image)),
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
                    ("Rotate 90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                    ("Rotate 180", cv2.rotate(image, cv2.ROTATE_180)),
                    ("Rotate 270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ]

                for method_name, processed_image in methods:
                    if not bft and successful_methods:
                        break
                    print(f"Trying zxing with method {method_name}")
                    if self.try_decode_zxingcpp(processed_image, method_name, output_dir, image_path):
                        successful_methods.append(method_name)
                        if bft:
                            continue
                        else:
                            break
                    print(f"Trying OpenCV with method {method_name}")
                    if self.try_decode_opencv(processed_image, method_name, output_dir, image_path):
                        successful_methods.append(method_name)
                        if bft:
                            continue
                        else:
                            break
                    print(f"Trying WeChat with method {method_name}")
                    if self.try_decode_wechat(processed_image, method_name, output_dir, image_path):
                        successful_methods.append(method_name)
                        if bft:
                            continue
                        else:
                            break

                    print(f"Trying QReader with method {method_name}")
                    if self.try_decode_qreader(processed_image, method_name, output_dir, image_path):
                        successful_methods.append(method_name)
                        if bft:
                            continue
                        else:
                            break

                if not successful_methods:
                    print("No QR codes found using the available methods.")
                else:
                    print(f"QR codes were successfully detected using the following methods: {', '.join(successful_methods)}")
                try:
                    gc.collect()
                except:
                    pass
            except Exception as e:
                print(f"An error occurred while processing the image: {e}")
        return self.results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance and decode QR codes from an image using multiple detectors.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input image")
    parser.add_argument("-o", "--output", required=False, default="/tmp", help="Directory to save the output images. Default is /tmp/ ")
    parser.add_argument("--model_dir", required=True, help="Directory containing the WeChat QR code model files")
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
    args = parser.parse_args()
    bft_qr_reader = BFTQRCodeReader(args.model_dir)
    results = bft_qr_reader.enhance_and_decode(args.input, args.output, args.bft, args.save_matched)
    if args.json_dump:
        output = os.path.join(args.output, "bft_dump.json")
        with open(output, "w") as f:
            f.write(json.dumps(results, indent=4))
    print(results)
