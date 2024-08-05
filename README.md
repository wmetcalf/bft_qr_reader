# bft_qr_reader
Blunt Force Trauma QR Reader. Tries a bunch of different methods with mutations (OpenCV,Zxing,WeChat,QReader)


```
usage: bft_qr_reader.py [-h] -i INPUT [-o OUTPUT] --model_dir MODEL_DIR [-b] [-s] [-j]

Enhance and decode QR codes from an image using multiple detectors.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input image
  -o OUTPUT, --output OUTPUT
                        Directory to save the output images. Default is /tmp/
  --model_dir MODEL_DIR
                        Directory containing the WeChat QR code model files
  -b, --bft             Blunt Force Trauma Keep Detecting even after first success https://www.youtube.com/watch?v=dtjGvBnAxVE
  -s, --save_matched    Save the image when a QR code is detected
  -j, --json_dump       Dump bft_dump.json to output directory
```
