# bft_qr_reader
Blunt Force Trauma QR Reader. Tries a bunch of different methods with mutations (OpenCV,Zxing,WeChat,QReader)

# Add req Packages
```
sudo apt-get install -y libzbar0 inkscape librsvg2-bin
pip3 install git+https://github.com/wmetcalf/bft_qr_reader.git

or

pip install -r requirements.txt
```

# Usage
```
bft_qr_reader [-h] [-i INPUT] [-o OUTPUT] [--model_dir MODEL_DIR] [--methods METHODS] [-b] [-s] [-j] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--webserver] [--port PORT]
                     [--recycle_workers RECYCLE_WORKERS] [--workers WORKERS]

Enhance and decode QR codes from an image using multiple detectors.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input image
  -o OUTPUT, --output OUTPUT
                        Directory to save the output images. Default is /tmp/
  --model_dir MODEL_DIR
                        Directory containing the WeChat QR code model files
  --methods METHODS
  -b, --bft             Blunt Force Trauma Keep Detecting even after first success https://www.youtube.com/watch?v=dtjGvBnAxVE
  -s, --save_matched    Save the image when a QR code is detected
  -j, --json_dump       Dump bft_dump.json to output directory
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level. Default is INFO.
  --webserver           Run as webserver
  --port PORT           Port for the webserver
  --recycle_workers RECYCLE_WORKERS
                        Recycle workers after this many requests default is 20
  --workers WORKERS     how many workers should we use default is 1
```
