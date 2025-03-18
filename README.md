# License Plate Recognition System

## Overview
This project implements an automated license plate recognition system using computer vision techniques and optical character recognition (OCR). The system can detect license plates in images and extract the alphanumeric characters from them.

## Features
- License plate detection using contour analysis
- Preprocessing for improved OCR accuracy
- Text extraction using Tesseract OCR
- Visual output of the detection process

## Prerequisites
- Python 3.6 or higher
- OpenCV (`cv2`)
- NumPy
- Tesseract OCR engine
- pytesseract (Python wrapper for Tesseract)
- PIL (Python Imaging Library)

## Installation

### 1. Install Python dependencies
```bash
pip install opencv-python numpy pytesseract pillow
```

### 2. Install Tesseract OCR
#### For Windows:
- Download and install the Tesseract installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
- Add the Tesseract installation directory to your system PATH

#### For macOS:
```bash
brew install tesseract
```

#### For Linux:
```bash
sudo apt-get install tesseract-ocr
```

### 3. Configure pytesseract
If Tesseract is not in your system PATH, you may need to set the path explicitly in your code:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
```

## Usage
1. Place your license plate image in the same directory as the script or provide the full path to the image
2. Modify the `image_path` variable in the `main()` function to point to your image
3. Run the script:
```bash
python license_plate_recognition.py
```

## How it works
1. The image is converted to grayscale and noise is reduced with a bilateral filter
2. Edges are detected using the Canny edge detector
3. Contours are found and filtered to identify potential license plates
4. The license plate region is isolated and prepared for OCR
5. Tesseract OCR extracts the text from the license plate
6. The result is displayed with the plate contour highlighted

## Limitations
- Works best with clear, well-lit, and frontally-oriented license plates
- May struggle with highly angled, blurry, or partially obscured plates
- OCR accuracy depends on image quality and plate condition
- Performance may vary across different license plate styles and formats

## Troubleshooting
- If you encounter issues with Tesseract OCR, try adjusting the configuration parameters
- For poor detection results, try adjusting the Canny edge detection thresholds
- If the plate is not being detected, you may need to adjust the contour filtering criteria
## Acknowledgments
- OpenCV for computer vision algorithms
- Tesseract OCR for text recognition
- The Python community for excellent libraries and documentation
