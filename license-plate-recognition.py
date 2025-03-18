import cv2
import numpy as np
import pytesseract
from PIL import Image
import os


def license_plate_recognition(image_path):
    """
    Extract and recognize license plate from an image
    
    Args:
        image_path: Path to the input image
    
    Returns:
        tuple: (processed_image, license_plate_text)
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Could not read image."

    # Make a copy of the original image
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Find edges
    edged = cv2.Canny(filtered, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # Variable to store license plate contour
    plate_contour = None
    
    # Loop over contours to find the license plate
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If the contour has 4 points, it's likely to be the license plate
        if len(approx) == 4:
            plate_contour = approx
            break
    
    if plate_contour is None:
        return original, "No license plate found."
    
    # Create a mask and draw the license plate contour on it
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    
    # Extract the license plate
    (x, y) = np.where(mask == 255)
    (top_x, top_y) = (np.min(x), np.min(y))
    (bottom_x, bottom_y) = (np.max(x), np.max(y))
    
    # Crop the license plate from the original image
    plate = original[top_x:bottom_x+1, top_y:bottom_y+1]
    
    # Draw the license plate contour on the original image
    cv2.drawContours(img, [plate_contour], -1, (0, 255, 0), 3)
    
    # Prepare the license plate for OCR
    # Convert to grayscale
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get a binary image
    _, plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Perform OCR on the license plate
    config = ('-l eng --oem 1 --psm 8')
    text = pytesseract.image_to_string(plate_binary, config=config)
    
    # Clean up the text
    text = ''.join(c for c in text if c.isalnum())
    
    return img, text

def main():
    """
    Main function to demonstrate license plate recognition
    """
    # Replace with your image path
    image_path = "license_plate.jpg"
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Perform license plate recognition
    processed_image, plate_text = license_plate_recognition(image_path)
    
    if processed_image is None:
        print(plate_text)
        return
    
    # Display the results
    print(f"Detected license plate: {plate_text}")
    
    # Resize the image for display
    scale_percent = 50  # percent of original size
    width = int(processed_image.shape[1] * scale_percent / 100)
    height = int(processed_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(processed_image, dim, interpolation=cv2.INTER_AREA)
    
    # Show the image
    cv2.imshow("License Plate Detection", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
