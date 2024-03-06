import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_redness_score(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert image to grayscale for face detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, return 0
    if len(faces) == 0:
        return 0, None

    # Get the first detected face
    x, y, w, h = faces[0]

    # Extract the region of interest (ROI) which is the face
    face_roi = img[y:y+h, x:x+w]

    # Convert the ROI to HSV color space
    face_hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    # Define a range for red color in HSV
    lower_red = np.array([0, 70, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask for red color in the ROI
    mask = cv2.inRange(face_hsv, lower_red, upper_red)

    # Calculate the percentage of redness in the face
    redness_percentage = (np.count_nonzero(mask) / (w * h))+  100 - 30

    # Display the original image with the detected face and red mask
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cv2.bitwise_and(face_roi, face_roi, mask=mask), cv2.COLOR_BGR2RGB))
    plt.title('Red Mask')

    plt.show()

    return redness_percentage, mask

# Specify the image path
image_path = 'main-qimg-8f68ebd077d3e8c95f1af2a120adbe90-lq.jpg'

# Calculate redness score and display the results
redness_score, red_mask = calculate_redness_score(image_path)
print(f'Redness Score: {redness_score:.2f}')


if red_mask is not None:
    cv2.imwrite('red_mask.jpg', red_mask)
