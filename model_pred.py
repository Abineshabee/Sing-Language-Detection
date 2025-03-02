import cv2


img = 'test_images/a.png'
# Read the image
image = cv2.imread(img)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('processed_pixels/processed1.png', gray_image)


import cv2
import numpy as np

# Define image dimensions
image_x, image_y = 64, 64

# Manually provide image path
img_path = img

# Read and process image
frame = cv2.imread(img_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Ensure background is black and object is white
# If the object appears black and background is white, invert it
white_pixels = np.sum(mask == 255)
black_pixels = np.sum(mask == 0)
if white_pixels > black_pixels:
    mask = cv2.bitwise_not(mask)

# Save the processed image
img_name = 'processed_pixels/processed1.png'
save_img = cv2.resize(mask, (image_x, image_y))
cv2.imwrite(img_name, save_img)

# Display result
cv2.imshow("Processed Image", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

