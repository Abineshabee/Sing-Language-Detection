import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
classifier = load_model('my_model.keras')

# Define image dimensions
image_x, image_y = 64, 64

# Function to predict letter from an image
def predictor(img_path):
    test_image = image.load_img(img_path, target_size=(image_x, image_y))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)  # Expand dimensions to match model input
    test_image /= 255.0  # Normalize pixel values
    result = classifier.predict(test_image)
    
    # Get predicted class index
    predicted_class = np.argmax(result)
    
    # Map index to corresponding letter
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return letters[predicted_class] if predicted_class < len(letters) else '?'  # '?' for invalid index

# Camera setup
cam = cv2.VideoCapture(0)
cv2.namedWindow("Live Feed")

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2)
    cv2.putText(frame, "Place hand inside the box", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Extract region of interest (ROI)
    imcrop = frame[102:298, 427:623]
    gray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save the processed image
    img_name = "captured.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    
    # Predict letter
    img_text = predictor(img_name)
    
    # Display result
    cv2.putText(frame, f"Predicted: {img_text}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.imshow("Live Feed", frame)
    cv2.imshow("Processed Image", mask)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cam.release()
cv2.destroyAllWindows()

