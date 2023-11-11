import cv2

def detect_human(image_path):
    # Load the pre-trained HOG+SVM model for human detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Load the image
    image = cv2.imread(image_path)

    # Resize the image (optional, but can improve performance)
    image = cv2.resize(image, (640, 480))

    # Detect humans in the image
    humans, _ = hog.detectMultiScale(image)

    # Draw rectangles around the detected humans
    for (x, y, w, h) in humans:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Human Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'Tp_D_CRN_S_N_nat00033_cha00086_11502.jpg'#'Tp_D_CRN_M_N_nat10129_cha00086_11522.jpg'
detect_human(image_path)
