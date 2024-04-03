import cv2
import numpy as np






def identify_fruit(image_path):
    # Use relative paths if the model files are in the same directory as your script
    prototxt_path = 'MobileNetSSD_deploy.prototxt'
    caffemodel_path = 'MobileNetSSD_deploy.caffemodel'

    # Load the pre-trained MobileNetSSD model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Load the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Preprocess the image for MobileNetSSD
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), 127.5)

    # Set the input to the neural network
    net.setInput(blob)

    # Initialize gray_fruit_roi outside the loop
    gray_fruit_roi = None

    # Run the forward pass to get predictions
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])

            # Check if the detected object is a fruit (you may need to adjust the class ID)
            if class_id == 56:  # Adjust the class ID based on the MobileNetSSD class labels

                # Scale bounding box coordinates to the original image size
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype(int)

                # Extract the region of interest (ROI) for the identified fruit
                fruit_roi = image[startY:endY, startX:endX]

                # Convert the ROI to grayscale
                gray_fruit_roi = cv2.cvtColor(fruit_roi, cv2.COLOR_BGR2GRAY)

                # Draw the bounding box and label on the original image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                label = f"Fruit: {confidence:.2f}"
                cv2.putText(image, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original and grayscale images side by side
    if gray_fruit_roi is not None:
        side_by_side = np.hstack((image, cv2.cvtColor(gray_fruit_roi, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Original vs Grayscale", side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Gray fruit ROI is None.")







# Replace 'path/to/your/image.jpg' with the actual path to your image
image_path = 'C:/Users/Huzaifa Khan/Desktop/College/Third_Year/IPCV/apple.jpg'
identify_fruit(image_path)

image = cv2.imread(image_path)
print("Image loaded successfully.")