import cv2
import numpy as np

def identify_fruit():
    # Use relative paths if the model files are in the same directory as your script
    prototxt_path = 'MobileNetSSD_deploy.prototxt'
    caffemodel_path = 'MobileNetSSD_deploy.caffemodel'

    # Load the pre-trained MobileNetSSD model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Open a connection to the camera (0 represents the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Preprocess the frame for MobileNetSSD
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Set the input to the neural network
        net.setInput(blob)

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

                    # Scale bounding box coordinates to the original frame size
                    box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (startX, startY, endX, endY) = box.astype(int)

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"Fruit: {confidence:.2f}"
                    cv2.putText(frame, label, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the result
        cv2.imshow("Identified Fruits", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing and identifying fruits from the camera
identify_fruit()
    