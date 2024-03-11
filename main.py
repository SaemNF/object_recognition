#Capture Live Video:  capture live video from your camera using OpenCV
import cv2
import numpy as np

# Open the default camera (typically the first camera connected)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Loop to continuously read frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Preprocess the frame for OpenVINO
    # Resize the frame to the input size expected by the neural network
    input_size = (300, 300)  # Example input size for a neural network
    preprocessed_frame = cv2.resize(frame, input_size)

    # Convert the frame to the format expected by OpenVINO
    # For example, convert from BGR (OpenCV default) to RGB
    preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)

    # Normalize the pixel values to the range [0, 1]
    preprocessed_frame = preprocessed_frame.astype(np.float32) / 255.0

    # Perform any additional preprocessing steps as required by your neural network model

    # Example: Add batch dimension to the frame
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

    # Now you can use the preprocessed_frame for inference with OpenVINO

    # Display the original captured frame
    #cv2.imshow('Original Frame', frame)

    # Display the preprocessed frame (for demonstration purposes)
    cv2.imshow('Preprocessed Frame', cv2.cvtColor(preprocessed_frame[0], cv2.COLOR_RGB2BGR))

    # Check for 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



#Perform Inference: use OpenVINO to perform inference tasks such as object detection, classification, or segmentation. 

#Post-Processing and Visualization:post-process the results as needed. Drawing bounding boxes around detected objects, annotating frames with inference results, and generating reports based on the analysis.

#Display or Stream Results: display the processed video with inference results in real-time or stream it to another destination as required by your application.

#add on idea:
# 1. add button on the popup video windows such as on and off.