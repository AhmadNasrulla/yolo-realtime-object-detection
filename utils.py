from ultralytics import YOLO
import cv2
import torch
from matplotlib import pyplot as plt


def detect_objects(source=0, model_path="models/yolo11l.pt"):
    """Function to perform object detection."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        results = model.predict(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def yolo_image_classify(model_path="models/yolo11l-cls.pt", image_path="C:/Users/ahmad/OneDrive/Desktop/yolo 11/images/img1.jpeg"):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Perform prediction
    results = model(image_path)
    
    # Visualize results
    annotated_image = results[0].plot()  # Get the annotated image with bounding boxes and labels
    
    # Display the image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
  

def yolo_realtime_classify():
    # Load the YOLOv5 model (e.g., 'yolov5s')
    model = torch.hub.load("models/yolo11l-cls.pt")

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference on the frame
        results = model(frame)

        # Display the results on the frame
        frame_with_results = results.render()[0]

        # Show the frame with detected objects
        cv2.imshow("YOLO Real-Time Classification", frame_with_results)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()




def oriented_detection_realtime(model_path="models/yolo11m-obb.pt", source=0, conf_threshold=0.3):
    model = YOLO(model_path)  # Replace with your model path

    cap = cv2.VideoCapture(0)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Loop to process the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Predict using YOLO
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()  # Annotate the frame with predictions

        # Display the frame
        cv2.imshow("YOLO Real-Time Detection", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def segment_objects(source=0, model_path="models/yolo11l-seg.pt"):
    """Function to perform object segmentation."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished processing video.")
            break

        results = model.predict(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Object Segmentation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
