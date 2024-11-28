from utils import detect_objects, yolo_image_classify , oriented_detection_realtime, segment_objects

def main():
    print("Select a mode:")
    print("1. Object Detection")
    print("2. Image Classification")
    print("3. Object Segmentation")
    print("4. Oriented Object Detection")  # Added for the oriented detection option

    choice = input("Enter your choice: ")

    if choice == "1":
        print("Starting object detection...")
        detect_objects()  # Calling the object detection function
    elif choice == "2":
        print("Starting image classification...")
        yolo_image_classify()
    elif choice == "3":
        print("Starting object segmentation...")
        segment_objects()  # Calling segmentation function
    elif choice == "4":
        print("Starting oriented object detection...")
        oriented_detection_realtime()  # Calling oriented detection
    else:
        print("Invalid choice. Exiting.")  # Handling invalid inputs

if __name__ == "__main__":
    main()
