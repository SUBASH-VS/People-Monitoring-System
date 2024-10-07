import cv2
from ultralytics import YOLO

# Step 1: Load the pre-trained YOLOv8 model (detects 'person' class)
model = YOLO('yolov8n.pt')  # YOLOv8 lightweight version

# Step 2: Load Haar Cascade face detector (for face detection within person detections)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # for face detection

# Step 3: Initialize webcam feed
cap = cv2.VideoCapture(0)  # 0 for webcam, or use video file path for CCTV feed

# Step 4: Run detection and display results in real-time
while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Step 5: Run YOLOv8 detection for 'person'
    results = model(frame)  # Detect objects in the frame
    people_count = 0  # Counter for people

    # Step 6: Filter detections for 'person' class (class 0 in COCO dataset)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 0:  # Class 0 corresponds to 'person'
                people_count += 1
                # Get the bounding box for 'person'
                bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                # Step 7: Crop the person area to detect faces within it
                person_roi = frame[y1:y2, x1:x2]  # Region of interest (ROI)

                # # Step 8: Detect faces in the person ROI
                # faces = face_cascade.detectMultiScale(person_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # for (fx, fy, fw, fh) in faces:
                #     # Draw face bounding box (adjust coordinates relative to the whole frame)
                #     cv2.rectangle(frame, (x1 + fx, y1 + fy), (x1 + fx + fw, y1 + fy + fh), (255, 0, 0), 2)

                # Draw person bounding box in green
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Step 9: Display the number of people in the frame
    cv2.putText(frame, f'People Count: {people_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Step 10: Show the frame with bounding boxes and face annotations
    cv2.imshow('YOLOv8 Face Detector', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 11: Release resources
cap.release()
cv2.destroyAllWindows()
