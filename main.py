import cv2
import numpy as np
import threading
import time

# Load YOLO
net = cv2.dnn.readNet(r"C:/Users/Isaac/Desktop/python/OpenCV/yolov3.weights", 
                      r"C:/Users/Isaac/Desktop/python/OpenCV/yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open(r"C:/Users/Isaac/Desktop/python/OpenCV/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Video file path
ip_camera_url = r"C:/Users/Isaac/Downloads/7092235-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame = None
paused = False  # This will track if the video is paused

def capture_video():
    global frame, paused
    while True:
        if not paused:  # Only capture video if not paused
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
            else:
                print("Frame capture failed or video ended.")
                break

# Start video capture in a separate thread
threading.Thread(target=capture_video, daemon=True).start()

frame_count = 0

while True:
    if frame is None:
        continue

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (500, 500))

    if frame_count % 2 == 0:  # Process every 2nd frame
        height, width, channels = resized_frame.shape
        blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        people_count = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if classes[class_ids[i]] == "person":
                    people_count += 1
                    cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print("People in the frame: ", people_count)

    # Display the current frame
    cv2.imshow("People Detection with YOLO", resized_frame)
    frame_count += 1

    # Wait for key press and detect if space is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord(' '):  # Press Space to pause
        paused = not paused  # Toggle paused state



cap.release()
cv2.destroyAllWindows()
