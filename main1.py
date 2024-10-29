import cv2
import numpy as np
import threading

# YOLO
net = cv2.dnn.readNet(r"C:/Users/Isaac/Desktop/python/OpenCV/yolov3.weights", 
                      r"C:/Users/Isaac/Desktop/python/OpenCV/yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open(r"C:/Users/Isaac/Desktop/python/OpenCV/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


ip_camera_url = "http://192.168.168.164:8080/video"#"C:/Users/Isaac/Desktop/python/OpenCV/people2.mp4"
cap = cv2.VideoCapture(ip_camera_url)

# if not cap.isOpened():
#     print("Xato: Videoni ochib bo'lmadi.")
#     exit()

frame = None
paused = False

def capture_video():
    global frame, paused
    while True:
        if not paused:
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
            else:
                print("Kadrni olishda xatolik yoki video tugadi.")
                break


threading.Thread(target=capture_video, daemon=True).start()

frame_count = 0

while True:
    if frame is None:
        continue


    resized_frame = cv2.resize(frame, (500, 500))

    # if frame_count % 2 == 0:
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
                if confidence > 0.5:  
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

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])  # Obyekt nomi
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(resized_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

   
    cv2.imshow("YOLO yordamida obyektlarni aniqlash", resized_frame)
    frame_count += 1


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord(' '):  
        paused = not paused

cap.release()
cv2.destroyAllWindows()
