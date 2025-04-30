import cv2
import torch
from ultralytics import YOLO

def process_image(results):
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0].item()

            if confidence > 0.5:
                boxes.append((x1, y1, x2, y2, label, confidence))

    return boxes

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture('assets/insane 4k.mp4')

moment = 1

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if moment % 2 == 0 or moment == 1:
        results = model(frame)
        boxes = process_image(results)

    for x1, y1, x2, y2, label, confidence in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)

    moment += 1

    if cv2.waitKey(int(1)) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()