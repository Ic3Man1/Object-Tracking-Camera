import cv2
import torch
from ultralytics import YOLO

def process_image(results, target_id):
    boxes = []
    coordinates = None
    for result in results:
        for box in result.boxes:
            if box.id is not None and int(box.id) == target_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0].item()

                if confidence > 0.5:
                    boxes.append((x1, y1, x2, y2, label, confidence))
                    print(f"ID {target_id} coords ({(x2-x1)/2}, {(y2-y1)/2})")
                    coordinates = ((x2-x1)/2, (y2-y1)/2)

    return boxes, coordinates

def give_move(x, y, h, w, hp1, hp2, wp1, wp2):
    move = 0
    if y < hp1 * h:
        move = 1
    elif y < hp1 * h and x > wp2 * w:
        move = 2
    elif x > wp2 * w:
        move = 3
    elif y > hp2 * h and x > wp2 * w:
        move = 4
    elif y > hp2 * h:
        move = 5
    elif y > hp2 * h and x < wp1 * w:
        move = 6
    elif x < wp1 * w:
        move = 7
    elif y < hp1 * h and x < wp1 * w:
        move = 8
    else:
        move = 0
    return move



model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture('assets/insane 4k.mp4')

moment = 1
target_id = None
class_id = -1

while(class_id < 0):
    print(list(model.names.values()))
    target = input("What object would you like to track (all possible above): ")
    class_ids = [k for k, v in model.names.items() if v == target]
    if class_ids:
        class_id = class_ids[0]
    else:
        print("WRONG OBJECT NAME!!!")

while cap.isOpened():
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    if not ret:
        break

    if moment % 2 == 0 or moment == 1:
        results = model.track(frame, classes=class_id, persist=True, verbose=False)
        if target_id is None:
            for result in results:
                if result.boxes:
                    first_id_box = next((b for b in result.boxes if b.id is not None), None)
                    if first_id_box:
                        target_id = int(first_id_box.id)
                        break
        boxes, coordinates = process_image(results, target_id)

    for x1, y1, x2, y2, label, confidence in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    hp1 = 0.25 # height parameter 1
    hp2 = 0.75 # height parameter 2
    wp1 = 0.2 # width parameter 1
    wp2 = 0.8 # width parameter 2

    if coordinates:
        x, y = coordinates[:2]
        camera_move = give_move(x, y, height, width, hp1, hp2, wp1, wp2)
        if moment % 2 == 0 or moment == 1:
            print(camera_move)

    cv2.line(frame, (int(width*wp1), 0), (int(width*wp1), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (int(width*wp2), 0), (int(width*wp2), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (0, int(height*hp1)), (width, int(height*hp1)), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (0, int(height*hp2)), (width, int(height*hp2)), color=(0, 255, 0), thickness=2)

    cv2.imshow("Frame", frame)

    moment += 1

    if cv2.waitKey(int(1)) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()