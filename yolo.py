import cv2
import torch
from ultralytics import YOLO
import time

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
                    print(f"ID {target_id} coords ({(x2+x1)/2}, {(y2+y1)/2})")
                    coordinates = ((x2+x1)/2, (y2+y1)/2)

    return boxes, coordinates

# def give_move(x, y, h, w, hp1, hp2, wp1, wp2):
#     if y < hp1 * h and x < wp1 * w:
#         print("Lewy gorny")
#         return 8  # lewy górny róg
#     elif y < hp1 * h and x > wp2 * w:
#         print("Prawy gorny")
#         return 2  # prawy górny róg
#     elif y > hp2 * h and x > wp2 * w:
#         print("Prawy dolny")
#         return 4  # prawy dolny róg
#     elif y > hp2 * h and x < wp1 * w:
#         print("Lewy dolny")
#         return 6  # lewy dolny róg
#     elif x < wp1 * w:
#         print("Lewy")
#         return 7  # lewa krawędź
#     elif y < hp1 * h:
#         print("Gora")
#         return 1  # górna krawędź
#     elif y > hp2 * h:
#         print("Dol")
#         return 5  # dolna krawędź
#     elif x > wp2 * w:
#         print("Prawo")
#         return 3  # prawa krawędź
#     else:
#         print("Srodek")
#         return 0  # środek
    
    
def give_move_horizontal(x, y, h, w, wp1, wp2, wp3, wp4, wp5, wp6):
    if x < wp1 * w:
        print("Mocno w prawo")
        return 1
    elif x < wp2 * w:
        print("W prawo")
        return 2
    elif x < wp3 * w:
        print("Troche w prawo")
        return 3   
    elif x < wp4 * w:
        print("Srodek")
        return 0
    elif x < wp5 * w:
        print("Troche w lewo")
        return 4
    elif x < wp6 * w:
        print("W lewo")
        return 5
    elif x >= wp6 * w:
        print("Mocno w lewo")
        return 6



model = YOLO("yolov8n.pt").to('cpu') # ma byc CPU TUAJ CHYBA ALE NA PC MI SIE NIE CHCIALO POBIERAC

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

# rtsp_url = 'rtsp://admin:admin123@192.168.5.190:554/main'

cap = cv2.VideoCapture('assets/face.mp4') # rtsp_url # 'assets/insane 4k.mp4'
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#cap = cv2.VideoCapture(gst, cv2.CAP_FFMPEG)
#time.sleep(2)

while cap.isOpened():
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    if not ret:
        break

    if moment % 10 == 0 or moment == 1:
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
    # horizontal and vetical parameters
    hp1 = 0.25 # height parameter 1
    hp2 = 0.75 # height parameter 2
    wp1 = 0.2 # width parameter 1
    wp2 = 0.8 # width parameter 2

    # only horizontal parameters
    hp1 = 1/7 # width parameter 1
    hp2 = 2/7 # width parameter 2
    hp3 = 3/7 # width parameter 3
    hp4 = 4/7 # width parameter 4
    hp5 = 5/7 # width parameter 5
    hp6 = 6/7 # width parameter 6


    if coordinates:
        x, y = coordinates[:2]
        # camera_move = give_move(x, y, height, width, hp1, hp2, wp1, wp2)
        camera_move = give_move_horizontal(x, y, height, width, hp1, hp2, hp3, hp4, hp5, hp6)
        if moment % 5 == 0 or moment == 1:
            print(camera_move)

    # DRAW HORIZONTAL AND VERTICAL LINES
    # cv2.line(frame, (int(width*wp1), 0), (int(width*wp1), height), color=(0, 255, 0), thickness=2)
    # cv2.line(frame, (int(width*wp2), 0), (int(width*wp2), height), color=(0, 255, 0), thickness=2)
    # cv2.line(frame, (0, int(height*hp1)), (width, int(height*hp1)), color=(0, 255, 0), thickness=2)
    # cv2.line(frame, (0, int(height*hp2)), (width, int(height*hp2)), color=(0, 255, 0), thickness=2)

    # DrAW ONLY HORIZONTAL LINES
    cv2.line(frame, (int(width*hp1), 0), (int(width*hp1), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (int(width*hp2), 0), (int(width*hp2), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (int(width*hp3), 0), (int(width*hp3), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (int(width*hp4), 0), (int(width*hp4), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (int(width*hp5), 0), (int(width*hp5), height), color=(0, 255, 0), thickness=2)
    cv2.line(frame, (int(width*hp6), 0), (int(width*hp6), height), color=(0, 255, 0), thickness=2) 

    cv2.imshow("Frame", frame)

    moment += 1

    if cv2.waitKey(int(1)) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()