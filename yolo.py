import cv2
import torch
from ultralytics import YOLO
import time
import math

def process_image(results, target_id, frame_width, frame_height):
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
                    
                    # Środek obiektu
                    object_center_x = (x2 + x1) / 2
                    object_center_y = (y2 + y1) / 2
                    
                    # Środek obrazu
                    frame_center_x = frame_width / 2
                    frame_center_y = frame_height / 2
                    
                    # Odległości od środka obrazu
                    distance_x = object_center_x - frame_center_x
                    distance_y = object_center_y - frame_center_y
                    
                    print(f"ID {target_id} - Odległość X: {distance_x:.1f}px, Odległość Y: {distance_y:.1f}px")
                    coordinates = (object_center_x, object_center_y)

    return boxes, coordinates

def give_move(x, y, h, w, hp1, hp2, wp1, wp2):
    if y < hp1 * h and x < wp1 * w:
        return 8  # lewy górny róg
    elif y < hp1 * h and x > wp2 * w:
        return 2  # prawy górny róg
    elif y > hp2 * h and x > wp2 * w:
        return 4  # prawy dolny róg
    elif y > hp2 * h and x < wp1 * w:
        return 6  # lewy dolny róg
    elif x < wp1 * w:
        return 7  # lewa krawędź
    elif y < hp1 * h:
        return 1  # górna krawędź
    elif y > hp2 * h:
        return 5  # dolna krawędź
    elif x > wp2 * w:
        return 3  # prawa krawędź
    else:
        return 0  # środek

def get_gps_coordinates(prompt):
    """Pobiera współrzędne GPS od użytkownika"""
    while True:
        try:
            coords = input(prompt)
            lat, lon = map(float, coords.split(','))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
            else:
                print("Błędne współrzędne! Szerokość geograficzna: -90 do 90, Długość geograficzna: -180 do 180")
        except ValueError:
            print("Błędny format! Wprowadź współrzędne w formacie: szerokość,długość (np. 52.2297,21.0122)")

def get_camera_orientation():
    """Pobiera orientację kamery od użytkownika"""
    orientations = {
        '1': ('N', 0),     # Północ
        '2': ('NE', 45),   # Północny wschód  
        '3': ('E', 90),    # Wschód
        '4': ('SE', 135),  # Południowy wschód
        '5': ('S', 180),   # Południe
        '6': ('SW', 225),  # Południowy zachód
        '7': ('W', 270),   # Zachód
        '8': ('NW', 315)   # Północny zachód
    }
    
    print("\nWybierz kierunek, w którym jest skierowana kamera:")
    print("1. Północ (N)")
    print("2. Północny wschód (NE)")
    print("3. Wschód (E)")
    print("4. Południowy wschód (SE)")
    print("5. Południe (S)")
    print("6. Południowy zachód (SW)")
    print("7. Zachód (W)")
    print("8. Północny zachód (NW)")
    
    while True:
        choice = input("Wprowadź numer opcji (1-8): ")
        if choice in orientations:
            direction, angle = orientations[choice]
            print(f"Wybrano: {direction} ({angle}°)")
            return angle
        else:
            print("Błędny wybór! Wprowadź numer od 1 do 8.")

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Oblicza kąt między dwoma punktami GPS (bearing)"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360  # Normalizacja do 0-360
    
    return bearing

def setup_gps_tracking():
    """Konfiguruje śledzenie GPS i oblicza wymagany kąt obrotu kamery"""
    print("=== KONFIGURACJA GPS KAMERY ===")
    
    # Pobierz współrzędne kamery
    camera_lat, camera_lon = get_gps_coordinates(
        "Wprowadź współrzędne GPS kamery (szerokość,długość): "
    )
    print(f"Pozycja kamery: {camera_lat}, {camera_lon}")
    
    # Pobierz orientację kamery
    camera_orientation = get_camera_orientation()
    
    # Pobierz współrzędne obiektu do śledzenia
    target_lat, target_lon = get_gps_coordinates(
        "Wprowadź współrzędne GPS obiektu do śledzenia (szerokość,długość): "
    )
    print(f"Pozycja obiektu: {target_lat}, {target_lon}")
    
    # Oblicz kąt do obiektu
    target_bearing = calculate_bearing(camera_lat, camera_lon, target_lat, target_lon)
    
    # Oblicz wymagany kąt obrotu
    rotation_angle = target_bearing - camera_orientation
    
    # Normalizacja kąta do zakresu -180 do 180
    if rotation_angle > 180:
        rotation_angle -= 360
    elif rotation_angle < -180:
        rotation_angle += 360
    
    print(f"\n=== WYNIKI OBLICZEŃ ===")
    print(f"Kierunek do obiektu: {target_bearing:.1f}°")
    print(f"Aktualna orientacja kamery: {camera_orientation}°")
    print(f"Wymagany obrót kamery: {rotation_angle:.1f}°")
    
    if rotation_angle > 0:
        print(f"Obróć kamerę w PRAWO o {rotation_angle:.1f}°")
    elif rotation_angle < 0:
        print(f"Obróć kamerę w LEWO o {abs(rotation_angle):.1f}°")
    else:
        print("Kamera jest już skierowana na obiekt!")
    
    # Potwierdzenie ustawienia kamery
    while True:
        confirmation = input("\nCzy kamera została już ustawiona zgodnie z wyliczeniami? (tak/nie): ").lower()
        if confirmation in ['tak', 't', 'yes', 'y']:
            print("Uruchamiam śledzenie obiektów...")
            return True
        elif confirmation in ['nie', 'n', 'no']:
            print("Proszę ustawić kamerę i uruchomić program ponownie.")
            return False
        else:
            print("Odpowiedz 'tak' lub 'nie'")


# Konfiguracja GPS - uruchom przed główną logiką
if not setup_gps_tracking():
    exit()

model = YOLO("yolov8n.pt").to('cpu')

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

rtsp_url = 'rtsp://admin:admin123@192.168.5.190:554/main'  # nieużywane
# cap = cv2.VideoCapture(rtsp_url) # 'assets/insane 4k.mp4'
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# cap = cv2.VideoCapture(gst, cv2.CAP_FFMPEG)

# AKTYWNE: Użycie kamery laptopa
cap = cv2.VideoCapture(0)  # 0 = domyślna kamera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(2)

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
        boxes, coordinates = process_image(results, target_id, width, height)

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