#!/usr/bin/env python3
import cv2
import torch
from ultralytics import YOLO
import time
import math
import numpy as np

# >>> NET: prosty broadcaster TCP (serwer na laptopie)
import socket, threading, json, time as _time
class ErrorBroadcaster:
    def __init__(self, host="0.0.0.0", port=5005):
        self.host = host
        self.port = port
        self.clients = set()
        self.lock = threading.Lock()
        self._srv_thread = threading.Thread(target=self._serve, daemon=True)
        self._srv_thread.start()
        print(f"[NET] Serwer uruchomiony na {host}:{port}")

    def _serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(10)
            while True:
                conn, addr = srv.accept()
                print(f"[NET] Klient połączony: {addr}")
                with self.lock:
                    self.clients.add(conn)
                threading.Thread(target=self._client_loop, args=(conn, addr), daemon=True).start()

    def _client_loop(self, conn, addr):
        try:
            # powitanie (opcjonalne)
            self._send_line(conn, json.dumps({"status": "connected"}))
            # nie czytamy danych — utrzymujemy połączenie do rozłączenia
            while True:
                data = conn.recv(1)
                if not data:
                    break
        except Exception:
            pass
        finally:
            with self.lock:
                if conn in self.clients:
                    self.clients.remove(conn)
            try:
                conn.close()
            except Exception:
                pass
            print(f"[NET] Klient rozłączony: {addr}")

    def _send_line(self, conn, line: str):
        conn.sendall((line + "\n").encode("utf-8"))

    def publish(self, error_x: float):
        msg = json.dumps({"error_x": float(error_x), "ts": _time.time()})
        print(f"[NET] Publikuję: {msg}")
        dead = []
        with self.lock:
            for c in list(self.clients):
                try:
                    self._send_line(c, msg)
                except Exception:
                    dead.append(c)
            for d in dead:
                try:
                    d.close()
                except Exception:
                    pass
                self.clients.discard(d)

# >>> NET: start serwera na porcie 5005
broadcaster = ErrorBroadcaster(host="0.0.0.0", port=5005)

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
                    coordinates = (object_center_x, object_center_y, distance_x, distance_y)

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
    camera_lat, camera_lon = get_gps_coordinates("Wprowadź współrzędne GPS kamery (szerokość,długość): ")
    print(f"Pozycja kamery: {camera_lat}, {camera_lon}")
    camera_orientation = get_camera_orientation()
    target_lat, target_lon = get_gps_coordinates("Wprowadź współrzędne GPS obiektu do śledzenia (szerokość,długość): ")
    print(f"Pozycja obiektu: {target_lat}, {target_lon}")
    target_bearing = calculate_bearing(camera_lat, camera_lon, target_lat, target_lon)
    rotation_angle = target_bearing - camera_orientation
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

def draw_arrow(img, start_point, end_point, color=(0, 0, 255), thickness=3, arrow_length=20):
    cv2.line(img, start_point, end_point, color, thickness)
    angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    arrow_angle = math.pi / 6  # 30 stopni
    x1 = int(end_point[0] - arrow_length * math.cos(angle - arrow_angle))
    y1 = int(end_point[1] - arrow_length * math.sin(angle - arrow_angle))
    x2 = int(end_point[0] - arrow_length * math.cos(angle + arrow_angle))
    y2 = int(end_point[1] - arrow_length * math.sin(angle + arrow_angle))
    cv2.line(img, end_point, (x1, y1), color, thickness)
    cv2.line(img, end_point, (x2, y2), color, thickness)

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
cap = cv2.VideoCapture(rtsp_url) # 'assets/insane 4k.mp4'
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
time.sleep(2)

boxes = []  # >>> ważne: zainicjalizuj boxes, by rysowanie nie sypało się przed pierwszym 'results'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]

    if moment % 10 == 0 or moment == 1:
        results = model.track(frame, classes=class_id, persist=True, verbose=False)
        # if target_id is None:
        if results is not None:
            for result in results:
                if result.boxes:
                    first_id_box = next((b for b in result.boxes if b.id is not None), None)
                    # if first_id_box:
                    #     target_id = int(first_id_box.id)
                    #     break
                    if first_id_box is not None:
                        boxes, coordinates = process_image(results, int(first_id_box.id), width, height)

    if boxes:
        x1, y1, x2, y2, label, confidence = boxes[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Środek okna
        frame_center_x = width // 2
        frame_center_y = height // 2
        
        # Krzyżyk w środku
        cv2.drawMarker(frame, (frame_center_x, frame_center_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 3)

        hp1, hp2 = 0.25, 0.75
        wp1, wp2 = 0.2, 0.8

        if coordinates and moment % 10 == 0:
            _, _, distance_x, _ = coordinates
            broadcaster.publish(distance_x)

        # Linie pomocnicze
        cv2.line(frame, (int(width*wp1), 0), (int(width*wp1), height), color=(0, 255, 0), thickness=2)
        cv2.line(frame, (int(width*wp2), 0), (int(width*wp2), height), color=(0, 255, 0), thickness=2)
        cv2.line(frame, (0, int(height*hp1)), (width, int(height*hp1)), color=(0, 255, 0), thickness=2)
        cv2.line(frame, (0, int(height*hp2)), (width, int(height*hp2)), color=(0, 255, 0), thickness=2)

    cv2.imshow("Frame", frame)
    moment += 1
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
