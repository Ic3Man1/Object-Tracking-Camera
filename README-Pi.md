# Object Tracking Camera - Raspberry Pi 4 Deployment

## 📋 Wymagania

- Raspberry Pi 4 (minimum 4GB RAM zalecane)
- Raspberry Pi OS (64-bit zalecane)
- Kamera USB lub kamera Pi
- Docker i Docker Compose

## 🚀 Szybkie uruchomienie

### Metoda 1: Automatyczny skrypt
```bash
chmod +x run-on-pi.sh
./run-on-pi.sh
```

### Metoda 2: Manualne uruchomienie

1. **Zainstaluj Docker:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

2. **Zainstaluj Docker Compose:**
```bash
sudo pip3 install docker-compose
```

3. **Zbuduj obraz:**
```bash
docker build -t object-tracking-camera:latest .
```

4. **Uruchom z Docker Compose:**
```bash
docker-compose up -d
```

## 🔧 Konfiguracja kamery

### Kamera USB
Domyślnie konfiguracja używa `/dev/video0`. Jeśli twoja kamera ma inny identyfikator:

```bash
# Sprawdź dostępne kamery
ls /dev/video*

# Edytuj docker-compose.yml i zmień urządzenie
nano docker-compose.yml
```

### Kamera Pi (CSI)
Dla kamery Pi dodaj do `docker-compose.yml`:
```yaml
devices:
  - /dev/video0:/dev/video0
  - /dev/vchiq:/dev/vchiq
```

## 📊 Monitorowanie

### Sprawdzenie statusu
```bash
docker-compose ps
```

### Logi w czasie rzeczywistym
```bash
docker-compose logs -f
```

### Statystyki zasobów
```bash
docker stats object-tracking-camera
```

## 🛠 Optymalizacja dla Raspberry Pi

### Wydajność
- Obraz używa wersji CPU PyTorch (lepsze dla ARM)
- OpenCV w wersji headless (mniej zasobów)
- Ograniczenia pamięci i CPU w docker-compose.yml

### Monitoring temperatury
```bash
# Sprawdź temperaturę Pi
vcgencmd measure_temp

# Sprawdź throttling
vcgencmd get_throttled
```

### GPU (opcjonalne)
Jeśli chcesz użyć GPU Pi (VideoCore):
```bash
# Dodaj do docker-compose.yml w sekcji devices:
- /dev/vchiq:/dev/vchiq
- /dev/vcsm-cma:/dev/vcsm-cma
```

## 🔄 Zarządzanie

### Restart kontejnera
```bash
docker-compose restart
```

### Zatrzymanie
```bash
docker-compose down
```

### Aktualizacja
```bash
docker-compose down
docker build -t object-tracking-camera:latest .
docker-compose up -d
```

### Backup konfiguracji
```bash
tar -czf backup-$(date +%Y%m%d).tar.gz *.py *.yml *.txt models/ output/
```

## 🐛 Rozwiązywanie problemów

### Brak dostępu do kamery
```bash
# Sprawdź uprawnienia
ls -la /dev/video*
sudo usermod -aG video $USER

# Restart Docker
sudo systemctl restart docker
```

### Problemy z pamięcią
```bash
# Zwiększ swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Problemy z GUI (jeśli używasz)
```bash
# Pozwól na połączenia X11
xhost +local:docker
export DISPLAY=:0
```

## 📁 Struktura katalogów

```
Object-Tracking-Camera/
├── Dockerfile              # Konfiguracja kontenera
├── docker-compose.yml      # Orchestracja
├── requirements.txt        # Zależności Python
├── yolo.py                 # Główna aplikacja
├── camera_position.py      # Pozycjonowanie kamery
├── run-on-pi.sh           # Skrypt automatycznego uruchamiania
├── models/                # Modele YOLO
├── output/                # Wyniki i logi
└── assets/                # Pliki testowe
```

## ⚡ Wskazówki wydajnościowe

1. **Rozdzielczość**: Użyj niższej rozdzielczości dla lepszej wydajności
2. **FPS**: Ogranicz FPS w kodzie do 15-20 dla Pi 4
3. **Model**: Rozważ yolov8n.pt (nano) zamiast większych modeli
4. **Cooling**: Zapewnij odpowiednie chłodzenie Pi

## 📝 Notatki

- Pierwszy build może potrwać 20-30 minut na Pi 4
- Modele YOLO będą pobrane automatycznie przy pierwszym uruchomieniu
- Kontener automatycznie restartuje się po rebootcie Pi
