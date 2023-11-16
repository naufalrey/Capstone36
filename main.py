import cv2
from object_detection import ObjectDetection
import firebase_admin
from firebase_admin import credentials, db
import schedule
from datetime import datetime

# Inisialisasi Firebase
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://testta-4e343-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Initialize Object Detection
od = ObjectDetection()

# Set up camera capture object
cap = cv2.VideoCapture('3x3.mp4')
frame_rate = 24.0  # Desired frame rate (24 frames per second)
cap.set(cv2.CAP_PROP_FPS, frame_rate)

# Inisialisasi penghitung kotak dan jumlah mobil
count = 0

# Inisialisasi nama lantai
lantai_names = ["1A1", "1A2", "1B1", "1B2", "2A1", "2A2", "G1", "G2", "Kosong"]

lantai_data = {
    "1A": 0,
    "1B": 0,
    "2A": 0,
    "GROUND": 0
}

def kirim_data_ke_firebase():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ref = db.reference('kapasitas_parkir')
    ref.set(lantai_data) 
    print("Data terkirim ke Firebase Realtime Database dengan timestamp:", timestamp)

# Pengaturan jadwal pengiriman data ke Firebase
schedule.every(5).seconds.do(kirim_data_ke_firebase)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Hitung ukuran frame
    height, width, _ = frame.shape

    # Bagi frame menjadi kotak-kotak 3x3
    kotak_height = height // 3
    kotak_width = width // 3

    mobil_count_1A = 0
    mobil_count_1B = 0
    mobil_count_2A = 0
    mobil_count_Ground = 0

    for row in range(3):
        for col in range(3):
            kotak_x1 = col * kotak_width
            kotak_x2 = (col + 1) * kotak_width
            kotak_y1 = row * kotak_height
            kotak_y2 = (row + 1) * kotak_height

            # Ambil kotak
            kotak_frame = frame[kotak_y1:kotak_y2, kotak_x1:kotak_x2]

            # Deteksi mobil dalam kotak
            (class_ids, scores, boxes) = od.detect(kotak_frame)

            # Menghitung jumlah mobil dalam kotak
            mobil_count = 0
            for score in scores:
                if score > 0.4:  # Ambang batas confidence
                    mobil_count += 1

            # Update data jumlah mobil tiap lantai
            lantai_name = lantai_names[row * 3 + col]
            if lantai_name in ["1A1", "1A2"]:
                mobil_count_1A += mobil_count
            elif lantai_name in ["1B1", "1B2"]:
                mobil_count_1B += mobil_count
            elif lantai_name in ["2A1", "2A2"]:
                mobil_count_2A += mobil_count
            elif lantai_name in ["G1", "G2"]:
                mobil_count_Ground += mobil_count

            # Gambar kotak deteksi
            cv2.rectangle(frame, (kotak_x1, kotak_y1), (kotak_x2, kotak_y2), (0, 255, 0), 2)

            # Tampilkan jumlah mobil di bagian bawah kotak
            kotak_center_x = (kotak_x1 + kotak_x2) // 2
            kotak_bottom_y = kotak_y2 - 10  # Menggeser teks ke bagian bawah kotak
            cv2.putText(frame, f'Lantai {lantai_name}: {mobil_count}', (kotak_center_x - 50, kotak_bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    count += 1

    # Update data jumlah mobil per lantai
    lantai_data["1A"] = 30 - mobil_count_1A
    lantai_data["1B"] = 30 - mobil_count_1B
    lantai_data["2A"] = 30 - mobil_count_2A
    lantai_data["GROUND"] = 30 - mobil_count_Ground

    # Tampilkan frame
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    schedule.run_pending()
    #cv2.imshow("Deteksi Mobil", frame)

cap.release()
cv2.destroyAllWindows()
