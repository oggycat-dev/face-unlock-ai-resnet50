# face_unlock.py
import cv2
import numpy as np
import tensorflow as tf
import json
import os
import argparse
from scipy.spatial.distance import cosine

# --- Cấu hình ---
MODEL_PATH = 'face_recognition_finetuned.h5'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DATABASE_PATH = 'user_database.json'
IMAGE_SIZE = (224, 224)

# --- 1. Tải mô hình và tạo mô hình trích xuất đặc trưng ---
print("Đang tải mô hình...")
# Tải mô hình đầy đủ đã được huấn luyện
full_model = tf.keras.models.load_model(MODEL_PATH)

# Tạo một mô hình mới chỉ để trích xuất đặc trưng (embedding)
# Chúng ta lấy output từ lớp ngay trước lớp phân loại cuối cùng
feature_extractor = tf.keras.Model(
    inputs=full_model.input,
    outputs=full_model.get_layer('global_average_pooling2d_1').output # Tên lớp có thể khác, cần kiểm tra
)

# Tải bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
print("Mô hình đã sẵn sàng.")

# --- 2. Các hàm chức năng ---

def preprocess_face(face_roi):
    """Tiền xử lý ảnh khuôn mặt để đưa vào mô hình."""
    resized_face = cv2.resize(face_roi, IMAGE_SIZE)
    normalized_face = resized_face / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    return input_face

def get_embedding(face_roi):
    """Tạo embedding từ ảnh khuôn mặt."""
    processed_face = preprocess_face(face_roi)
    embedding = feature_extractor.predict(processed_face, verbose=0)
    return embedding.flatten() # Làm phẳng vector

def load_database():
    """Tải cơ sở dữ liệu người dùng từ file JSON."""
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_database(db):
    """Lưu cơ sở dữ liệu người dùng vào file JSON."""
    with open(DATABASE_PATH, 'w') as f:
        json.dump(db, f, indent=4)

# --- 3. Chế độ Đăng ký (Enrollment) ---
def enroll_user(name):
    """Chụp ảnh và lưu embedding của người dùng mới."""
    cap = cv2.VideoCapture(0)
    embeddings = []
    capture_count = 0
    required_captures = 20 # Số lượng ảnh cần chụp

    while capture_count < required_captures:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Chỉ chụp khi khuôn mặt đủ lớn và rõ
            if w > 100 and h > 100:
                face_roi = frame[y:y+h, x:x+w]
                embedding = get_embedding(face_roi)
                embeddings.append(embedding.tolist())
                capture_count += 1
                
                # Hiển thị thông báo tiến trình
                progress_text = f"Da chup: {capture_count}/{required_captures}"
                cv2.putText(frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow(f"Dang ky - {name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == required_captures:
        # Tính embedding trung bình làm "master embedding"
        master_embedding = np.mean(embeddings, axis=0)
        
        db = load_database()
        db[name] = master_embedding.tolist()
        save_database(db)
        print(f"Da dang ky thanh cong cho nguoi dung: {name}")
    else:
        print("Dang ky that bai. Vui long thu lai.")

# --- 4. Chế độ Xác thực (Verification) ---
def verify_user():
    """Xác thực người dùng qua camera."""
    db = load_database()
    if not db:
        print("Chua co nguoi dung nao trong co so du lieu. Vui long dang ky truoc.")
        return

    cap = cv2.VideoCapture(0)
    # Ngưỡng tương đồng, có thể điều chỉnh
    # Càng nhỏ càng khắt khe
    similarity_threshold = 0.2 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            current_embedding = get_embedding(face_roi)
            
            best_match_name = "Unknown"
            min_distance = float('inf')

            # So sánh với tất cả người dùng trong CSDL
            for name, master_embedding in db.items():
                distance = cosine(current_embedding, np.array(master_embedding))
                if distance < min_distance:
                    min_distance = distance
                    best_match_name = name
            
            # Kiểm tra xem có khớp không
            if min_distance < similarity_threshold:
                display_name = best_match_name
                color = (0, 255, 0) # Xanh lá: Khớp
                # TẠI ĐÂY BẠN CÓ THỂ THÊM HÀNH ĐỘNG MỞ KHÓA
            else:
                display_name = "Unknown"
                color = (0, 0, 255) # Đỏ: Không khớp

            text = f"{display_name} (Dist: {min_distance:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Xac thuc - Nhan 'q' de thoat", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# --- 5. Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="He thong mo khoa bang khuon mat")
    parser.add_argument("--mode", type=str, required=True, choices=['enroll', 'verify'],
                        help="Chon che do: 'enroll' de dang ky, 'verify' de xac thuc.")
    parser.add_argument("--name", type=str, help="Ten nguoi dung de dang ky.")
    args = parser.parse_args()

    if args.mode == "enroll":
        if not args.name:
            print("Loi: Can cung cap ten voi co che do --mode enroll.")
        else:
            enroll_user(args.name)
    elif args.mode == "verify":
        verify_user()
