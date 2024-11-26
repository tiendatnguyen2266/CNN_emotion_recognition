import cv2
import os

video_folder = '/content/gdrive/MyDrive/Project_CSUD_AI/input_dataset'
dataset_folder = '/content/gdrive/MyDrive/Project_CSUD_AI/dataset'

# Định dạng nhãn của các video
labels = ['BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe']
# Số lượng ảnh cho train và test
num_train_images = 20
num_test_images = 10

# Tạo các thư mục train và test trong dataset_folder nếu chưa tồn tại
for label in labels:
    train_folder = os.path.join(dataset_folder, 'train', label)
    test_folder = os.path.join(dataset_folder, 'test', label)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

# Load model phát hiện khuôn mặt của OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm để trích xuất ảnh từ video và nhận diện khuôn mặt
def extract_faces_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Tính toán khoảng cách giữa các khung để đạt đủ số lượng ảnh cho train và test
    interval = max(total_frames // (num_train_images + num_test_images), 1)

    count = 0
    saved_train_images = 0
    saved_test_images = 0

    while cap.isOpened() and (saved_train_images < num_train_images or saved_test_images < num_test_images):
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển khung hình sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Duyệt qua từng khuôn mặt được phát hiện
        for (x, y, w, h) in faces:
            # Cắt khuôn mặt và thay đổi kích thước
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))

            # Lưu ảnh cho thư mục train
            if saved_train_images < num_train_images and count % interval == 0:
                train_image_path = os.path.join(dataset_folder, 'train', label, f"{os.path.basename(video_path).split('.')[0]}_{saved_train_images}.jpg")
                cv2.imwrite(train_image_path, face_resized)
                saved_train_images += 1

            # Lưu ảnh cho thư mục test
            if saved_test_images < num_test_images and count % interval == 0:
                test_image_path = os.path.join(dataset_folder, 'test', label, f"{os.path.basename(video_path).split('.')[0]}_{saved_test_images}.jpg")
                cv2.imwrite(test_image_path, face_resized)
                saved_test_images += 1

        count += 1

    cap.release()
    print(f"Đã lưu {saved_train_images} ảnh vào thư mục train và {saved_test_images} ảnh vào thư mục test từ video {video_path} vào nhãn {label}.")

# Lặp qua từng nhãn và video, trích xuất ảnh
for label in labels:
    label_videos_folder = os.path.join(video_folder, label)

    # Kiểm tra xem thư mục chứa video có tồn tại không
    if not os.path.exists(label_videos_folder):
        print(f"Thư mục {label_videos_folder} không tồn tại. Vui lòng kiểm tra lại.")
        continue

    # Lấy danh sách các video trong thư mục
    videos = [v for v in os.listdir(label_videos_folder) if v.endswith(('.mp4', '.avi', '.mov'))]

    # Kiểm tra nếu không có video nào trong thư mục
    if len(videos) == 0:
        print(f"Không tìm thấy video nào trong thư mục {label_videos_folder}.")
        continue

    # Trích xuất ảnh từ mỗi video
    for video in videos:
        video_path = os.path.join(label_videos_folder, video)
        extract_faces_from_video(video_path, label)
