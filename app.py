import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array


# Cấu hình customtkinter
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

# Tạo cửa sổ chính
window = ctk.CTk()
window.geometry('850x550+500+150')
window.title("AI PROJECT - NHẬN DIỆN CẢM XÚC KHUÔN MẶT")
window.iconbitmap('logo/icon.ico')
window.resizable(width=False, height=False)

# Các nhãn tiêu đề
title_frame = ctk.CTkFrame(window, corner_radius=15)
title_frame.pack(pady=10, fill="x")

title_label = ctk.CTkLabel(title_frame, text="PROJECT MÔN CƠ SỞ VÀ ỨNG DỤNG AI", 
                           font=("Arial", 30, "bold"), text_color="#001949")
title_label.pack(pady=5)

subtitle_label = ctk.CTkLabel(title_frame, text="ĐỀ TÀI: NHẬN DIỆN CẢM XÚC KHUÔN MẶT",
                              font=("Arial", 28, "bold"), text_color="#001949")
subtitle_label.pack()

group_label = ctk.CTkLabel(title_frame, text="NHÓM 14", font=("Arial", 28, "bold"), text_color="#001949")
group_label.pack(pady=5)

# Logo
img = Image.open("logo/logo.png")
logo_rz = img.resize((105, 120))
logo = ImageTk.PhotoImage(logo_rz)
logo_label = ctk.CTkLabel(window, image=logo, text="")
logo_label.place(relx=0.01, rely=0.02, anchor="nw")

# Load mô hình
model = model_from_json(open("model/model_arch.json", "r").read())
model.load_weights("model/my_model.h5")
face_haar_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Hàm Import File Ảnh Và Nhận Diện
def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        res, frame = cap.read()
        height, width, channel = frame.shape
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        try: 
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                image_pixels = img_to_array(roi_gray)
                image_pixels = np.expand_dims(image_pixels, axis=0)
                image_pixels /= 255
                predictions = model.predict(image_pixels)
                max_index = np.argmax(predictions[0])
                emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(frame, emotion_prediction, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (0, 255, 0), 1, cv2.LINE_AA)
        except:
            pass
        frame[0:int(height/1000), 0:int(width)] = res
        cv2.imshow('AI PROJECT - NHAN DIEN CAM XUC KHUON MAT', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows() 

# Hàm Mở Camera Và Nhận Diện
def detect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        res, frame = cap.read()
        if not res:
            break
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(frame, emotion_prediction, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        cv2.imshow('AI PROJECT - NHAN DIEN CAM XUC KHUON MAT', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Hàm Nhận Diện Và Ghi Video
def detectRec():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    op = cv2.VideoWriter('videoluulai.mp4', fourcc, 9.0, (640, 480))
    while cap.isOpened():
        res, frame = cap.read()
        if not res:
            break
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
            roi_gray = gray_image[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('BinhThuong', 'Buon', 'GianDu', 'NgacNhien', 'VuiVe')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(frame, emotion_prediction, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        op.write(frame)
        cv2.imshow('AI PROJECT - NHAN DIEN CAM XUC KHUON MAT - DETECT & RECORD', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    op.release()
    cap.release()
    cv2.destroyAllWindows()

# Hàm thoát chương trình
def exitt():
    window.destroy()

# Các nút chức năng
button_frame = ctk.CTkFrame(window, corner_radius=15)
button_frame.pack(pady=20, fill="x")

# Điều chỉnh kích thước và font các nút
button_size = {
    'width': 800,
    'height': 75,
    'font': ("Arial", 20, "bold")
}

button1 = ctk.CTkButton(button_frame, text="ĐƯA FILE VÀO ĐỂ NHẬN DIỆN", 
                        command=UploadAction, **button_size)
button1.pack(pady=10)

button2 = ctk.CTkButton(button_frame, text="NHẬN DIỆN TRỰC TIẾP BẰNG CAMERA", 
                        command=detect, **button_size)
button2.pack(pady=10)

button3 = ctk.CTkButton(button_frame, text="NHẬN DIỆN TRỰC TIẾP VÀ LƯU VIDEO LẠI", 
                        command=detectRec, **button_size)
button3.pack(pady=10)

button4 = ctk.CTkButton(button_frame, text="THOÁT", command=exitt, 
                        **button_size, fg_color="red")
button4.pack(pady=10)

window.mainloop()