import cv2
import time
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Load model YOLOv8
model = YOLO("yolov8.pt")

# Định nghĩa khoảng cách thực tế (ví dụ: 5 mét giữa hai điểm đã biết trong khung hình)
real_distance = 5  # Đơn vị: mét


# Hàm để tính tốc độ di chuyển
def calculate_speed(distance_in_pixels, time_elapsed, pixel_distance):
    # Chuyển đổi khoảng cách pixel thành mét
    distance_in_meters = (distance_in_pixels / pixel_distance) * real_distance
    # Tính tốc độ (mét/giây)
    speed = distance_in_meters / time_elapsed
    return speed


# Hàm xử lý video
def process_frame(frame, previous_positions, pixel_distance):
    results = model(frame)
    current_time = time.time()
    speed_info = []

    for obj in results[0].boxes:
        cls_id = obj.cls.cpu().numpy()[0]
        bbox = obj.xyxy.cpu().numpy()[0]

        # Giả sử chỉ quan tâm đến xe ô tô (car), cls_id có thể thay đổi tùy vào việc model của bạn nhận diện loại phương tiện nào
        if cls_id == 2:  # ID của xe ô tô trong COCO dataset là 2
            # Tính toán tâm của bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Nếu phương tiện này đã được theo dõi từ trước
            if cls_id in previous_positions:
                previous_x, previous_y, previous_time = previous_positions[cls_id]
                distance_in_pixels = ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5
                time_elapsed = current_time - previous_time

                # Tính tốc độ
                speed = calculate_speed(distance_in_pixels, time_elapsed, pixel_distance)
                speed_info.append(f"Tốc độ của phương tiện ID {cls_id}: {speed:.2f} m/s")

            # Cập nhật vị trí và thời gian trước đó
            previous_positions[cls_id] = (center_x, center_y, current_time)

    annotated_frame = results[0].plot()
    return annotated_frame, speed_info


# Streamlit phần giao diện
st.title("Vehicle Detection and Speed Calculation")

# Biến lưu trữ vị trí trước đó của các phương tiện
previous_positions = {}
pixel_distance = 100  # Khoảng cách giữa hai điểm trong ảnh tính bằng pixel, cần được tính toán

# Tạo nút bắt đầu và dừng
start_button = st.button("Bắt đầu")
stop_button = st.button("Dừng")

if start_button:
    # Đọc video từ webcam
    cap = cv2.VideoCapture(0)
    frame_window = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Không thể lấy dữ liệu từ camera.")
            break

        # Xử lý khung hình
        annotated_frame, speed_info = process_frame(frame, previous_positions, pixel_distance)

        # Hiển thị khung hình đã được chú thích
        frame_window.image(annotated_frame, channels="BGR")

        # Hiển thị thông tin tốc độ
        for info in speed_info:
            st.write(info)

        # Kiểm tra nút "Dừng"
        if stop_button:
            break

    # Giải phóng tài nguyên
    cap.release()
