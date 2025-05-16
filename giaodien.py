import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
yolo_model = YOLO("/kaggle/input/code-cnn-va-yolov8/yolov8trained.pt")

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="/kaggle/input/code-cnn-va-yolov8/code_cnn_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Danh sách nhãn món ăn
class_names = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com trang", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

# Bảng giá
food_prices = {
    "Ca hu kho": 10000,
    "Canh cai": 8000,
    "Canh chua": 8000,
    "Com trang": 5000,
    "Dau hu sot ca": 7000,
    "Ga chien": 12000,
    "Rau muong xao": 6000,
    "Thit kho": 12000,
    "Thit kho trung": 14000,
    "Trung chien": 7000
}

def classify_image(image):
    results = yolo_model(image)
    detections = results[0].boxes.data.cpu().numpy()

    predicted_classes = set()

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        class_id = int(class_id)
        if score < 0.3:
            continue

        crop = image[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (224, 224))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(output_data))
        predicted_label = class_names[predicted_index]

        predicted_classes.add(predicted_label)

    if not predicted_classes:
        return "⚠️ Không phát hiện được món ăn", "0đ"

    result_text = "\n".join([
        f"🍽️ {food}: {food_prices[food]:,}đ" for food in sorted(predicted_classes)
    ])
    total_price = sum([food_prices[food] for food in predicted_classes])
    return result_text, f"💰 Tổng cộng: {total_price:,}đ"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <h1 style='text-align: center;'>🍽️ Hệ Thống Tự Động Nhận Diện Món Ăn & Tính Tiền</h1>
    <p style='text-align: center;'>Ứng dụng YOLOv8 kết hợp CNN để phát hiện và phân loại món ăn từ ảnh khay cơm. Hiển thị tên món, giá và tổng tiền.</p>
    """)

    with gr.Row():
        image_input = gr.Image(type="numpy", label="📸 Tải ảnh khay cơm")

    with gr.Row():
        with gr.Column():
            food_output = gr.Textbox(label="📋 Danh sách món ăn & đơn giá (VNĐ)", lines=10, interactive=False)
        with gr.Column():
            total_output = gr.Textbox(label="🧾 Tổng tiền cần thanh toán", interactive=False)

    btn = gr.Button("🚀 Phân tích ảnh & Tính tiền")
    btn.click(classify_image, inputs=image_input, outputs=[food_output, total_output])

    gr.Markdown("---")
    gr.Markdown("<p style='text-align: center;'>© 2025 - Đồ án AI | UEH - Đại học Kinh tế TP. HCM</p>")

demo.launch(debug=True)
