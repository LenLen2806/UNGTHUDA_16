import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from PIL import Image
import json

# Cấu hình Streamlit
st.set_page_config(page_title="Chẩn đoán Ung thư Da", layout="wide", page_icon="🧬")

# Load mô hình và ánh xạ lớp
model = tf.keras.models.load_model("ungthuda.h5")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Thông tin chi tiết về bệnh
disease_info = {
    "akiec": {
        "ten_ta": "Actinic Keratoses",
        "ten_tv": "Dày sừng quang hóa",
        "nguyennhan": "Tiếp xúc lâu dài với tia cực tím (UV).",
        "dauhieu": "Vùng da thô ráp, có vảy, sạm màu.",
        "dieutri": "Điều trị bằng laser, lạnh, kem bôi, hoặc phẫu thuật nhỏ."
    },
    "bcc": {
        "ten_ta": "Basal Cell Carcinoma",
        "ten_tv": "Ung thư biểu mô tế bào đáy",
        "nguyennhan": "Tia UV từ ánh nắng mặt trời.",
        "dauhieu": "Nốt sáp bóng, dễ chảy máu.",
        "dieutri": "Phẫu thuật, xạ trị, hoặc điều trị tại chỗ."
    },
    "bkl": {
        "ten_ta": "Benign Keratosis-like lesions",
        "ten_tv": "Tổn thương lành tính giống dày sừng",
        "nguyennhan": "Thay đổi da do lão hóa hoặc di truyền.",
        "dauhieu": "Vùng da sẫm màu, phẳng hoặc gồ nhẹ.",
        "dieutri": "Không cần điều trị, có thể loại bỏ vì thẩm mỹ."
    },
    "df": {
        "ten_ta": "Dermatofibroma",
        "ten_tv": "U xơ da",
        "nguyennhan": "Phản ứng da sau tổn thương nhỏ như vết cắn.",
        "dauhieu": "U cứng, nhỏ, màu nâu hoặc đỏ tím.",
        "dieutri": "Không cần điều trị. Có thể phẫu thuật nếu gây khó chịu."
    },
    "mel": {
        "ten_ta": "Melanoma",
        "ten_tv": "U hắc tố ác tính",
        "nguyennhan": "Tổn thương DNA tế bào hắc tố (thường do tia UV).",
        "dauhieu": "Nốt ruồi bất thường về màu, hình dạng, kích thước.",
        "dieutri": "Phẫu thuật, điều trị miễn dịch, hóa trị. Phát hiện sớm rất quan trọng."
    },
    "nv": {
        "ten_ta": "Melanocytic Nevi",
        "ten_tv": "Nốt ruồi sắc tố",
        "nguyennhan": "Tăng sinh tế bào hắc tố lành tính.",
        "dauhieu": "Nốt tròn nhỏ, màu nâu hoặc đen, đối xứng.",
        "dieutri": "Không cần điều trị. Theo dõi nếu có thay đổi bất thường."
    },
    "vasc": {
        "ten_ta": "Vascular lesions",
        "ten_tv": "Tổn thương mạch máu",
        "nguyennhan": "Bất thường mạch máu bẩm sinh hoặc mắc phải.",
        "dauhieu": "Bớt đỏ, u máu, da đổi màu.",
        "dieutri": "Điều trị bằng laser hoặc theo dõi không can thiệp nếu lành tính."
    }
}

# Kiểm tra ảnh có phải da người không
def is_skin_image(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    skin_mask = cv2.inRange(img, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = img.shape[0] * img.shape[1]
    return skin_pixels / total_pixels > 0.2

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/9716/9716586.png", width=150)


st.sidebar.title("📤Tải ảnh vùng da")
method = st.sidebar.radio("Chọn phương thức:", ["📁 Tải ảnh", "📷 Chụp ảnh"])
img_file = st.sidebar.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"]) if method == "📁 Tải ảnh" else st.sidebar.camera_input("Chụp ảnh")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:16px; line-height:1.6;'>
    📌 <b>Định dạng hỗ trợ:</b> <code>.jpg</code>, <code>.png</code>, <code>.jpeg</code><br>
    🧠 <b>Mô hình:</b> MobileNetV2<br>
    👨‍🏫 Dành cho mục đích nghiên cứu và giáo dục.
</div>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown("<h1 style='text-align: center; color: #155724;'>🔬 Hệ thống hỗ trợ chẩn đoán Ung thư Da</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chọn ảnh để hệ thống AI phân tích và hiển thị thông tin bệnh</p>", unsafe_allow_html=True)
st.markdown("---")

if img_file:
    img = Image.open(img_file).convert("RGB")
    
    # Dự đoán trước để chuẩn bị thông tin
    # if not is_skin_image(img):
    #     st.warning("⚠️ Ảnh không phải da người. Vui lòng thử ảnh khác.")
    #     st.stop()

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0]

    pred_idx = int(np.argmax(prediction))
    pred_label = index_to_class[pred_idx]
    confidence = float(np.max(prediction)) * 100
    info = disease_info.get(pred_label, {})

    # Tạo bố cục 2 cột
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown("### 🖼️ Ảnh vùng da")
        st.image(img, use_container_width=True)
  
    with col2:
        st.subheader("📊 Kết quả chẩn đoán")

        if not is_skin_image(img):
            st.warning("⚠️ Ảnh không phải da người. Vui lòng thử ảnh khác.")
            st.stop()

        # Khối dự đoán
        st.markdown(f"""
        <div style='
            background-color: #fff;
            border-left: 4px solid #155724;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        '>
            <h3 style='color: #155724;'>✅ Dự đoán: {info.get("ten_ta", pred_label.upper())}</h3>
            <p style='color: #155724; font-size: 18px;'>Độ chính xác: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Khối thông tin bệnh học
        st.markdown("""
        <div style='
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        '>
        """, unsafe_allow_html=True)

        st.markdown(f"### 🩺 Tên tiếng Việt: `{info.get('ten_tv', 'Không có thông tin')}`")
        st.markdown(f"**📚 Nguyên nhân:** {info.get('nguyennhan', 'Chưa rõ')}")
        st.markdown(f"**🔍 Dấu hiệu nhận biết:** {info.get('dauhieu', 'Không có')}")
        st.markdown(f"**💊 Hướng điều trị:** {info.get('dieutri', 'Chưa rõ')}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Credit dưới cùng toàn màn hình
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>© 2025 | AI Chẩn đoán Ung thư Da | Ứng dụng demo được phát triển bởi nhóm 16 AI - 2025 | Dataset: HAM10000</p>", unsafe_allow_html=True)
else:
    st.info("📌 Vui lòng chọn ảnh để bắt đầu phân tích.")
    st.markdown("""
                <p style='text-align: center; color: grey; margin-top: 100px;'>
                © 2025 | AI Chẩn đoán Ung thư Da | Ứng dụng demo được phát triển bởi nhóm 16 AI - 2025 | Dataset: HAM10000
                </p>
                """, unsafe_allow_html=True)
