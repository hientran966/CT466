import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def save_base64_image(data, path):
    """Chuyển base64 thành ảnh và lưu lại."""
    header, encoded = data.split(',', 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))
    image.save(path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Lưu ảnh gốc
        image_file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
        
        # Lưu mask từ dữ liệu base64
        mask_data = request.form["mask"]
        mask_path = os.path.join(UPLOAD_FOLDER, "mask.png")
        save_base64_image(mask_data, mask_path)

        # Xử lý ảnh
        result_path = os.path.join(RESULT_FOLDER, "colorized.png")
        colorize_image(image_path, mask_path, result_path)
        
        return render_template("index.html", result_url=result_path)
    
    return render_template("index.html", result_url=None)

def colorize_image(image_path, mask_path, result_path):
    """Xử lý ảnh bằng OpenCV."""
    # Load model
    prototxt = "model/colorization_deploy_v2.prototxt"
    model = "model/colorization_release_v2.caffemodel"
    points = "model/pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    # Load ảnh và mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Tách những đường màu đỏ (vẽ của người dùng)
    img = cv2.imread(mask_path)
    red_pixels = (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 255)  # Red color detection
    
    # Tạo một mask mới với những điểm đỏ chuyển thành trắng và những điểm khác thành đen
    custom_mask = np.zeros_like(mask)
    custom_mask[red_pixels] = 255  # Những điểm màu đỏ thành trắng

    # Resize mask để khớp với kích thước ảnh
    custom_mask = cv2.resize(custom_mask, (image.shape[1], image.shape[0]))

    # Inpainting với mask đã chỉnh sửa
    inpainted = cv2.inpaint(image, custom_mask, 3, flags=cv2.INPAINT_NS)
    scaled = inpainted.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize & xử lý ảnh
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    
    # Ghép lại ảnh màu
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    # Lưu ảnh kết quả
    cv2.imwrite(result_path, colorized)

if __name__ == "__main__":
    app.run(debug=True)
