# api.py
from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms
import one_hot
import model
from model import mymodel
import common

app = Flask(__name__)

# 实例化模型并加载权重
model = mymodel()

# 加载权重文件
checkpoint = torch.load("best_model.pth", map_location=torch.device('cpu'))
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# 定义图片预处理函数
def preprocess_image(image):
    tensor_img = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 添加归一化
    ])
    img = tensor_img(image)
    img = torch.reshape(img, (1, 1, 60, 160))  # 修改为正确的形状
    return img

# 定义接口
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # 解析 Base64 编码的图片
        image_data = data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        # 预处理图片
        img = preprocess_image(image)

        # 模型预测
        with torch.no_grad():
            outputs = model(img)
        outputs = outputs.view(-1, len(common.captcha_array))
        outputs_label = one_hot.vectotext(outputs)

        # 返回预测结果
        return jsonify({'captcha': outputs_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)