from PIL import Image
from torch.utils.data import DataLoader
import one_hot
import model
import torch
import common
import my_datasets2
from model import mymodel
from torchvision import transforms

def test_pred():
    model_path = "./best_model.pth"
    # 1. 实例化模型
    model = mymodel().cuda()
    
    # 2. 加载保存的权重
    try:
        # 尝试加载整个模型（旧格式）
        model.load_state_dict(torch.load(model_path))
    except:
        # 尝试加载域适应模型的权重
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # 3. 加载测试集
    test_data = my_datasets2.mydatasets("./dataset/test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()

    # 4. 测试
    correct = 0
    results = []
    for i, (imgs, labels, labels_text) in enumerate(test_dataloader):
        imgs = imgs.cuda()
        labels = labels.cuda()
    
        with torch.no_grad():
            predict_outputs = model(imgs)
        predict_outputs = predict_outputs.squeeze(0)
        predict_labels = one_hot.vectotext(predict_outputs.cpu())
        labels_text = labels_text[0]  # 获取标签文本
        if predict_labels.lower() == labels_text.lower():
            correct += 1
            print("预测正确：正确值:{}, 预测值:{}".format(labels_text, predict_labels))
        else:
            print("预测失败:正确值:{}, 预测值:{}".format(labels_text, predict_labels))
    
    print("正确率: {:.2f}%".format(correct / test_length * 100))

def pred_pic(pic_path):
    img = Image.open(pic_path)
    tensor_img = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 添加归一化
    ])
    img = tensor_img(img).cuda()
    print(img.shape)
    img = torch.reshape(img, (1, 1, 60, 160))  # 修改为正确的形状
    print(img.shape)
    model = mymodel().cuda()  # 使用新的模型实例化
    model.load_state_dict(torch.load("best_model.pth"))  # 加载模型权重
    model.eval()  # 设置为评估模式
    outputs = model(img)
    outputs = outputs.view(-1, len(common.captcha_array))
    outputs_label = one_hot.vectotext(outputs)
    print(outputs_label)

if __name__ == '__main__':
    test_pred()
    #pred_pic("./dataset/aalR_f0f492.png")