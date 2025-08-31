import torch.onnx
from model import mymodel


pytorch_net_path = 'model.pth'
onnx_net_path = 'model.onnx'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = mymodel().to(device)


checkpoint = torch.load(pytorch_net_path, map_location=device)


if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

input_tensor = torch.randn(1, 1, 60, 160).to(device)

torch.onnx.export(model, input_tensor, onnx_net_path, opset_version=11, verbose=False)
