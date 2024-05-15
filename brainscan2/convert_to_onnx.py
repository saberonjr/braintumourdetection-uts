
import torch
from torchvision.models import mobilenet_v2

img_size = (640, 640)
batch_size = 1
onnx_model_path = 'model.onnx'

model_path = 'braintumourdetection/brainscan2/runs/detect/train2/weights/best.pt'
model = Model()
model.load_state_dict(torch.load(pt_model_path, map_location='cpu')).eval()
model.eval()

sample_input = torch.rand((batch_size, 3, *img_size))

y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)