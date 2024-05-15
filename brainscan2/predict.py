from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train2/weights/best.pt')  # load a brain-tumor fine-tuned model

# Inference using the model
results = model.predict("./datasets/brain-tumor/valid/images/val_1 (114).jpg")