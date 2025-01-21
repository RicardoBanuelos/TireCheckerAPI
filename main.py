from fastapi import FastAPI, UploadFile, HTTPException
from io import BytesIO
from PIL import Image

import torch
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights

app = FastAPI()

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('model.pth', weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),            
    transforms.Normalize(             
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]    
    )
])

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def process_image(image: UploadFile):
    img = Image.open(BytesIO(image.file.read()))
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    return img_tensor

    
@app.post("/tire/check")
async def check_image(upload_file: UploadFile):
    if upload_file.file is None or upload_file.filename == "" or upload_file.content_type != "image/jpeg":
        raise HTTPException(status_code=400, detail="Invalid or empty file")
    
    
    img_tensor = process_image(upload_file)

    # Make the prediction
    with torch.no_grad():
        output = model(img_tensor)

    # Get the predicted class (0 = defective, 1 = good)
    _, predicted_class = torch.max(output, 1)

    # Map predicted class to label
    labels = ['good', 'defective']
    predicted_label = labels[predicted_class.item()]

    return {"predicted_label": predicted_label}