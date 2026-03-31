from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# IMPORTANT: This allows your React frontend to talk to this Python backend
origins = [
    "http://localhost:3000",
    "https://lung-cancer-deploy-frontend.vercel.app", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model you just saved
model = tf.keras.models.load_model('lung_cancer_model.h5')

# These must match the order in your notebook's train_gen.class_indices
CLASSES = ['Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image
    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert('RGB')
    
    # 2. Preprocess (Must match your notebook: 224x224 and rescale 1/255)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Inference
    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return {
        "prediction": CLASSES[idx],
        "confidence": confidence,
        "is_cancer": "benign" not in CLASSES[idx].lower()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)