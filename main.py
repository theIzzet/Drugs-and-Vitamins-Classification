import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uvicorn
from fastapi.staticfiles import StaticFiles 
from starlette.responses import FileResponse
app = FastAPI(title="Vitamin Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model ve sınıf isimleri
model = None
class_names = [
    "Alaxan", "Bactidol", "Bioflu", "Biogesic", "DayZinc",
    "Decolgen", "Fish Oil", "Kremil S", "Medicol", "Neozep"
]

def build_model():
    """Orijinal model yapısını yeniden oluştur"""
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout
    
    pretrained_model = MobileNetV2(
        input_shape=(224, 224, 3),
        pooling="avg",
        weights="imagenet",
        include_top=False
    )
    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = Dense(256, activation="relu")(pretrained_model.output)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def load_model():
    global model
    try:
        model = build_model()
        
        # drug_cnn de oluşturulan ağırlıkların yüklenmesi
        model.load_weights("checkpoint.weights.h5")
        
       
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        print("Model ve ağırlıklar başarıyla yüklendi!")
        
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        raise e

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

@app.on_event("startup")
async def startup_event():
    load_model()



@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/classes")
async def get_classes():
    return {"classes": class_names}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Dosya bir resim olmalıdır")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        all_probabilities = {
            class_names[i]: float(predictions[0][i]) for i in range(len(class_names))
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin sırasında hata: {str(e)}")



app.mount("/static", StaticFiles(directory="static",html=True), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse('static/index.html')
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)