import io
import json
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from datetime import datetime, timedelta
from typing import List, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, EmailStr, Field
from motor.motor_asyncio import AsyncIOMotorClient
from jose import jwt
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Monument Recognition & Travel API")

# --- 1. CONFIGURATION ---
MONGO_URL = "mongodb+srv://Epics:v0XiOpYEwBW3SdOp@cluster0.f74wvvx.mongodb.net/?appName=Cluster0"
SECRET_KEY = "VIT_BHOPAL_SECRET_KEY" 
ALGORITHM = "HS256"

# --- 2. DATABASE & MODEL INITIALIZATION ---
client = AsyncIOMotorClient(MONGO_URL)
db = client["My-database"]

try:
    model = tf.keras.models.load_model("trained_model.h5")
    with open("class_names.json") as f:
        class_names = json.load(f)
    print("✅ AI Model & My-database Connected Successfully")
except Exception as e:
    print(f"⚠️ Load Error: {e}")



class SignupSchema(BaseModel):
    fullName: str
    email: EmailStr
    password: str
    confirmPassword: str

class TransactionSchema(BaseModel):
    userId: str
    transactionId: str
    plan: dict = {"planName": "Standard"}
    phoneNumber: str
    amount: float
    paymentMethod: str

class TripOrItinerarySchema(BaseModel):
    userId: str
    tripName: str
    destinations: List[str]
    startDate: datetime
    endDate: datetime
    travelers: List[Any] = []
    transport: Optional[dict] = None
    budget: Optional[dict] = None
    weatherForecast: Optional[dict] = None
    costBreakdown: Optional[dict] = None
    tripStatus: str = "Planned"
    itinerary: Optional[dict] = None 

# --- 4. CORE AUTHENTICATION ---

def hash_pass(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

@app.post("/signup")
async def signup(user: SignupSchema):
    if user.password != user.confirmPassword:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    existing = await db["signup"].find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = {"fullName": user.fullName, "email": user.email, "password": hash_pass(user.password)}
    result = await db["signup"].insert_one(new_user)
    uid = str(result.inserted_id)
    
    # Auto-initialize 'profile' collection
    await db["profile"].insert_one({
        "userId": uid, "fullName": user.fullName, "bio": "CS Student",
        "location": "Bhopal", "tipsHistory": [], "stats": {"trips": 0},
        "badges": [], "posts": []
    })
    return {"status": "success", "userId": uid}

@app.post("/login")
async def login(email: str, password: str):
    db_user = await db["signup"].find_one({"email": email})
    if not db_user or db_user["password"] != hash_pass(password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode({"sub": db_user["email"], "id": str(db_user["_id"])}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "userId": str(db_user["_id"]), "fullName": db_user["fullName"]}

# --- 5. MONUMENT & TRANSACTION LOGIC ---

@app.post("/predict")
async def predict_monument(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    image_arr = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    label = class_names[np.argmax(model.predict(image_arr, verbose=0))]
    
    # Fetch from 'Monuments' collection
    info = await db["Monuments"].find_one({"Monumentes": label})
    if info:
        info["_id"] = str(info["_id"])
        return info
    return {"monument": label, "msg": "No DB details found."}

@app.post("/transaction")
async def add_transaction(txn: TransactionSchema):
    txn_data = txn.dict()
    txn_data["transactionDateTime"] = datetime.utcnow().isoformat()
    result = await db["Transaction"].insert_one(txn_data)
    
    # Save to 'users collection' link as per your schema
    await db["users collection"].insert_one({
        "userId": txn.userId,
        "fullName": "User", 
        "transactionId": txn.transactionId
    })
    return {"status": "success", "db_id": str(result.inserted_id)}

@app.get("/profile/{userId}")
async def get_profile(userId: str):
    profile = await db["profile"].find_one({"userId": userId})
    if not profile: raise HTTPException(status_code=404)
    profile["_id"] = str(profile["_id"])
    return profile

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
