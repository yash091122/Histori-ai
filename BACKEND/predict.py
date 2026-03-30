import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model("trained_model_2.h5")

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

# Load itinerary
df = pd.read_excel("Itineary.xlsx")

def normalize(name):
    return name.lower().replace("_", " ").strip()

df['norm'] = df['Monuments'].apply(normalize)

itinerary_map = {
    row['norm']: row for _, row in df.iterrows()
}

# Main Function
def predict(image_path):
    
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)
    pred_index = np.argmax(preds)
    predicted_class = class_names[pred_index]
    
    predicted_norm = normalize(predicted_class)
    itinerary = itinerary_map.get(predicted_norm)
    
    if itinerary is not None:
        return {
            "monument": predicted_class,
            "location": itinerary['Location'],
            "rating": itinerary['Rating'],
            "region": itinerary['Region'],
            "country": itinerary['Country'],
            "year_built": itinerary['Year Built'],
            "visiting_hours": itinerary['Visiting Hours'],
            "ticket_price": itinerary['Ticket Price'],
            "description": itinerary['Description'],
            "fun_facts": itinerary['Fun Facts'],
            "historical_context": itinerary['Historical Context'],
            "review_count": itinerary['Review Count'],
        }
    
    return {
        "monument": predicted_class,
        "itinerary": None,
    }
