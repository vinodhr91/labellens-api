import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the rules JSON
with open("rules.json", "r") as f:
    RULES = json.load(f)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.post("/api/analyze")
async def analyze_label(
    age_group: str = Form(...),
    gender: str = Form("Unknown"), 
    image1: UploadFile = File(...), 
    image2: Optional[UploadFile] = File(None) # Optional second image
):
    if age_group not in RULES["age_brackets"]:
        raise HTTPException(status_code=400, detail="Invalid age group selected.")

    image1_bytes = await image1.read()
    image2_bytes = await image2.read() if image2 else None

    prompt = """
    You are a precise food label OCR extraction engine. Analyze the provided image(s) of a food label.
    Extract the following fields strictly as JSON.
    
    CRITICAL INSTRUCTIONS:
    1. Look for "Added Sugars" first. If not found, look for "Total Sugars".
    2. Look for "Saturated Fat", "Total Fat", "Protein", "Carbohydrates".
    3. Look for "Sodium" (in mg).
    4. Extract all values PER SERVING. If the label only shows "per 100g", calculate the per-serving value based on the serving size.
    
    {
      "product_name": "string",
      "serving_size": "string",
      "sugar_g_per_serving": float,
      "sat_fat_g_per_serving": float,
      "sodium_mg_per_serving": float,
      "protein_g_per_serving": float,
      "carbs_g_per_serving": float,
      "total_fat_g_per_serving": float,
      "ingredients_list": "string of raw text"
    }
    If a field is missing, set numerical values to 0.0 and strings to 'Unknown'.
    """

    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    
    try:
        content = [prompt, {"mime_type": image1.content_type, "data": image1_bytes}]
        if image2_bytes:
            content.append({"mime_type": image2.content_type, "data": image2_bytes})
            
        response = model.generate_content(content)
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(cleaned_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    limits = RULES["age_brackets"][age_group]
    
    sugar_val = extracted_data.get("sugar_g_per_serving", 0.0)
    sat_fat_val = extracted_data.get("sat_fat_g_per_serving", 0.0)
    sodium_val = extracted_data.get("sodium_mg_per_serving", 0.0)
    
    max_sugar = limits["max_sugar_g"]
    max_sat_fat = limits["max_sat_fat_g"]
    max_sodium = limits["max_sodium_mg"]
    
    sugar_pct = round((sugar_val / max_sugar) * 100, 1) if max_sugar > 0 else (100 if sugar_val > 0 else 0)
    fat_pct = round((sat_fat_val / max_sat_fat) * 100, 1) if max_sat_fat > 0 else (100 if sat_fat_val > 0 else 0)
    sodium_pct = round((sodium_val / max_sodium) * 100, 1) if max_sodium > 0 else (100 if sodium_val > 0 else 0)
    teaspoons = round(sugar_val / 4.0, 1)

    flags = []
    if sugar_pct > 50: flags.append("High Sugar")
    if fat_pct > 50: flags.append("High Saturated Fat")
    if sodium_pct > 50: flags.append("High Sodium")

    if max_sugar == 0 and (sugar_val > 0 or sodium_val > 0):
        verdict = "🚨 NOT RECOMMENDED FOR INFANTS"
    elif flags:
        verdict = "⚠️ EXCEEDS DAILY LIMIT: " + ", ".join(flags)
    else:
        verdict = "✅ WITHIN DAILY LIMITS"

    return {
        "product": extracted_data.get("product_name"),
        "serving_size": extracted_data.get("serving_size"),
        "demographic": limits.get("label"),
        "verdict": verdict,
        "breakdown": {
            "sugar": {"value_g": sugar_val, "limit_g": max_sugar, "percentage": sugar_pct, "teaspoons": teaspoons},
            "sat_fat": {"value_g": sat_fat_val, "limit_g": max_sat_fat, "percentage": fat_pct},
            "sodium": {"value_mg": sodium_val, "limit_mg": max_sodium, "percentage": sodium_pct}
        },
        "other_nutrients": {
            "protein_g": extracted_data.get("protein_g_per_serving", 0.0),
            "carbs_g": extracted_data.get("carbs_g_per_serving", 0.0),
            "total_fat_g": extracted_data.get("total_fat_g_per_serving", 0.0)
        },
        "raw_ingredients": extracted_data.get("ingredients_list"),
        "citations": RULES.get("governing_bodies", []), 
        "disclaimer": "This is an educational tool based on public WHO/ICMR/FSSAI guidelines. It is not medical advice."
    }

# ==========================================
# HUMAN FEEDBACK LOOP ENDPOINT
# ==========================================
class Feedback(BaseModel):
    product_name: str
    user_comment: str
    is_accurate: bool

@app.post("/api/feedback")
async def submit_feedback(feedback: Feedback):
    # In the future, we will save this to Supabase
    # For now, we just print it to the Render logs so you can see it working.
    print(f"--- NEW FEEDBACK ---")
    print(f"Product: {feedback.product_name}")
    print(f"Accurate: {feedback.is_accurate}")
    print(f"Comment: {feedback.user_comment}")
    print(f"--------------------")
    
    return {"status": "success", "message": "Thank you for your feedback!"}