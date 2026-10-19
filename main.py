import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend to connect
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
    image: UploadFile = File(...)
):
    if age_group not in RULES["age_brackets"]:
        raise HTTPException(status_code=400, detail="Invalid age group selected.")

    image_bytes = await image.read()

    prompt = """
    You are a precise food label OCR extraction engine. Analyze the provided image of a food label.
    Extract the following fields strictly as JSON.
    
    CRITICAL INSTRUCTIONS:
    1. Look for "Added Sugars" first. If not found, look for "Total Sugars".
    2. Look for "Saturated Fat".
    3. Look for "Sodium" (in mg).
    4. Extract all values PER SERVING. If the label only shows "per 100g", calculate the per-serving value based on the serving size.
    
    {
      "product_name": "string",
      "serving_size": "string (e.g., 20g, 1 scoop, 250ml)",
      "sugar_g_per_serving": float,
      "sat_fat_g_per_serving": float,
      "sodium_mg_per_serving": float,
      "ingredients_list": "string of raw text"
    }
    If a field is missing, set numerical values to 0.0 and strings to 'Unknown'.
    """

    model = genai.GenerativeModel('gemini-3.5-flash-lite')
    
    try:
        response = model.generate_content([
            prompt,
            {"mime_type": image.content_type, "data": image_bytes}
        ])
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        extracted_data = json.loads(cleaned_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

    limits = RULES["age_brackets"][age_group]
    
    # Extracted values
    sugar_val = extracted_data.get("sugar_g_per_serving", 0.0)
    sat_fat_val = extracted_data.get("sat_fat_g_per_serving", 0.0)
    sodium_val = extracted_data.get("sodium_mg_per_serving", 0.0)
    
    # Limits
    max_sugar = limits["max_sugar_g"]
    max_sat_fat = limits["max_sat_fat_g"]
    max_sodium = limits["max_sodium_mg"]
    
    # Math (Avoid division by zero for infants)
    sugar_pct = round((sugar_val / max_sugar) * 100, 1) if max_sugar > 0 else (100 if sugar_val > 0 else 0)
    fat_pct = round((sat_fat_val / max_sat_fat) * 100, 1) if max_sat_fat > 0 else (100 if sat_fat_val > 0 else 0)
    sodium_pct = round((sodium_val / max_sodium) * 100, 1) if max_sodium > 0 else (100 if sodium_val > 0 else 0)

    teaspoons = round(sugar_val / 4.0, 1)

    # Neutral Verdict Logic
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
            "sugar": {
                "value_g": sugar_val,
                "limit_g": max_sugar,
                "percentage": sugar_pct,
                "teaspoons": teaspoons
            },
            "sat_fat": {
                "value_g": sat_fat_val,
                "limit_g": max_sat_fat,
                "percentage": fat_pct
            },
            "sodium": {
                "value_mg": sodium_val,
                "limit_mg": max_sodium,
                "percentage": sodium_pct
            }
        },
        "raw_ingredients": extracted_data.get("ingredients_list"),
        "citations": RULES["governing_bodies"],
        "disclaimer": "This is an educational tool based on public WHO/ICMR/FSSAI guidelines. It is not medical advice."
    }