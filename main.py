import asyncio
import json
import os
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

load_dotenv()
app = FastAPI(title="LabelLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("rules.json", "r") as f:
    RULES = json.load(f)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Get Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

PROMPT = """
You are a precise food label OCR extraction engine. Analyze the provided image of a food label.
Extract the following fields strictly as JSON.

CRITICAL INSTRUCTIONS:
1. Look for "Added Sugars" first. If not found, look for "Total Sugars".
2. Look for "Saturated Fat", "Total Fat", "Protein", "Carbohydrates".
3. Look for "Sodium" (in mg).
4. Extract all values PER SERVING. If the label only shows "per 100g", calculate the per-serving value based on the serving size.
5. Look for the total net weight or net quantity of the product (e.g., 'Net Weight: 150g', 'Net Qty: 100g'). If not found, set to 'Unknown'.

{
  "product_name": "string",
  "serving_size": "string",
  "net_weight": "string (e.g. 150g, 100g, or Unknown)",
  "servings_per_pack": float,
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

HIDDEN_INGREDIENTS_DB = {
    "invert sugar": "A highly processed liquid sugar that spikes blood sugar faster than table sugar.",
    "liquid glucose": "A cheap sweetener made from starch. It adds empty calories and raises blood sugar rapidly.",
    "maltodextrin": "A highly processed starch used as a filler and thickener. It has a higher glycemic index than table sugar.",
    "palmolein": "A cheap industrial palm oil high in saturated fats, often used to cut costs.",
    "high fructose corn syrup": "A highly refined sweetener linked to increased risk of obesity and metabolic issues."
}

async def extract_label_data(image_bytes: bytes, mime_type: str) -> dict:
    content = [PROMPT, {"mime_type": mime_type, "data": image_bytes}]
    last_err = None
    for model_name in ['gemini-3.5-flash-lite', 'gemini-1.5-flash', 'gemini-2.5-flash']:
        try:
            model = genai.GenerativeModel(model_name)
            response = await asyncio.to_thread(model.generate_content, content)
            cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_text)
        except Exception as e:
            last_err = e
            continue
    raise HTTPException(status_code=500, detail=f"Failed to process image: {str(last_err)}")

def calculate_product_metrics(extracted_data: dict, limits: dict) -> dict:
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

    max_pct = max(sugar_pct, fat_pct, sodium_pct)

    # Verdict Logic:
    # If the highest metric is >100%, show '🚨 EXCEEDS DAILY LIMITS'
    # If highest metric 25% - 99%: '⚠️ HIGH LEVELS DETECTED'
    # If all metrics < 25%: '✅ WITHIN DAILY LIMITS'
    if max_sugar == 0 and (sugar_val > 0 or sodium_val > 0):
        verdict = "🚨 NOT RECOMMENDED FOR INFANTS"
    elif max_pct > 100:
        verdict = "🚨 EXCEEDS DAILY LIMITS"
    elif max_pct >= 25:
        verdict = "⚠️ HIGH LEVELS DETECTED"
    else:
        verdict = "✅ WITHIN DAILY LIMITS"

    raw_ingredients_text = extracted_data.get("ingredients_list", "").lower()
    detected_flags = []

    for key, meaning in HIDDEN_INGREDIENTS_DB.items():
        if key in raw_ingredients_text:
            display_name = key.title()
            detected_flags.append({"name": display_name, "meaning": meaning})

    if "sugar" in raw_ingredients_text and not any(f["name"] == "Sugar" for f in detected_flags):
        detected_flags.append({
            "name": "Sugar",
            "meaning": "Standard added sugar. High intake is linked to weight gain and tooth decay."
        })

    return {
        "product": extracted_data.get("product_name") or "Scanned Food Product",
        "serving_size": extracted_data.get("serving_size") or "Unknown",
        "net_weight": extracted_data.get("net_weight") or "Unknown",
        "servings_per_pack": extracted_data.get("servings_per_pack", 0.0),
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
        "hidden_ingredients": detected_flags,
        "raw_ingredients": extracted_data.get("ingredients_list"),
    }

@app.post("/api/analyze")
async def analyze_label(
    age_group: str = Form(...),
    gender: str = Form("Unknown"), 
    image1: UploadFile = File(...), 
    image2: Optional[UploadFile] = File(None)
):
    if age_group not in RULES["age_brackets"]:
        raise HTTPException(status_code=400, detail="Invalid age group selected.")

    limits = RULES["age_brackets"][age_group]

    image1_bytes = await image1.read()
    image2_bytes = await image2.read() if image2 else None

    # If image2 is provided, process both images concurrently
    if image2_bytes:
        extracted_a, extracted_b = await asyncio.gather(
            extract_label_data(image1_bytes, image1.content_type),
            extract_label_data(image2_bytes, image2.content_type)
        )
        product_a = calculate_product_metrics(extracted_a, limits)
        product_b = calculate_product_metrics(extracted_b, limits)
    else:
        extracted_a = await extract_label_data(image1_bytes, image1.content_type)
        product_a = calculate_product_metrics(extracted_a, limits)
        product_b = None

    # LOG TO SUPABASE DIRECTLY VIA HTTPX
    if SUPABASE_URL and SUPABASE_KEY:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        try:
            async with httpx.AsyncClient() as client:
                payload_a = {
                    "product_name": product_a.get("product"),
                    "age_group": age_group,
                    "sugar_pct": product_a["breakdown"]["sugar"]["percentage"],
                    "fat_pct": product_a["breakdown"]["sat_fat"]["percentage"],
                    "sodium_pct": product_a["breakdown"]["sodium"]["percentage"],
                    "verdict": product_a["verdict"]
                }
                await client.post(f"{SUPABASE_URL}/rest/v1/scans", headers=headers, json=payload_a)

                if product_b:
                    payload_b = {
                        "product_name": product_b.get("product"),
                        "age_group": age_group,
                        "sugar_pct": product_b["breakdown"]["sugar"]["percentage"],
                        "fat_pct": product_b["breakdown"]["sat_fat"]["percentage"],
                        "sodium_pct": product_b["breakdown"]["sodium"]["percentage"],
                        "verdict": product_b["verdict"]
                    }
                    await client.post(f"{SUPABASE_URL}/rest/v1/scans", headers=headers, json=payload_b)
        except Exception as e:
            print(f"Supabase log error: {e}")

    return {
        "product_a": product_a,
        "product_b": product_b,
        "demographic": limits.get("label"),
        "citations": RULES.get("governing_bodies", []),
        "disclaimer": "This is an educational tool based on public WHO/ICMR/FSSAI guidelines. It is not medical advice.",
        # Backwards compatibility top-level fields for single scan
        "product": product_a.get("product"),
        "serving_size": product_a.get("serving_size"),
        "net_weight": product_a.get("net_weight"),
        "servings_per_pack": product_a.get("servings_per_pack"),
        "verdict": product_a.get("verdict"),
        "breakdown": product_a.get("breakdown"),
        "other_nutrients": product_a.get("other_nutrients"),
        "hidden_ingredients": product_a.get("hidden_ingredients"),
        "raw_ingredients": product_a.get("raw_ingredients"),
    }

class Feedback(BaseModel):
    product_name: str
    user_comment: str
    is_accurate: bool

@app.post("/api/feedback")
async def submit_feedback(feedback: Feedback):
    if not (SUPABASE_URL and SUPABASE_KEY):
        return {"status": "success", "message": "Thank you for your feedback!"}

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    payload = {
        "product_name": feedback.product_name,
        "is_accurate": feedback.is_accurate,
        "user_comment": feedback.user_comment
    }
    
    try:
        async with httpx.AsyncClient() as client:
            db_response = await client.post(f"{SUPABASE_URL}/rest/v1/feedback", headers=headers, json=payload)
            if db_response.status_code >= 400:
                print(f"Feedback Error: {db_response.text}")
                raise HTTPException(status_code=500, detail="Failed to save feedback")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")
        
    return {"status": "success", "message": "Thank you for your feedback!"}
