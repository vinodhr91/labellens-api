import asyncio
import json
import os
import sys
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

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

# Get Supabase credentials with resilient normalization
raw_supabase_url = os.getenv("SUPABASE_URL", "").strip().rstrip('/')
raw_supabase_key = os.getenv("SUPABASE_KEY", "").strip()

if raw_supabase_url.startswith("sb_") or (not raw_supabase_url.startswith("http") and raw_supabase_key.startswith("http")):
    SUPABASE_URL = raw_supabase_key.rstrip('/')
    SUPABASE_KEY = raw_supabase_url
else:
    SUPABASE_URL = raw_supabase_url
    SUPABASE_KEY = raw_supabase_key

if SUPABASE_URL.endswith("/rest/v1"):
    SUPABASE_URL = SUPABASE_URL[:-8]

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

async def log_scan_to_supabase(
    product_name: str,
    age_group: str,
    verdict: str,
    sugar_pct: float,
    fat_pct: float,
    sodium_pct: float,
    profile_name: Optional[str] = None,
):
    """
    Inserts a row into the Supabase 'scans' table for every scan.
    Wrapped in try/except so Supabase failures print to Render logs but never crash the app.
    """
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("❌ SCAN LOG FAILED: Missing SUPABASE_URL or SUPABASE_KEY on server.")
        return

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # Returns inserted rows in response body to verify insert succeeded
    }

    # Keys in dictionary exactly match the column names in the Supabase 'scans' table:
    # product_name, age_group, sugar_pct, fat_pct, sodium_pct, verdict
    data = {
        "product_name": str(product_name or "Unknown Product"),
        "age_group": str(age_group or "Unknown"),
        "sugar_pct": float(sugar_pct) if sugar_pct is not None else 0.0,
        "fat_pct": float(fat_pct) if fat_pct is not None else 0.0,
        "sodium_pct": float(sodium_pct) if sodium_pct is not None else 0.0,
        "verdict": str(verdict or "Unknown"),
    }

    endpoint = f"{SUPABASE_URL}/rest/v1/scans"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            http_response = await client.post(endpoint, headers=headers, json=data)

            # Parse Supabase response
            try:
                raw_json = http_response.json()
            except Exception:
                raw_json = None

            class SupabaseResponse:
                def __init__(self, data, status_code, text):
                    self.data = (
                        data
                        if (isinstance(data, list) and status_code < 400)
                        else ([data] if (isinstance(data, dict) and status_code < 400) else [])
                    )
                    self.status_code = status_code
                    self.text = text

                def __repr__(self):
                    return f"SupabaseResponse(status_code={self.status_code}, data={self.data}, text={self.text})"

            response = SupabaseResponse(
                data=raw_json,
                status_code=http_response.status_code,
                text=http_response.text,
            )

            # Print loud error to the console if response.data is empty
            if http_response.status_code >= 400 or not response.data:
                print('❌ SCAN LOG FAILED:', response)
            else:
                print(f"✅ SCAN LOGGED SUCCESSFULLY! (Product: '{data['product_name']}', Age Group: '{data['age_group']}')")
    except Exception as e:
        print('❌ SCAN LOG FAILED:', str(e))

@app.post("/api/analyze")
async def analyze_label(
    age_group: str = Form(...),
    gender: str = Form("Unknown"),
    profile_name: Optional[str] = Form("Default Profile"),
    image1: UploadFile = File(...), 
    image2: Optional[UploadFile] = File(None)
):
    # Normalize age group if formatted as 0-12 months or 10+ years
    if age_group in ("0-12m", "0-12 months", "0 – 12 months", "0-12", "infant"):
        age_group = "under-2"
    elif age_group in ("10+ years", "10+ yrs", "10+years", "10plus", "10+"):
        age_group = "10+"

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

    # Log to Supabase scans table inside an isolated try/except block so it NEVER crashes the scan route
    try:
        active_profile = (profile_name or "").strip() or "Default Profile"
        sugar_a = product_a.get("breakdown", {}).get("sugar", {}).get("percentage", 0.0)
        fat_a = product_a.get("breakdown", {}).get("sat_fat", {}).get("percentage", 0.0)
        sodium_a = product_a.get("breakdown", {}).get("sodium", {}).get("percentage", 0.0)

        await log_scan_to_supabase(
            product_name=product_a.get("product") or "Scanned Food Product",
            age_group=age_group,
            verdict=product_a.get("verdict") or "Unknown",
            sugar_pct=sugar_a,
            fat_pct=fat_a,
            sodium_pct=sodium_a,
            profile_name=active_profile,
        )

        if product_b:
            sugar_b = product_b.get("breakdown", {}).get("sugar", {}).get("percentage", 0.0)
            fat_b = product_b.get("breakdown", {}).get("sat_fat", {}).get("percentage", 0.0)
            sodium_b = product_b.get("breakdown", {}).get("sodium", {}).get("percentage", 0.0)

            await log_scan_to_supabase(
                product_name=product_b.get("product") or "Scanned Food Product",
                age_group=age_group,
                verdict=product_b.get("verdict") or "Unknown",
                sugar_pct=sugar_b,
                fat_pct=fat_b,
                sodium_pct=sodium_b,
                profile_name=active_profile,
            )
    except Exception as log_err:
        print('❌ SCAN LOG FAILED:', str(log_err))

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
    product_name: Optional[str] = "Unknown"
    user_comment: Optional[str] = None
    reason: Optional[str] = None
    is_accurate: bool

@app.post("/api/feedback")
async def submit_feedback(feedback: Feedback):
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("❌ FEEDBACK INSERT FAILED: Missing SUPABASE_URL or SUPABASE_KEY environment variables on server.")
        return {"status": "error", "message": "Supabase configuration missing on server"}

    # 1. Ensure is_accurate is sent as a Python Boolean (True or False), not a string
    is_accurate_bool: bool = bool(feedback.is_accurate)

    # 2. Ensure the data dictionary being inserted exactly matches column names: product_name, is_accurate, reason
    reason_text = feedback.reason or feedback.user_comment or ("Accurate" if is_accurate_bool else "Inaccurate")
    data = {
        "product_name": feedback.product_name or "Unknown Product",
        "is_accurate": is_accurate_bool,  # Python boolean True / False
        "reason": reason_text,
    }

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"  # Returns inserted rows in response body to verify insert succeeded
    }

    endpoint = f"{SUPABASE_URL}/rest/v1/feedback"

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            http_response = await client.post(endpoint, headers=headers, json=data)

            # Parse Supabase response
            try:
                raw_json = http_response.json()
            except Exception:
                raw_json = None

            class SupabaseResponse:
                def __init__(self, data, status_code, text):
                    self.data = (
                        data
                        if (isinstance(data, list) and status_code < 400)
                        else ([data] if (isinstance(data, dict) and status_code < 400) else [])
                    )
                    self.status_code = status_code
                    self.text = text

                def __repr__(self):
                    return f"SupabaseResponse(status_code={self.status_code}, data={self.data}, text={self.text})"

            response = SupabaseResponse(
                data=raw_json,
                status_code=http_response.status_code,
                text=http_response.text,
            )

            # 3. Capture the Supabase response. If response.data is empty, print bright error message
            if http_response.status_code >= 400 or not response.data:
                print('❌ FEEDBACK INSERT FAILED:', response)

                # Check for RLS (Row Level Security) permissions
                # REMINDER: If the error mentions RLS or permissions, check Supabase Insert Policy:
                # Supabase Dashboard -> Table Editor -> 'feedback' -> Add RLS policy allowing anon insert.
                lower_err = http_response.text.lower()
                if (
                    "row-level security" in lower_err
                    or "rls" in lower_err
                    or "permission denied" in lower_err
                    or http_response.status_code == 403
                ):
                    print("🚨 SUPABASE RLS ERROR: Row Level Security is blocking the insert! Please check your Supabase Insert Policy for table 'feedback'.")

                raise HTTPException(status_code=500, detail=f"Failed to save feedback: {http_response.text}")

            # 4. If it succeeds, print success message
            print('✅ FEEDBACK SAVED SUCCESSFULLY!')
            return {"status": "success", "data": response.data, "message": "Thank you for your feedback!"}

    except HTTPException:
        raise
    except Exception as e:
        print('❌ FEEDBACK INSERT FAILED:', str(e))
        raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")
