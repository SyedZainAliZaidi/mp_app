import json
import os
import uuid as uuid_lib

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from pydantic import BaseModel

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = "gemini-3.6-flash"
BUCKET_NAME = "attachments"

REST_HEADERS = {
    "apikey": SUPABASE_SECRET_KEY,
    "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

app = FastAPI(title="M&P Service Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServiceRequestCreate(BaseModel):
    user_id: str
    device_brand: str
    device_model: str | None = None
    issue_type: str
    warranty_status: str = "unknown"
    description: str | None = None


class DescribeProblemRequest(BaseModel):
    user_id: str
    message: str


class CourierBookingCreate(BaseModel):
    request_id: str
    pickup_address: str
    courier_provider: str | None = "TBD"


class StatusUpdate(BaseModel):
    status: str
    note: str | None = None


def db_insert(table: str, payload: dict) -> list:
    res = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=REST_HEADERS, json=payload, timeout=15)
    if res.status_code not in (200, 201):
        raise Exception(f"{res.status_code}: {res.text}")
    return res.json()


def db_select(table: str, filters: str = "", order: str = "") -> list:
    url = f"{SUPABASE_URL}/rest/v1/{table}?select=*"
    if filters:
        url += f"&{filters}"
    if order:
        url += f"&order={order}"
    res = requests.get(url, headers=REST_HEADERS, timeout=15)
    if res.status_code != 200:
        raise Exception(f"{res.status_code}: {res.text}")
    return res.json()


def db_update(table: str, filters: str, payload: dict) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{table}?{filters}"
    res = requests.patch(url, headers=REST_HEADERS, json=payload, timeout=15)
    if res.status_code not in (200, 201, 204):
        raise Exception(f"{res.status_code}: {res.text}")
    return res.json() if res.text else []


def ask_gemini_json(prompt: str) -> str:
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    response = model.generate_content(prompt)
    return response.text


@app.get("/")
def health():
    return {"status": "M&P service assistant backend running"}


@app.post("/service-requests")
def create_service_request(req: ServiceRequestCreate):
    try:
        result = db_insert("service_requests", {
            "user_id": req.user_id,
            "device_brand": req.device_brand,
            "device_model": req.device_model,
            "issue_type": req.issue_type,
            "warranty_status": req.warranty_status,
            "description": req.description,
            "status": "submitted",
        })

        if not result:
            raise HTTPException(status_code=400, detail="Could not create service request")

        request_id = result[0]["id"]

        db_insert("request_status_history", {
            "request_id": request_id,
            "status": "submitted",
            "note": "Request received from customer",
        })

        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@app.get("/service-requests/{request_id}")
def get_service_request(request_id: str):
    try:
        result = db_select("service_requests", filters=f"id=eq.{request_id}")
        if not result:
            raise HTTPException(status_code=404, detail="Request not found")
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@app.get("/service-centers")
def list_service_centers():
    try:
        return db_select("service_centers", filters="is_active=eq.true")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@app.post("/describe-problem")
def describe_problem(req: DescribeProblemRequest):
    try:
        prompt = (
            f"A customer contacted M&P device service support and described their problem in their own words.\n\n"
            f'Their message: "{req.message}"\n\n'
            f"Read this and extract structured information. Respond ONLY with valid JSON, no other text, "
            f"in exactly this shape:\n"
            f'{{"device_brand": "guess the brand or unknown", "device_model": "guess the model or null", '
            f'"issue_type": "one of screen_replacement, battery, software, physical_damage, warranty_claim, other", '
            f'"warranty_status": "one of warranty, non_warranty, unknown", '
            f'"summary": "a short one sentence clean summary of the issue"}}'
        )
        raw = ask_gemini_json(prompt)
        extracted = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

    try:
        result = db_insert("service_requests", {
            "user_id": req.user_id,
            "device_brand": extracted.get("device_brand", "unknown"),
            "device_model": extracted.get("device_model"),
            "issue_type": extracted.get("issue_type", "other"),
            "warranty_status": extracted.get("warranty_status", "unknown"),
            "description": extracted.get("summary", req.message),
            "status": "submitted",
        })

        if not result:
            raise HTTPException(status_code=400, detail="Database insert failed (check RLS policies)")

        request_id = result[0]["id"]

        db_insert("request_status_history", {
            "request_id": request_id,
            "status": "submitted",
            "note": "Request received and understood from customer's own description",
        })

        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Supabase DB Error: {str(e)}")


@app.post("/service-requests/{request_id}/attachments")
def upload_attachment(request_id: str, file_type: str = Form(...), file: UploadFile = File(...)):
    try:
        file_bytes = file.file.read()
        file_extension = file.filename.split(".")[-1] if file.filename and "." in file.filename else "bin"
        storage_path = f"{request_id}/{uuid_lib.uuid4()}.{file_extension}"

        upload_res = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{BUCKET_NAME}/{storage_path}",
            headers={"apikey": SUPABASE_SECRET_KEY, "Authorization": f"Bearer {SUPABASE_SECRET_KEY}", "Content-Type": file.content_type or "application/octet-stream"},
            data=file_bytes,
            timeout=30,
        )
        if upload_res.status_code not in (200, 201):
            raise Exception(f"Upload failed: {upload_res.text}")

        result = db_insert("request_attachments", {
            "request_id": request_id,
            "file_url": storage_path,
            "file_type": file_type,
        })

        if not result:
            raise HTTPException(status_code=400, detail="Could not save attachment record")

        signed_res = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/sign/{BUCKET_NAME}/{storage_path}",
            headers=REST_HEADERS,
            json={"expiresIn": 604800},
            timeout=15,
        )
        signed_url = None
        if signed_res.status_code == 200:
            signed_url = f"{SUPABASE_URL}/storage/v1{signed_res.json().get('signedURL', '')}"

        return {"attachment": result[0], "temporary_view_url": signed_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage Error: {str(e)}")


@app.get("/service-requests/{request_id}/attachments")
def list_attachments(request_id: str):
    try:
        return db_select("request_attachments", filters=f"request_id=eq.{request_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@app.post("/courier-bookings")
def create_courier_booking(req: CourierBookingCreate):
    try:
        result = db_insert("courier_bookings", {
            "request_id": req.request_id,
            "pickup_address": req.pickup_address,
            "courier_provider": req.courier_provider or "TBD",
            "status": "pending",
        })

        if not result:
            raise HTTPException(status_code=400, detail="Could not create courier booking")

        db_update("service_requests", f"id=eq.{req.request_id}", {"status": "courier_booked"})

        db_insert("request_status_history", {
            "request_id": req.request_id,
            "status": "courier_booked",
            "note": f"Courier pickup scheduled from {req.pickup_address}",
        })

        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@app.patch("/service-requests/{request_id}/status")
def update_status(request_id: str, update: StatusUpdate):
    try:
        result = db_update("service_requests", f"id=eq.{request_id}", {"status": update.status})

        if not result:
            raise HTTPException(status_code=404, detail="Request not found")

        db_insert("request_status_history", {
            "request_id": request_id,
            "status": update.status,
            "note": update.note,
        })

        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")


@app.get("/service-requests/{request_id}/timeline")
def get_timeline(request_id: str):
    try:
        return db_select("request_status_history", filters=f"request_id=eq.{request_id}", order="created_at")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")
