import json
import os
import uuid as uuid_lib

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from supabase import create_client

load_dotenv()

# Environment Configurations & Clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"
BUCKET_NAME = "attachments"

app = FastAPI(title="M&P Service Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
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


# Helper Functions
def ask_groq_json(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=30,
    )
    return response.choices[0].message.content


def extract_signed_url(signed_res: dict | object) -> str | None:
    if isinstance(signed_res, dict):
        return signed_res.get("signedURL") or signed_res.get("signedUrl")
    return getattr(signed_res, "signed_url", None)


# API Routes
@app.get("/")
def health():
    return {"status": "M&P service assistant backend running"}


@app.post("/service-requests")
def create_service_request(req: ServiceRequestCreate):
    result = (
        supabase.table("service_requests")
        .insert({
            "user_id": req.user_id,
            "device_brand": req.device_brand,
            "device_model": req.device_model,
            "issue_type": req.issue_type,
            "warranty_status": req.warranty_status,
            "description": req.description,
            "status": "submitted",
        })
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=400, detail="Could not create service request")

    request_id = result.data[0]["id"]

    supabase.table("request_status_history").insert({
        "request_id": request_id,
        "status": "submitted",
        "note": "Request received from customer",
    }).execute()

    return result.data[0]


@app.get("/service-requests/{request_id}")
def get_service_request(request_id: str):
    result = supabase.table("service_requests").select("*").eq("id", request_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Request not found")
    return result.data[0]


@app.get("/service-centers")
def list_service_centers():
    result = supabase.table("service_centers").select("*").eq("is_active", True).execute()
    return result.data


@app.post("/describe-problem")
def describe_problem(req: DescribeProblemRequest):
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

    raw = ask_groq_json(prompt)

    try:
        extracted = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not understand the problem, please try rephrasing")

    result = (
        supabase.table("service_requests")
        .insert({
            "user_id": req.user_id,
            "device_brand": extracted.get("device_brand", "unknown"),
            "device_model": extracted.get("device_model"),
            "issue_type": extracted.get("issue_type", "other"),
            "warranty_status": extracted.get("warranty_status", "unknown"),
            "description": extracted.get("summary", req.message),
            "status": "submitted",
        })
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=400, detail="Could not create service request")

    request_id = result.data[0]["id"]

    supabase.table("request_status_history").insert({
        "request_id": request_id,
        "status": "submitted",
        "note": "Request received and understood from customer's own description",
    }).execute()

    return result.data[0]


@app.post("/service-requests/{request_id}/attachments")
def upload_attachment(request_id: str, file_type: str = Form(...), file: UploadFile = File(...)):
    file_bytes = file.file.read()
    file_extension = file.filename.split(".")[-1] if file.filename and "." in file.filename else "bin"
    storage_path = f"{request_id}/{uuid_lib.uuid4()}.{file_extension}"

    supabase.storage.from_(BUCKET_NAME).upload(
        storage_path,
        file_bytes,
        {"content-type": file.content_type}
    )

    signed_res = supabase.storage.from_(BUCKET_NAME).create_signed_url(storage_path, 60 * 60 * 24 * 7)

    result = supabase.table("request_attachments").insert({
        "request_id": request_id,
        "file_url": storage_path,
        "file_type": file_type,
    }).execute()

    if not result.data:
        raise HTTPException(status_code=400, detail="Could not save attachment record")

    return {
        "attachment": result.data[0],
        "temporary_view_url": extract_signed_url(signed_res)
    }


@app.get("/service-requests/{request_id}/attachments")
def list_attachments(request_id: str):
    result = supabase.table("request_attachments").select("*").eq("request_id", request_id).execute()

    attachments = []
    for row in result.data:
        signed_res = supabase.storage.from_(BUCKET_NAME).create_signed_url(row["file_url"], 60 * 60)
        attachments.append({
            **row,
            "temporary_view_url": extract_signed_url(signed_res)
        })

    return attachments


@app.post("/courier-bookings")
def create_courier_booking(req: CourierBookingCreate):
    result = (
        supabase.table("courier_bookings")
        .insert({
            "request_id": req.request_id,
            "pickup_address": req.pickup_address,
            "courier_provider": req.courier_provider or "TBD",
            "status": "pending",
        })
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=400, detail="Could not create courier booking")

    supabase.table("service_requests").update({"status": "courier_booked"}).eq("id", req.request_id).execute()

    supabase.table("request_status_history").insert({
        "request_id": req.request_id,
        "status": "courier_booked",
        "note": f"Courier pickup scheduled from {req.pickup_address}",
    }).execute()

    return result.data[0]


@app.patch("/service-requests/{request_id}/status")
def update_status(request_id: str, update: StatusUpdate):
    result = (
        supabase.table("service_requests")
        .update({"status": update.status})
        .eq("id", request_id)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=404, detail="Request not found")

    supabase.table("request_status_history").insert({
        "request_id": request_id,
        "status": update.status,
        "note": update.note,
    }).execute()

    return result.data[0]


@app.get("/service-requests/{request_id}/timeline")
def get_timeline(request_id: str):
    result = (
        supabase.table("request_status_history")
        .select("*")
        .eq("request_id", request_id)
        .order("created_at")
        .execute()
    )
    return result.data
