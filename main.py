import os
import json
import uuid as uuid_lib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client, create_client
from groq import Groq


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# Check configuration
# ============================================================

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set in .env"
    )

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY must be set in .env"
    )


# ============================================================
# Create clients
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"
BUCKET_NAME = "attachments"


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="M&P Service Assistant API",
    description="FastAPI + Supabase + Groq backend for the M&P warranty and repair service assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request bodies
# ============================================================

class DescribeProblemRequest(BaseModel):
    user_id: str
    message: str


class ServiceRequestCreate(BaseModel):
    user_id: str
    device_brand: str
    device_model: str | None = None
    issue_type: str
    warranty_status: str = "unknown"
    description: str | None = None


class CourierBookingCreate(BaseModel):
    request_id: str
    pickup_address: str
    courier_provider: str | None = None


class StatusUpdate(BaseModel):
    status: str
    note: str | None = None


# ============================================================
# Root / Server Health Check
# ============================================================

@app.get("/")
def root():
    return {
        "success": True,
        "message": "Server is running"
    }


# ============================================================
# Test Supabase Connection
# ============================================================

@app.get("/test-connection")
def test_connection():
    try:
        response = (
            supabase
            .table("profiles")
            .select("id")
            .limit(1)
            .execute()
        )

        return {
            "success": True,
            "message": "Supabase connection successful",
            "data": response.data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Supabase connection failed: {str(e)}"
        )


# ============================================================
# Profiles
# ============================================================

@app.get("/profiles")
def get_profiles():
    try:
        response = (
            supabase
            .table("profiles")
            .select("*")
            .execute()
        )

        return {
            "success": True,
            "profiles": response.data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch profiles: {str(e)}"
        )


@app.get("/profile/{profile_id}")
def get_profile(profile_id: str):
    try:
        response = (
            supabase
            .table("profiles")
            .select("*")
            .eq("id", profile_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Profile not found"
            )

        return {
            "success": True,
            "profile": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch profile: {str(e)}"
        )


@app.get("/profile/phone/{phone}")
def get_profile_by_phone(phone: str):
    try:
        response = (
            supabase
            .table("profiles")
            .select("*")
            .eq("phone", phone)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Profile not found"
            )

        return {
            "success": True,
            "profile": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch profile: {str(e)}"
        )


# ============================================================
# AI powered problem intake
# ============================================================

def ask_groq_json(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        timeout=30
    )
    return response.choices[0].message.content


@app.post("/describe-problem")
def describe_problem(req: DescribeProblemRequest):
    try:
        prompt = (
            f"A customer contacted M&P device service support and described their problem in their own words.\n\n"
            f"Their message: \"{req.message}\"\n\n"
            f"Read this and extract structured information. Respond ONLY with valid JSON, no other text, "
            f"in exactly this shape:\n"
            f'{{"device_brand": "guess the brand or unknown", "device_model": "guess the model or null", '
            f'"issue_type": "one of screen_replacement, battery, software, physical_damage, warranty_claim, other", '
            f'"warranty_status": "one of warranty, non_warranty, unknown", '
            f'"summary": "a short one sentence clean summary of the issue"}}'
        )

        raw = ask_groq_json(prompt)
        extracted = json.loads(raw)

        response = (
            supabase
            .table("service_requests")
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

        if not response.data:
            raise HTTPException(status_code=400, detail="Could not create service request")

        request_id = response.data[0]["id"]

        supabase.table("request_status_history").insert({
            "request_id": request_id,
            "status": "submitted",
            "note": "Request received and understood from customer's own description",
        }).execute()

        return {
            "success": True,
            "service_request": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process problem description: {str(e)}"
        )


# ============================================================
# Service requests
# ============================================================

@app.post("/service-requests")
def create_service_request(req: ServiceRequestCreate):
    try:
        response = (
            supabase
            .table("service_requests")
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

        if not response.data:
            raise HTTPException(status_code=400, detail="Could not create service request")

        request_id = response.data[0]["id"]

        supabase.table("request_status_history").insert({
            "request_id": request_id,
            "status": "submitted",
            "note": "Request received from customer",
        }).execute()

        return {
            "success": True,
            "service_request": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create service request: {str(e)}"
        )


@app.get("/service-requests/{request_id}")
def get_service_request(request_id: str):
    try:
        response = (
            supabase
            .table("service_requests")
            .select("*")
            .eq("id", request_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Request not found")

        return {
            "success": True,
            "service_request": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch service request: {str(e)}"
        )


@app.patch("/service-requests/{request_id}/status")
def update_request_status(request_id: str, update: StatusUpdate):
    try:
        response = (
            supabase
            .table("service_requests")
            .update({"status": update.status})
            .eq("id", request_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Request not found")

        supabase.table("request_status_history").insert({
            "request_id": request_id,
            "status": update.status,
            "note": update.note,
        }).execute()

        return {
            "success": True,
            "service_request": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update status: {str(e)}"
        )


@app.get("/service-requests/{request_id}/timeline")
def get_timeline(request_id: str):
    try:
        response = (
            supabase
            .table("request_status_history")
            .select("*")
            .eq("request_id", request_id)
            .order("created_at")
            .execute()
        )

        return {
            "success": True,
            "timeline": response.data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch timeline: {str(e)}"
        )


# ============================================================
# Attachments
# ============================================================

@app.post("/service-requests/{request_id}/attachments")
def upload_attachment(request_id: str, file_type: str = Form(...), file: UploadFile = File(...)):
    try:
        file_bytes = file.file.read()
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "bin"
        storage_path = f"{request_id}/{uuid_lib.uuid4()}.{file_extension}"

        supabase.storage.from_(BUCKET_NAME).upload(
            storage_path,
            file_bytes,
            {"content-type": file.content_type}
        )

        signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(storage_path, 60 * 60 * 24 * 7)

        response = (
            supabase
            .table("request_attachments")
            .insert({
                "request_id": request_id,
                "file_url": storage_path,
                "file_type": file_type,
            })
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=400, detail="Could not save attachment record")

        return {
            "success": True,
            "attachment": response.data[0],
            "temporary_view_url": signed_url.get("signedURL")
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload attachment: {str(e)}"
        )


@app.get("/service-requests/{request_id}/attachments")
def list_attachments(request_id: str):
    try:
        response = (
            supabase
            .table("request_attachments")
            .select("*")
            .eq("request_id", request_id)
            .execute()
        )

        attachments = []
        for row in response.data:
            signed = supabase.storage.from_(BUCKET_NAME).create_signed_url(row["file_url"], 60 * 60)
            attachments.append({**row, "temporary_view_url": signed.get("signedURL")})

        return {
            "success": True,
            "attachments": attachments
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch attachments: {str(e)}"
        )


# ============================================================
# Courier bookings
# ============================================================

@app.post("/courier-bookings")
def create_courier_booking(req: CourierBookingCreate):
    try:
        response = (
            supabase
            .table("courier_bookings")
            .insert({
                "request_id": req.request_id,
                "pickup_address": req.pickup_address,
                "courier_provider": req.courier_provider,
                "status": "pending",
            })
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=400, detail="Could not book courier")

        supabase.table("service_requests").update({"status": "courier_booked"}).eq("id", req.request_id).execute()

        supabase.table("request_status_history").insert({
            "request_id": req.request_id,
            "status": "courier_booked",
            "note": "Pickup scheduled",
        }).execute()

        return {
            "success": True,
            "courier_booking": response.data[0]
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to book courier: {str(e)}"
        )


# ============================================================
# Service centers
# ============================================================

@app.get("/service-centers")
def list_service_centers():
    try:
        response = (
            supabase
            .table("service_centers")
            .select("*")
            .eq("is_active", True)
            .execute()
        )

        return {
            "success": True,
            "service_centers": response.data
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch service centers: {str(e)}"
        )
