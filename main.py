import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from supabase import Client, create_client


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# ============================================================
# Check Supabase configuration
# ============================================================

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY must be set in .env"
    )


# ============================================================
# Create Supabase client
# ============================================================

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Profile API",
    description="FastAPI + Supabase Profile API",
    version="1.0.0"
)


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
    """
    Tests the connection between FastAPI and Supabase.

    No Bearer token required.
    """

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
# Get All Profiles
# ============================================================

@app.get("/profiles")
def get_profiles():
    """
    Get all profiles from the profiles table.

    No Bearer token required.
    """

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


# ============================================================
# Get Single Profile
# ============================================================

@app.get("/profile/{profile_id}")
def get_profile(profile_id: str):
    """
    Get one profile by ID.

    No Bearer token required.
    """

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


# ============================================================
# Get Profile by Phone
# ============================================================

@app.get("/profile/phone/{phone}")
def get_profile_by_phone(phone: str):
    """
    Get profile using phone number.

    No Bearer token required.
    """

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
