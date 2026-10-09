"""
Anywear VTO - Compatibility Entry Point
Redirects to the production Step 2 pipeline in app.py.
"""

from app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
