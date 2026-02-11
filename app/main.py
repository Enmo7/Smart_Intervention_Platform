import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# Import the engine logic using absolute path
from app.engine import analyze_message

app = FastAPI(title="Guardian AI - Intervention Platform")

# Configure static files and template engine
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def index(request: Request):
    """Route to serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(message: str = Form(...)):
    """API endpoint to receive text and return AI-driven risk analysis."""
    if not message.strip():
        return {"error": "Message content is empty"}
        
    analysis_result = analyze_message(message)
    return analysis_result

if __name__ == "__main__":
    # Start server with auto-reload for development
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)