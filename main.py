import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
from typing import List

# Local imports
from models import RequestConfig
from ai import get_openai_client, format_messages, call_openai_structured, store_case_analysis
from constants import SYSTEM_PROMPT


load_dotenv()

web_app = FastAPI(
    title="Tinhk-webapp",
    docs_url="/docs",
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

app = modal.App(
    "Tinhk-webapp",
    image=image,
    secrets= [
        modal.Secret.from_name("openai"),
        modal.Secret.from_name("postgres-secret")
    ]
)

@app.function(keep_warm=1, timeout=3600)
@modal.asgi_app()
def fastapi_app():
    logger.success("Started FastAPI")
    return web_app

@web_app.post("/analyze_case")
async def analyze_case(
    case_title: str,
    screenshot_urls: List[str],
    additional_context: str = None,
    config: RequestConfig = None
):
    # Data Validation
    if not case_title:
        raise HTTPException(status_code=400, detail="Case title is required")
        
    if not config:
        config = RequestConfig()

    if not screenshot_urls or len(screenshot_urls) == 0:
        raise HTTPException(status_code=400, detail="At least one screenshot URL is required")
    
    for url in screenshot_urls:
        if not isinstance(url, str) or not url.strip():
            raise HTTPException(status_code=400, detail="Invalid screenshot URL provided")
    
    client = get_openai_client()
    messages = format_messages(
        system_prompt=SYSTEM_PROMPT(config),
        user_prompt=f"Case title: {case_title}\nAdditional context: {additional_context}",
        image_urls=screenshot_urls
    )
    
    try:
        case_analysis = call_openai_structured(client, messages)
        try:
            case_id = store_case_analysis(case_title, screenshot_urls, case_analysis)
            return {"case_id": case_id, "analysis": case_analysis}
        except Exception as e:
            logger.error(f"Error storing case analysis: {e}")
            # Return analysis even if storage failed
            return {"case_id": None, "analysis": case_analysis}
    except Exception as e:
        logger.error(f"Error analyzing case: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while analyzing the case")