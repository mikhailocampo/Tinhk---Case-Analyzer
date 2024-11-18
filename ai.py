import os
import json
from openai import OpenAI
from typing import List
from loguru import logger
from sqlalchemy import create_engine, text
from datetime import date

from constants import CaseAnalysisSchema


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_messages(
    system_prompt: str,
    user_prompt: str,
    image_urls: List[str] = None
):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    if image_urls:
        # Add image URLs to the user's content list
        messages[1]["content"].extend([
            {"type": "image_url", "image_url": {"url": url}} 
            for url in image_urls
        ])

    return messages


def call_openai_structured(client: OpenAI, messages: List[dict]) -> CaseAnalysisSchema:
    logger.debug(f"Starting OpenAI Structured Call")
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=messages,
        response_format=CaseAnalysisSchema,
    )
    logger.info(f"Total tokens: {response.usage.total_tokens}")
    return response.choices[0].message.parsed


def store_case_analysis(
    title: str,
    image_urls: List[str],
    case_analysis: CaseAnalysisSchema
    ):
    query = text("""
        INSERT INTO viet_cases (
            title,
            images,
            summary,
            keypoints,
            translations,
            created_at
        ) VALUES (
            :title,
            CAST(:images AS JSONB),
            :summary,
            CAST(:keypoints AS JSONB),
            CAST(:translations AS JSONB),
            :created_at
        )
        RETURNING id
    """)
    
    # Convert Python objects to JSON strings first, which PostgreSQL will then cast to JSONB
    params = {
        "title": title,
        "images": json.dumps(image_urls),
        "summary": case_analysis.summary,
        "keypoints": json.dumps(case_analysis.key_points),
        "translations": json.dumps([translation.model_dump() for translation in case_analysis.translations]),
        "created_at": date.today().isoformat()
    }
    db_url = os.getenv("PGSQL_URL")
    with create_engine(db_url).connect() as connection:
        with connection.begin():
            logger.info(f"Storing generated case analysis: {title}")
            result = connection.execute(query, params)
            case_id = result.fetchone()[0]
            logger.success(f"Successfully stored case analysis with ID: {case_id}")
            return case_id
