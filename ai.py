import os
import json
import base64
import requests
import boto3
import uuid
import asyncio
from openai import OpenAI
from typing import List
from loguru import logger
from sqlalchemy import create_engine, text
from datetime import date
from concurrent.futures import ThreadPoolExecutor

from constants import CaseAnalysisSchema


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_s3_client() -> boto3.client:
    return boto3.client('s3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))


async def upload_to_storage(base64_data: str, key: str) -> str:
    """Upload base64 image to S3 and return permanent URL."""
    try:
        logger.debug(f"Starting upload for key: {key}")
        start_time = asyncio.get_event_loop().time()
        
        # Default content type
        content_type = 'image/jpeg'
        
        # Check if base64 data has a prefix and extract content type
        if ',' in base64_data:
            header, base64_data = base64_data.split(',', 1)
            if ';base64' in header:
                content_type = header.split(':')[1].split(';')[0]
        
        # Validate content type
        allowed_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
        if content_type not in allowed_types:
            raise ValueError(f"Unsupported image format: {content_type}. Allowed formats: {allowed_types}")
        
        # Convert base64 to binary
        binary_data = base64.b64decode(base64_data)
        
        # Use ThreadPoolExecutor for the blocking S3 operation
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            s3_client = get_s3_client()
            bucket_name = os.getenv('AWS_BUCKET_NAME')
            
            logger.debug(f"Initiating S3 upload for key: {key}")
            await loop.run_in_executor(
                pool,
                lambda: s3_client.put_object(
                    Bucket=bucket_name,
                    Key=key,
                    Body=binary_data,
                    ContentType=content_type,
                    ContentDisposition='inline',
                    ACL='public-read'
                )
            )
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        logger.debug(f"Completed upload for key: {key} in {duration:.2f} seconds")
        
        # Return permanent URL
        return f"https://{bucket_name}.s3.amazonaws.com/{key}"
    except Exception as e:
        logger.error(f"Failed to upload to storage for key {key}: {e}")
        raise

async def format_messages(
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

    permanent_urls = []  # List to store permanent URLs

    if image_urls:
        upload_tasks = []
        for url in image_urls:
            try:
                # Handle both prefixed and raw base64 data
                if url.startswith('data:image'):
                    base64_data = url
                elif url.startswith('/9j/') or url.startswith('iVBOR'):
                    base64_data = f"data:image/jpeg;base64,{url}"
                else:
                    logger.warning(f"Unsupported image format: {url[:30]}...")
                    continue

                key = f"cases/{date.today().strftime('%Y/%m/%d')}/{uuid.uuid4()}.jpg"
                upload_tasks.append(upload_to_storage(base64_data, key))
            except Exception as e:
                logger.error(f"Failed to process image data: {e}")
                continue

        # Wait for all uploads to complete in parallel
        if upload_tasks:
            try:
                permanent_urls = await asyncio.gather(*upload_tasks)
                processed_images = [
                    {"type": "image_url", "image_url": {"url": url}}
                    for url in permanent_urls
                ]
                messages[1]["content"].extend(processed_images)
            except Exception as e:
                logger.error(f"Failed during parallel upload: {e}")

    return messages, permanent_urls


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
            :keypoints,
            CAST(:translations AS JSONB),
            :created_at
        )
        RETURNING id
    """)
    
    params = {
        "title": title,
        "images": json.dumps(image_urls),
        "summary": case_analysis.summary,
        "keypoints": case_analysis.key_points,
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
