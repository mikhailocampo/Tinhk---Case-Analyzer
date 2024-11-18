import os
import json
import base64
import requests
import boto3
import uuid
from openai import OpenAI
from typing import List
from loguru import logger
from sqlalchemy import create_engine, text
from datetime import date

from constants import CaseAnalysisSchema


def get_openai_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def convert_blob_to_base64(blob_url: str) -> str:
    """Convert blob URL to base64 string."""
    try:
        # Remove 'blob:' prefix if present
        actual_url = blob_url.replace('blob:', '')
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        response = requests.get(actual_url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Get the actual content type from the response headers
        content_type = response.headers.get('content-type', '').lower().split(';')[0]
        
        # If we received HTML instead of an image, try to extract the image URL
        if content_type == 'text/html':
            raise ValueError("Cannot access blob URL directly. Please provide a direct image URL or base64 data.")
        
        # Validate content type before proceeding
        allowed_types = ['image/png', 'image/jpeg', 'image/gif', 'image/webp']
        if content_type not in allowed_types:
            raise ValueError(f"Unsupported image format: {content_type}. Allowed formats: {allowed_types}")
        
        # Convert the image data to base64
        image_data = base64.b64encode(response.content).decode('utf-8')
        return f"data:{content_type};base64,{image_data}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error accessing blob URL: {e}")
        raise ValueError(f"Failed to access image URL: {str(e)}")
    except Exception as e:
        logger.error(f"Error converting blob URL to base64: {e}")
        raise


def get_s3_client() -> boto3.client:
    return boto3.client('s3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))


def upload_to_storage(base64_data: str, key: str) -> str:
    """Upload base64 image to S3 and return permanent URL."""
    try:
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
        
        # Log the content type for debugging
        logger.debug(f"Uploading with Content-Type: {content_type}")
        
        # Convert base64 to binary
        binary_data = base64.b64decode(base64_data)
        
        # Upload to S3 with correct metadata
        s3_client = get_s3_client()
        bucket_name = os.getenv('AWS_BUCKET_NAME')
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=binary_data,
            ContentType=content_type,
            ContentDisposition='inline',  # Ensure inline display
            ACL='public-read'
        )
        
        # Return permanent URL
        return f"https://{bucket_name}.s3.amazonaws.com/{key}"
    except Exception as e:
        logger.error(f"Failed to upload to storage: {e}")
        raise


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
        processed_images = []
        for url in image_urls:
            try:
                # Now we expect base64 data directly
                if url.startswith('data:image'):
                    key = f"cases/{date.today().strftime('%Y/%m/%d')}/{uuid.uuid4()}.jpg"
                    permanent_url = upload_to_storage(url, key)
                    processed_images.append({"type": "image_url", "image_url": {"url": permanent_url}})
                else:
                    logger.warning(f"Unsupported image format: {url[:30]}...")
                    continue
            except Exception as e:
                logger.error(f"Failed to process image data: {e}")
                continue
        
        messages[1]["content"].extend(processed_images)

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
