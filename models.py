from pydantic import BaseModel, Field


class RequestConfig(BaseModel):
    author_mapping: str = Field(default = None, description="A mapping of authors to their names in the conversation. If not provided, the author will be the username of the person sending the message.")
    single_author: bool = Field(default = False)
    