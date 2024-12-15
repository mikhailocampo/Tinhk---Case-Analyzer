from pydantic import BaseModel, Field
from typing import List, Literal


SYSTEM_PROMPT = lambda config: f"""
You are a vietnamese to english translator given a series of screenshots of messages. It is often that you will find the messages written with informal slang or its messages are out of order in context. Sometimes you know some context about the extracted messages to give you a further instructions.

Author: {config.single_author} (When true, give a summary focusing on the single author. Otherwise, give a summary focusing on the conversation as a whole.)
Username: {config.author_mapping} (A mapping of author to their username in the conversation. If not provided, the author will be the username of the person sending the message.)

Notes:
- Given a single screenshot, the vietnamese text and english text should point to the screenshot's content rather than individual chat messages within the screenshot. Suppose a screenshot contains 3 messages (chat bubbles), both the english and vietnamese text should point to the combined content of the 3 messages separated by newline per message.
- When single author is true, the author field should be the {config.author_mapping}. Extra messages from other authors should be ignored.
"""

class Translation(BaseModel):
    index: int = Field(description="The index of the screenshot")
    author: str = Field(description="The username of the author of the message.")
    vietnamese_text: str = Field(description="The original text to be translated from vietnamese to english extracted from the screenshot")
    english_text: str = Field(description="The translated text in English corresponding to the original text")
    confidence: Literal['high', 'medium', 'low'] = Field(description="The confidence in the translation's accuracy")

class CaseAnalysisSchema(BaseModel):
    translations: List[Translation] = Field(description="A list of translations of the conversation in english")
    summary: str = Field(description="A detailed summary of the conversation in english, highlighting key takeaways, lessons learned, and any other important information. Also share any additional context that you think is important. Briefly also reference screenshot numbers to help direct readers, especially non-vietnamese speakers, to note-worthy parts of the conversation. An example would be: 'Screenshot 4 and 5 are great to read in detail because they contain the most important information about the case.'")
    key_points: str = Field(description="A list of key points of the conversation in english, highlighting the most important information written in markdown format.")
    