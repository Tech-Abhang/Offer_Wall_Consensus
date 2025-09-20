from openai import OpenAI
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class OFConsensus:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info("OFConsensus initialized")