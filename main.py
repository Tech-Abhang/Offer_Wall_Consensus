from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import openai
import logging
import os
from consensusAPI import OFConsensus
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Consensus API", description="API for consensus validation")

class QuestionAnswers(BaseModel):
    answerType: List[str]  # Array of "Yes" or "No" answers
    answer: str  # Answer for the second question

class ValidationResponse(BaseModel):
    result: str  # "pass" or "fail"

# Initialize OpenAI consensus handler
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables")

consensus_handler = None
if api_key:
    consensus_handler = OFConsensus(api_key=api_key)

@app.post("/validate-answers", response_model=ValidationResponse)
async def validate_answers(answers: QuestionAnswers):
    try:
        if len(answers.answerType) != 2:
            logger.info(f"Invalid answerType length: {len(answers.answerType)}, expected 2")
            return ValidationResponse(result="fail")
        
        if answers.answerType[0].lower() != "yes" or answers.answerType[1].lower() != "no":
            logger.info(f"Invalid answerType sequence: {answers.answerType}, expected ['yes', 'no']")
            return ValidationResponse(result="fail")
        
        # If first validation passes, validate second using OpenAI
        if not consensus_handler:
            logger.error("OpenAI API key not configured")
            raise HTTPException(status_code=500, detail="OpenAI API not configured")
        
        # Call OpenAI to validate the answer against expected answer 
        validation_result = validate_with_openai(answers.answer)
        
        return ValidationResponse(result=validation_result)
        
    except Exception as e:
        logger.error(f"Error validating answers: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def validate_with_openai(user_answer: str) -> str:
    try:
        expected_answer = "A black BMW car convertible with orange seat covers"
        prompt = f"""
        You are a validator that checks if a user's answer matches an expected answer.
        Expected answer: "{expected_answer}"
        User's answer: "{user_answer}"
        Determine if the user's answer is semantically equivalent to or correctly describes the expected answer.
        Respond with exactly one word: either "pass" or "fail"
        """
        
        response = consensus_handler.client.chat.completions.create(
            model=consensus_handler.model,
            messages=[
                {"role": "system", "content": "You are a precise validator. Respond only with 'pass' or 'fail'."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Ensure the response is either "pass" or "fail"
        if result not in ["pass", "fail"]:
            logger.warning(f"Unexpected OpenAI response: {result}, defaulting to 'fail'")
            return "fail"
        
        logger.info(f"OpenAI validation result: {result} for answer: {user_answer}")
        return result
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return "fail"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
