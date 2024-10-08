from typing import List, Dict
from openai import OpenAI
from pydantic import BaseModel
from persona import LLMInterface

class PostRatingSchema(BaseModel):
    index: List[int]
    ratings: List[int]

class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06", system_prompt: str = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt 

    def predict_preferences(self, inputs: List[str], indices: List[int]) -> Dict[str, List[int]]:
        numbered_inputs = [f"{indices[i]}. {text}" for i, text in enumerate(inputs)]
        prompt = " make sure to rate each post based on the persona you are imagining when rating.\n".join(numbered_inputs)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=PostRatingSchema,
            )

            parsed_response = completion.choices[0].message.parsed
            return {
                "index": parsed_response.index,
                "ratings": parsed_response.ratings
            }
        except Exception as e:
            print(f"Error in predict_preferences: {e}")
            # Return empty lists if there's an error
            return {"index": [], "ratings": []}