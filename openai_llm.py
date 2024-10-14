from typing import List, Dict
from openai import OpenAI
from pydantic import BaseModel
from persona import LLMInterface

class PostRatingSchema(BaseModel):
    index: List[int]
    ratings: List[int]

class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06", system_prompt: str = None, prompt: str = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.message_history = []  # Initialize an empty list to store message history

    def predict_preferences(self, inputs: List[str], indices: List[int]) -> Dict[str, List[int]]:
        numbered_inputs = [f"{indices[i]}. {text}" for i, text in enumerate(inputs)]
        prompts = "Make sure to rate each post based on the persona you are imagining when rating.\n" + self.prompt + "\n".join(numbered_inputs)

        # Add the current prompt to the message history
        self.message_history.append({"role": "user", "content": prompts})

        try:
            # Prepare messages including system prompt and message history
            messages = [{"role": "system", "content": self.system_prompt}] + self.message_history

            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=PostRatingSchema,
            )

            parsed_response = completion.choices[0].message.parsed
            
            # Add the LLM's response to the message history
            self.message_history.append({"role": "assistant", "content": str(parsed_response)})
            
            # Limit the message history to the last N messages (e.g., 10) to prevent it from growing too large and consuming too much tokens
            max_history = 10
            self.message_history = self.message_history[-max_history:]

            # Check if the number of returned indices and ratings match the input
            if len(parsed_response.index) != len(indices) or len(parsed_response.ratings) != len(indices):
                raise ValueError(f"Mismatch in returned data. Expected {len(indices)} items, got {len(parsed_response.index)} indices and {len(parsed_response.ratings)} ratings.")
            
            return {
                "index": parsed_response.index,
                "ratings": parsed_response.ratings
            }
        except Exception as e:
            print(f"Error in predict_preferences: {e}")
            # Raise the exception to be handled by the caller
            raise

    def clear_history(self):
        """Clear the message history"""
        self.message_history = []
