from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd

class LLMInterface(ABC):
    @abstractmethod
    def predict_preferences(self, inputs: List[str], indices: List[int]) -> Dict[str, List[int]]:
        """
        Predict user preferences for a batch of inputs.
        
        Args:
            inputs: List of strings containing post descriptions
            indices: List of indices corresponding to the inputs
        
        Returns:
            Dictionary with 'index' and 'ratings' lists
        """
        pass

class PreferencePredictor:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'Titles' not in data.columns:
            raise KeyError("The input DataFrame must contain a 'Titles' column.")
        if 'Index' not in data.columns:
            raise KeyError("The input DataFrame must contain an 'Index' column.")

        inputs = data['Titles'].tolist()
        indices = data['Index'].tolist()
        predictions = self.llm.predict_preferences(inputs, indices)
        
        # Create a mapping of index to rating
        index_to_rating = dict(zip(predictions['index'], predictions['ratings']))
        
        # Apply the ratings to the DataFrame
        data['Preference'] = data['Index'].map(index_to_rating)
        
        return data