import pandas as pd
import os
import yaml
from dotenv import load_dotenv
from persona import PreferencePredictor, LLMInterface
from openai_llm import OpenAILLM

def load_config(config_file: str = 'config.yaml') -> dict:
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def process_csv(config: dict, llm: LLMInterface):
    predictor = PreferencePredictor(llm)

    # Load your dataset
    df = pd.read_csv(config['input_file'])

    # Check if 'Titles' column exists
    if 'Titles' not in df.columns:
        raise KeyError("The input CSV file must contain a 'Titles' column.")

    # Add an index column if not present
    if 'Index' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Index'}, inplace=True)

    # Predict preferences in batches
    results = []
    
    max_rows = min(config['samples'], len(df))  
    for i in range(0, max_rows, config['batch_size']):
        batch = df.iloc[i:i + config['batch_size']]
        result_batch = predictor.predict_batch(batch)
        results.append(result_batch)
        print(f"Processed batch {i // config['batch_size'] + 1}")

    # Concatenate all results and save to a new CSV
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(config['output_file'], index=False)

    print(f"All batches processed and saved to '{config['output_file']}'")

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    config = load_config()
    
    llm = OpenAILLM(api_key, model=config['model'], system_prompt=config['system_prompt'])
    process_csv(config, llm)