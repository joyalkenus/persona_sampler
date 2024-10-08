# User Persona Preference Predictor

This project uses an AI model to predict preferences for user posts on social media based on a specified persona. It processes a CSV file containing post titles and generates a new CSV file with predicted preferences.

## Setup

1. Clone this repository to your local machine.

2. Install the required Python packages:
   ```
   pip install pandas openai pyyaml python-dotenv
   ```

3. Create a `.env` file in the root directory of the project and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Prepare your input CSV file (default name: `reddit_posts.csv`) with at least a 'Titles' column containing the post titles.

5. Review and adjust the `config.yaml` file to set your desired parameters:
   - `input_file`: Name of your input CSV file
   - `output_file`: Name for the output CSV file with predictions
   - `batch_size`: Number of posts to process in each batch
   - `model`: OpenAI model to use
   - `samples`: Maximum number of rows to process from the input file
   - `persona_characteristics`: Description of the persona for preference prediction
   - `system_prompt_template`: Template for the system prompt

## Usage

1. Make sure your input CSV file is in the same directory as the Python scripts.

2. Run the main script:
   ```
   python main.py
   ```

3. The script will process the input file in batches and generate an output CSV file with the predicted preferences.

## File Structure

- `main.py`: The main script that orchestrates the preference prediction process
- `persona.py`: Contains the abstract base class for LLM interfaces and the PreferencePredictor class
- `openai_llm.py`: Implements the OpenAI LLM interface
- `config.yaml`: Configuration file for setting parameters
- `.env`: (Not included in repo) Store your OpenAI API key here

## Customization

- To use a different LLM, create a new class that implements the `LLMInterface` in `persona.py`, and update `main.py` to use your new LLM class.
- Adjust the `persona_characteristics` in `config.yaml` to change the persona for preference prediction.
- Modify the `system_prompt_template` in `config.yaml` to alter how the persona is instructed.
- Adjust the `batch_size` and `samples` in `config.yaml` to control processing speed and the amount of data processed.

## Output

The script will generate a new CSV file (default: `reddit_posts_with_preferences.csv`) containing all the original columns from the input file, plus a new 'Preference' column with the predicted preferences (0 for dislike, 1 for like).

## Note

Ensure you have sufficient API credits and are aware of the OpenAI API usage costs when running this script, especially for large datasets.