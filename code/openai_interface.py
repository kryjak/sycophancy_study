"""
For OpenAI fine-tuning API, the required format is:
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
The file with such dictionaries should be saved in .jsonl format.
"""

import openai
import os
import json
import pandas as pd

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = "You are a helpful assistant answering questions to the best of your ability."

def get_completion(prompt, system_prompt=SYSTEM_PROMPT, model="gpt-4o-mini-2024-07-18", max_tokens=100, temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content

def create_fine_tuning_data(df: pd.DataFrame, output_file: str) -> None:
    """Create data for fine-tuning and submit a fine-tuning job the the API"""
    # Create a list to hold the data for fine-tuning
    fine_tuning_data = []

    # loop through the DataFrame and create the required data structure:
    for _, row in df.iterrows():
        fine_tuning_data.append({
            "messages": [
                {"role": "system", "content": "SYSTEM_PROMPT"},
                {"role": "user", "content": row['prompt']},
                {"role": "assistant", "content": row['expected_answer']}
            ]
        })

    # Save the data to a file in .jsonl format
    with open(output_file, 'w') as f:
        for item in fine_tuning_data:
            f.write(json.dumps(item) + '\n')

def upload_files(file_path: str) -> str:
    """Upload a file to OpenAI and return the file ID"""
    response = client.files.create(
        file=open(file_path, "rb"),
        purpose='fine-tune'
    )
    return response.id

def submit_fine_tuning_job(training_file: str, model: str = "gpt-4o-mini-2024-07-18", **kwargs) -> str:
    """Submit a fine-tuning job to the API"""
    response = client.fine_tuning.jobs.create(
        training_file=training_file,
        model=model,
        **kwargs
    )

    return response.id

def list_fine_tuning_jobs(n_jobs: int = 10) -> list:
    """List all fine-tuning jobs"""
    # client.fine_tuning.jobs returns some weird OpenAI object that does not have len() method
    # when doing list() on it, for some reason it ignores the limit and returns all jobs
    # but we can avoid this issue if we retrieve just the .data attribute of this object
    return client.fine_tuning.jobs.list(limit=n_jobs).data

def retrieve_fine_tuning_job(job_id: str) -> dict:
    """Retrieve a fine-tuning job"""
    return client.fine_tuning.jobs.retrieve(job_id)

