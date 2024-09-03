import openai
import os

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

