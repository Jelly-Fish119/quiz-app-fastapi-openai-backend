from openai import OpenAI
from app.core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_topic_from_text(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant that extracts the main topic of a document page."},
            {"role": "user", "content": f"Extract the topic from the following page:\n\n{text}"}
        ],
        temperature=0.3,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()
