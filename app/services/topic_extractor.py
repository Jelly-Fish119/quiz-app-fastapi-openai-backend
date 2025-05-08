from app.core.openai_client import extract_topic_from_text

def extract_topics_per_page(pages: list[str]) -> list[str]:
    return [extract_topic_from_text(page) for page in pages]
