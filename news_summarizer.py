import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Use a smaller model for better performance on CPUs
model_name = "google/flan-t5-base"  # Change to "flan-t5-large" if you want better summaries
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ✅ Replace with your own API key from NewsAPI.org
NEWS_API_KEY = "Add your API key here"

def fetch_news(topic):
    """Fetch latest news articles on a given topic from NewsAPI"""
    url = f"https://newsapi.org/v2/everything?q={topic}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "articles" not in data:
        return []

    # Extract the top 5 news headlines
    return [article["title"] for article in data["articles"][:5]]

def summarize_news(topic):
    """Summarize the latest news on a topic using FLAN-T5"""
    news = fetch_news(topic)
    
    if not news:
        return "No news found for this topic."

    news_text = "\n".join(news)
    prompt = f"Summarize these news headlines about {topic}:\n\n{news_text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    topic = input("Enter a news topic: ")
    summary = summarize_news(topic)
    print("\n📰 AI-Generated Summary:\n", summary)
