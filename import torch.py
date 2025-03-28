from transformers import pipeline

# Test a simple sentiment analysis pipeline
classifier = pipeline('sentiment-analysis')

result = classifier("I love this!")
print(result)
