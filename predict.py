from evaluate import predict_sentiment

if __name__ == '__main__':
    test_reviews = [
        "This movie was absolutely amazing! I loved every second of it.",
        "Terrible film. Complete waste of time. Do not watch.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen. Highly recommend!",
        "ok watch but second half was boring.",
        'Absolutely terrific film enjoyed every part of it.',
        'I want a refund for watching this garbage'
    ]
    
    print("=" * 70)
    print("SENTIMENT CLASSIFIER — CUSTOM REVIEW PREDICTIONS")
    print("=" * 70)
    
    for i, review in enumerate(test_reviews, 1):
        sentiment, confidence = predict_sentiment(review)
        print(f"\nReview {i}:")
        print(f"  Text: {review}")
        print(f"  Prediction: {sentiment.upper()} (confidence: {confidence:.2%})")
    
    print("\n" + "=" * 70)