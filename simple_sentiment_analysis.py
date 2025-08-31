#!/usr/bin/env python3
"""
Simple Twitter Sentiment Analysis with Bar Chart
Creates a clean bar graph showing positive vs negative sentiment distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    try:
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'

def main():
    # Load dataset
    print("Loading Twitter dataset...")
    df = pd.read_csv('twitter_dataset.csv')
    print(f"Loaded {len(df)} tweets")
    
    # Perform sentiment analysis
    print("Analyzing sentiment...")
    df['Text'] = df['Text'].astype(str)
    df['sentiment'] = df['Text'].apply(analyze_sentiment)
    
    # Create simple bar chart
    print("Creating visualization...")
    plt.figure(figsize=(10, 6))
    
    # Get sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    
    # Define colors
    colors = {'Positive': '#2E8B57', 'Negative': '#DC143C', 'Neutral': '#FFD700'}
    bar_colors = [colors.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
    
    # Create bar chart
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
    
    # Customize chart
    plt.title('Twitter Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Tweets', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=14, fontweight='bold')
    
    # Add percentage labels
    total_tweets = len(df)
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        percentage = (count / total_tweets) * 100
        plt.annotate(f'({percentage:.1f}%)',
                    xy=(i, count),
                    xytext=(0, -25),
                    textcoords="offset points",
                    ha='center', va='top',
                    fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nSentiment Analysis Results:")
    print("=" * 40)
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_tweets) * 100
        print(f"{sentiment}: {count:,} tweets ({percentage:.1f}%)")
    
    # Save results
    df.to_csv('sentiment_results.csv', index=False)
    print(f"\nResults saved to: sentiment_results.csv")

if __name__ == "__main__":
    main()
