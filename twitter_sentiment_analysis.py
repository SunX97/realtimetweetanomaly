#!/usr/bin/env python3
"""
Twitter Sentiment Analysis Tool
Analyzes sentiment of tweets and creates visualization for Google Colab
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for Google Colab compatibility
plt.style.use('default')
sns.set_palette("husl")

def load_twitter_data(file_path):
    """Load and preprocess Twitter dataset"""
    print("Loading Twitter dataset...")
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return 'Neutral'

def analyze_sentiment_vader(text):
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

def perform_sentiment_analysis(df):
    """Perform sentiment analysis on tweets"""
    print("Performing sentiment analysis...")
    
    # Clean the text column
    df['Text'] = df['Text'].astype(str)
    
    # Apply sentiment analysis using both methods
    print("Analyzing with TextBlob...")
    df['sentiment_textblob'] = df['Text'].apply(analyze_sentiment_textblob)
    
    print("Analyzing with VADER...")
    df['sentiment_vader'] = df['Text'].apply(analyze_sentiment_vader)
    
    # Create a combined sentiment (using VADER as primary)
    df['sentiment'] = df['sentiment_vader']
    
    print("Sentiment analysis completed!")
    return df

def create_sentiment_visualization(df):
    """Create bar chart visualization of sentiment distribution"""
    print("Creating sentiment visualization...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Twitter Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. VADER Sentiment Distribution (Main Chart)
    sentiment_counts = df['sentiment_vader'].value_counts()
    colors = ['#2E8B57', '#DC143C', '#FFD700']  # Green, Red, Gold
    bars1 = ax1.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
    ax1.set_title('Sentiment Distribution (VADER)', fontweight='bold')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Number of Tweets')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 2. TextBlob Sentiment Distribution
    sentiment_counts_tb = df['sentiment_textblob'].value_counts()
    bars2 = ax2.bar(sentiment_counts_tb.index, sentiment_counts_tb.values, color=colors)
    ax2.set_title('Sentiment Distribution (TextBlob)', fontweight='bold')
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Number of Tweets')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    # 3. Percentage Distribution (Pie Chart)
    sentiment_percentages = df['sentiment_vader'].value_counts(normalize=True) * 100
    ax3.pie(sentiment_percentages.values, labels=sentiment_percentages.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax3.set_title('Sentiment Percentage Distribution', fontweight='bold')
    
    # 4. Sentiment by Engagement (Likes vs Sentiment)
    sentiment_likes = df.groupby('sentiment_vader')['Likes'].mean()
    bars4 = ax4.bar(sentiment_likes.index, sentiment_likes.values, color=colors)
    ax4.set_title('Average Likes by Sentiment', fontweight='bold')
    ax4.set_xlabel('Sentiment')
    ax4.set_ylabel('Average Likes')
    
    # Add value labels on bars
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_sentiment_summary(df):
    """Print detailed sentiment analysis summary"""
    print("\n" + "="*60)
    print("TWITTER SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    
    total_tweets = len(df)
    print(f"Total tweets analyzed: {total_tweets}")
    
    print("\n--- VADER Sentiment Analysis ---")
    vader_counts = df['sentiment_vader'].value_counts()
    for sentiment, count in vader_counts.items():
        percentage = (count / total_tweets) * 100
        print(f"{sentiment}: {count} tweets ({percentage:.1f}%)")
    
    print("\n--- TextBlob Sentiment Analysis ---")
    textblob_counts = df['sentiment_textblob'].value_counts()
    for sentiment, count in textblob_counts.items():
        percentage = (count / total_tweets) * 100
        print(f"{sentiment}: {count} tweets ({percentage:.1f}%)")
    
    print("\n--- Engagement Analysis ---")
    engagement_by_sentiment = df.groupby('sentiment_vader')[['Likes', 'Retweets']].mean()
    print("Average engagement by sentiment (VADER):")
    print(engagement_by_sentiment.round(2))
    
    print("\n--- Sample Tweets by Sentiment ---")
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        if sentiment in df['sentiment_vader'].values:
            sample_tweet = df[df['sentiment_vader'] == sentiment]['Text'].iloc[0]
            print(f"\n{sentiment} tweet example:")
            print(f"  \"{sample_tweet[:100]}...\"")

def main():
    """Main function to run the complete sentiment analysis"""
    
    # Load the dataset
    df = load_twitter_data('twitter_dataset.csv')
    
    if df is None:
        print("Failed to load dataset. Exiting...")
        return
    
    # Perform sentiment analysis
    df_analyzed = perform_sentiment_analysis(df)
    
    # Create visualization
    create_sentiment_visualization(df_analyzed)
    
    # Print summary
    print_sentiment_summary(df_analyzed)
    
    # Save results
    output_file = 'twitter_sentiment_results.csv'
    df_analyzed.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df_analyzed

if __name__ == "__main__":
    # Run the analysis
    results = main()
