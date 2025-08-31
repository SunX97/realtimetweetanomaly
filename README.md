# Real-Time Tweet Anomaly Detection & Sentiment Analysis

A comprehensive sentiment analysis system for Twitter data with visualization capabilities, designed for real-time anomaly detection in social media sentiment patterns.

## 🎯 Overview

This project provides tools for analyzing Twitter sentiment data and detecting anomalies in social media patterns. It's particularly useful for:
- **FinTech Applications**: Monitor sentiment around financial markets, cryptocurrencies, and blockchain technologies
- **Social Media Analytics**: Track public opinion and sentiment trends
- **Anomaly Detection**: Identify unusual patterns in tweet sentiment that may indicate market events or social phenomena

## 🚀 Features

- **Dual Sentiment Analysis**: Uses both VADER and TextBlob for robust sentiment detection
- **Interactive Visualizations**: Bar charts, pie charts, and engagement analysis
- **Google Colab Integration**: Ready-to-use Jupyter notebook for cloud execution
- **Real-time Processing**: Designed for scalable tweet analysis
- **Financial Context Awareness**: Special handling for FinTech and blockchain-related content

## 📊 Sentiment Analysis Results

Current dataset analysis shows:
- **Positive Sentiment**: 79.3% (7,934 tweets)
- **Negative Sentiment**: 16.8% (1,682 tweets)
- **Neutral Sentiment**: 3.8% (384 tweets)

## 🛠️ Installation

### Local Setup

```bash
pip install pandas matplotlib seaborn textblob vaderSentiment jupyter
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('brown')"
```

### Google Colab Setup

```python
!pip install textblob vaderSentiment matplotlib seaborn pandas
import nltk
nltk.download('punkt')
nltk.download('brown')
nltk.download('vader_lexicon')
```

## 📁 Project Structure

```
realtimetweetanomaly/
├── README.md                           # Project documentation
├── twitter_sentiment_analysis.py       # Complete analysis script
├── simple_sentiment_analysis.py        # Streamlined version
├── Twitter_Sentiment_Analysis.ipynb    # Google Colab notebook
├── twitter_dataset.csv                 # Sample dataset
├── twitter_sentiment_results.csv       # Analysis results
└── requirements.txt                    # Python dependencies
```

## 🚀 Quick Start

### Option 1: Run Local Analysis

```bash
python twitter_sentiment_analysis.py
```

### Option 2: Simple Bar Chart Only

```bash
python simple_sentiment_analysis.py
```

### Option 3: Google Colab

1. Upload `Twitter_Sentiment_Analysis.ipynb` to Google Colab
2. Upload your dataset when prompted
3. Run all cells to see complete analysis

## 📈 Visualization Examples

The system generates multiple visualizations:

1. **Sentiment Distribution Bar Chart**: Shows positive, negative, and neutral tweet counts
2. **Percentage Pie Chart**: Displays sentiment distribution as percentages
3. **Engagement Analysis**: Correlates sentiment with likes and retweets
4. **Method Comparison**: Compares VADER vs TextBlob results

## 🔧 Usage

### Basic Usage

```python
import pandas as pd
from twitter_sentiment_analysis import main

# Run complete analysis
results = main()
```

### Custom Analysis

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply to your data
df['sentiment'] = df['text_column'].apply(analyze_sentiment)
```

## 💡 Applications

### FinTech & Blockchain
- **Market Sentiment Monitoring**: Track sentiment around cryptocurrencies and financial instruments
- **Investment Decision Support**: Use sentiment trends to inform trading strategies
- **Risk Assessment**: Detect negative sentiment spikes that may indicate market volatility

### Social Media Analytics
- **Brand Monitoring**: Track public sentiment about companies or products
- **Crisis Detection**: Identify unusual negative sentiment patterns
- **Engagement Optimization**: Understand which sentiment types generate more engagement

## 🔍 Technical Details

### Sentiment Analysis Methods

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**
   - Optimized for social media text
   - Handles emojis, slang, and informal language
   - Returns compound score from -1 (negative) to +1 (positive)

2. **TextBlob**
   - General-purpose sentiment analysis
   - Returns polarity score from -1 to +1
   - Good for formal text analysis

### Thresholds
- **Positive**: Compound score ≥ 0.05
- **Negative**: Compound score ≤ -0.05
- **Neutral**: -0.05 < Compound score < 0.05

## 📋 Requirements

```
pandas>=2.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
textblob>=0.17.0
vaderSentiment>=3.3.0
jupyter>=1.0.0
nltk>=3.8
```

## 🚀 Future Enhancements

- [ ] Real-time streaming analysis
- [ ] Machine learning model training
- [ ] Advanced anomaly detection algorithms
- [ ] Integration with Twitter API v2
- [ ] Blockchain sentiment correlation analysis
- [ ] Multi-language sentiment support

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Related Technologies

This project complements blockchain and FinTech development by providing sentiment analysis capabilities that can be integrated into:
- **Smart Contracts** for sentiment-based decision making
- **DeFi Protocols** for market sentiment analysis
- **Trading Bots** for sentiment-driven strategies
- **Risk Management Systems** for social sentiment monitoring

---

**Built with ❤️ for the FinTech and Blockchain community**
