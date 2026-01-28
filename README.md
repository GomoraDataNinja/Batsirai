# Batsirai – YouTube Topic Sentiment Bot

Batsirai is a Streamlit web app that helps users understand what people are saying on YouTube about any topic.

A user types a topic.

Batsirai searches YouTube for many videos related to that topic.

It samples comments, runs sentiment analysis, then ranks the best videos.

It produces a Top 10 list and clear reasons for why each video ranks where it does.

---

## What Batsirai Does

- Searches YouTube for videos based on a topic you enter
- Scans a chosen number of videos
- Samples a chosen number of comments per video
- Labels comments as Positive, Neutral, or Negative
- Scores each video using:
  - engagement
  - sentiment
  - topic match
  - recency
- Displays:
  - Top 10 videos
  - a bot-style answer in point form
  - what people keep mentioning
  - what people like
  - what people complain about
  - questions people keep asking
- Exports CSV files for:
  - rankings
  - sampled comments
- Includes a password wall for controlled access

---

## App Tabs

### Search
- Type your topic
- Choose how many videos to scan
- Choose how many comments per video to sample
- Choose a time window
- Optional region code filter

### Top 10
- Shows the top 10 ranked videos
- Shows bot answer in clear bullet points
- Shows theme points with example comments
- Shows the Top 10 table
- Shows recommendations and why each video ranked

### Explore
- Lets you pick a single video
- Shows key metrics
- Shows sampled comments and sentiment filtering

### Export
- Download rankings CSV
- Download sampled comments CSV

---

## How Scoring Works

Batsirai calculates a final score per video using:

- Engagement score  
  likes + comment count compared to views

- Sentiment score  
  average sentiment of sampled comments

- Topic match score  
  compares topic words with title and description words

- Recency score  
  newer videos get a small boost

Batsirai combines these into one final score, then sorts videos from best to worst.

---

## Requirements

The app uses:

- streamlit
- pandas
- numpy
- plotly
- textblob
- google-api-python-client

---

## Setup Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
