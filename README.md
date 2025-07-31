# AI-Powered Book Recommendations for School District

An AI-powered book recommendation system that helps librarians provide personalized reading suggestions to students based on their borrowing history.

## Project Structure

```
read-to-succeed/
├── data/
│   ├── book_catalog.csv      # Sample book metadata
│   └── rental_history.csv    # Sample borrowing records
├── src/
│   └── process_data.py       # Offline data processing script
├── config/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

1. Copy `.env.example` to `.env`
2. Fill in your Elasticsearch and AWS Bedrock credentials

### 3. Data Processing

Run the offline processing script to generate embeddings and index books:

```bash
cd src
python process_data.py
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Usage

1. Enter a student ID (e.g., `student_001`)
2. The system will show their recent reading history
3. Get personalized recommendations with AI-generated explanations

## Features

- **Content-based filtering** using sentence transformers
- **Collaborative filtering** using matrix factorization
- **AI-generated recommendation snippets** via AWS Bedrock
- **Vector similarity search** with Elasticsearch
- **Interactive web interface** with Streamlit

## Sample Data

The system includes sample data with 16 popular books and borrowing history for 10 students, including classics like Harry Potter, The Hobbit, and Wonder.