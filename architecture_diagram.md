# AI Book Recommendations - Architecture Overview

## System Architecture (Horizontal Layout)

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   DATA SOURCES  │────▶│  DATA PROCESSING  │────▶│ VECTOR DATABASE │◀────│ APPLICATION     │◀────│ USER INTERFACE  │
│                 │     │                   │     │                 │     │                 │     │                 │
│ • book_catalog  │     │ BookRecommend     │     │ OpenSearch      │     │ LibraryDatabase │     │ Streamlit App   │
│   (27 books)    │     │ Processor         │     │ Docker:9200     │     │ App             │     │                 │
│ • rental_hist   │     │                   │     │                 │     │                 │     │ • Rental Hist   │
│   (54 records)  │     │ • Embeddings      │     │ Index: "books"  │     │ • find_similar  │     │ • Recommend     │
│                 │     │   384-dim         │     │ Settings:       │     │ • get_history   │     │ • Browse Books  │
│                 │     │ • Collaborative   │     │   knn=true      │     │                 │     │                 │
│                 │     │   50-dim SVD      │     │                 │     │                 │     │                 │
└─────────────────┘     └───────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                                    │
                                                                                    ▼
                                                                          ┌─────────────────┐
                                                                          │  AI SERVICES    │
                                                                          │                 │
                                                                          │ AWS Bedrock     │
                                                                          │ Claude 3 Haiku  │
                                                                          └─────────────────┘
```

## Data Flow

```
OFFLINE: CSV Files ──▶ Generate Embeddings ──▶ Index in OpenSearch ──▶ Ready for Search
         
RUNTIME: User Select ──▶ Get Reading History ──▶ KNN Search ──▶ Generate AI Snippets ──▶ Display Results
```

## Tech Stack Summary

**Frontend:** Streamlit | **Backend:** Python, pandas, sentence-transformers, scikit-surprise  
**Database:** OpenSearch 2.11.0 (Docker) | **AI:** AWS Bedrock Claude 3 Haiku | **ML:** HNSW KNN, SVD Collaborative Filtering