# Project Plan: AI-Powered Book Recommendations for a School District

## Introduction

This document outlines a project to develop an AI-powered book recommendation system for a school district. The primary goal is to increase student reading by providing personalized, data-driven suggestions. We'll leverage Large Language Models (LLMs) to enhance the capabilities of the district's librarians, moving beyond their current reliance on anecdotal data. This plan details the project's architecture, the technical implementation for a proof-of-concept (POC), and the key components required to build a successful solution.

## Problem Statement

During initial interviews, librarians noted that students often read books they personally recommend. However, these recommendations are based on a librarian's memory of what other students have enjoyed, which is not scalable and can lead to a narrow range of suggestions. The librarians have access to the complete history of borrowed books and their electronic card catalog but lack the tools to effectively use this data. The core challenge is to augment the librarians' expertise with a system that can systematically analyze the entire dataset to uncover deeper patterns and provide more diverse and personalized recommendations.

## Proposed LLM-Based Solution

We propose a hybrid recommendation system that combines the strengths of collaborative filtering and content-based filtering, both powered by LLMs.

1. **Rich Content Embeddings**: We'll use an LLM (a Sentence-Transformer model) to read the descriptions and summaries of all books in the electronic card catalog. This creates a rich, semantic understanding of each book's content, allowing the system to identify thematically similar books far more effectively than simple keyword or genre matching.
2. **Collaborative Filtering**: We'll use a matrix factorization model on the book rental history to learn the latent tastes of students and the characteristics of books based on borrowing patterns. This helps students discover new interests by finding books liked by peers with similar reading histories.
3. **Dynamic & Explainable Recommendations**: When the system suggests a book, a generative LLM (like GPT or Gemini) will create a custom, one-sentence snippet explaining why the book is a good match (e.g., "Because you liked the magical school setting in Harry Potter, you might enjoy The Name of the Wind."). This makes the recommendations more compelling and transparent.

# System Architecture
The architecture is divided into two parts: an offline pipeline for data processing and an online application for serving real-time recommendations.

## Offline Processing Pipeline
* **Data Sources**: The system will ingest the **Electronic Card Catalog** (for book metadata) and the **Book Rental History** (for user-item interactions).
* **Embedding Generation**:
    * A Python script using **Sentence-Transformers** will generate content-based embeddings from book descriptions.
    * A second Python script using the **Surprise** library will generate collaborative filtering embeddings from the rental data using **SVD (Matrix Factorization)**.
* **Data Indexing**: A final script will process and load all book metadata and their learned embeddings into an Elasticsearch index. Elasticsearch will act as our vector database.

## Online Serving Application
* **Frontend UI**: A **Streamlit** web application will provide a simple, interactive user interface for librarians and students.
* **Backend & Logic**:
    * When a user enters a book they liked, the app retrieves that book's embedding from Elasticsearch.
    * It then performs a **vector similarity search** in Elasticsearch to find the 'N' most similar books based on their embeddings.
    * The app makes a real-time call to a generative LLM to create the personalized "Why you'll like this" snippet for each recommendation.
    * The final recommendations are displayed in the Streamlit UI.

# Implementation Plan
Here's a high-level overview of the technical steps to create the POC.
0. **Pre-requisites**:
    * Setup an AWS account with free tier access to AWS Bedrock available for API access to a LLM for chat.
    * Setup a Elasticsearch serverless environment complient with their 14-day free trial.
1. **Environment Setup**:
    * Set up a Python virtual environment.
    * Install the necessary libraries: pandas, streamlit, sentence-transformers, scikit-surprise, elasticsearch, and an LLM provider library (e.g., Amazon Bedrock).
2. **Data Preperation**:
    * Create an Elasticsearch index named books with a mapping that includes a dense_vector field for the embeddings.
    * Generate sample test data for the book catalog and rental history suitable to demo this application (balance this tradeoff: small enough for simplicity, large enough to show value).
    * Write a script to read the book catalog and rental history into Pandas DataFrames.
3. **Offline Embedding Generation & Indexing**:
    * Write a Python script (process_data.py) to:
        1. Generate content embeddings for all book descriptions using a Sentence-Transformer model.
        2. (Optional for initial POC, but explainable) Generate collaborative embeddings using the Surprise library on the rental data.
        3. Combine the book metadata and the embeddings into a single JSON document for each book.
        4. Upload these documents to the books index in Elasticsearch.
4. **Build the Streamlit Application**:
    * Create the app.py for your Streamlit app.
    * Implement the UI with a text input box for a librarian to enter a users name.
    * Write the backend logic to:
        1. Query Elasticsearch to get the embedding vector for the last 3 books the user read
        2. Use Elasticsearch's knn search functionality to find similar books based on vector similarity
        3. For each of those 3 books in the users reading history, call a generative LLM API to create a custom recommendation snippet.
        4. Display the results, including book titles, authors, and the generated snippets, in a clean and readable format.
