import streamlit as st
import pandas as pd
from opensearchpy import OpenSearch
import boto3
import json
import os
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class LibraryDatabaseApp:
    def __init__(self):
        self.client = None
        self.bedrock_client = None
        self.setup_connections()
        self.load_data()
    
    def load_data(self):
        self.rental_history = pd.read_csv('data/rental_history.csv')
        self.book_catalog = pd.read_csv('data/book_catalog.csv')
        
        # Convert dates
        self.rental_history['checkout_date'] = pd.to_datetime(self.rental_history['checkout_date'])
        self.rental_history['return_date'] = pd.to_datetime(self.rental_history['return_date'])
        
        # Ensure book_id columns have consistent data types
        self.rental_history['book_id'] = self.rental_history['book_id'].astype(int)
        self.book_catalog['book_id'] = self.book_catalog['book_id'].astype(int)
    
    def setup_connections(self):
        host = os.getenv('OPENSEARCH_HOST', 'localhost')
        port = int(os.getenv('OPENSEARCH_PORT', '9200'))
        use_ssl = os.getenv('OPENSEARCH_USE_SSL', 'false').lower() == 'true'
        
        if use_ssl:
            # AWS managed OpenSearch
            region = os.getenv('AWS_REGION', 'us-east-2')
            credentials = boto3.Session().get_credentials()
            awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
            
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=None,
            )
        else:
            # Local OpenSearch
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                use_ssl=False,
                verify_certs=False,
            )
        
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-2'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def get_user_reading_history(self, user_id):
        rental_history = pd.read_csv('data/rental_history.csv')
        user_books = rental_history[rental_history['user_id'] == user_id].tail(3)
        
        book_details = []
        for _, row in user_books.iterrows():
            try:
                response = self.client.get(index="books", id=row['book_id'])
                book_details.append(response['_source'])
            except Exception as e:
                st.error(f"Error fetching book {row['book_id']}: {e}")
        
        return book_details
    
    def find_similar_books(self, book_embedding, num_recommendations=5):
        query = {
            "query": {
                "knn": {
                    "content_embedding": {
                        "vector": book_embedding,
                        "k": num_recommendations
                    }
                }
            }
        }
        
        try:
            response = self.client.search(index="books", body=query)
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            st.error(f"Error searching for similar books: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, num_recommendations=5):
        try:
            # Query for books with collaborative_features
            query = {
                "query": {"match_all": {}},
                "size": 100
            }
            
            response = self.client.search(index="books", body=query)
            books_with_collab = [hit['_source'] for hit in response['hits']['hits'] 
                               if 'collaborative_features' in hit['_source']]
            
            if not books_with_collab:
                return []
            
            # Get user's reading history book IDs
            user_books = self.rental_history[self.rental_history['user_id'] == user_id]['book_id'].tolist()
            
            # Score books based on collaborative features
            recommendations = []
            for book in books_with_collab:
                if book['book_id'] not in user_books:
                    # Use collaborative_features as similarity score
                    collab_score = sum(book['collaborative_features'])
                    recommendations.append((book, collab_score))
            
            # Sort by collaborative score and return top recommendations
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return [book for book, score in recommendations[:num_recommendations]]
            
        except Exception as e:
            st.error(f"Error getting collaborative recommendations: {e}")
            return []

    def generate_recommendation_snippet(self, user_book, recommended_book):
        prompt = f"""
        Create a short, engaging recommendation snippet (1-2 sentences) explaining why someone who enjoyed "{user_book['title']}" by {user_book['author']} would like "{recommended_book['title']}" by {recommended_book['author']}.
        
        User's book: {user_book['description']}
        Recommended book: {recommended_book['description']}
        
        Start with "Because you liked..." and make it personal and specific.
        """
        
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId="us.anthropic.claude-3-haiku-20240307-v1:0",
                accept="application/json",
                contentType="application/json"
            )
            
            response_body = json.loads(response.get('body').read())
            return response_body['content'][0]['text'].strip()
        
        except Exception as e:
            return f"You might enjoy this book based on your interest in {user_book['genre']} stories."
    
    def show_rental_history(self):
        st.header("📚 Rental History")
        
        # Get unique students for filter
        students = ['All Students'] + sorted(self.rental_history['student_name'].unique().tolist())
        selected_student = st.selectbox("Filter by Student:", students)
        
        # Filter data
        if selected_student != 'All Students':
            filtered_history = self.rental_history[self.rental_history['student_name'] == selected_student]
        else:
            filtered_history = self.rental_history
        
        # Merge with book details
        history_with_books = filtered_history.merge(
            self.book_catalog[['book_id', 'title', 'author', 'genre']], 
            on='book_id', 
            how='left'
        )
        
        # Sort by checkout date (latest first)
        history_with_books = history_with_books.sort_values('checkout_date', ascending=False)
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rentals", len(history_with_books))
        with col2:
            st.metric("Unique Students", len(filtered_history['student_name'].unique()))
        with col3:
            st.metric("Unique Books", len(filtered_history['book_id'].unique()))
        with col4:
            current_rentals = filtered_history[filtered_history['return_date'] > datetime.now()]
            st.metric("Current Rentals", len(current_rentals))
        
        st.divider()
        
        # Display rental history table
        st.subheader("Recent Rentals")
        
        # Format the display dataframe
        display_df = history_with_books[['student_name', 'title', 'author', 'genre', 'checkout_date', 'return_date']].copy()
        display_df['checkout_date'] = display_df['checkout_date'].dt.strftime('%Y-%m-%d')
        display_df['return_date'] = display_df['return_date'].dt.strftime('%Y-%m-%d')
        display_df.columns = ['Student', 'Book Title', 'Author', 'Genre', 'Checkout Date', 'Return Date']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    
    def show_recommendations(self):
        st.header("🎯 AI Book Recommendations")
        st.write("Get personalized book recommendations based on reading history!")
        
        try:
            self.client.info()
        except Exception:
            st.error("Cannot connect to OpenSearch. Please check your configuration.")
            return
        
        # Get unique user IDs for dropdown
        user_ids = sorted(self.rental_history['user_id'].unique().tolist())
        user_names = []
        for uid in user_ids:
            name = self.rental_history[self.rental_history['user_id'] == uid]['student_name'].iloc[0]
            user_names.append(f"{name} ({uid})")
        
        selected_user = st.selectbox("Select Student:", user_names)
        
        if selected_user and st.button("Get Recommendations"):
            user_id = selected_user.split('(')[1].split(')')[0]
            
            with st.spinner("Finding your perfect next read..."):
                reading_history = self.get_user_reading_history(user_id)
                
                if not reading_history:
                    st.warning("No reading history found for this student.")
                    return
                
                st.subheader("📖 Recent Reading History")
                for book in reading_history:
                    st.write(f"• **{book['title']}** by {book['author']}")
                
                st.subheader("🎯 Based on your reading history:")
                
                all_recommendations = []
                for user_book in reading_history:
                    similar_books = self.find_similar_books(
                        user_book['content_embedding'], 
                        num_recommendations=3
                    )
                    
                    for rec_book in similar_books:
                        if rec_book['book_id'] not in [b['book_id'] for b in reading_history]:
                            if rec_book not in all_recommendations:
                                all_recommendations.append((user_book, rec_book))
                
                seen_books = set()
                unique_recommendations = []
                for user_book, rec_book in all_recommendations:
                    if rec_book['book_id'] not in seen_books:
                        unique_recommendations.append((user_book, rec_book))
                        seen_books.add(rec_book['book_id'])
                
                for i, (user_book, rec_book) in enumerate(unique_recommendations[:5]):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{rec_book['title']}**")
                            st.write(f"*by {rec_book['author']} ({rec_book['publication_year']})*")
                            st.write(f"**Genre:** {rec_book['genre']}")
                            
                            snippet = self.generate_recommendation_snippet(user_book, rec_book)
                            st.write(f"💡 {snippet}")
                            
                            with st.expander("Book Description"):
                                st.write(rec_book['description'])
                        
                        with col2:
                            st.write(f"**ISBN:** {rec_book['isbn']}")
                        
                        st.divider()
                
                # Collaborative filtering recommendations
                st.subheader("👥 Based on other readers like you:")
                
                collab_recommendations = self.get_collaborative_recommendations(user_id, num_recommendations=5)
                
                if collab_recommendations:
                    for rec_book in collab_recommendations:
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**{rec_book['title']}**")
                                st.write(f"*by {rec_book['author']} ({rec_book['publication_year']})*")
                                st.write(f"**Genre:** {rec_book['genre']}")
                                
                                st.write("💡 Readers with similar tastes enjoyed this book.")
                                
                                with st.expander("Book Description"):
                                    st.write(rec_book['description'])
                            
                            with col2:
                                st.write(f"**ISBN:** {rec_book['isbn']}")
                            
                            st.divider()
                else:
                    st.info("No collaborative filtering data available yet.")
    
    def show_book_browser(self):
        st.header("📖 Book Browser")
        st.write("Browse all books in the database and view their stored data")
        
        try:
            self.client.info()
        except Exception:
            st.error("Cannot connect to OpenSearch. Please check your configuration.")
            return
        
        # Get all books from OpenSearch
        try:
            response = self.client.search(
                index="books",
                body={
                    "size": 100,
                    "query": {"match_all": {}},
                    "sort": [{"book_id": {"order": "asc"}}]
                }
            )
            
            books = [hit['_source'] for hit in response['hits']['hits']]
            
            if not books:
                st.warning("No books found in the database.")
                return
            
            st.subheader(f"Found {len(books)} books")
            
            # Create dropdown with book titles and IDs
            book_options = [f"{book['book_id']}: {book['title']} by {book['author']}" for book in books]
            selected_book_option = st.selectbox("Select a book to view details:", book_options)
            
            if selected_book_option:
                # Extract book_id from selection
                book_id = int(selected_book_option.split(':')[0])
                selected_book = next(book for book in books if book['book_id'] == book_id)
                
                st.divider()
                
                # Display book details in a nice format
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"{selected_book['title']}")
                    st.write(f"**Author:** {selected_book['author']}")
                    st.write(f"**Genre:** {selected_book['genre']}")
                    st.write(f"**Publication Year:** {selected_book['publication_year']}")
                    st.write(f"**ISBN:** {selected_book['isbn']}")
                    
                    with st.expander("Description"):
                        st.write(selected_book['description'])
                
                with col2:
                    st.metric("Book ID", selected_book['book_id'])
                    
                    if 'content_embedding' in selected_book:
                        embedding_len = len(selected_book['content_embedding'])
                        st.metric("Content Embedding", f"{embedding_len} dims")
                    
                    if 'collaborative_features' in selected_book:
                        collab_len = len(selected_book['collaborative_features'])
                        st.metric("Collaborative Features", f"{collab_len} dims")
                
                # Show raw JSON data
                st.subheader("Raw OpenSearch Document")
                st.json(selected_book)
                
        except Exception as e:
            st.error(f"Error loading books: {e}")
    
    def run_app(self):
        st.title("🏢 Readington Library Management System")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to:", ["Rental History", "Book Recommendations", "Book Browser"])
        
        if page == "Rental History":
            self.show_rental_history()
        elif page == "Book Recommendations":
            self.show_recommendations()
        elif page == "Book Browser":
            self.show_book_browser()

def main():
    st.set_page_config(
        page_title="Library Database System",
        page_icon="📚",
        layout="wide"
    )
    
    app = LibraryDatabaseApp()
    app.run_app()

if __name__ == "__main__":
    main()