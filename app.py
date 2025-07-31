import streamlit as st
import pandas as pd
from opensearchpy import OpenSearch
import boto3
import json
import os
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv

load_dotenv()

class BookRecommendationApp:
    def __init__(self):
        self.client = None
        self.bedrock_client = None
        self.setup_connections()
    
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
            "size": num_recommendations,
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
    
    def run_app(self):
        st.title("📚 AI-Powered Book Recommendations")
        st.write("Get personalized book recommendations based on your reading history!")
        
        try:
            self.client.info()
        except Exception:
            st.error("Cannot connect to OpenSearch. Please check your configuration.")
            return
        
        user_id = st.text_input("Enter Student ID:", placeholder="e.g., student_001")
        
        if st.button("Get Recommendations") and user_id:
            with st.spinner("Finding your perfect next read..."):
                reading_history = self.get_user_reading_history(user_id)
                
                if not reading_history:
                    st.warning("No reading history found for this student ID.")
                    return
                
                st.subheader("📖 Your Recent Reading History")
                for book in reading_history:
                    st.write(f"• **{book['title']}** by {book['author']}")
                
                st.subheader("🎯 Recommended Books")
                
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

def main():
    st.set_page_config(
        page_title="Book Recommendations",
        page_icon="📚",
        layout="wide"
    )
    
    app = BookRecommendationApp()
    app.run_app()

if __name__ == "__main__":
    main()