# AI Book Recommendations - Project Instructions

## Tech Stack & Architecture
- **Data Processing**: Python with pandas, sentence-transformers, scikit-surprise
- **Vector Database**: OpenSearch (local Docker or AWS managed)
- **LLM Integration**: AWS Bedrock (Claude 3 Haiku)
- **Frontend**: Streamlit web app
- **Data Flow**: Offline processing → OpenSearch indexing → Real-time recommendations

## Key Implementation Details
- Use `opensearch-py` client (not elasticsearch)
- OpenSearch index mapping uses `knn_vector` type with `dimension: 384` for embeddings
- Content embeddings: SentenceTransformer model `all-MiniLM-L6-v2`
- Collaborative filtering: SVD with scikit-surprise library
- Model ID for Bedrock: `us.anthropic.claude-3-haiku-20240307-v1:0`
- Environment variables in `.env` file for configuration

## Development Guidelines
- NEVER create files unless absolutely necessary for achieving your goal
- ALWAYS prefer editing existing files to creating new ones
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- Run `python src/process_data.py` to generate embeddings and index data
- Run `streamlit run app.py` to start the web application