# Meilisearch RAG - Conversational Search 

A simple, extensible conversational search system built with **retrieval-augmented generation (RAG)**. It combines:

- ⚡ **Meilisearch** – Lightweight, fast, hybrid vector + keyword search
- 🧠 **LangChain** – Chainable framework for LLM-powered applications
- 🚀 **FastAPI** – Modern, high-performance Python web framework

---

### Features 
- ✅ **Modular Architecture** - Meilisearch and LangChain can easily be swapped out for other providers. 
- ✅ **Conversational Search** - Combines the power of hyrbrid search and summarization for human-friendly responses
- ✅ **Document indexing** — Index raw documents, chunk and embed them, and search via similarity or hybrid queries

It also allows users to index documents (including the creation of embeddings), and perform similarity and semantic searches. 

## Quick Start 

### Prerequisites   
* UV installed 
* Docker/colima or alternative running 

### Quick start setup instructions 
1. Clone repo 
2. Ensure docker/colima is running 
3. Create .env file `touch .env.docker` (`docker compose up` expects .docker extension)
4. Populate .env file with required variables: 
```bash
MODEL_ID=your_model_id
EMBEDDER_NAME=your_model_name
MEILI_MASTER_KEY=your_master_key
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
REGION=your_region
meilisearch_url=http://meilisearch:7700
```
1. Launch using docker `docker compose up`
2. Populate Meilisearch with data `uv run ./scripts/index_documents.py` (requires UV - This will chunk your content, embed it, and index it in Meilisearch using a dataset of BBC articles from 2005.)
3. Have a play with the API using the `/docs` route (Open your browser at: http://localhost:8000/docs)

If running locally, replace `.env.docker` with `.env.local`

### Example use 
#### Request 
Call the conversational search endpoint: 
```bash 
curl -X 'POST' \
  'http://localhost:8000/search/conversational' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Tell me about chelsea",
  "limit": 5
}' 
```
#### Reponse: 
```json
{
  "summary": "Chelsea Football Club has been making headlines ...",
  "sources": [
    {
      "id": "342__0",
      "chuck": "mourinho plots \\ remaining output truncated... " 
```

### Setting up env 
The project uses UV, so all that should be required is: `uv venv`. You may have to `uv sync` too. 

### Configuration 

All settings are managed via `pydantic-settings (BaseSettings)` in `src/conf/`

### Project Structure 
```
src/
├── app.py # FastAPI routes
├── services/ # Business logic
├── infrastructure/ # LangChain, Meilisearch, + base classes etc.
├── domain/ # schemas, dataclasses, etc.
└── conf/ # Settings and config
```
## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit your changes
4. Open a pull request

Please write tests and follow PEP8-style conventions.

## To-do 
* Make async 
* improve test coverage 
* abstract langchain embedder implementation away from search_service. 
* Allow users to dynamically index content of different shapes into different indexes defined at runtime 
* Add automation/github actions 
