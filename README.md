## PDF RAG

This project loads PDF files, cleans extracted text, recursively chunks it,
stores embeddings in persistent Chroma, retrieves relevant chunks, and
generates answers using OpenAI.

### Setup

Install dependencies with `uv sync`, then configure `.env`:

```env
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=your-embedding-model
OPENAI_CHAT_MODEL=your-chat-model
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Command line

Index multiple banking PDFs:

```powershell
uv run project index .\src\data\banking --recursive --reset
```

Put redacted statements, loan agreements, card terms, fee schedules, and KYC
policies in `src/data/banking`. The `--reset` option clears old vectors before
building a fresh collection.

Ask a question:

```powershell
uv run project ask "What is the main topic?"
```

The Chroma database is stored in `chroma_db/` and is ignored by Git.

### API

Add `AUTH_USERNAME`, `AUTH_PASSWORD`, and a long random `JWT_SECRET_KEY` to
`.env`, then start the API:

```powershell
uv run rag-api
```

Login:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/login -Method Post -ContentType 'application/json' -Body '{"username":"admin","password":"change-this-password"}'
```

Use the returned bearer token for chatbot requests:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -Headers @{ Authorization = 'Bearer YOUR_TOKEN' } -ContentType 'application/json' -Body '{"question":"What is the main topic?","k":4}'
```
