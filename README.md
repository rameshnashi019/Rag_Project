## Document RAG

This project loads PDF and plain-text files, cleans extracted text, recursively
chunks it, stores embeddings in persistent Chroma, retrieves relevant chunks,
and generates answers using OpenAI.

### What the project does

This is a retrieval-augmented generation (RAG) application for asking
document-grounded questions about engineering and quality records. Documents
are loaded from a file or directory, converted into searchable chunks, stored
in a persistent vector database, and retrieved as context for each answer.

The application supports:

- PDF files, with one source document created per page
- Plain-text `.txt` files, with one source document created per file
- Recursive directory indexing
- Configurable text cleaning for headers, footers, watermarks, and replacements
- Source metadata in answers, including the source file and page when available
- A command-line interface and an authenticated FastAPI web API

### Processing flow

```text
PDF / TXT files
	|
	v
Document loading -> Text cleaning -> Recursive chunking
	|                   |                  |
	+-------------------+------------------+
			    v
		    Chroma vector database
			    |
			    v
		  Retrieval -> OpenAI answer
```

### System architecture

```mermaid
flowchart TB
	classDef source fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
	classDef process fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
	classDef model fill:#fff3e0,stroke:#f57c00,color:#e65100
	classDef storage fill:#fce4ec,stroke:#c2185b,color:#880e4f
	classDef api fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c

	CLI[CLI: project index / ask<br/>Python argparse] --> INPUT
	WEB[Web UI<br/>FastAPI + Uvicorn] --> AUTH

	subgraph S1[1. Data sources]
		INPUT[PDF and TXT files<br/>src/data or user directory]
	end

	subgraph S2[2. Ingestion and parsing]
		LOAD[PDF: LangChain PyPDFLoader + pypdf<br/>TXT: pathlib UTF-8 reader<br/>Output: LangChain Document]
	end

	subgraph S3[3. Cleaning and chunking]
		CLEAN[Text normalization<br/>Unicode, headers, footers, watermarks]
		CHUNK[RecursiveCharacterTextSplitter<br/>chunk size, overlap, start index]
		META[Metadata<br/>source, page, chunk index, chunk count]
		CLEAN --> CHUNK --> META
	end

	subgraph S4[4. Embedding and indexing]
		EMBED[Primary: OpenAIEmbeddings<br/>Model: OPENAI_EMBEDDING_MODEL]
		FALLBACK[Fallback: HuggingFaceEmbeddings<br/>Model: HF_EMBEDDING_MODEL]
		DB[(Chroma vector database<br/>langchain-chroma, cosine distance<br/>persistent storage: chroma_db)]
		EMBED -. failure .-> FALLBACK
		EMBED --> DB
		FALLBACK --> DB
	end

	subgraph S5[5. Query and retrieval]
		AUTH[JWT authentication<br/>python-jose, /login]
		QUESTION[Question<br/>CLI or POST /chat]
		DECOMPOSE[Optional query decomposition<br/>ChatOpenAI, OPENAI_CHAT_MODEL]
		SEARCH[Chroma similarity search<br/>or MMR re-ranking<br/>k and fetch_k]
		CONTEXT[Top chunks formatted with<br/>source and page metadata]
		AUTH --> QUESTION --> DECOMPOSE --> SEARCH --> CONTEXT
		QUESTION --> SEARCH
	end

	subgraph S6[6. Grounded answer generation]
		LLM[ChatOpenAI<br/>Model: OPENAI_CHAT_MODEL<br/>temperature: 0]
		RESPONSE[Answer with source citations<br/>or explicit no-answer message]
		LLM --> RESPONSE
	end

	INPUT --> LOAD --> CLEAN
	META --> EMBED
	DB --> SEARCH
	CONTEXT --> LLM
	RESPONSE --> WEB
	RESPONSE --> CLI

	class INPUT source
	class LOAD,CLEAN,CHUNK,META,QUESTION,DECOMPOSE,SEARCH,CONTEXT process
	class EMBED,FALLBACK,LLM model
	class DB storage
	class CLI,WEB,AUTH,RESPONSE api
```

The vertical flow separates the indexing path from the question path. During
indexing, source files are parsed, cleaned, chunked, enriched with metadata,
embedded, and persisted in Chroma. During a question, the API authenticates the
user, optionally decomposes the query, retrieves relevant chunks, and sends
only that context to `ChatOpenAI` for a grounded response.

### Project structure

| Path | Purpose |
| --- | --- |
| `src/ingestion` | Loads PDF and text source files |
| `src/preprocessing` | Normalizes and cleans extracted text |
| `src/chunking` | Splits documents into searchable chunks |
| `src/vectordb` | Stores and queries Chroma vectors |
| `src/reterieval` | Retrieves relevant chunks for a question |
| `src/generation` | Generates grounded answers with OpenAI |
| `src/project` | Provides the pipeline and CLI entry point |
| `src/api` | Serves the FastAPI backend and web interface |
| `src/data` | Example engineering and quality documents |
| `chroma_db` | Local persistent vector database |

### Setup

Install dependencies with `uv sync`, then configure `.env`:

```env
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=your-embedding-model
OPENAI_CHAT_MODEL=your-chat-model
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Command line workflow

Run the indexing command only when you add or change source documents, cleaning rules,
chunking settings, or the embedding model. You do not need to run it for every
question.

For a fresh index, clear the old vectors and process all supported documents:

```powershell
uv run project index .\src\data --recursive --reset
```

Place supported PDF and text files in `src/data` or another directory. The
`--reset` option clears old vectors before building a fresh collection.

After indexing, start the API:

```powershell
uv run rag-api
```

Then open the chatbot:

```text
http://127.0.0.1:8000
```

For command-line questions, use:

```powershell
uv run project ask "What is the main topic?"
```

### Example questions

After indexing the documents, you can ask questions such as:

- What material is currently approved for the P104 pump housing?
- Why is Polymer Grade B being considered as a replacement?
- What are the highest risks of replacing Polymer Grade A with Polymer Grade B?
- What are the operating temperature and pressure requirements for P104?
- What validation tests are required before approving the material change?
- Why is extended thermal cycling required for Polymer Grade B?
- What happened during the previous CP-420 Polymer Grade B validation?
- Did Polymer Grade B pass pressure and vibration testing?
- What caused the historical housing cracks?
- What supplier-quality issue was found with Supplier-Z?
- What corrective actions did Supplier-Z implement?
- What inspections are required after coolant aging?
- What are the acceptance criteria for the P104 material-change validation?
- Is Polymer Grade B approved for the P104 housing yet?
- What evidence is required before the material change can be released?

The Chroma database is stored in `chroma_db/` and is ignored by Git.

### Retrieval evaluation

Use labeled questions to measure whether retrieval returns the expected source
documents. The evaluator counts each source file once, even when multiple
chunks from that file are returned.

```python
from evalution import RetrievalCase, evaluate_retrieval, summarize_retrieval
from project import RAGPipeline
from reterieval import retrieve

pipeline = RAGPipeline(persist_directory="chroma_db_staging")
cases = [
	RetrievalCase.from_sources(
		"What material is currently approved for the P104 pump housing?",
		["SPEC-P104-REV6.txt"],
	),
	RetrievalCase.from_sources(
		"What happened during the previous CP-420 Polymer Grade B validation?",
		["TR-1845_CP420_Thermal_Validation.txt", "FA-2218_CP420_Housing_Crack.txt"],
	),
]

scores = evaluate_retrieval(
	cases,
	lambda query, k: retrieve(
		query, pipeline.store, k=k, decompose=False, use_mmr=False
	).documents,
	k=4,
)
summary = summarize_retrieval(scores)
print(f"precision@4: {summary.mean_precision_at_k:.2%}")
print(f"recall@4: {summary.mean_recall_at_k:.2%}")
```

Precision@k is the fraction of the top `k` results that belong to an expected
source. Recall@k is the fraction of expected sources found in the top `k`.
Track these values on a fixed evaluation set whenever documents, chunking,
embeddings, or retrieval settings change.

### Configuration

The application reads settings from `.env`. `OPENAI_API_KEY` is required for
OpenAI embeddings or answer generation. `HF_EMBEDDING_MODEL` is used by the
local Hugging Face embedding configuration when selected. Chroma data is
stored in `CHROMA_PERSIST_DIRECTORY` when that setting is provided; otherwise
the default directory is `chroma_db`.

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

