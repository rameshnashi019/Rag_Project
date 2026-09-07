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
flowchart LR
	classDef source fill:#e8f5e9,stroke:#388e3c,color:#1b5e20
	classDef process fill:#e3f2fd,stroke:#1976d2,color:#0d47a1
	classDef storage fill:#fff3e0,stroke:#f57c00,color:#e65100
	classDef output fill:#f3e5f5,stroke:#7b1fa2,color:#4a148c

	subgraph Sources[1. Source Documents]
		PDF[PDF files]
		TXT[TXT files]
		DATA[src/data directory]
		PDF --> DATA
		TXT --> DATA
	end

	subgraph Ingestion[2. Ingestion and Processing]
		LOAD[Document loader]
		CLEAN[Text cleaning]
		CHUNK[Recursive chunking]
		META[Source and page metadata]
		LOAD --> CLEAN --> CHUNK
		CHUNK --> META
	end

	subgraph Indexing[3. Embedding and Storage]
		EMBED[Embedding provider]
		CHROMA[(Persistent Chroma DB)]
		EMBED --> CHROMA
	end

	subgraph Query[4. Retrieval]
		QUESTION[User question]
		RETRIEVE[Similarity retrieval]
		CONTEXT[Relevant chunks and metadata]
		QUESTION --> RETRIEVE --> CONTEXT
	end

	subgraph Answer[5. Response Generation]
		OPENAI[OpenAI generator]
		RESPONSE[Grounded answer with sources]
		OPENAI --> RESPONSE
	end

	DATA --> LOAD
	META --> EMBED
	CHROMA --> RETRIEVE
	CONTEXT --> OPENAI

	CLI[CLI: project index / ask] --> DATA
	WEB[FastAPI and web UI] --> QUESTION
	RESPONSE --> WEB

	class PDF,TXT,DATA source
	class LOAD,CLEAN,CHUNK,META,QUESTION,RETRIEVE,CONTEXT process
	class EMBED,CHROMA storage
	class OPENAI,RESPONSE,CLI,WEB output
```

The indexing path runs from the source directory through loading, cleaning,
chunking, embedding, and persistent Chroma storage. The question path retrieves
relevant chunks and passes them to the answer generator along with source
metadata so responses can be traced back to the input documents.

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

