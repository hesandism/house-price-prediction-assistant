# House Price Intelligence Assistant

A tool-calling LLM agent for the Sri Lankan residential property market. It
answers questions by choosing between two tools:

- **`predict_price`** — a Random Forest regressor trained on 20,000 Sri Lankan
  listings, for "what would this house cost" questions.
- **`retrieve_docs`** — RAG over Central Bank of Sri Lanka reports (Real Estate
  Market Analysis + Financial Stability Reviews), for "why did prices move"
  questions.

It can use both in one answer — an estimate plus the market reasoning behind it.

```
frontend/  Next.js 16 chat UI          -> talks to the API over HTTP
api/       FastAPI wrapper             -> exposes the agent as POST /ask
src/       agent, tools, training, ETL
data/      house_prices_srilanka.csv   (20k rows)
docs/      source PDFs for retrieval
models/    saved model + encoder (.joblib)
chroma_db/ persisted vector store
```

## Prerequisites

- Python 3.12+
- Node.js 20+
- A **Groq** API key — free at <https://console.groq.com/keys>

## Setup

### 1. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your key

```powershell
Copy-Item .env.example .env           # macOS/Linux: cp .env.example .env
```

Then edit `.env` and set `GROQ_API_KEY` to your own key.

> **Model IDs go stale.** Groq retires models without notice. If a request fails
> with `404 model_not_found`, pick a current tool-calling model from
> <https://console.groq.com/docs/models> and update `HOUSE_AGENT_MODEL`.

### 3. Build the artifacts

Both are committed as generated output, so you only need this if `models/` or
`chroma_db/` is empty, or after changing the dataset or the PDFs.

```powershell
python src\eda.py             # optional: summary stats + plots to notebooks/eda_outputs/
python src\train_model.py     # -> models/price_model.joblib, models/model_columns.joblib
python src\ingest_docs.py     # -> chroma_db/ (embeds docs/*.pdf; a few minutes, CPU-only)
```

`train_model.py` fits both Linear Regression and Random Forest, prints a
comparison, and saves whichever has the lower RMSE. Expect Random Forest to win
at about **R² 0.955**.

### 4. Frontend dependencies

```powershell
cd frontend
npm install
```

## Running it

You need **two terminals**. Start the API first — the UI is useless without it.

**Terminal 1 — API** (from the project root):

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --port 8000
```

Check it: <http://127.0.0.1:8000/health> reports the resolved model and where
that value came from —

```json
{"status":"ok","model":"groq:openai/gpt-oss-120b","model_source":".env"}
```

Interactive API docs are at <http://127.0.0.1:8000/docs>.

> **If `model_source` says `shell environment`,** a `HOUSE_AGENT_MODEL` exported
> in your terminal is outranking `.env` — `load_dotenv()` never overwrites a
> variable that is already set. Editing `.env` will appear to do nothing. Clear
> it and restart:
>
> ```powershell
> Remove-Item Env:\HOUSE_AGENT_MODEL -ErrorAction SilentlyContinue
> ```
>
> Also note `--reload` watches `.py` files only, so `.env` edits always need a
> manual restart.

**Terminal 2 — frontend:**

```powershell
cd frontend
npm run dev
```

Open <http://localhost:3000> and ask a question.

The UI calls `http://localhost:8000` by default. To point it elsewhere, create
`frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

If you change the API's port or host, add that origin to `ALLOWED_ORIGINS` in
[api/main.py](api/main.py) too, or the browser will block the request.

## Using it without the browser

The agent has a REPL, which is the fastest way to test changes:

```powershell
python src\agent.py              # interactive; 'quit' to exit
python src\agent.py --examples   # run three canned questions and exit
```

You can also exercise either tool on its own, bypassing the LLM entirely:

```powershell
python src\tools.py
```

Or hit the API directly:

```powershell
curl -X POST http://127.0.0.1:8000/ask `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"How have Colombo house prices changed recently?\"}'
```

## Things worth knowing

**Size means land extent.** The dataset's only real size measure is `perch`
(land area, ~272.25 sqft), which is also how Sri Lankan listings quote size.
`predict_price` takes `area_sqft` and converts it to perches. Do **not** map
house area onto `kitchen_area_sqft` — that column is the kitchen alone
(35–250 sqft), so any real house area falls far outside the training range and
the tree model saturates, returning the same price for 900 and 1800 sqft.

**The model refuses to extrapolate quietly.** Outside the trained perch range it
still returns a number, but appends an explicit unreliability caveat, because
tree models collapse to a boundary leaf rather than extending the trend.

**Never feed `price_lkr` in as a feature.** It is the target. Including it in
`NUMERIC_COLUMNS` in [src/train_model.py](src/train_model.py) makes the model
return its own input (R² = 1.0, zero error) while learning nothing.
`predict_price` detects this in the saved artifact and refuses to report a price
rather than passing off a lookup as a prediction.

**Both Chroma collections must share one embedding model.** `housing_docs`
(PDFs, from `ingest_docs.py`) and `markdown_docs` (from `build_vectorstore.py`)
are queried together and their cosine distances are merged and re-ranked. They
both use `all-MiniLM-L6-v2` with normalised embeddings; changing that in one
place and not the others makes the merged ranking meaningless.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `404 model_not_found` | `HOUSE_AGENT_MODEL` names a retired Groq model. Check `/health` for `model_source` — a stale shell variable outranks `.env`, and `--reload` ignores `.env` edits. |
| `503 Agent is not configured` | Provider package missing or no API key. `pip install langchain-groq` and check `.env`. |
| `Model artifact not found` | Run `python src\train_model.py`. |
| `No Chroma store at ...` | Run `python src\ingest_docs.py`. |
| UI shows "Could not reach the API" | The API isn't running, or its origin isn't in `ALLOWED_ORIGINS`. |
| `Prediction unavailable: the saved model is invalid` | The artifact was trained with the price leak. Retrain. |
