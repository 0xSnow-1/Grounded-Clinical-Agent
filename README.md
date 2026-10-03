<div align="center">

# Grounded Clinical Agent
### Deterministic, Self-Correcting Clinical Decision-Support RAG Agent

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_19_+_TypeScript-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Qdrant](https://img.shields.io/badge/Vector_DB-Qdrant-dc2626?style=flat-square)](https://qdrant.tech)
[![AWS Bedrock](https://img.shields.io/badge/Inference-AWS_Bedrock-232F3E?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/bedrock/)

<p align="center">
  <a href="#key-capabilities">Key Capabilities</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#ui-showcase">UI Showcase</a> •
  <a href="#system-economics">Economics & Latency</a> •
  <a href="#evaluation--benchmarks">Evaluation & Benchmarks</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#roadmap">Roadmap</a>
</p>

</div>

---

> **Domain Scope:** Grounded clinical guidance strictly indexed across institutional oral healthcare guidelines: **CDC Oral Health Surveillance**, **WHO Global Oral Health 2030 Strategies**, **USPSTF Pediatric Caries Guidelines**, and **ADA/AAPD Pit & Fissure Sealant Protocols**.

> **Status:** The benchmark numbers below (baseline $B_0$, run `r009`) were measured on the pre-hybrid, dense-only pipeline at commit `083e93f`. A hybrid-vectorstore rewrite has landed since (commit `406b5a0`): it adds a dual dense + sparse storage layer in `rag/vectorstore.py`, but the retrieval path (`rag/retrieval.py` is currently a stub) and the ingest/agent entry points that depend on it are not yet reconnected, so that work is a work-in-progress on `main` and the B0 numbers remain the only measured baseline.

---

## Key Capabilities

* **Citation-Verified Claim Generation:** Every factual sentence requires explicit citation linking to retrieved clinical passages; the measured hallucination rate is **5.0%** (faithfulness 95.0%) on the 40-question B0 benchmark (`evals/benchmarks.json` `r009`).
* **Automated Self-Correction Loop:** Untraceable claims trigger feedback loops back to the generator (up to 3 retries) before escalating to human review.
* **Dual-LLM Judge Decoupling:** Generation uses **Claude Haiku 4.5** for fast, low-cost drafting; validation uses **Claude Sonnet 4.6** to prevent self-preference bias.
* **Clinical Boundary Safety Containment:** an 8-category robustness suite (role drift, unauthorized pediatric prescriptions, format suppression, and more); measured **80.0%** adversarial defense at baseline B0, with 100% as a Phase-3 target, not a measured value.
* **Production Full-Stack Architecture:** REST API and CopilotKit AG-UI mounting on FastAPI, state checkpointers on PostgreSQL, and a custom React + TypeScript client.

---

## System Architecture

```mermaid
flowchart TD
    UserQuery([Clinician / User Query]) --> Router{Query Router}
    
    Router -->|Non-Clinical / Conversational| StandardAgent[Conversational Agent]
    Router -->|Clinical Inquiry| Retrieval[Qdrant Retrieval: MedEmbed-small-v0.1]
    
    Retrieval --> MedicalAgent[Medical Agent Node: Claude Haiku 4.5]
    MedicalAgent --> Checker{Groundedness Checker: Sonnet 4.6}
    
    Checker -->|Claim Verified| Success([Verified Output + Citations])
    Checker -->|Untraceable Claim & Retry < 3| MedicalAgent
    Checker -->|Untraceable Claim & Retry >= 3| Escalation([Human Clinical Escalation Fallback])
    
    StandardAgent --> ConvEnd([Conversational Output])
    
    style Success fill:#1f7a5f,stroke:#bff3dd,color:#fff
    style Escalation fill:#d4a024,stroke:#fff,color:#000
    style Checker fill:#141415,stroke:#2f6fec,color:#fff
```

### Execution Lifecycle

1. **Deterministic Intent Routing:** Filters out-of-domain and non-clinical conversations from the RAG graph.
2. **Biomedical Vector Retrieval:** Extracts relevant guideline passages using domain-specific `abhinand/MedEmbed-small-v0.1` embeddings.
3. **Structured Claim Generation:** Emits structured `MedicalAnswer` models with itemized claims and confidence ratings.
4. **Independent Groundedness Auditing:** Reviews claims against source evidence and injects corrective critique on failure.
5. **PostgreSQL Checkpointing:** Persists thread execution state and conversation histories across distributed sessions.

---

## UI Showcase

The web interface is built with React 19, TypeScript, and a vanilla CSS design system inspired by [Beautiful UI](https://beautifului.dev), featuring streaming text reveals, tool invocation chips, and interactive evidence inspectors.

<div align="center">

| Verified Clinical Response with Citations | Interactive Claim Evidence Expansion |
| :---: | :---: |
| <img src="docs/images/ui-medical-answer.png" width="450" alt="Verified Response"> | <img src="docs/images/ui-medical-citation-expanded.png" width="450" alt="Expanded Evidence Dropdown"> |

</div>

---

## System Economics

Haiku generation pricing below is Anthropic's published API list price per 1M tokens (accessed 2026-10-02); AWS Bedrock rates can differ by region, so the figures are indicative. **No Sonnet figure is stated:** Claude Sonnet 4.6 is not listed on Anthropic's public pricing page, and Bedrock's current public list stops at Sonnet 4.5, so a verifiable number exists for neither. The previous "% cheaper" comparison was removed because this design runs the Sonnet 4.6 judge on **every** query, so a single-shot Haiku-vs-Sonnet price differential would be misleading. Per-query token consumption is not yet instrumented; latency and cost instrumentation are Phase-5 roadmap items.

| Metric | Haiku 4.5 generator (per 1M tokens) | Basis |
|---|---|---|
| **Input Pricing** | $1.00 | Anthropic published API list price (2026-10-02) |
| **Output Pricing** | $5.00 | Anthropic published API list price (2026-10-02) |
| **Hallucination Rate** | **5.0%** on 40-question benchmark | File-backed: `evals/benchmarks.json` `r009` |

---

## Architectural Decision Records (ADRs)

| Subsystem | Chosen Technology | Alternatives Considered | Trade-Off & Decision Rationale |
|---|---|---|---|
| **Embeddings** | `abhinand/MedEmbed-small-v0.1` (dense) + `Qdrant/bm25` sparse storage (hybrid WIP) | OpenAI `text-embedding-3-small`, `all-MiniLM-L6-v2` | General embeddings underperform on specialized dental ontology (*edentulism*, *caries*, *amoxicillin prophylaxis*). `MedEmbed` captures clinical nomenclature with zero external API latency. Hybrid storage (dense + sparse named vectors, commit `406b5a0`) is merged; the RRF fusion retrieval path is not yet implemented in `rag/retrieval.py` (see Status note). |
| **Orchestration** | LangGraph (Cyclic DAG) | LangChain Linear Chains, LlamaIndex | Linear pipelines cannot loop back to regenerate when claims fail validation. LangGraph enables cyclic state flow between generator and groundness checker. |
| **State Persistence** | PostgreSQL (`PostgresSaver`) | In-memory `MemorySaver`, Redis | In-memory storage drops state on container restart. PostgreSQL ensures persistent audit trails across distributed workers. |
| **Document Parsing** | Docling Chunker | Naive Character Splitter, Recursive Token Splitter | Standard splitters sever clinical tables and dosage matrices across boundaries. Docling preserves markdown table hierarchies and guideline headers. |
| **Dual-LLM Configuration** | Agent: Haiku 4.5<br>Judge: Sonnet 4.6 | Single LLM for both generation and eval | Using the same model to evaluate its own output causes severe self-preference bias. Separating the judge ensures unbiased scoring. |

---

## Evaluation & Benchmarks

The repository includes two automated evaluation suites measuring both retrieval precision and generation faithfulness:

### 1. Comprehensive 40-Question Benchmark
Evaluates **Retrieval HitRate@3**, **MRR**, **Faithfulness**, **Answer Relevance**, and **Safety Defense** across all 5 guideline documents using a unified judge prompt (~$0.40 per full 40-question run):

```bash
python evals/run_comprehensive_eval.py
```

Display historical benchmark ledger:
```bash
python evals/run_comprehensive_eval.py --show
```

### 2. Boundary Robustness & Jailbreak Test
Deterministic evaluation testing 8 adversarial attack categories (instruction override, role drift, format suppression, hypothetical procedure elicitation):

```bash
python evals/run_robustness_eval.py
```

### Current Empirical Benchmark: Baseline B0 (40 Clinical Cases)

The initial baseline evaluation ($B_0$) measures the system using **Naive Dense Vector Search (`MedEmbed-small-v0.1`, $k=5$)** at commit `083e93f`, prior to the hybrid-search work that landed later (see Status note — those hybrid changes are not yet wired end-to-end, so there is no $B_1$ yet):

| Metric | Score (Baseline $B_0$) | Phase 3 Target ($B_1$ Hybrid) | Engineering Rationale |
|---|---|---|---|
| **Retrieval HitRate@3** | **92.5%** | **> 96.0%** | Dense search pulls general documents but misses specific dosage/data tables. |
| **Retrieval MRR** | **0.872** | **> 0.930** | Measures position of ground-truth context; lower on multi-hop queries. |
| **Generation Faithfulness** | **95.0%** | **> 97.0%** | High grounding, but vulnerable to numerical hallucinations when tables are missed. |
| **Answer Relevance** | **85.4%** | **> 92.0%** | **Key Bottleneck (14.6% Gap):** Partial retrieval forces defensive refusals, leaving queries unanswered. |
| **Adversarial Safety Defense** | **80.0%** | **100.0%** | Containment rate against prompt injections and role drift. |

> Detailed failure mode analysis and per-question diagnostics are documented in [`evals/baseline_b0_report.md`](evals/baseline_b0_report.md).

<details>
<summary><b>View Historical Benchmark Ledger</b></summary>

```
==========================================================================================
  HISTORICAL BENCHMARK EVALUATION LEDGER
==========================================================================================
Run    Date        Commit   #Q   Hit@3   Faithful  Relevance  Safety   Notes                    
------------------------------------------------------------------------------------------
r004   2026-08-08  35228b6  20   N/A     0.930     N/A        N/A      Post-checker baseline
r007   2026-08-12  a76db62  20   N/A     0.808     N/A        N/A      Pre-UI baseline
r009   2026-08-16  083e93f  40   0.925   0.950     0.854      80.0%    Baseline B0 (Naive RAG)
==========================================================================================
```

</details>

---

## Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/0xSnow-1/Grounded-Clinical-Agent.git
cd Grounded-Clinical-Agent

# Virtual environment setup
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Configure your `.env` file:

```ini
AWS_BEARER_TOKEN_BEDROCK=your_token_here
BEDROCK_REGION=us-east-1
BEDROCK_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
JUDGE_MODEL_ID=global.anthropic.claude-sonnet-4-6
DB_URI=postgresql://user:password@neon-db-host/dbname?sslmode=require
```

### 2. Ingest Guidelines & Index Vector Store

```bash
python -m rag.ingest
```

### 3. Launch API & Interactive UI

```bash
uvicorn app.main:app --reload --port 8000
```
Open **http://localhost:8000** in your browser.

### 4. Run the offline test suite

```bash
pytest -q --continue-on-collection-errors
```

**Current honest state:** the suite defines 7 test functions across 3 modules, but only `tests/test_faithfulness.py` collects and passes (1 passed). `tests/test_vectorstore.py` and `tests/test_hybrid_vectorstore.py` fail at import because the hybrid rewrite (commit `406b5a0`) removed `build_vectorstore`/`load_vectorstore` from `rag/vectorstore.py` and left `rag/retrieval.py` as a stub, so `from rag.vectorstore import build_vectorstore` and `from rag.retrieval import hybrid_retrieve_chunks` no longer resolve. This is the same disconnect the Status note describes; the two vectorstore modules will import again when the hybrid retrieval path is reconnected (Roadmap Phase 3). The two vectorstore modules also need a local Qdrant reachable at `localhost:6333` (`docker run -p 6333:6333 qdrant/qdrant`) because `rag/vectorstore.py` creates its collection at import time.

<details>
<summary><b>Docker Deployment Instructions</b></summary>

```bash
# Build and run container in background
docker compose up --build -d

# Check live logs
docker compose logs -f

# Verify API endpoints
curl -I http://localhost:8000/docs
```

</details>

---

## Repository Structure

```
├── agent/                           # CopilotKit contribution skills registry (3rd-party, unrelated to the core agent)
├── app/
│   └── main.py                     # FastAPI REST API + CopilotKit AG-UI protocol + static mount
├── data/                           # Clinical guideline storage (PDFs, parsed markdown, Qdrant vectors)
├── docs/                            # Walkthrough + UI screenshots
├── evals/
│   ├── benchmark_40.json           # 40-question comprehensive clinical benchmark dataset
│   ├── eval_metrics.py             # Deterministic IR metrics (HitRate@3, MRR) + Unified Sonnet judge
│   ├── run_comprehensive_eval.py   # 4-metric evaluation harness with historical ledger tracking
│   ├── robustness_prompts.json     # 8 boundary/adversarial test cases
│   ├── run_robustness_eval.py      # Deterministic boundary test runner
│   └── benchmarks.json             # Historical versioned evaluation ledger
├── frontend/                       # React + TypeScript client (Beautiful UI design language)
│   ├── src/
│   │   ├── components/             # ChatMessage, LoadingIndicator, PromptBar, Sidebar
│   │   ├── App.tsx
│   │   └── index.css               # Vanilla CSS design tokens & animations
│   ├── package.json
│   └── vite.config.ts
├── rag/                            # RAG pipeline (hybrid WIP — see Status note)
│   ├── content_processor.py        # Section & table chunker
│   ├── doc_parser.py               # Multi-format doc converter
│   ├── ingest.py                   # Corpus indexing entrypoint (wires to hybrid vectorstore)
│   ├── retrieval.py                # Stub: retrieval functions not yet reimplemented
│   ├── vectorstore.py              # Hybrid storage (dense MedEmbed + sparse Qdrant/bm25) — WIP
│   └── vectorstore_wip.py          # Pre-hybrid dense-only reference (the B0 baseline path)
├── src/                            # LangGraph agent orchestration
│   ├── prompts/                    # Decoupled system prompts
│   ├── agent.py                    # StateGraph with cyclic verification & Postgres checkpointer
│   ├── states.py                   # Pydantic state models (MedicalAnswer, CitedClaim)
│   └── tools.py                    # Qdrant evidence retrieval tool (imports the stub retrieval)
├── tests/                          # Pytest test suite (vectorstore, hybrid-vectorstore, faithfulness)
├── Architectural_Decisions.md      # ADR log
├── Dockerfile                      # Production container spec
├── docker-compose.yml              # Local/cloud orchestration
└── pyproject.toml                  # Python package metadata
```

---

<div id="roadmap"></div>

## Engineering Roadmap: What I Will Do Next

The next development phase focuses on evolving this architecture into an enterprise-scale clinical intelligence platform:

1. **Finish Hybrid Retrieval & Cross-Encoder Reranking (Phase 3, in progress):**
   - Hybrid storage layer (dense `MedEmbed-small-v0.1` + sparse `Qdrant/bm25` named vectors in Qdrant) is merged (commit `406b5a0`), but the retrieval path is a stub: `rag/retrieval.py` still needs `retrieve_chunks_with_scores` and `hybrid_retrieve_chunks`, and the index/build functions behind `rag/ingest.py` / `rag/vectorstore.py` must be reconnected before the agent tool (`src/tools.py`), the hybrid tests, and the ingest entrypoint work again.
   - Then implement Reciprocal Rank Fusion (RRF) to merge candidate pools and add FlashRank cross-encoder reranking to optimize context precision on exact drug dosages and acronyms.
2. **Multi-Agent Specialist Taskforce (Phase 4 Upgrade):**
   - Deconstruct the monolithic medical node into specialized sub-agents: **Triage & Intake**, **Guideline Researcher**, **Drug & Allergy Specialist**, **Groundedness Auditor**, and **Patient Communication Node**.
3. **Multi-Format Ingestion Engine Expansion:**
   - Extend the Docling parsing engine to support automated ingestion of clinical PDFs, DOCX guidelines, and structured clinical database feeds.
4. **CI/CD Quality Gates:**
   - Deploy automated GitHub Actions workflows running unit tests, retrieval evaluation regression checks, and frontend TypeScript build validation on every pull request.
