# Materials and Methods

## 1. Dataset and Data Processing

### 1.1 Dataset Description

This study utilizes the **Chicago Crime Dataset (2001–Present)**, sourced from the City of Chicago's official open data portal (https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2). The dataset comprises approximately **7.8 million structured crime records** spanning over two decades. Each record contains 22 attributes:

**Categorical Attributes:** CRIME_TYPE, DESCRIPTION, LOCATION_DESCRIPTION, IUCR, BEAT, WARD, DISTRICT, COMMUNITY_AREA  
**Temporal Attributes:** DATE, UPDATED_ON, YEAR, MONTH  
**Geospatial Attributes:** LATITUDE, LONGITUDE, X_COORDINATE, Y_COORDINATE, LOCATION (point geometry)  
**Metadata Attributes:** ID, CASE_NUMBER, ARREST (Boolean), DOMESTIC (Boolean)

**Data Type:** Structured tabular data in CSV format  
**Collection Method:** Weekly API pulls from Chicago Data Portal via Apache Airflow orchestration  
**Storage Infrastructure:** PostgreSQL v14+ database containerized with Docker for reproducibility

### 1.2 Data Processing Pipeline

#### Schema Standardization
All column names were transformed from mixed case with spaces (e.g., "Primary Type") to uppercase with underscores (e.g., "CRIME_TYPE"). This standardization:
- Eliminates the need for constant double-quoting in SQL queries
- Enforces consistent database naming conventions
- Simplifies LLM-generated SQL accuracy by reducing syntax complexity

#### Data Type Casting
- **DATETIME Conversion:** String-formatted timestamps (e.g., "01/15/2020 14:30:00") were cast to PostgreSQL TIMESTAMP type. This enables temporal range queries, aggregations by time intervals, and date arithmetic operations (e.g., crimes between 2:00 AM–4:00 AM, crimes per week).
- **NUMERIC Conversion:** X_COORDINATE and Y_COORDINATE fields were cast from strings to FLOAT8 for geospatial proximity calculations and mapping operations.
- **BOOLEAN Conversion:** ARREST and DOMESTIC fields were standardized from mixed text/numeric representations (e.g., "Y", "N", 1, 0, "TRUE") to PostgreSQL BOOLEAN type, enabling efficient filtering.

#### Null Value Handling Strategy

| Column Group | Specific Columns | Handling Strategy | Rationale |
|---|---|---|---|
| **Identifiers** | ID, CASE_NUMBER | Preserved as NOT NULL constraints | Missing identifiers indicate corrupted/phantom records; enforces data integrity |
| **Categorical** | CRIME_TYPE, DESCRIPTION, LOCATION_DESCRIPTION | Retained as NULL or empty strings | Permits downstream COALESCE operations in SQL; respects implicit "unknown" status |
| **Geospatial** | LATITUDE, LONGITUDE, X_COORDINATE, Y_COORDINATE | Preserved as NULL | Prevents incorrect georeferencing at coordinate (0,0); allows spatial queries to filter valid records |
| **Temporal** | DATE, UPDATED_ON, YEAR | Validation gate during casting | Malformed dates trigger query failures, acting as a quality gate; ensures temporal consistency |
| **Logical** | ARREST, DOMESTIC | Retained as NULL | Preserves "Unknown" status distinction (not executed vs. unknown arrest status) |

**Quality Assurance:** Weekly automated schema validation via Airflow confirms structural consistency before LLM interactions.

---

## 2. Problem Formulation

### 2.1 Task Definition

This project addresses the **Text-to-SQL translation problem**—converting natural language questions into executable SQL queries. Formally:

- **Task Type:** Sequential retrieval + code generation task (Semantic parsing)
- **Input:** Natural language question from non-technical stakeholders (e.g., "How many theft crimes occurred in Chicago during 2020?")
- **Output:** (1) Syntactically valid PostgreSQL SELECT statement; (2) Result set (tabular data); (3) Natural language summary
- **Output Format:** Structured triple: `{sql_query: str, results: List[Tuple], answer: str}`

### 2.2 Problem Constraints

- **Read-Only Operations Only:** System must reject INSERT, UPDATE, DELETE, DROP, and ALTER statements
- **Schema Scope:** Questions are constrained to the Chicago Crime database; cross-domain questions (e.g., weather, economics) are rejected
- **Query Complexity:** SELECT-only queries with support for JOINs, GROUP BY, HAVING, ORDER BY, LIMIT, and subqueries
- **Assumptions:**
  - Column names and types are stable during agent runtime (validated weekly)
  - Crime data is refreshed weekly; real-time queries reflect 7-day lag
  - One database connection; no distributed queries

---

## 3. Model / System Architecture

### 3.1 Overall System Workflow

```
Weekly Data Ingestion (Airflow)  
    ↓  
PostgreSQL Database (Docker Container)  
    ↓  
[User Question] → LangGraph AI Agent → [Natural Language Answer]
```

### 3.2 Six-Node StateGraph Architecture

The agent is implemented as a **multi-node agentic workflow** using LangGraph's StateGraph paradigm. Each node performs a specific task and passes state to the next node via conditional or deterministic edges.

**State Definition (TypedDict):**
```
{
  question: str,                    # Original user query
  sql_query: Optional[str],         # Generated or validated SQL
  schema: Optional[str],            # Database schema information
  results: Optional[List],          # Query result set
  answer: Optional[str],            # Final natural language response
  error_log: List[str],             # Execution trace for debugging
  retry_count: int,                 # Query generation retry counter
  is_relevant: bool                 # Relevance check flag
}
```

#### Node 1: Relevance Checker
**Purpose:** Validate that user questions are database-related before expending LLM compute.

**Implementation:**
- Fast-path heuristic: Regex pattern matching + keyword detection (no LLM call)
- Detects irrelevant patterns: math expressions (`r"what\s+is\s+\d+\s*[\+\-\*/]\s*\d+"`), definitions, general knowledge questions
- Verifies presence of query action verbs ("how many", "list", "top") AND data context words ("crime", "chicago", "year")
- On rejection: Invokes LLM to generate contextual explanation (e.g., "I can only help with database queries")
- **Output:** `is_relevant` flag; routes to Schema Fetcher or END

**Latency:** <5ms for irrelevant detection (regex-only path)

#### Node 2: Schema Fetcher
**Purpose:** Retrieve current database schema for SQL generation.

**Implementation:**
- Calls LangChain's `SQLDatabase.get_table_info()` to fetch full schema
- Returns column names, data types, constraints for all accessible tables
- Handles connection errors gracefully; logs failures for debugging
- **Output:** Complete schema string to state

**Latency:** 50–200ms (includes database round-trip)

#### Node 3: Query Generator
**Purpose:** Generate SQL query using LLM with PostgreSQL-specific constraints.

**Model:** Groq LLM (mixtral-8x7b-32768 or openai/gpt-oss-120b)

**Prompt Engineering (System Prompt):**
The system prompt enforces strict rules to prevent invalid SQL:
```
"You are a PostgreSQL expert. Write a precise SELECT query...
CRITICAL RULES:
- CASE SENSITIVITY: Database is case-sensitive for identifiers
- You MUST enclose ALL column and table names in double quotes exactly as they appear
- Use single quotes for string literals ONLY
- Limit results to at most 3 rows unless user asks otherwise
- NEVER issue DML statements (INSERT, UPDATE, DELETE, DROP, ALTER)
- Return ONLY the SQL query without explanations or markdown
- If question cannot be answered with SELECT, respond with: INVALID_REQUEST"
```

**Key Hyperparameters:**
- Temperature: 0 (deterministic output; no randomness)
- Max Tokens: 500 (sufficient for complex queries; prevents token bloat)

**Output:** SQL string or INVALID_REQUEST marker

#### Node 4: Query Validator
**Purpose:** Verify SQL syntax without executing against live data.

**Implementation:**
- Executes `EXPLAIN <query>` (query plan only, no data fetched)
- Regex-validates absence of DML statements: `r'\b(UPDATE|DELETE|DROP|INSERT|ALTER)\b'`
- On validation failure: Increments retry counter; returns to Query Generator
- Max retries: 3 attempts before terminating with user-friendly error
- **Output:** Validated sql_query or NULL for retry

#### Node 5: Query Runner
**Purpose:** Execute validated query and capture results.

**Implementation:**
- Calls `db.run(sql_query, fetch="all")`
- Wraps execution in try-except; logs errors without crashing
- Limits result set to 10 rows for synthesis (avoids token bloat in LLM)
- Logs row count and execution metadata
- **Output:** `results` array to state

#### Node 6: Answer Synthesizer
**Purpose:** Convert SQL results into human-readable natural language summary.

**Implementation:**
- If pre-set answer exists (e.g., from relevance rejection), return as-is
- Format results as JSON string (up to 10 rows)
- Prompt LLM to generate clear, concise summary
- System prompt: "You are a data analyst. Provide clear, concise natural language summary..."
- **Output:** Final `answer` string to user

### 3.3 Conditional Edges

- **Relevance Checker → Schema Fetcher (if is_relevant) OR END**
  - Irrelevant questions terminate immediately with explanation
  - Relevant questions proceed to schema fetching

- **Query Validator → Query Runner (if valid SQL) OR Query Generator (if invalid)**
  - Valid queries proceed to execution
  - Invalid queries loop back to generator (max 3 retries)
  - After 3 failures, route to Answer Synthesizer with error message

### 3.4 Tools and Frameworks

| Component | Technology | Rationale |
|---|---|---|
| **LLM** | Groq API (mixtral-8x7b-32768) | Fast inference (<1s/query); cost-effective; supports long context |
| **Agentic Framework** | LangGraph | Explicit state management; transparent execution flow; conditional routing |
| **Database Driver** | LangChain SQLDatabase + SQLAlchemy | Abstraction over PostgreSQL dialect; built-in schema introspection |
| **Database** | PostgreSQL v14+ | Robust; supports complex queries; production-ready |
| **Containerization** | Docker | Environment reproducibility; isolation; scaling |
| **Data Pipeline** | Apache Airflow | Scheduled weekly ingestion; automatic error handling; audit trail |
| **Language** | Python 3.11+ | Mature ML ecosystem; LangChain/LangGraph native support |

### 3.5 Why This Approach is Suitable

1. **Relevance Pre-filtering (Efficiency):** Regex-based fast path rejects irrelevant questions in <5ms, avoiding wasteful LLM API calls that cost latency and token credits. LLM invoked only for natural explanation.

2. **Explicit State Management (Debuggability):** StateGraph enforces deterministic execution; every node outputs to shared state. Full execution trace available in `error_log` for diagnosing failures.

3. **Retry Logic (Robustness):** Query Validator loop allows LLM self-correction without user intervention. Max 3 retries balances user experience (fast response) vs. accuracy (multiple attempts).

4. **Production Architecture (Scalability):** Separation of concerns:
   - **Data Layer:** Apache Airflow (managed ingestion) + PostgreSQL (persistent storage)
   - **Inference Layer:** LangGraph (stateful orchestration) + Groq (fast LLM)
   - **Logic Layer:** Six modular nodes (easy to swap, test, debug independently)

---

## 4. Training Strategy

### 4.1 Approach

The agent employs **iterative prompt engineering** as its primary optimization mechanism. No model fine-tuning is performed (frozen LLM constraint); all learning occurs via system prompt refinement.

### 4.2 Iterative Optimization Phases

**Phase 1: Baseline Prompt**
- Initial system prompt: "Write SELECT queries based on schema and question"
- Issues: High syntax errors (~40%); incorrect column quoting; DML statements leaked through
- Lesson learned: LLMs need explicit, detailed instructions for SQL generation

**Phase 2: PostgreSQL-Specific Constraints**
- Added rules: "Double-quote ALL identifiers"; "No UPDATE/DELETE/DROP"
- Regex pattern in validator catches DML statements at post-generation stage
- Result: DML violations <2%; syntax errors ~25%
- Lesson learned: Explicit LLM constraints + validator combination is effective

**Phase 3: Schema Context & Real Examples**
- Embedded actual Chicago Crime column names and example queries in prompt
- Added contextual notes about data types (e.g., "CRIME_TYPE is categorical")
- Result: JOIN accuracy improved; LIMIT properly enforced; syntax errors ~8%
- Lesson learned: In-context examples reduce hallucinations

**Phase 4: Relevance Checker & Full Integration**
- Separated relevance logic (fast-path heuristics) from query generation
- Refined error messages for user feedback
- Added retry mechanism with EXPLAIN validation
- Final result: Invalid SQL rate <3%; DML leakage ~0%; user-friendly error messages

### 4.3 Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| **Temperature** | 0 | Deterministic SQL; reproducibility; no random hallucinations |
| **Max Tokens** | 500 | Sufficient for complex multi-table queries; prevents token waste |
| **Retry Limit** | 3 | Balances accuracy (multiple attempts) vs. latency (not excessive) |
| **Relevance Threshold** | 2+ action verbs + data context | Prevents false positives (generic questions) |
| **Result Limit** | 3 rows (default) | Concise answers; token efficiency; user-friendly output |
| **LLM Model** | mixtral-8x7b-32768 | Fast inference; good quality; supports 32k context |

---

## 5. Evaluation Framework

### 5.1 Evaluation Metrics

**Primary Metrics (Directly assess task performance):**

1. **Execution Accuracy (EX)** — Percentage of LLM-generated queries returning identical result sets to ground-truth SQL
   - Calculation: (Queries with matching result sets / Total queries) × 100%
   - Comparison method: Row-for-row deterministic comparison (sort-invariant)
   - **Target: >85%**

2. **Valid SQL Rate (VSR)** — Percentage of generated queries that execute without triggering SQL syntax errors
   - Calculation: (Queries executed successfully / Total queries) × 100%
   - Includes validator stage; DML statement detection counts as invalid
   - **Target: >95%**

3. **Synthesis Quality (SQ)** — Natural language summary clarity and accuracy (1–5 Likert scale)
   - 1 = Incorrect/misleading summary (misrepresents data)
   - 2 = Inaccurate (missing key info)
   - 3 = Accurate but unclear (confusing phrasing)
   - 4 = Clear and accurate (good summary)
   - 5 = Excellent (professional, concise, complete)
   - Evaluation method: LLM-as-a-Judge for 80% of samples; manual annotation for random 20%
   - **Target: Average ≥4.0**

**Secondary Metrics (Optional, for deeper analysis):**
- **Recall@k:** Percentage of schema columns correctly identified in top-k retrieved results (for debugging query generation failures)
- **Mean Reciprocal Rank (MRR):** Average ranking position of correct SQL among k candidate queries (useful for ranking alternative query plans)

### 5.2 Test Dataset

- **Size:** 100 hand-crafted natural language questions with ground-truth SQL queries
- **Stratification by Difficulty:**
  - **Easy (30 questions):** Single-table queries, basic aggregations (COUNT, SUM, MAX), simple WHERE filters, no JOINs
    - Examples: "How many crimes in 2020?", "Show top 10 districts by crime count"
  - **Medium (40 questions):** Multi-table JOINs, GROUP BY with HAVING, date range filters, DISTINCT, ORDER BY
    - Examples: "List crimes by type and district with counts", "Arrests per month in Q4 2021"
  - **Hard (30 questions):** Nested subqueries, complex aggregations, window functions, date arithmetic, NULL handling edge cases
    - Examples: "Show crimes with year-over-year change in theft incidents", "Top districts missing geospatial data"

### 5.3 Baselines

- **Baseline 1 (Rule-Based Lexical Matching):** Hard-coded SQL templates matched via regex. Simple questions only. Expected EX: ~40%
- **Baseline 2 (Zero-Shot LLM without Database-Specific Rules):** Standard LLM prompt without PostgreSQL constraints or example templates. Expected EX: ~60%
- **Proposed System (Optimized Agent):** Full agentic architecture with prompt engineering. Target EX: >85%

### 5.4 Validation Strategy

**Execution Phase:**
1. Query the system with each of 100 test questions
2. Capture LLM-generated SQL, validator output, results, and synthesis

**Comparison Phase:**
1. Execute both LLM-generated and ground-truth queries against read-only PostgreSQL replica
2. Retrieve result sets; normalize by sorting rows lexicographically
3. Perform deterministic equality check: `generated_results == ground_truth_results`
4. Log execution time; flag queries exceeding 5-second threshold as timeouts

**Grading Phase:**
1. **EX Score:** Percentage of queries with matching result sets
2. **VSR Score:** Percentage of queries that pass validator without syntax errors
3. **SQ Score:** Random sample 20 questions → manual annotation; remaining 80 → LLM-as-a-Judge

**Error Analysis:**
- Categorize failures by type: syntax error, wrong table, incorrect joins, missing WHERE clause, etc.
- Stratify by difficulty (Easy/Medium/Hard) to identify weak spots
- Provide confusion matrix (TP/FP/FN/TN) for error types

### 5.5 Reporting Format

Results will be presented as:
- **Summary Table:** EX, VSR, SQ scores overall + stratified by difficulty
- **Confusion Matrix:** Error classification (e.g., 5 syntax errors, 3 missing JOINs, 2 incorrect aggregations)
- **Baseline Comparison:** Proposed system vs. Baseline 1, Baseline 2
- **Sample Outputs:** 3–5 example Q&A pairs showcasing correct and incorrect system behaviors

---

## 6. Implementation Details

### 6.1 Software Environment

**Core Language & Dependencies:**
```
Python 3.11+
LangChain (v0.1+) — LLM orchestration and SQL utilities
LangGraph (v0.2+) — Agentic state management and DAG execution
Groq SDK (v0.9+) — LLM API client
SQLAlchemy (v2.0+) — Database abstraction layer
psycopg2-binary (v2.9+) — PostgreSQL driver
apache-airflow (v2.8+) — Data pipeline orchestration
python-dotenv (v1.0+) — Environment variable management
```

### 6.2 Infrastructure

| Component | Specification | Notes |
|---|---|---|
| **Database** | PostgreSQL v14+ (Docker image: `postgres:14-alpine`) | In-memory indexes for fast queries; ~15 GB storage for full dataset |
| **LLM API** | Groq (groq.com) | Rate limit: 30 req/min; external dependency; requires internet |
| **Inference** | CPU-based | Groq handles GPU acceleration server-side |
| **Container Runtime** | Docker Desktop or Docker Engine | Ensures reproducibility across machines |
| **Orchestration** | Apache Airflow (local DAG scheduler) | Runs data ingestion weekly; can be deployed to cloud (GCP, AWS) |

### 6.3 Hardware & Constraints

- **Compute:** CPU-only machine sufficient (GPU not needed; LLM inference on Groq servers)
- **Storage:** ~15 GB for PostgreSQL database + ~2 GB for application code/notebooks
- **Network:** Stable internet connection required for:
  - Weekly Airflow data pulls from Chicago Data API
  - Real-time LLM API calls to Groq
  - Estimated bandwidth: <100 MB/week for data ingestion
- **Latency Target:** <10 seconds per user query (schema fetch ~100ms + LLM generation ~1000ms + query execution ~1000ms + synthesis ~1000ms)

### 6.4 Environment Setup

**Required Environment Variables (.env file):**
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
DB_USER=admin
DB_PASSWORD=pswd
DB_HOST=100.xx.xx.xx
DB_PORT=xxxx
DB_NAME=xxxxxxxxx
```

**Never commit .env file; add to .gitignore:**
```
.env
.env.local
```

### 6.5 Assumptions & Limitations

**Assumptions:**
1. PostgreSQL schema (column names, types) remains static during agent runtime; validated weekly via Airflow
2. Chicago Crime dataset updated weekly; real-time queries reflect 7-day lag
3. Groq API available at ≤30 req/minute rate limit
4. Single database connection; no connection pooling for distributed queries

**Limitations:**
1. **Scope:** Agent restricted to Chicago Crime database; no cross-database or domain-specific questions
2. **Query Complexity:** Limited support for window functions, CTEs, and advanced analytics
3. **Real-time Data:** Weekly refresh lag; cannot answer "crimes today" with current data
4. **Concurrency:** Single-user agent; no built-in multi-user session management

---
