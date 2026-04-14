"""
sql_agent.py  ─  LangGraph SQL Agent
=====================================
Answers natural-language questions against a SQL database by chaining LLM
calls with real SQL tool execution.

Graph flow:
    START
      └─► list_tables       (hard-coded: lists all tables, no LLM)
            └─► call_get_schema  (LLM picks relevant tables, fetches DDL)
                  └─► get_schema       (tool executor: runs sql_db_schema)
                        └─► generate_query  (LLM writes the SQL query)
                              ├─► END         (if LLM gives a final answer without a query)
                              └─► check_query (LLM reviews the SQL)
                                    └─► run_query  (tool executor: runs sql_db_query)
                                          └─► generate_answer  (LLM interprets results → END)

Public API:
    build_agent(model, db, tools) → compiled LangGraph agent
    run_agent(agent, question)    → stream and print results
"""

import json
import os
import re
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Optional Groq + SQL utilities for turnkey setup
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_sql(text: str) -> str | None:
    """Pull the first SQL statement out of a plain-text string."""
    m = re.search(
        r"(?:SELECT|WITH|INSERT|UPDATE|DELETE).*?(?=;|\Z)",
        text or "", re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    # Strip markdown code fences if present
    sql = re.sub(r"^```sql\n?|```\s*$", "", m.group(0).strip(), flags=re.IGNORECASE).strip()
    return sql or None


def _parse_tool_json(text: str, expected_name: str) -> dict | None:
    """
    Parse a JSON-like tool call from plain model text.

    Why this exists:
        llama3.1 (and some other Ollama models) sometimes ignore tool_choice="any"
        and instead print a raw JSON object in the message body, e.g.:
            {"name": "sql_db_query", "parameters": {"query": "SELECT ..."}}
        This function converts that blob into a LangChain tool_call dict.

    Returns None if:
        - no JSON block is found
        - JSON is malformed
        - name doesn't match expected_name
        - required args keys are missing
    """
    block = re.search(r"\{[\s\S]*\}", text or "")
    if not block:
        return None
    try:
        obj = json.loads(block.group(0))
    except Exception:
        return None

    name = obj.get("name")
    # Support both "args" (LangChain style) and "parameters" (Ollama plain-text style)
    args = obj.get("args") or obj.get("parameters")

    if name != expected_name or not isinstance(args, dict):
        return None

    # Normalise: some models use "table" instead of "table_names"
    if name == "sql_db_schema" and "table_names" not in args:
        if not args.get("table"):
            return None
        args = {"table_names": args["table"]}

    # A query tool call without an actual query string is useless
    if name == "sql_db_query" and not args.get("query"):
        return None

    return {"name": name, "args": args, "id": "parsed_from_text", "type": "tool_call"}


# ── Agent builder ──────────────────────────────────────────────────────────────

def build_agent(model, db, tools):
    """
    Build and compile a LangGraph SQL agent.

    Parameters
    ----------
    model  : LangChain chat model (ChatOllama, ChatGoogleGenerativeAI, …)
    db     : LangChain SQLDatabase wrapper
    tools  : list returned by SQLDatabaseToolkit.get_tools()

    Returns
    -------
    CompiledStateGraph  (call .stream() or .invoke() on it)
    """

    # ── Resolve tools ────────────────────────────────────────────────────────
    get_schema_tool  = next(t for t in tools if t.name == "sql_db_schema")
    run_query_tool   = next(t for t in tools if t.name == "sql_db_query")
    list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")

    # ToolNode auto-routes AIMessage.tool_calls to the matching tool
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")
    run_query_node  = ToolNode([run_query_tool],  name="run_query")

    # ── System prompts ───────────────────────────────────────────────────────
    QUERY_PROMPT = (f"You have run a SQL query to answer a user's question.\n"
        "Give a concise, direct answer to the user's question based on the result above. "
        "Do not run any more queries.")

    QUERY_PROMPT_1 = (
        f"""You are a {db.dialect} SQL expert. 
            Write a precise SELECT query for the question and call sql_db_query.
            Use only column names that exist in the provided schema.
            Use only the table names that are relevant to the question, as indicated by the schema tool results.
            Never modify data (no INSERT / UPDATE / DELETE / DROP).
            Make sure to use proper joins, group by, order by, and limit clauses as needed to answer the question accurately and concisely.
            Create new columns as needed with WITH statements or subqueries, but avoid unnecessary complexity.

            Rules:
            - CASE SENSITIVITY: This database is case-sensitive for identifiers. 
            - You MUST enclose ALL column names and table names in double quotes (e.g., "Year", "Primary Type") exactly as they appear in the schema.
            - Use double quotes for string literals (e.g., WHERE "Primary Type" = 'THEFT').
            - Limit results to at most 3 rows unless the user asks otherwise.
            - Only select the columns that are relevant to the question.
            - Order results by a relevant column when it helps the answer.
            - NEVER issue DML statements (INSERT, UPDATE, DELETE, DROP, etc.).
            """
    )

    CHECK_PROMPT = (
        f"""You are a SQL expert with a strong attention to detail.
            Double-check the {db.dialect} query for common mistakes:
            - NOT IN with NULL values
            - UNION vs UNION ALL
            - BETWEEN for exclusive ranges
            - Data type mismatches in predicates
            - Unquoted / mis-quoted identifiers
            - Wrong number of function arguments
            - Incorrect JOIN columns

            If mistakes are found, rewrite the query.
            Otherwise, reproduce the original query exactly.
            Then call the run_query tool to execute it."""
    )

    # ── Node definitions (closures — capture model, db, tools from outer scope) ─

    def list_tables(state: MessagesState):
        """
        Step 1 — Hard-coded: list all DB tables without calling the LLM.
        Injects the table list into the message history as context for later nodes.
        """
        tc = {"name": "sql_db_list_tables", "args": {}, "id": "list_tables", "type": "tool_call"}
        result = list_tables_tool.invoke(tc)
        return {"messages": [
            AIMessage(content="", tool_calls=[tc]),             # synthetic tool-call trigger
            result,                                             # tool response (the table list)
            AIMessage(content=f"Available tables: {result.content}"),  # readable summary
        ]}

    def call_get_schema(state: MessagesState):
        """
        Step 2 — LLM chooses which tables are relevant and fetches their DDL.
        Fallback: if the model ignores tool_choice="any", select all tables.
        """
        response = model.bind_tools([get_schema_tool], tool_choice="any").invoke(state["messages"])

        if not response.tool_calls:
            parsed = _parse_tool_json(response.content or "", "sql_db_schema")
            response.tool_calls = [parsed] if parsed else [{
                "name": "sql_db_schema",
                "args": {"table_names": ", ".join(db.get_usable_table_names())},
                "id": "fallback_schema",
                "type": "tool_call",
            }]
        return {"messages": [response]}

    # Step 3 is get_schema_node (ToolNode) — no Python function needed.

    def generate_query(state: MessagesState):
        """
        Step 4 — LLM writes a SQL query based on the schema and question.

        Three-pass fallback for Ollama models that don't reliably use tool calls:
          Pass 1: bind_tools call — model should call sql_db_query with the query
          Pass 2: if query is missing/empty, try extracting SQL from response content
          Pass 3: if still nothing, call the model without tool binding and extract SQL
                  from plain text (most Ollama models write SQL reliably in plain text)
        """
        msgs = [{"role": "system", "content": QUERY_PROMPT}] + state["messages"]
        response = model.bind_tools([run_query_tool], tool_choice="any").invoke(msgs)

        # Try to get a SQL string from any source in the response
        sql = (response.tool_calls[0].get("args") or {}).get("query") if response.tool_calls else None
        if not sql:
            parsed = _parse_tool_json(response.content or "", "sql_db_query")
            sql    = (parsed or {}).get("args", {}).get("query") or _extract_sql(response.content or "")

        # Pass 3: model returned nothing usable — ask for SQL as plain text
        if not sql:
            plain_prompt = (
                QUERY_PROMPT
                + "\nWrite ONLY the SQL query. No explanation, no JSON, no markdown."
            )
            plain = model.invoke([{"role": "system", "content": plain_prompt}] + state["messages"])
            sql = _extract_sql(plain.content or "")

        if sql:
            response.tool_calls = [{
                "name": "sql_db_query",
                "args": {"query": sql},
                "id": "final_query",
                "type": "tool_call",
            }]
        return {"messages": [response]}

    def check_query(state: MessagesState):
        """
        Step 5 — LLM reviews the SQL for mistakes before it runs.

        Root cause of KeyError: 'query'
        --------------------------------
        should_continue previously routed here whenever ANY tool call existed,
        including sql_db_schema calls (which have "table_names", not "query").
        Fix 1: should_continue now only routes here for sql_db_query with a query.
        Fix 2: this node uses .get() instead of direct key access as a second guard.
        """
        last_tc = state["messages"][-1].tool_calls[0]
        query = (last_tc.get("args") or {}).get("query") or ""  # safe: never raises KeyError

        if not query:
            # No valid SQL to review — return early with no tool_calls → should_continue → END
            resp = AIMessage(content="Could not find a SQL query to review.", tool_calls=[])
            resp.id = state["messages"][-1].id
            return {"messages": [resp]}

        response = model.bind_tools([run_query_tool], tool_choice="any").invoke([
            {"role": "system", "content": CHECK_PROMPT},
            {"role": "user",   "content": query},
        ])

        if not response.tool_calls:
            parsed = _parse_tool_json(response.content or "", "sql_db_query")
            sql    = (parsed or {}).get("args", {}).get("query") or _extract_sql(response.content or "") or query
            response.tool_calls = [{
                "name": "sql_db_query",
                "args": {"query": sql},
                "id": "fallback_checked",
                "type": "tool_call",
            }]

        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def generate_answer(state: MessagesState):
        """
        Final step — LLM reads the query results and gives a natural-language answer.
        No tool binding: the model must not generate another query here.
        """
        ANSWER_PROMPT = (
            "You have just executed a SQL query. Based on the results in the conversation, "
            "give a concise, direct answer to the user's original question. "
            "Do not run any more queries."
        )
        msgs = [{"role": "system", "content": ANSWER_PROMPT}] + state["messages"]
        response = model.invoke(msgs)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["check_query", "__end__"]:
        """
        Conditional edge after generate_query.

        Route to check_query ONLY when:
            - the last message contains a tool call
            - AND that call is sql_db_query
            - AND it has a non-empty query string

        This is the primary fix for KeyError: 'query' — we never enter
        check_query unless a valid query is confirmed to exist.
        """
        last = state["messages"][-1]
        if last.tool_calls:
            tc = last.tool_calls[0]
            if tc.get("name") == "sql_db_query" and (tc.get("args") or {}).get("query"):
                return "check_query"
        return END  # LLM gave a final text answer or no valid query was produced

    # ── Wire and compile the graph ───────────────────────────────────────────
    g = StateGraph(MessagesState)

    g.add_node(list_tables)
    g.add_node(call_get_schema)
    g.add_node(get_schema_node, "get_schema")   # tool executor: sql_db_schema
    g.add_node(generate_query)
    g.add_node(check_query)
    g.add_node(run_query_node, "run_query")      # tool executor: sql_db_query
    g.add_node(generate_answer)

    g.add_edge(START,             "list_tables")
    g.add_edge("list_tables",     "call_get_schema")
    g.add_edge("call_get_schema", "get_schema")
    g.add_edge("get_schema",      "generate_query")
    g.add_edge("check_query",     "run_query")
    g.add_edge("run_query",       "generate_answer")  # results → final answer, no loop
    g.add_edge("generate_answer", END)

    g.add_conditional_edges("generate_query", should_continue)

    return g.compile()

import os
from dotenv import load_dotenv

load_dotenv()



# ── Convenience builders ─────────────────────────────────────────────────────

def build_groq_agent(
    db_uri: str,
    model_name: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    api_key: Optional[str] = None,
):
    """Build a SQL agent powered by Groq.

    Parameters
    ----------
    db_uri : str
        Database connection string (sqlite:///..., postgresql://..., etc.).
    model_name : str
        Groq model identifier (e.g., "mixtral-8x7b-32768", "gemma-7b-it").
    temperature : float
        Sampling temperature; keep low for deterministic SQL.
    api_key : str, optional
        Groq API key; falls back to GROQ_API_KEY environment variable.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    groq_key = os.getenv("GROQ_API_KEY") #api_key or os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("Set GROQ_API_KEY or pass api_key to build_groq_agent().")

    model = ChatGroq(model=model_name, temperature=temperature, api_key=groq_key)
    db = SQLDatabase.from_uri(db_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    return build_agent(model, db, tools)


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_agent(agent, question: str):
    """
    Stream the agent for a question and print each step.
    SQL being executed and query results are highlighted for easy inspection.
    """
    print(f"\n{'='*60}\nQuestion: {question}\n{'='*60}")

    for step_no, step in enumerate(
        agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ),
        start=1,
    ):
        msg = step["messages"][-1]
        msg.pretty_print()

        # Highlight SQL whenever the agent is about to execute a query
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "sql_db_query" and tc.get("args", {}).get("query"):
                print(f"\n── [Step {step_no}] SQL to execute ──────────────────────")
                print(tc["args"]["query"])

        # Show actual results returned by the database
        if msg.__class__.__name__ == "ToolMessage" and getattr(msg, "name", "") == "sql_db_query":
            print(f"\n── [Step {step_no}] Query result ────────────────────────")
            print(msg.content)
