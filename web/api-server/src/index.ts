import express from 'express'
import cors from 'cors'
import Groq from 'groq-sdk'
import { Pool } from 'pg'
import { config } from 'dotenv'
import { resolve } from 'path'

// Load env vars from web/.env.local (one level up from api-server/)
config({ path: resolve(__dirname, '../../.env.local') })

const PORT = parseInt(process.env.PORT ?? '3001', 10)
const MODEL = process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile'

// ── Singletons ────────────────────────────────────────────────────────────────

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY?.trim() })

const pool = new Pool({
  connectionString: process.env.DATABASE_URL?.trim(),
  ssl: process.env.DATABASE_SSL === 'true' ? { rejectUnauthorized: false } : undefined,
  max: 5,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
})

// ── Relevance checker ─────────────────────────────────────────────────────────

const ACTION_VERBS = [
  'how many', 'how often', 'what are the', 'what is the', 'what percentage',
  'list', 'show', 'find', 'get', 'count', 'total', 'sum', 'average',
  'top', 'most', 'least', 'rank', 'filter', 'between', 'after', 'before',
  'during', 'in the year', 'which', 'for each', 'compare', 'distribution',
  'percentage', 'median', 'calculate', 'identify', 'breakdown',
]

const DATA_WORDS = [
  'crime', 'chicago', 'crimes', 'report', 'reported', 'database',
  'table', 'records', 'data', 'district', 'year', 'date', 'type',
  'arrest', 'victim', 'incidents', 'cases', 'statistics',
  'breakdown', 'distribution',
]

const IRRELEVANT_PATTERNS = [
  /what\s+is\s+\d+\s*[\+\-\*/]\s*\d+/,
  /what is the meaning/,
  /^what is\s+\w+\s*\?*\s*$/,
  /^define\s+/,
  /^explain\s+/,
  /^tell me about\s+/,
  /who (are|is|was|were)/,
  /when (are|is|was|were)/,
  /why (are|is|was|were)/,
]

function isRelevant(question: string): boolean {
  const q = question.toLowerCase().trim()
  for (const p of IRRELEVANT_PATTERNS) if (p.test(q)) return false
  return ACTION_VERBS.some(v => q.includes(v)) && DATA_WORDS.some(w => q.includes(w))
}

// ── Schema fetcher ────────────────────────────────────────────────────────────

async function fetchSchema(): Promise<string> {
  const client = await pool.connect()
  try {
    const { rows } = await client.query(`
      SELECT table_name, column_name, data_type, character_maximum_length
      FROM information_schema.columns
      WHERE table_schema = 'public'
      ORDER BY table_name, ordinal_position
    `)
    const tables: Record<string, string[]> = {}
    for (const r of rows) {
      if (!tables[r.table_name]) tables[r.table_name] = []
      const type = r.character_maximum_length
        ? `${r.data_type}(${r.character_maximum_length})`
        : r.data_type
      tables[r.table_name].push(`  "${r.column_name}" ${type}`)
    }
    return Object.entries(tables)
      .map(([t, cols]) => `CREATE TABLE ${t} (\n${cols.join(',\n')}\n)`)
      .join('\n\n')
  } finally {
    client.release()
  }
}

// ── SQL generator ─────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a PostgreSQL expert. Write a precise SELECT query for the question.
Use only column names that exist in the provided schema.
Never modify data (no INSERT / UPDATE / DELETE / DROP).

=== DATABASE RELATIONSHIPS ===
The crime_incidents table stores location as numeric codes:
  crime_incidents."DISTRICT"       (integer) → district_names."district_id"
  crime_incidents."COMMUNITY_AREA" (integer) → community_area_names."community_area_id"
  crime_incidents."WARD"           (integer) → ward_names."ward_id"

Lookup tables:
  district_names      : "district_id" (int PK), "district_name" (text)
  community_area_names: "community_area_id" (int PK), "community_area_name" (text)
  ward_names          : "ward_id" (int PK), "ward_name" (text)

=== CRITICAL RULES ===
1. QUOTING IDENTIFIERS: Enclose ALL column and table names in double quotes:
     "YEAR", "DISTRICT", "CRIME_TYPE", "crime_incidents", "district_names"

2. QUOTING STRING VALUES: ALWAYS use single quotes for string/text values:
     WHERE "CRIME_TYPE" = 'THEFT'      ← CORRECT
     WHERE "CRIME_TYPE" = "THEFT"      ← WRONG (double quotes = column name, not a value)
   Numeric values need no quotes: WHERE "YEAR" = 2020

3. RESOLVING LOCATION NAMES — most important rule:
   When the user mentions a district, community area, or ward BY NAME (e.g. "Central",
   "Lincoln Park", "Austin"), you MUST join with the lookup table and filter by name.
   NEVER hardcode numeric IDs or guess them.

   CORRECT example for "central district":
     SELECT COUNT(*) FROM "crime_incidents" ci
     JOIN "district_names" dn ON ci."DISTRICT" = dn."district_id"
     WHERE dn."district_name" ILIKE 'Central'
       AND ci."YEAR" = 2020;

   WRONG example:
     WHERE "DISTRICT" = '001'   ← hardcoded code, wrong

4. Use ILIKE for name matching so capitalisation does not matter.
5. Do NOT add a LIMIT clause unless the question explicitly asks for top-N or a specific number.
6. Use CTEs (WITH clauses) for multi-step aggregations or window function pipelines.
7. For percentage calculations use: ROUND(100.0 * numerator / NULLIF(denominator, 0), 2).
8. Return ONLY the SQL query — no explanations, no markdown fences.
9. If the question cannot be answered with SELECT, respond with exactly: INVALID_REQUEST`

async function generateSQL(question: string, schema: string, lastError?: string): Promise<string | null> {
  let userContent = `Schema:\n${schema}\n\nQuestion: ${question}`
  if (lastError) {
    userContent += `\n\nPrevious query failed validation with this error:\n${lastError}\nPlease fix the query.`
  }
  const res = await groq.chat.completions.create({
    model: MODEL,
    temperature: 0,
    messages: [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: userContent },
    ],
  })
  let sql = res.choices[0].message.content?.trim() ?? ''
  if (sql.includes('INVALID_REQUEST')) return null
  sql = sql.replace(/^```(?:sql)?\n?|```\s*$/gi, '').trim()
  return sql || null
}

// ── SQL validator ─────────────────────────────────────────────────────────────

async function validateSQL(sql: string): Promise<{ valid: boolean; error?: string }> {
  if (/\b(UPDATE|DELETE|DROP|INSERT|ALTER)\b/i.test(sql))
    return { valid: false, error: 'DML statement detected' }
  const client = await pool.connect()
  try {
    await client.query(`EXPLAIN ${sql}`)
    return { valid: true }
  } catch (e) {
    return { valid: false, error: e instanceof Error ? e.message : String(e) }
  } finally {
    client.release()
  }
}

// ── Query runner ──────────────────────────────────────────────────────────────

async function runQuery(sql: string): Promise<Record<string, unknown>[]> {
  const client = await pool.connect()
  try {
    const { rows } = await client.query(sql)
    return rows
  } finally {
    client.release()
  }
}

// ── Answer synthesizer ────────────────────────────────────────────────────────

async function synthesizeAnswer(
  question: string,
  results: Record<string, unknown>[],
): Promise<string> {
  const sample = JSON.stringify(results.slice(0, 10), null, 2)
  const res = await groq.chat.completions.create({
    model: MODEL,
    temperature: 0.3,
    messages: [
      {
        role: 'system',
        content:
          'You are a data analyst. Give a clear, concise natural-language summary of the SQL query results. Be direct and informative.',
      },
      {
        role: 'user',
        content: `Question: ${question}\n\nSQL Results (${results.length} rows total):\n${sample}\n\nSummarise.`,
      },
    ],
  })
  return res.choices[0].message.content ?? 'No summary available.'
}

async function irrelevantResponse(question: string): Promise<string> {
  const res = await groq.chat.completions.create({
    model: MODEL,
    temperature: 0.3,
    messages: [
      {
        role: 'system',
        content:
          "You are a helpful assistant focused on Chicago crime data analysis. The user's question is not related to the database. Politely explain that their question is outside your scope. Be concise but friendly.",
      },
      { role: 'user', content: `User question: ${question}` },
    ],
  })
  return res.choices[0].message.content ?? 'I can only answer questions about Chicago crime data.'
}

// ── Express app ───────────────────────────────────────────────────────────────

const app = express()
app.use(cors())
app.use(express.json())

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', model: MODEL, timestamp: new Date().toISOString() })
})

app.post('/api/chat', async (req, res) => {
  try {
    const { question } = req.body as { question?: string }

    if (!question?.trim()) {
      res.status(400).json({ error: 'Question is required.' })
      return
    }

    console.log(`[query] ${question.slice(0, 80)}`)

    if (!isRelevant(question)) {
      const answer = await irrelevantResponse(question)
      res.json({ answer, sql_query: null, results: null })
      return
    }

    let schema: string
    try {
      schema = await fetchSchema()
    } catch (e) {
      console.error('[schema] DB connection failed:', e)
      res.status(503).json({ error: 'Database connection failed.' })
      return
    }

    let sql: string | null = null
    let lastError: string | undefined
    for (let attempt = 0; attempt < 3 && !sql; attempt++) {
      const candidate = await generateSQL(question, schema, lastError)
      if (!candidate) break
      const { valid, error } = await validateSQL(candidate)
      if (valid) {
        sql = candidate
      } else {
        lastError = error
      }
    }

    if (!sql) {
      res.json({
        answer: 'Unable to generate a valid SQL query after 3 attempts.',
        sql_query: null,
        results: null,
      })
      return
    }

    let results: Record<string, unknown>[] = []
    try {
      results = await runQuery(sql)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      res.json({ answer: `Query error: ${msg}`, sql_query: sql, results: null })
      return
    }

    const answer = await synthesizeAnswer(question, results)
    console.log(`[done]  ${results.length} rows — ${answer.slice(0, 60)}…`)
    res.json({ answer, sql_query: sql, results })
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    console.error('[chat] error:', msg)
    res.status(500).json({ error: 'Internal server error.', details: msg })
  }
})

app.listen(PORT, () => {
  console.log('\n  Chicago Crime — API Server')
  console.log('  ──────────────────────────────────────────────────')
  console.log(`  Local:    http://localhost:${PORT}`)
  console.log(`  Health:   http://localhost:${PORT}/health`)
  console.log('  ──────────────────────────────────────────────────')
  console.log('  To expose publicly via Tailscale Funnel:')
  console.log(`    /Applications/Tailscale.app/Contents/MacOS/Tailscale funnel --bg ${PORT}`)
  console.log('  Check your public URL:')
  console.log('    /Applications/Tailscale.app/Contents/MacOS/Tailscale funnel status')
  console.log('')
})
