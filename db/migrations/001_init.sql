
-- No manual migrations needed.
-- LangGraph's PostgresStore.setup() creates its own tables automatically.
-- -- Long-term memory: conversation summaries
-- CREATE TABLE IF NOT EXISTS memory_summaries (
--     id SERIAL PRIMARY KEY,
--     user_id TEXT NOT NULL,
--     session_id TEXT,
--     summary TEXT NOT NULL,
--     created_at TIMESTAMPTZ DEFAULT NOW(),
--     updated_at TIMESTAMPTZ DEFAULT NOW()
-- );

-- CREATE INDEX IF NOT EXISTS idx_memory_summaries_user_id ON memory_summaries (user_id);


-- -- Long-term memory: structured facts (key-value)
-- CREATE TABLE IF NOT EXISTS memory_facts (
--     id SERIAL PRIMARY KEY,
--     user_id TEXT NOT NULL,
--     fact_key TEXT NOT NULL,
--     fact_value TEXT NOT NULL,
--     created_at TIMESTAMPTZ DEFAULT NOW(),
--     updated_at TIMESTAMPTZ DEFAULT NOW(),
--     UNIQUE (user_id, fact_key)
-- );

-- CREATE INDEX IF NOT EXISTS idx_memory_facts_user_id ON memory_facts (user_id);