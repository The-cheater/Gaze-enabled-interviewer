-- Clear Examiney.AI Database (retain schema & admin credentials)
-- This script removes all data from tables while preserving the table structure
-- Admin credentials (SUPABASE_URL, SUPABASE_KEY) are stored as environment variables,
-- so they are unaffected by this script.

-- Truncate all tables in order (respecting foreign key relationships)
-- ON DELETE CASCADE is set up, but we truncate in correct order for safety

TRUNCATE TABLE error_logs CASCADE;
TRUNCATE TABLE ocean_reports CASCADE;
TRUNCATE TABLE video_signals CASCADE;
TRUNCATE TABLE question_responses CASCADE;
TRUNCATE TABLE candidate_credentials CASCADE;
TRUNCATE TABLE sessions CASCADE;

-- Confirm all tables are now empty
SELECT 
    'sessions' as table_name, COUNT(*) as row_count FROM sessions
UNION ALL
SELECT 'candidate_credentials', COUNT(*) FROM candidate_credentials
UNION ALL
SELECT 'question_responses', COUNT(*) FROM question_responses
UNION ALL
SELECT 'video_signals', COUNT(*) FROM video_signals
UNION ALL
SELECT 'ocean_reports', COUNT(*) FROM ocean_reports
UNION ALL
SELECT 'error_logs', COUNT(*) FROM error_logs;
