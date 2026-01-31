"""
Database Initialization Script.

Creates the codelumen database, schemas, and initial tables.
This script is self-contained and does not rely on external SQL files.
"""

import logging
import sys
import psycopg2
from psycopg2 import sql
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if it exists
from utils.config.env_loader import load_env
load_env()

from services.db_config import get_db_config, CODELUMEN_DATABASE
from services.postgresql_connection_pool import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SQL DEFINITIONS
# =============================================================================

SQL_CREATE_SCHEMAS = """
-- Create Schemas for CodeLumen
-- This script creates the codelumen and codelumen_cache schemas

-- Create codelumen schema for main application tables
CREATE SCHEMA IF NOT EXISTS codelumen;

-- Create codelumen_cache schema for cache metadata
CREATE SCHEMA IF NOT EXISTS codelumen_cache;

-- Grant permissions to postgres user
GRANT USAGE ON SCHEMA codelumen TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA codelumen TO postgres;

GRANT USAGE ON SCHEMA codelumen_cache TO postgres;
GRANT ALL PRIVILEGES ON SCHEMA codelumen_cache TO postgres;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA codelumen GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA codelumen_cache GRANT ALL ON TABLES TO postgres;
"""

SQL_CREATE_TABLES = """
-- Create Tables for CodeLumen
-- This script creates all tables in the codelumen schema

-- Table: scan_history
-- Purpose: Stores analysis scan history and results (Phase 1)
CREATE TABLE IF NOT EXISTS codelumen.scan_history (
    id SERIAL PRIMARY KEY,
    project_path VARCHAR(500) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    files_scanned INTEGER DEFAULT 0,
    chunks_analyzed INTEGER DEFAULT 0,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result_json JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_scan_history_status CHECK (status IN ('completed', 'failed', 'in_progress', 'cancelled'))
);

-- Table: experiences
-- Purpose: Stores agent experiences for learning (Phase 3)
CREATE TABLE IF NOT EXISTS codelumen.experiences (
    experience_id VARCHAR PRIMARY KEY,
    agent_name VARCHAR NOT NULL,
    goal TEXT NOT NULL,
    goal_understanding_json JSONB NOT NULL,
    strategy_used_json JSONB NOT NULL,
    outcome_json JSONB NOT NULL,
    performance_metrics_json JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success_level VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_experiences_success_level CHECK (success_level IN ('full', 'partial', 'failed') OR success_level IS NULL)
);

-- Table: knowledge_base
-- Purpose: Stores shared knowledge for agent collaboration (Phase 4)
CREATE TABLE IF NOT EXISTS codelumen.knowledge_base (
    knowledge_id VARCHAR PRIMARY KEY,
    knowledge_type VARCHAR NOT NULL,
    content_json JSONB NOT NULL,
    metadata_json JSONB,
    agent_name VARCHAR NOT NULL DEFAULT 'Unknown',
    relevance_tags TEXT[],
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    expires_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    file_hash VARCHAR,
    project_path VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_knowledge_base_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT chk_knowledge_base_access_count CHECK (access_count >= 0)
);

-- Table: swarm_skills
-- Purpose: Stores swarm-level skills and patterns (Phase 5)
CREATE TABLE IF NOT EXISTS codelumen.swarm_skills (
    skill_id VARCHAR PRIMARY KEY,
    skill_type VARCHAR NOT NULL,
    skill_category VARCHAR NOT NULL,
    content TEXT NOT NULL,
    context_json JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    success_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP,
    metadata_json JSONB,
    architecture_type VARCHAR,
    CONSTRAINT chk_swarm_skills_usage_count CHECK (usage_count >= 0),
    CONSTRAINT chk_swarm_skills_success_rate CHECK (success_rate >= 0.0 AND success_rate <= 1.0),
    CONSTRAINT chk_swarm_skills_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Table: swarm_skillbook
-- Purpose: Stores swarm-level skills and patterns (Phase 5)
CREATE TABLE IF NOT EXISTS codelumen.swarm_skillbook (
    skill_id VARCHAR PRIMARY KEY,
    skill_name VARCHAR NOT NULL,
    skill_type VARCHAR NOT NULL,
    pattern_json JSONB NOT NULL,
    usage_count INTEGER DEFAULT 0,
    success_rate DOUBLE PRECISION,
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_swarm_skillbook_usage_count CHECK (usage_count >= 0),
    CONSTRAINT chk_swarm_skillbook_success_rate CHECK (success_rate IS NULL OR (success_rate >= 0.0 AND success_rate <= 1.0))
);

-- Table: swarm_reflections
-- Purpose: Stores reflections on swarm analysis effectiveness (Phase 5)
CREATE TABLE IF NOT EXISTS codelumen.swarm_reflections (
    reflection_id VARCHAR PRIMARY KEY,
    analysis_id VARCHAR NOT NULL,
    architecture_type VARCHAR NOT NULL,
    roles_selected TEXT[] NOT NULL,
    roles_effectiveness_json JSONB NOT NULL,
    prompt_quality_json JSONB NOT NULL,
    synthesis_quality DOUBLE PRECISION NOT NULL,
    token_efficiency DOUBLE PRECISION NOT NULL,
    key_insights TEXT[],
    helpful_patterns TEXT[],
    harmful_patterns TEXT[],
    recommendations TEXT[],
    timestamp TIMESTAMP NOT NULL,
    metadata_json JSONB
);

-- Table: skill_usage_history
-- Purpose: Tracks skill usage history (Phase 5)
CREATE TABLE IF NOT EXISTS codelumen.skill_usage_history (
    usage_id VARCHAR PRIMARY KEY,
    skill_id VARCHAR NOT NULL,
    analysis_id VARCHAR NOT NULL,
    effectiveness DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (skill_id) REFERENCES codelumen.swarm_skills(skill_id) ON DELETE CASCADE,
    CONSTRAINT chk_skill_usage_effectiveness CHECK (effectiveness >= 0.0 AND effectiveness <= 1.0)
);

-- Table: prompts
-- Purpose: Stores prompt library and generated prompts (Phase 5)
CREATE TABLE IF NOT EXISTS codelumen.prompts (
    prompt_id VARCHAR PRIMARY KEY,
    prompt_name VARCHAR NOT NULL,
    prompt_content TEXT NOT NULL,
    metadata_json JSONB,
    role VARCHAR,
    architecture_hash VARCHAR,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_prompts_usage_count CHECK (usage_count >= 0)
);

-- Table: analysis_metrics
-- Purpose: Stores metrics linked to specific scans (Phase 1)
CREATE TABLE IF NOT EXISTS codelumen.analysis_metrics (
    id SERIAL PRIMARY KEY,
    scan_id INTEGER NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES codelumen.scan_history(id) ON DELETE CASCADE
);

-- Table: metrics_history
-- Purpose: Stores historical metrics for trend analysis (Phase 6)
CREATE TABLE IF NOT EXISTS codelumen.metrics_history (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    project_path VARCHAR(500),
    agent_name VARCHAR(100),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    metadata_json JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: cache_metadata
-- Purpose: Tracks cache state and metadata (Phase 2)
CREATE TABLE IF NOT EXISTS codelumen_cache.cache_metadata (
    cache_key VARCHAR PRIMARY KEY,
    cache_type VARCHAR NOT NULL,
    ttl_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP,
    CONSTRAINT chk_cache_metadata_access_count CHECK (access_count >= 0),
    CONSTRAINT chk_cache_metadata_ttl CHECK (ttl_seconds IS NULL OR ttl_seconds > 0)
);

-- Table: user_settings
-- Purpose: Stores user-specific application settings
CREATE TABLE IF NOT EXISTS codelumen.user_settings (
    user_id VARCHAR(255) NOT NULL,
    setting_key VARCHAR(255) NOT NULL,
    setting_value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, setting_key),
    CONSTRAINT chk_user_settings_user_id CHECK (LENGTH(user_id) > 0),
    CONSTRAINT chk_user_settings_setting_key CHECK (LENGTH(setting_key) > 0)
);

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION codelumen.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_scan_history_updated_at ON codelumen.scan_history;
CREATE TRIGGER update_scan_history_updated_at
    BEFORE UPDATE ON codelumen.scan_history
    FOR EACH ROW
    EXECUTE FUNCTION codelumen.update_updated_at_column();

DROP TRIGGER IF EXISTS update_knowledge_base_updated_at ON codelumen.knowledge_base;
CREATE TRIGGER update_knowledge_base_updated_at
    BEFORE UPDATE ON codelumen.knowledge_base
    FOR EACH ROW
    EXECUTE FUNCTION codelumen.update_updated_at_column();

DROP TRIGGER IF EXISTS update_prompts_updated_at ON codelumen.prompts;
CREATE TRIGGER update_prompts_updated_at
    BEFORE UPDATE ON codelumen.prompts
    FOR EACH ROW
    EXECUTE FUNCTION codelumen.update_updated_at_column();
"""

SQL_CREATE_INDEXES = """
-- Create Indexes for CodeLumen
-- This script creates all indexes for optimal query performance

-- ============================================================================
-- Indexes for scan_history
-- ============================================================================

-- Project path lookup
CREATE INDEX IF NOT EXISTS idx_scan_history_project_path 
    ON codelumen.scan_history(project_path);

-- Recent scans (DESC for most recent first)
CREATE INDEX IF NOT EXISTS idx_scan_history_timestamp 
    ON codelumen.scan_history(timestamp DESC);

-- Agent filtering
CREATE INDEX IF NOT EXISTS idx_scan_history_agent_name 
    ON codelumen.scan_history(agent_name);

-- Status filtering
CREATE INDEX IF NOT EXISTS idx_scan_history_status 
    ON codelumen.scan_history(status);

-- Composite: Project history (most common query)
CREATE INDEX IF NOT EXISTS idx_scan_history_project_timestamp 
    ON codelumen.scan_history(project_path, timestamp DESC);

-- ============================================================================
-- Indexes for experiences
-- ============================================================================

-- Agent filtering
CREATE INDEX IF NOT EXISTS idx_experiences_agent_name 
    ON codelumen.experiences(agent_name);

-- Recent experiences
CREATE INDEX IF NOT EXISTS idx_experiences_timestamp 
    ON codelumen.experiences(timestamp DESC);

-- Success level filtering
CREATE INDEX IF NOT EXISTS idx_experiences_success_level 
    ON codelumen.experiences(success_level);

-- Full-text search on goal
CREATE INDEX IF NOT EXISTS idx_experiences_goal_fts 
    ON codelumen.experiences USING gin(to_tsvector('english', goal));

-- JSONB queries
CREATE INDEX IF NOT EXISTS idx_experiences_goal_understanding_json 
    ON codelumen.experiences USING gin(goal_understanding_json);

CREATE INDEX IF NOT EXISTS idx_experiences_strategy_used_json 
    ON codelumen.experiences USING gin(strategy_used_json);

-- ============================================================================
-- Indexes for knowledge_base
-- ============================================================================

-- Type filtering
CREATE INDEX IF NOT EXISTS idx_knowledge_base_type 
    ON codelumen.knowledge_base(knowledge_type);

-- Agent filtering
CREATE INDEX IF NOT EXISTS idx_knowledge_base_agent_name 
    ON codelumen.knowledge_base(agent_name);

-- Relevance tags array overlap queries (GIN index for array operations)
CREATE INDEX IF NOT EXISTS idx_knowledge_base_relevance_tags 
    ON codelumen.knowledge_base USING gin(relevance_tags);

-- Relevance sorting (using confidence instead of relevance_score)
CREATE INDEX IF NOT EXISTS idx_knowledge_base_confidence 
    ON codelumen.knowledge_base(confidence DESC NULLS LAST);

-- Expiration cleanup (partial index, using expires_at instead of expiration_timestamp)
CREATE INDEX IF NOT EXISTS idx_knowledge_base_expiration 
    ON codelumen.knowledge_base(expires_at) 
    WHERE expires_at IS NOT NULL;

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_knowledge_base_content_fts 
    ON codelumen.knowledge_base USING gin(to_tsvector('english', content_json::text));

-- JSONB queries
CREATE INDEX IF NOT EXISTS idx_knowledge_base_content_json 
    ON codelumen.knowledge_base USING gin(content_json);

-- ============================================================================
-- Indexes for swarm_skillbook
-- ============================================================================

-- Name lookup
CREATE INDEX IF NOT EXISTS idx_swarm_skillbook_name 
    ON codelumen.swarm_skillbook(skill_name);

-- Type filtering
CREATE INDEX IF NOT EXISTS idx_swarm_skillbook_type 
    ON codelumen.swarm_skillbook(skill_type);

-- Popular skills
CREATE INDEX IF NOT EXISTS idx_swarm_skillbook_usage 
    ON codelumen.swarm_skillbook(usage_count DESC);

-- Best skills
CREATE INDEX IF NOT EXISTS idx_swarm_skillbook_success_rate 
    ON codelumen.swarm_skillbook(success_rate DESC NULLS LAST);

-- Recently used
CREATE INDEX IF NOT EXISTS idx_swarm_skillbook_last_used 
    ON codelumen.swarm_skillbook(last_used DESC NULLS LAST);

-- JSONB queries
CREATE INDEX IF NOT EXISTS idx_swarm_skillbook_pattern_json 
    ON codelumen.swarm_skillbook USING gin(pattern_json);

-- ============================================================================
-- Indexes for prompts
-- ============================================================================

-- Name lookup
CREATE INDEX IF NOT EXISTS idx_prompts_name 
    ON codelumen.prompts(prompt_name);

-- Role filtering
CREATE INDEX IF NOT EXISTS idx_prompts_role 
    ON codelumen.prompts(role);

-- Architecture filtering
CREATE INDEX IF NOT EXISTS idx_prompts_architecture_hash 
    ON codelumen.prompts(architecture_hash);

-- Popular prompts
CREATE INDEX IF NOT EXISTS idx_prompts_usage 
    ON codelumen.prompts(usage_count DESC);

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_prompts_content_fts 
    ON codelumen.prompts USING gin(to_tsvector('english', prompt_content));

-- JSONB queries
CREATE INDEX IF NOT EXISTS idx_prompts_metadata_json 
    ON codelumen.prompts USING gin(metadata_json);

-- ============================================================================
-- Indexes for analysis_metrics
-- ============================================================================

-- Scan ID lookup (for retrieving metrics for a scan)
CREATE INDEX IF NOT EXISTS idx_analysis_metrics_scan_id 
    ON codelumen.analysis_metrics(scan_id);

-- ============================================================================
-- Indexes for metrics_history
-- ============================================================================

-- Time-series queries
CREATE INDEX IF NOT EXISTS idx_metrics_history_timestamp 
    ON codelumen.metrics_history(timestamp DESC);

-- Project filtering
CREATE INDEX IF NOT EXISTS idx_metrics_history_project 
    ON codelumen.metrics_history(project_path);

-- Agent filtering
CREATE INDEX IF NOT EXISTS idx_metrics_history_agent 
    ON codelumen.metrics_history(agent_name);

-- Metric name filtering
CREATE INDEX IF NOT EXISTS idx_metrics_history_metric_name 
    ON codelumen.metrics_history(metric_name);

-- Composite: Analytics queries (most common)
CREATE INDEX IF NOT EXISTS idx_metrics_history_analytics 
    ON codelumen.metrics_history(timestamp DESC, project_path, agent_name, metric_name);

-- ============================================================================
-- Indexes for cache_metadata
-- ============================================================================

-- Type filtering
CREATE INDEX IF NOT EXISTS idx_cache_metadata_type 
    ON codelumen_cache.cache_metadata(cache_type);

-- Expiration cleanup (partial index)
CREATE INDEX IF NOT EXISTS idx_cache_metadata_expires_at 
    ON codelumen_cache.cache_metadata(expires_at) 
    WHERE expires_at IS NOT NULL;

-- LRU eviction
CREATE INDEX IF NOT EXISTS idx_cache_metadata_last_accessed 
    ON codelumen_cache.cache_metadata(last_accessed DESC NULLS LAST);

-- ============================================================================
-- Indexes for user_settings
-- ============================================================================

-- User ID lookup (most common query)
CREATE INDEX IF NOT EXISTS idx_user_settings_user_id 
    ON codelumen.user_settings(user_id);

-- Setting key lookup
CREATE INDEX IF NOT EXISTS idx_user_settings_setting_key 
    ON codelumen.user_settings(setting_key);

-- Updated timestamp for cleanup/audit
CREATE INDEX IF NOT EXISTS idx_user_settings_updated_at 
    ON codelumen.user_settings(updated_at);
"""

# =============================================================================
# MAIN INITIALIZATION LOGIC
# =============================================================================

def create_database() -> bool:
    """
    Create the codelumen database if it doesn't exist.
    
    Returns:
        True if successful, False otherwise
    """
    config = get_db_config().postgresql
    
    try:
        # Connect to default postgres database to create new database
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            user=config.user,
            password=config.password,
            database="postgres"  # Connect to default database
        )
        conn.autocommit = True  # Required for CREATE DATABASE
        
        with conn.cursor() as cursor:
            # Check if database exists
            cursor.execute("""
                SELECT 1 FROM pg_database WHERE datname = %s
            """, (CODELUMEN_DATABASE,))
            
            exists = cursor.fetchone()
            
            if exists:
                logger.info(f"Database '{CODELUMEN_DATABASE}' already exists")
            else:
                # Create database
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(CODELUMEN_DATABASE))
                )
                logger.info(f"Created database '{CODELUMEN_DATABASE}'")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Error creating database: {e}")
        return False


def initialize_schema() -> bool:
    """
    Initialize the schema by executing the SQL definitions.
    """
    config = get_db_config().postgresql
    schema = config.schema
    cache_schema = config.cache_schema
    
    # Helper to replace schema placeholders
    def prepare_sql(sql_template: str) -> str:
        # Replace schema references
        sql = sql_template.replace("codelumen.", f"{schema}.")
        sql = sql_template.replace("codelumen_cache.", f"{cache_schema}.")
        
        # Replace schema definitions and permissions
        sql = sql.replace("SCHEMA codelumen", f"SCHEMA {schema}")
        sql = sql.replace("SCHEMA codelumen_cache", f"SCHEMA {cache_schema}")
        
        return sql

    try:
        with get_db_connection(CODELUMEN_DATABASE, read_only=False) as conn:
            with conn.cursor() as cursor:
                # 1. Create schemas
                logger.info("Executing Schema Creation...")
                cursor.execute(prepare_sql(SQL_CREATE_SCHEMAS))
                
                # 2. Create tables
                logger.info("Executing Table Creation...")
                cursor.execute(prepare_sql(SQL_CREATE_TABLES))
                
                # 3. Create indexes
                logger.info("Executing Index Creation...")
                cursor.execute(prepare_sql(SQL_CREATE_INDEXES))
                
                # 4. Initialize schema_version table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        description VARCHAR
                    )
                """)
                
                # Check current version
                cursor.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                current_version = result[0] if result else 0
                
                if current_version is None:
                    current_version = 0

                # If version is 0 (fresh install), mark as initial setup
                if current_version == 0:
                     logger.info("Setting initial schema version...")
                     cursor.execute("INSERT INTO schema_version (version, description) VALUES (1, 'Initial Setup via init_database.py')")
                
            conn.commit()
        
        logger.info("Schema initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing schema: {e}")
        return False


def main():
    """Main initialization function."""
    logger.info("Initializing database for CodeLumen...")
    
    # Step 1: Create database
    logger.info("Step 1: Creating database...")
    if not create_database():
        logger.error("Failed to create database")
        return 1
    
    # Step 2: Initialize Schema (Schemas, Tables, Indexes)
    logger.info("Step 2: Initializing Schema...")
    if not initialize_schema():
        logger.error("Failed to initialize schema")
        return 1
    
    logger.info("Database initialization completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
