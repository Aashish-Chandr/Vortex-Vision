-- Initialization script for VortexVision PostgreSQL
-- Creates the vortexvision database (already created by POSTGRES_DB env var)
-- and grants all privileges to the vortex user.

-- MLflow uses the same DB with its own tables, no separate schema needed.
GRANT ALL PRIVILEGES ON DATABASE vortexvision TO vortex;
