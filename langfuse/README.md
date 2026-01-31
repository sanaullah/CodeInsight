# Langfuse Setup

This directory contains the Docker Compose configuration for running Langfuse locally. It serves as the primary guide for initializing and configuring the Langfuse observability platform for the CodeInsight project.

## Important: Configuration Files

**⚠️ NOTE**: The `.env` file in this directory is **ONLY** for Docker Compose to run the Langfuse server itself. 

**The LLM Framework uses `config.yaml` in the project root**, not `.env files. All framework configuration (API keys, Langfuse client credentials, etc.) should be in `config.yaml`.

## Quick Start

1. **Edit `.env` file** (in this directory) to set your initial Langfuse server user credentials (if needed)

2. **Start Langfuse**:
   ```powershell
   docker-compose up -d
   ```

3. **Access Langfuse**:
   - Web UI: http://localhost:3000
   - **First time**: If you set credentials in `.env`, use those. Otherwise, Langfuse will prompt you to create an admin account.
   - **Subsequent logins**: Use the credentials you set initially

## Initial User Setup

When Langfuse starts for the first time, it needs initial admin credentials. You have two options:

### Option 1: Set Credentials via .env File (Recommended)

Create a `.env` file in the `langfuse/` directory with your initial credentials:

```env
LANGFUSE_INIT_USER_EMAIL=admin@example.com
LANGFUSE_INIT_USER_NAME=Admin User
LANGFUSE_INIT_USER_PASSWORD=your-secure-password
LANGFUSE_INIT_ORG_NAME=My Organization
LANGFUSE_INIT_PROJECT_NAME=My Project
```

**⚠️ IMPORTANT**: 
- Change these values before starting Langfuse in production!
- If you don't set these, Langfuse will prompt you to create an account on first login
- These credentials are only used on the **first run** when the database is empty

## Headless Initialization

For automated deployments or infrastructure-as-code setups, you can perform a "headless" initialization by setting the following environment variables. If these are set, Langfuse will automatically create the organization, project, and user on startup if they don't exist.

**⚠️ Note**: These variables are only acted upon if the resources do not verify exist.

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `LANGFUSE_INIT_ORG_ID` | Unique ID for the organization | Yes | `my-org` |
| `LANGFUSE_INIT_ORG_NAME` | Display name for the organization | No | `My Organization` |
| `LANGFUSE_INIT_PROJECT_ID` | Unique ID for the project | Yes | `my-project` |
| `LANGFUSE_INIT_PROJECT_NAME` | Display name for the project | No | `My Project` |
| `LANGFUSE_INIT_USER_EMAIL` | Admin user email | Yes | `admin@example.com` |
| `LANGFUSE_INIT_USER_NAME` | Admin user name | No | `Admin User` |
| `LANGFUSE_INIT_USER_PASSWORD` | Admin user password | Yes | `secure-password` |
| `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` | Public key for the project | No | `pk-lf-...` |
| `LANGFUSE_INIT_PROJECT_SECRET_KEY` | Secret key for the project | No | `sk-lf-...` |

### Example `.env` for Headless Setup

```env
# Organization
LANGFUSE_INIT_ORG_ID=codeinsight-org
LANGFUSE_INIT_ORG_NAME=CodeInsight Org

# Project
LANGFUSE_INIT_PROJECT_ID=codeinsight-v6
LANGFUSE_INIT_PROJECT_NAME=CodeInsight v6 API
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-1234567890
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-1234567890

# User
LANGFUSE_INIT_USER_EMAIL=admin@codeinsight.dev
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=secure_password_here
```

## Environment Variables (Docker Compose Only)

The `.env` file in this directory contains the following variables that docker-compose automatically loads for running the Langfuse server:

### Initial User Setup (First Run Only)
- `LANGFUSE_INIT_USER_EMAIL` - Initial admin user email (required for auto-setup)
- `LANGFUSE_INIT_USER_NAME` - Initial admin user name (optional)
- `LANGFUSE_INIT_USER_PASSWORD` - Initial admin user password (required for auto-setup)
- `LANGFUSE_INIT_ORG_NAME` - Organization name (optional)
- `LANGFUSE_INIT_PROJECT_NAME` - Project name (optional)
- `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` - Project public key (optional, auto-generated if not set)
- `LANGFUSE_INIT_PROJECT_SECRET_KEY` - Project secret key (optional, auto-generated if not set)

**Note**: These variables are only used on the **first run** when the database is empty. After the initial setup, changing them won't affect existing users.

### Server Configuration
- `NEXTAUTH_SECRET` - Secret for NextAuth.js (change this!)
- `DATABASE_URL` - PostgreSQL connection string
- `SALT` - Encryption salt (change this!)
- `ENCRYPTION_KEY` - Encryption key (change this! Generate with: `openssl rand -hex 32`)
- And many more (see docker-compose.yml for full list)

## Getting API Keys for Framework

After logging into Langfuse (http://localhost:3000):

1. Go to **Settings** → **API Keys**
2. Create a new API key or use an existing one
3. Copy the **Public Key** (starts with `pk-lf-...`)
4. Copy the **Secret Key** (starts with `sk-lf-...`)
5. Add these to your `config.yaml` in the project root:

```yaml
langfuse:
  enabled: true
  public_key: "pk-lf-..."  # From Langfuse dashboard
  secret_key: "sk-lf-..."   # From Langfuse dashboard
  host: "http://localhost:3000"
```

## Updating Initial Credentials

**Note**: Initial user credentials (`LANGFUSE_INIT_*`) only work on first run. To change existing user passwords:

1. Log into Langfuse web UI
2. Go to **Settings** → **Profile** or **Users**
3. Update password there

To reset everything and start fresh:

```powershell
docker-compose down -v  # Removes all data
# Edit .env with new credentials
docker-compose up -d     # Starts fresh
```


## Upgrading Langfuse Docker Images

If you want to update Langfuse to the latest Docker images while keeping your existing data:

1. **Change to the `langfuse` directory** (where `docker-compose.yml` lives):

   ```powershell
   cd langfuse
   ```

2. **Stop the existing containers** (you already did this if you ran it before):

   ```powershell
   docker-compose down
   ```

3. **Pull the latest images**:

   ```powershell
   docker-compose pull
   ```

4. **Start Langfuse again with the updated images**:

   ```powershell
   docker-compose up -d
   ```

5. **Verify that everything is running**:

   ```powershell
   docker-compose ps
   ```

6. **(Optional) View logs if something looks off**:

   ```powershell
   docker-compose logs -f
   ```

> **Note**: This keeps your existing PostgreSQL / ClickHouse / MinIO volumes, so all Langfuse data is preserved.  
> To completely reset everything (including data), see the section on removing volumes below.

## Stopping Langfuse

```powershell
docker-compose down
```

To also remove volumes (deletes all data):

```powershell
docker-compose down -v
```

