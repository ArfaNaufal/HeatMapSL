# EyGazeHeatmap Deployment & Development Guide

This guide covers the instructions for building production Docker images and setting up a local development environment.

---

## 1. Production Workflow (Docker)
Use this workflow to build and deploy the finalized application using Docker images.

### Step 0: Setup git clone repo 
Clone the latest version of repo in your specific folder
```bash
git clone https://github.com/Eyapa/EyGazeHeatmap_Frontend_V1 .
```

### Step 1: Prepare Environment Variables
Since the `.env` file is ignored by Git for security, you must create it from the template provided in the repository.

```bash
# Copy the example template to a real .env file
cp .env.example .env

# Open and edit the .env file to add your secrets (e.g., JWT_SECRET)
# On Windows, you can open it with Notepad. On Linux/WSL, use nano:
nano .env
```

### Step 2: Build and Deploy
This project uses a base `compose.yml` and a production override `compose.prod.yml`. You must use both to ensure the correct production settings are applied.

```bash
# Build the images (using the production overrides)
docker compose -f compose.prod.yml build --no-cache

# Start the containers in detached mode
docker compose -f compose.prod.yml up -d
```

### Step 3: Verify Deployment
- Frontend: Accessible at `http://localhost:80` (or your configured domain).
- Backend API: Accessible at `http://localhost:8000/docs` (Swagger UI).

# 2. Development Workflow (Local)
Use this workflow for active coding. It provides **Hot Reloading** and uses the **Vite Proxy** to communicate with the backend.

## Prerequisites
- **Node.js (v18+)**
- **Python (3.9+)**

### Step 1: Backend Setup
0. Git clone latest main version repo.
```bash
git clone https://github.com/Eyapa/EyGazeHeatmap_Frontend_V1 .
```


1. Navigate to the backend folder:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the API:
```bash
uvicorn main:app --reload --port 8000
# Or
python -m main
```

### Step 2: Frontend Setup
1. Open a new terminal and navigate to the frontend folder:
```bash
cd frontend
```

2. Install packages:
```bash
npm install
```
3. Use vite.config.dev.ts for vite configuration.
```bash
mv vite.config.ts vite.config.prod.ts
cp vite.config.dev.ts vite.config.ts
```

4. Start the development server (HTTPS enabled for WebGazer):
```bash
npm run dev
```

5. Open `https://localhost:3000` in your browser.
- *Note: Accept the "Unsafe/Advanced" certificate warning to allow camera access.*

# 3. Maintenance Commands
## Resetting the Environment
If you encounter the "folder instead of file" error with `database.db`, run these commands to wipe and restart:
```bash
# Stop containers
docker compose down

# Remove the 'fake' database folder if it exists
rm -rf backend/database.db

# Create an empty file to satisfy the mount
touch backend/database.db

# Restart
docker compose -f compose.prod.yml up -d
```

## Project Structure Notes
- `backend/db/`: Recommended location for SQLite files using the `.gitkeep` strategy
- `frontend/nginx.conf`: Contains the reverse proxy logic for production.
- `frontend/vite.config.ts`: Contains the proxy and HTTPS logic for development.
