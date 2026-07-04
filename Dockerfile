# =========================================================================
# 1. BASE IMAGE (FROM)
# =========================================================================
# We start with a stable, official Python 3.10 slim image. Using "slim" instead
# of the full Python image reduces our container size from ~1GB to <150MB,
# which speeds up deployment and improves security.
FROM python:3.10-slim

# =========================================================================
# 2. SYSTEM ENVIRONMENT VARIABLES (ENV)
# =========================================================================
# - PYTHONUNBUFFERED: Forces Python to flush stdout/stderr buffer immediately
#   so we get real-time application logs in Docker Desktop / Kubernetes.
# - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk,
#   keeping our container image clean.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# =========================================================================
# 3. CONTAINER WORKING DIRECTORY (WORKDIR)
# =========================================================================
# Set the default path where all subsequent commands will run inside the container.
# If the directory doesn't exist, Docker will create it automatically.
WORKDIR /app

# =========================================================================
# 4. INSTALL DEPENDENCIES (COPY & RUN)
# =========================================================================
# We copy ONLY requirements.txt first. This is a critical Docker best practice
# called "Layer Caching". As long as requirements.txt does not change, Docker 
# will reuse the cached package installations, speeding up subsequent builds.
COPY requirements.txt .

# Install all packages defined in requirements.txt.
# We include "--no-cache-dir" to prevent pip from storing cache files inside
# the container, minimizing the final image size.
RUN pip install --no-cache-dir -r requirements.txt

# =========================================================================
# 5. COPY APPLICATION ARTIFACTS & SOURCE CODE (COPY)
# =========================================================================
# Copy the rest of our local project directories into the container's /app folder.
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY static/ ./static/
COPY config/ ./config/
COPY start.sh .

# Make the start script executable so Docker can run it.
RUN chmod +x start.sh

# =========================================================================
# 6. EXPOSE PORTS (EXPOSE)
# =========================================================================
# Inform Docker that the container listens on these specified ports at runtime:
# - Port 8000: Exposes our high-performance FastAPI REST API.
# - Port 8501: Exposes our Streamlit Analytical UI Dashboard.
EXPOSE 8000
EXPOSE 8501

# =========================================================================
# 7. CONTAINER ENTRYPOINT COMMAND (CMD)
# =========================================================================
# The default command that executes when the container spins up. 
# We call our start.sh script, which concurrently launches both our FastAPI 
# and Streamlit services.
CMD ["./start.sh"]
