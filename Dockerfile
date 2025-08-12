# Use Python 3.10 base image
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Install necessary utilities
RUN apt-get update && apt-get install -y curl iputils-ping net-tools ffmpeg libsndfile1

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chmod 777 /app/logs

# Create temp directory for audio file processing
RUN mkdir -p /app/temp_uploads && chmod 777 /app/temp_uploads

# Copy all your files to the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openai-whisper ffmpeg-python pydub
RUN pip install python-multipart pillow pandas openpyxl PyPDF2 python-docx


# Set environment variables for the LM Studio connection
ENV LM_STUDIO_HOST=host.docker.internal
ENV LM_STUDIO_PORT=1234

# Update to use HTTP protocol instead of WebSocket
ENV LM_STUDIO_PROTOCOL=http
ENV PYTHONUNBUFFERED=1

# Expose port 8000 to the outside world
EXPOSE 8000

# Create a startup script with better log handling
RUN echo '#!/bin/bash\n\
echo "Container starting..."\n\
\n\
# Ensure logs directory exists with proper permissions\n\
mkdir -p /app/logs\n\
chmod 777 /app/logs\n\
\n\
# Create initial log entry\n\
LOG_FILE="/app/logs/llm_api_$(date +%Y-%m-%d).log"\n\
echo "$(date -Iseconds) - Docker container started - Creating initial log entry" > "$LOG_FILE"\n\
echo "Created log file: $LOG_FILE"\n\
\n\
# Network checks\n\
echo "Checking connection to LM Studio at $LM_STUDIO_HOST:$LM_STUDIO_PORT..."\n\
if curl -s "http://$LM_STUDIO_HOST:$LM_STUDIO_PORT/v1/models" > /dev/null; then\n\
  echo "$(date -Iseconds) - LM Studio connection successful" >> "$LOG_FILE"\n\
  echo "LM Studio connection successful"\n\
else\n\
  echo "$(date -Iseconds) - WARNING: Cannot connect to LM Studio" >> "$LOG_FILE"\n\
  echo "WARNING: Cannot connect to LM Studio"\n\
fi\n\
\n\
# Start the FastAPI server\n\
echo "Starting API server..."\n\
exec uvicorn textgen:app --host 0.0.0.0 --port 8000 --log-level debug\n\
' > /app/start.sh && chmod +x /app/start.sh
# Start the FastAPI server using the wrapper script
CMD ["/app/start.sh"] 