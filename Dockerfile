FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY static/ static/
COPY user_provided_codex/ user_provided_codex/

# HF Spaces uses port 7860; override with PORT env var for other hosts
ENV PORT=7860

EXPOSE 7860

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
