FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
EXPOSE 8080

CMD ["python", "-m", "main"]
