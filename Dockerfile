# Real-time SV volatility estimator & risk engine -- REST service image.
#
#   docker build -t volest .
#   docker run -p 8000:8000 volest
#   open http://localhost:8000/docs        (interactive Swagger UI)
#
FROM python:3.11-slim

WORKDIR /app

# install deps first so the layer caches across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# application code
COPY sv_filter.py pmmh.py risk_engine.py online.py data.py app.py ./

EXPOSE 8000

# single worker: sessions live in-process memory (see app.py notes for scaling)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
