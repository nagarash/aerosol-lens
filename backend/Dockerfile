FROM python:3.12-slim

WORKDIR /srv

# agent/ is a top-level package shared with the backend
COPY agent/ ./agent/
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# copy the whole backend package (app.py, grid.py, earthdata_auth.py, ...)
COPY backend/ ./backend/

# Runtime data: backend/geocode.py resolves place names against
# data/gazetteer.sample.json, so the image needs it or /ask 500s on any
# named location. Only the small sample JSONs land here -- data/kerchunk/
# is gitignored and lives on the mounted volume at /data instead.
COPY data/ ./data/

ENV PYTHONPATH=/srv
EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
