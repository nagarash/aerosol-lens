FROM python:3.12-slim

WORKDIR /srv

# agent/ is a top-level package shared with the backend
COPY agent/ ./agent/
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# copy the whole backend package (app.py, grid.py, earthdata_auth.py, ...)
COPY backend/ ./backend/

ENV PYTHONPATH=/srv
EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
