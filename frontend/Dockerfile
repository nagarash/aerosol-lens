# Static frontend: plain HTML/JS/CSS, no build step. Served by nginx.
#
# Deploy from this directory: fly deploy -c fly.toml -a aerosol-lens
# The build context is this directory (frontend/), so COPY . below copies
# index.html, app.js, config.js, styles.css and js/ -- never the repo root.
# (.dockerignore keeps the Dockerfile itself and test files out.)
FROM nginx:alpine

COPY . /usr/share/nginx/html/

EXPOSE 80
