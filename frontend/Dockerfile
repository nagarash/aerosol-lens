# Static frontend: plain HTML/JS/CSS, no build step. Served by nginx.
#
# Deploy from frontend/ per fly.toml (`fly deploy -c fly.toml -a aerosol-lens`):
# the build context is frontend/ itself, so COPY paths are relative to it.
# .dockerignore in frontend/ is respected (it sits at the context root).
# Only the served files are copied -- Dockerfile, fly.toml, and tests stay
# out of the image.
FROM nginx:alpine

COPY index.html app.js config.js styles.css /usr/share/nginx/html/
COPY js/grid-render.js /usr/share/nginx/html/js/

EXPOSE 80
