# Static frontend: plain HTML/JS/CSS, no build step. Served by nginx, same
# pattern as docker-compose.yml's frontend service, packaged for Fly.io.
#
# NOTE (verified empirically, do not "fix" this back to `COPY .`): Fly
# resolves `dockerfile` relative to this app dir (frontend/) but the build
# CONTEXT sent to the builder is the repo root (git root). A bare `COPY .`
# here copies the WHOLE REPO -- .git included -- into the public web root.
#
# Explicit file-by-file COPY, not `COPY frontend/`: a .dockerignore placed
# in frontend/ is inert (Docker only reads one at the context root, i.e.
# the repo root), so Dockerfile/fly.toml/the test file would otherwise
# land in the served tree too. Nothing secret in them, but no reason to
# ship them.
FROM nginx:alpine

COPY frontend/index.html frontend/app.js frontend/config.js frontend/styles.css \
     /usr/share/nginx/html/
COPY frontend/js/grid-render.js /usr/share/nginx/html/js/

EXPOSE 80
