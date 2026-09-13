# Static frontend: plain HTML/JS/CSS, no build step. Served by nginx, same
# pattern as docker-compose.yml's frontend service, packaged for Fly.io.
#
# NOTE: Fly resolves `dockerfile` relative to this app dir (frontend/) but
# the build CONTEXT sent to the builder is the repo root (git root) --
# confirmed empirically, not documented. `COPY frontend/` (not `COPY .`)
# is deliberate: a bare `COPY .` here copies the WHOLE REPO, .git included,
# into the public web root.
FROM nginx:alpine

COPY frontend/ /usr/share/nginx/html/

EXPOSE 80
