#!/bin/bash
# Runs PostgreSQL, the RAG API, Open WebUI, and Nginx.
set -euo pipefail

install -d -o app -g app -m 0700 /var/lib/open-webui
nginx -t

# PostgreSQL listens on loopback only.
docker-entrypoint.sh postgres -c listen_addresses=127.0.0.1 &
pg_pid=$!
pids=("$pg_pid")

term() {
  kill -TERM "${pids[@]}" 2>/dev/null || true
  wait
}
trap term TERM INT

# The postgres entrypoint first runs a socket-only init server; wait for TCP.
until pg_isready -q -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"; do
  kill -0 "$pg_pid" 2>/dev/null || { echo "PostgreSQL exited" >&2; exit 1; }
  sleep 1
done

gosu app python -m notebook.migration

if [ "$#" -gt 0 ]; then
  gosu app "$@" &
else
  gosu app uvicorn notebook.api:app --host 127.0.0.1 --port "$PORT" &
fi
pids+=("$!")

(
  cd /var/lib/open-webui
  exec gosu app env \
    DATA_DIR=/var/lib/open-webui \
    DATABASE_URL=sqlite:////var/lib/open-webui/webui.db \
    OPENAI_API_BASE_URL="http://127.0.0.1:${PORT}/v1" \
    OPENAI_API_KEY=not-needed \
    ENABLE_OLLAMA_API=false \
    ENABLE_TITLE_GENERATION=false \
    ENABLE_TAGS_GENERATION=false \
    ENABLE_FOLLOW_UP_GENERATION=false \
    ENABLE_AUTOCOMPLETE_GENERATION=false \
    open-webui serve --host 127.0.0.1 --port 3000
) &
pids+=("$!")

nginx -g 'daemon off;' &
pids+=("$!")

# Exit (and let the platform restart us) if any process dies.
status=0
wait -n "${pids[@]}" || status=$?
term
exit "$status"
