#!/bin/bash
# Runs PostgreSQL (pgvector) and the RAG API in one container.
set -euo pipefail

# PostgreSQL listens on loopback only; only the API port is exposed.
docker-entrypoint.sh postgres -c listen_addresses=127.0.0.1 &
pg_pid=$!

term() {
  kill -TERM "${api_pid:-}" 2>/dev/null || true
  kill -TERM "$pg_pid" 2>/dev/null || true
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
  gosu app uvicorn notebook.api:app --host 0.0.0.0 --port "$PORT" &
fi
api_pid=$!

# Exit (and let the platform restart us) if either process dies.
wait -n "$pg_pid" "$api_pid"
status=$?
term
exit "$status"
