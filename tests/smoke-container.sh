#!/bin/bash
# Run against a freshly built Notebook image: bash tests/smoke-container.sh IMAGE
set -euo pipefail
container=$(docker run -d --rm -p 127.0.0.1::8080 \
  -e CHAT_MODEL=openai:gpt-4o \
  -e EMBEDDING_MODEL=openai:text-embedding-3-small \
  -e OPENAI_API_KEY=smoke-test \
  -e OFFLINE_MODE=true \
  -e RAG_EMBEDDING_ENGINE=openai \
  "$1")
trap 'docker stop "$container" >/dev/null 2>&1 || true' EXIT
origin=http://$(docker port "$container" 8080/tcp)
for ((attempt=0; attempt<90; attempt++)); do
  if curl -fsS --max-time 3 "$origin/health" >/dev/null 2>&1; then
    curl -fsS "$origin/api/config" | grep -q 'Open WebUI'
    curl -fsS "$origin/citation-assets/view" | grep -q 'PDF citation'
    docker exec "$container" python -c \
      'import json, urllib.request; models = json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models")); assert models["data"][0]["id"] == "notebook-rag"'
    exit 0
  fi
  sleep 2
done
docker logs "$container"
echo "Open WebUI did not become reachable through Nginx" >&2
exit 1
