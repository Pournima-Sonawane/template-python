#!/usr/bin/env bash

set -o errexit
set -o nounset

generate_repo_index() {
  REPO_URL="$1"

  REPO_INDEX_FILE="${REPO_URL}/index.yaml"

  if wget -q "$REPO_INDEX_FILE" -O index.yaml; then
    echo "File ${REPO_INDEX_FILE} downloaded. Add a new release to index.yaml."
    helm repo index --merge index.yaml .
  elif [ $? -eq 8 ]; then
    echo "File ${REPO_INDEX_FILE} does not exist. Create new index.yaml."
    helm repo index .
  else
    echo "Error: File ${REPO_INDEX_FILE} is not available."
    exit 1
  fi
}

main() {
  REPO_URL="$1"
  generate_repo_index "$REPO_URL"
}

main "$@"
