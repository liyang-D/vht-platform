#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

env_keys() {
  sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$1" | sort -u
}

compose_keys() {
  grep -hoE '\$\{[A-Z][A-Z0-9_]*' "$@" | sed 's/${//' | sort -u
}

check_duplicates() {
  local env_file=$1
  local duplicates
  duplicates=$(sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' "$env_file" | sort | uniq -d)
  if [[ -n "$duplicates" ]]; then
    echo "Duplicate keys in ${env_file#$repo_root/}:" >&2
    echo "$duplicates" >&2
    return 1
  fi
}

check_environment() {
  local name=$1
  local override_file="$repo_root/infra/compose.$name.yml"
  local example_file="$repo_root/infra/env/$name.example.env"
  local real_file="$repo_root/infra/env/$name.env"

  check_duplicates "$example_file"

  if ! diff -u \
    <(compose_keys "$repo_root/infra/compose.yml" "$override_file") \
    <(env_keys "$example_file"); then
    echo "$name example keys do not match its Compose configuration." >&2
    return 1
  fi

  if [[ -f "$real_file" ]]; then
    check_duplicates "$real_file"
    if ! diff -u <(env_keys "$example_file") <(env_keys "$real_file"); then
      echo "$name real env keys do not match its example schema." >&2
      return 1
    fi
  fi
}

check_environment dev
check_environment prod

echo "Environment schemas are consistent."
