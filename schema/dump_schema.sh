#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/schema.sql"
read -r -p "MySQL username: " MYSQL_USER
if [ -z "$MYSQL_USER" ]; then
  echo "Username is required." >&2
  exit 1
fi
read -r -p "Database name: " MYSQL_DB
if [ -z "$MYSQL_DB" ]; then
  echo "Database name is required." >&2
  exit 1
fi
mysqldump -h localhost -u "$MYSQL_USER" -p \
  --no-data --no-tablespaces --routines --triggers --events --single-transaction \
  --column-statistics=0 \
  "$MYSQL_DB" -r "$OUT"
echo "Wrote $OUT"
