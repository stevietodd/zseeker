#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/schema.sql"
GUARD_INJECT="$DIR/schema_guard_inject.sql"
# After mysqldump, the guard block from schema_guard_inject.sql is inserted so
# DROP TABLE remains opt-in (@zseeker_allow_schema_drops). Edit that file, not
# the splice logic here, when changing the guard.

if [[ ! -f "$GUARD_INJECT" ]]; then
  echo "Missing guard inject file: $GUARD_INJECT" >&2
  exit 1
fi

read -r -p "MySQL username: " MYSQL_USER
if [[ -z "$MYSQL_USER" ]]; then
  echo "Username is required." >&2
  exit 1
fi
read -r -p "Database name: " MYSQL_DB
if [[ -z "$MYSQL_DB" ]]; then
  echo "Database name is required." >&2
  exit 1
fi

TMP_DUMP="$(mktemp "$DIR/.schema_mysqldump.XXXXXX.sql")"
STAGED="${OUT}.new.$$"
cleanup() { rm -f "$TMP_DUMP" "$STAGED"; }
trap cleanup EXIT

mysqldump -h localhost -u "$MYSQL_USER" -p \
  --no-data --no-tablespaces --routines --triggers --events --single-transaction \
  --column-statistics=0 \
  "$MYSQL_DB" -r "$TMP_DUMP"

awk -v injectfile="$GUARD_INJECT" '
BEGIN { marker = "/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;" }
$0 == marker {
  print
  while ((getline line < injectfile) > 0) {
    print line
  }
  if (close(injectfile) != 0) {
    print "awk: failed to read inject file" > "/dev/stderr"
    exit 2
  }
  next
}
{ print }
' "$TMP_DUMP" > "$STAGED"

if ! grep -Fq "zseeker_schema_guard" "$STAGED"; then
  echo "Guard splice failed: output does not contain zseeker_schema_guard." >&2
  echo "Expected mysqldump prelude line:" >&2
  echo "  /*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;" >&2
  exit 1
fi

mv -f "$STAGED" "$OUT"
trap - EXIT
rm -f "$TMP_DUMP"
echo "Wrote $OUT (with guard from $(basename "$GUARD_INJECT"))"
