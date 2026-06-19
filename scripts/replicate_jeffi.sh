#!/usr/bin/env bash
# replicate_jeffi.sh — full nightly replication of Jeffi RDS into local Postgres on Razer.
#
# Flow:
#   1. Open SSH tunnel via jeffi-ec2: localhost:15432 -> RDS:5432
#   2. pg_dump (custom format) the live DB through the tunnel
#   3. Restore into a fresh staging DB
#   4. Atomically swap: jeffi_replica <-> jeffi_replica_old
#   5. POST status to Jeffi admin /api/admin/replication/log
#
# Secrets live in /etc/jeffistores-replication/.env (mode 600).
# Logs to journalctl when run via the systemd timer; stderr otherwise.
#
# Manual run:
#   sudo -u aloysjehwin bash scripts/replicate_jeffi.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Config — load secrets from a file outside the repo
# -----------------------------------------------------------------------------
SECRETS_FILE="${SECRETS_FILE:-/etc/jeffistores-replication/.env}"
if [[ ! -r "$SECRETS_FILE" ]]; then
    echo "[replicate] FATAL: cannot read $SECRETS_FILE" >&2
    exit 2
fi
# shellcheck disable=SC1090
set -a; source "$SECRETS_FILE"; set +a

: "${RDS_HOST:?missing in $SECRETS_FILE}"
: "${RDS_PORT:=5432}"
: "${RDS_DATABASE:?missing}"
: "${RDS_MASTER_USER:=postgres}"
: "${RDS_MASTER_PASSWORD:?missing}"
: "${EC2_SSH_HOST:?missing (e.g. jeffi-ec2)}"
: "${EC2_SSH_USER:=ec2-user}"
: "${EC2_SSH_KEY:?missing (path to jeffi-stores-key.pem)}"
: "${LOCAL_DB:=jeffi_replica}"
: "${LOCAL_USER:=jeffi_replica}"
: "${LOCAL_TUNNEL_PORT:=15432}"
: "${ADMIN_LOG_URL:=}"           # e.g. https://admin.jeffistores.in/api/admin/replication/log
: "${ADMIN_LOG_TOKEN:=}"          # CRON_SECRET shared with the app

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
START_EPOCH=$(date -u +%s)
RUN_ID="repl-$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR=$(mktemp -d -t jeffi-repl-XXXXXX)
DUMP_FILE="$WORKDIR/jeffi.dump"
TUNNEL_PID=""

log()  { printf '[%s] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$*"; }
fail() { log "FAIL: $*"; finish "failed" "$*"; exit 1; }

cleanup() {
    if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    rm -rf "$WORKDIR" 2>/dev/null || true
}
trap cleanup EXIT

post_admin_log() {
    local status="$1" message="$2" duration="$3" rows="${4:-null}" dump_bytes="${5:-null}"
    [[ -z "$ADMIN_LOG_URL" || -z "$ADMIN_LOG_TOKEN" ]] && return 0
    local payload
    payload=$(printf '{"run_id":"%s","status":"%s","message":%s,"duration_seconds":%s,"row_count":%s,"dump_bytes":%s,"source":"razer"}' \
        "$RUN_ID" "$status" "$(jq -Rn --arg m "$message" '$m')" "$duration" "$rows" "$dump_bytes")
    curl --silent --show-error --max-time 30 \
        -H "Authorization: Bearer $ADMIN_LOG_TOKEN" \
        -H "Content-Type: application/json" \
        -X POST -d "$payload" \
        "$ADMIN_LOG_URL" >/dev/null 2>&1 || log "warn: admin log POST failed (non-fatal)"
}

finish() {
    local status="$1" message="${2:-}"
    local now duration
    now=$(date -u +%s)
    duration=$((now - START_EPOCH))
    log "STATUS=$status DURATION=${duration}s"
    post_admin_log "$status" "$message" "$duration" "${ROW_COUNT:-null}" "${DUMP_SIZE:-null}"
}

# -----------------------------------------------------------------------------
# 1. Open SSH tunnel via EC2 to RDS
# -----------------------------------------------------------------------------
log "Opening SSH tunnel: localhost:$LOCAL_TUNNEL_PORT -> $EC2_SSH_HOST -> $RDS_HOST:$RDS_PORT"
ssh -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ExitOnForwardFailure=yes \
    -i "$EC2_SSH_KEY" \
    -N -L "127.0.0.1:$LOCAL_TUNNEL_PORT:$RDS_HOST:$RDS_PORT" \
    "$EC2_SSH_USER@$EC2_SSH_HOST" &
TUNNEL_PID=$!

# Wait for tunnel
for _ in {1..15}; do
    if (echo > "/dev/tcp/127.0.0.1/$LOCAL_TUNNEL_PORT") 2>/dev/null; then
        log "Tunnel ready (pid $TUNNEL_PID)"
        break
    fi
    sleep 1
done
(echo > "/dev/tcp/127.0.0.1/$LOCAL_TUNNEL_PORT") 2>/dev/null || fail "tunnel never opened"

# -----------------------------------------------------------------------------
# 2. pg_dump from RDS via tunnel
# -----------------------------------------------------------------------------
log "Dumping $RDS_DATABASE via tunnel (custom format, compressed)..."
PGPASSWORD="$RDS_MASTER_PASSWORD" pg_dump \
    -h 127.0.0.1 -p "$LOCAL_TUNNEL_PORT" \
    -U "$RDS_MASTER_USER" -d "$RDS_DATABASE" \
    --format=custom --compress=3 \
    --no-owner --no-privileges \
    --jobs=1 \
    --exclude-table=admin_audit_log \
    --exclude-table=merchant_sync_log \
    --exclude-table=product_ai_enrichment_log \
    -f "$DUMP_FILE" || fail "pg_dump failed"

DUMP_SIZE=$(stat -c %s "$DUMP_FILE")
log "Dump complete: $(numfmt --to=iec --suffix=B "$DUMP_SIZE")"

# -----------------------------------------------------------------------------
# 3. Restore into fresh staging DB
# -----------------------------------------------------------------------------
STAGING_DB="${LOCAL_DB}_staging"
log "Restoring into $STAGING_DB ..."

# Read local password from ~/.pgpass (format: host:port:db:user:pass)
LOCAL_PASS=$(awk -F: -v u="$LOCAL_USER" -v d="$LOCAL_DB" '$3==d && $4==u {print $5; exit}' "$HOME/.pgpass")
if [[ -z "$LOCAL_PASS" ]]; then
    fail "could not find local password in ~/.pgpass for $LOCAL_USER@$LOCAL_DB"
fi

sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DROP DATABASE IF EXISTS ${STAGING_DB};
CREATE DATABASE ${STAGING_DB} OWNER ${LOCAL_USER};
SQL

PGPASSWORD="$LOCAL_PASS" pg_restore \
    -h localhost -U "$LOCAL_USER" -d "$STAGING_DB" \
    --no-owner --no-privileges \
    --jobs=4 \
    --exit-on-error \
    "$DUMP_FILE" || fail "pg_restore failed"

# Row count for the log (sum across user tables)
ROW_COUNT=$(PGPASSWORD="$LOCAL_PASS" psql -h localhost -U "$LOCAL_USER" -d "$STAGING_DB" -tAc "
SELECT COALESCE(SUM(n_live_tup), 0)
FROM pg_stat_user_tables
WHERE schemaname = 'public';
")
log "Staging restore complete: $ROW_COUNT rows across public tables"

# -----------------------------------------------------------------------------
# 4. Atomic swap
# -----------------------------------------------------------------------------
log "Swapping $LOCAL_DB <-> ${LOCAL_DB}_old ..."
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
-- Kick off any open sessions so the rename can proceed.
SELECT pg_terminate_backend(pid) FROM pg_stat_activity
 WHERE datname IN ('${LOCAL_DB}', '${LOCAL_DB}_old') AND pid <> pg_backend_pid();
DROP DATABASE IF EXISTS ${LOCAL_DB}_old;
ALTER DATABASE ${LOCAL_DB} RENAME TO ${LOCAL_DB}_old;
ALTER DATABASE ${STAGING_DB} RENAME TO ${LOCAL_DB};
SQL

# -----------------------------------------------------------------------------
# 5. Done
# -----------------------------------------------------------------------------
NOW=$(date -u +%s); DURATION=$((NOW - START_EPOCH))
log "OK: duration=${DURATION}s rows=$ROW_COUNT dump=$(numfmt --to=iec --suffix=B "$DUMP_SIZE")"
post_admin_log "ok" "rows=$ROW_COUNT dump=$DUMP_SIZE" "$DURATION" "$ROW_COUNT" "$DUMP_SIZE"
