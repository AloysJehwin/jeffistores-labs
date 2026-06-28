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
: "${ADMIN_LOG_TOKEN:=}"          # CRON_SECRET (legacy fallback)
: "${RAZER_CLIENT_CERT:=}"        # path to client cert PEM (preferred mTLS auth)
: "${RAZER_CLIENT_KEY:=}"         # path to client key PEM

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

post_admin_log_http() {
    # Returns 0 if HTTP POST got a 2xx; non-zero otherwise.
    local payload="$1"
    [[ -z "$ADMIN_LOG_URL" ]] && return 1
    local curl_auth=()
    if [[ -n "$RAZER_CLIENT_CERT" && -n "$RAZER_CLIENT_KEY" ]]; then
        curl_auth=(--cert "$RAZER_CLIENT_CERT" --key "$RAZER_CLIENT_KEY")
    elif [[ -n "$ADMIN_LOG_TOKEN" ]]; then
        curl_auth=(-H "Authorization: Bearer $ADMIN_LOG_TOKEN")
    else
        return 1
    fi
    local body_file http_code curl_rc=0
    body_file=$(mktemp -t jeffi-admin-log-XXXXXX)
    http_code=$(curl --silent --show-error --max-time 30 \
        "${curl_auth[@]}" \
        -H "Content-Type: application/json" \
        -X POST -d "$payload" \
        -o "$body_file" -w '%{http_code}' \
        "$ADMIN_LOG_URL" 2>>"$body_file") || curl_rc=$?
    local body_snippet
    body_snippet=$(tr -d '\n' <"$body_file" | head -c 300)
    rm -f "$body_file"
    if (( curl_rc != 0 )); then
        log "admin log HTTP error (rc=$curl_rc): $body_snippet"
        return 1
    fi
    if [[ "$http_code" =~ ^2 ]]; then
        log "admin log POST ok (HTTP $http_code)"
        return 0
    fi
    log "admin log HTTP rejected (HTTP $http_code): $body_snippet"
    return 1
}

post_admin_log_db() {
    # Direct INSERT into public.replication_runs via the still-open SSH tunnel.
    # Returns 0 on success, non-zero on failure. Idempotent on run_id.
    local status="$1" message="$2" duration="$3" rows="$4" dump_bytes="$5"
    if [[ -z "${TUNNEL_PID:-}" ]] || ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        log "admin log DB fallback skipped: tunnel not available"
        return 1
    fi
    # Normalize "null" sentinels from the bash side into SQL NULLs.
    local rows_sql="${rows}" dump_sql="${dump_bytes}" duration_sql="${duration}"
    [[ "$rows_sql" == "null" || -z "$rows_sql" ]] && rows_sql="NULL"
    [[ "$dump_sql" == "null" || -z "$dump_sql" ]] && dump_sql="NULL"
    [[ "$duration_sql" == "null" || -z "$duration_sql" ]] && duration_sql="NULL"
    local started_iso
    started_iso=$(date -u -d "@$START_EPOCH" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u +%Y-%m-%dT%H:%M:%SZ)
    if PGPASSWORD="$RDS_MASTER_PASSWORD" psql \
            -h 127.0.0.1 -p "$LOCAL_TUNNEL_PORT" \
            -U "$RDS_MASTER_USER" -d "$RDS_DATABASE" \
            -v ON_ERROR_STOP=1 --quiet \
            -v run_id="$RUN_ID" \
            -v status="$status" \
            -v message="$message" \
            -v started_at="$started_iso" \
            >/dev/null 2>&1 <<SQL
INSERT INTO public.replication_runs
    (run_id, source, status, started_at, duration_seconds, row_count, dump_bytes, message)
VALUES
    (:'run_id', 'razer', :'status', :'started_at'::timestamptz, ${duration_sql}, ${rows_sql}, ${dump_sql}, :'message')
ON CONFLICT (run_id) DO UPDATE SET
    status = EXCLUDED.status,
    duration_seconds = EXCLUDED.duration_seconds,
    row_count = EXCLUDED.row_count,
    dump_bytes = EXCLUDED.dump_bytes,
    message = EXCLUDED.message,
    recorded_at = now();
SQL
    then
        log "admin log DB fallback ok (INSERT/UPDATE replication_runs)"
        return 0
    fi
    log "admin log DB fallback failed (psql INSERT errored)"
    return 1
}

post_admin_log() {
    local status="$1" message="$2" duration="$3" rows="${4:-null}" dump_bytes="${5:-null}"
    local payload
    payload=$(printf '{"run_id":"%s","status":"%s","message":%s,"duration_seconds":%s,"row_count":%s,"dump_bytes":%s,"source":"razer"}' \
        "$RUN_ID" "$status" "$(jq -Rn --arg m "$message" '$m')" "$duration" "$rows" "$dump_bytes")
    if post_admin_log_http "$payload"; then
        return 0
    fi
    log "admin log HTTP path failed — falling back to direct DB write via tunnel"
    if post_admin_log_db "$status" "$message" "$duration" "$rows" "$dump_bytes"; then
        return 0
    fi
    log "warn: admin log POST and DB fallback both failed (non-fatal)"
    return 0
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
    -f "$DUMP_FILE" || fail "pg_dump failed"

DUMP_SIZE=$(stat -c %s "$DUMP_FILE")
log "Dump complete: $(numfmt --to=iec --suffix=B "$DUMP_SIZE")"

# Read local password early — needed for embeddings pre-dump in step 2b
LOCAL_PASS_PRE=$(awk -F: -v u="$LOCAL_USER" -v d="$LOCAL_DB" '$3==d && $4==u {print $5; exit}' "$HOME/.pgpass")

# -----------------------------------------------------------------------------
# 2b. Preserve existing embeddings before overwriting jeffi_replica
# -----------------------------------------------------------------------------
EMBED_DUMP_FILE="$WORKDIR/embeddings.dump"
EMBED_COUNT=0
if PGPASSWORD="$LOCAL_PASS_PRE" psql -h localhost -U "$LOCAL_USER" -d "$LOCAL_DB" \
        -tAc "SELECT 1 FROM pg_tables WHERE schemaname='public' AND tablename='embeddings'" 2>/dev/null | grep -q 1; then
    EMBED_COUNT=$(PGPASSWORD="$LOCAL_PASS_PRE" psql -h localhost -U "$LOCAL_USER" -d "$LOCAL_DB" \
        -tAc "SELECT COUNT(*) FROM embeddings" 2>/dev/null || echo 0)
    log "Dumping $EMBED_COUNT existing embeddings rows..."
    PGPASSWORD="$LOCAL_PASS_PRE" pg_dump \
        -h localhost -U "$LOCAL_USER" -d "$LOCAL_DB" \
        --format=custom --compress=3 \
        --no-owner --no-privileges \
        -t embeddings \
        -f "$EMBED_DUMP_FILE" || { log "warn: embeddings dump failed — will rebuild from scratch"; EMBED_DUMP_FILE=""; }
else
    log "No existing embeddings table — will be built fresh by sync_incremental"
    EMBED_DUMP_FILE=""
fi

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
# 4b. Restore preserved embeddings into new jeffi_replica
# -----------------------------------------------------------------------------
if [[ -n "${EMBED_DUMP_FILE:-}" && -f "$EMBED_DUMP_FILE" ]]; then
    log "Restoring $EMBED_COUNT embeddings rows into $LOCAL_DB ..."
    PGPASSWORD="$LOCAL_PASS" pg_restore \
        -h localhost -U "$LOCAL_USER" -d "$LOCAL_DB" \
        --no-owner --no-privileges \
        --exit-on-error \
        "$EMBED_DUMP_FILE" \
    && log "Embeddings restored OK" \
    || log "warn: embeddings restore failed — sync_incremental will rebuild"
else
    log "No embeddings dump to restore — sync_incremental will build from scratch"
fi

# -----------------------------------------------------------------------------
# 5. Done
# -----------------------------------------------------------------------------
NOW=$(date -u +%s); DURATION=$((NOW - START_EPOCH))
log "OK: duration=${DURATION}s rows=$ROW_COUNT dump=$(numfmt --to=iec --suffix=B "$DUMP_SIZE")"
post_admin_log "ok" "rows=$ROW_COUNT dump=$DUMP_SIZE" "$DURATION" "$ROW_COUNT" "$DUMP_SIZE"
