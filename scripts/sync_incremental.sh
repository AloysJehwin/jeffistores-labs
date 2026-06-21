#!/usr/bin/env bash
# sync_incremental.sh — incremental upsert of changed rows from Jeffi RDS → Razer replica.
#
# Instead of pg_dump + full restore, this script:
#   1. Opens SSH tunnel via EC2 → RDS (same as replicate_jeffi.sh)
#   2. For each tracked table: copies rows changed since last successful sync
#      using updated_at > watermark (or created_at for append-only tables)
#   3. Upserts via psql COPY → temp table → INSERT ... ON CONFLICT DO UPDATE
#   4. Records watermark in a local state file after each table succeeds
#   5. Creates/refreshes the embeddings table (pgvector) for RAG
#   6. POSTs status to admin log endpoint (same as full replication script)
#
# Secrets from /etc/jeffistores-replication/.env (same file as replicate_jeffi.sh).
#
# Manual run:
#   sudo -u aloysjehwin bash scripts/sync_incremental.sh
#
# Systemd timer: see systemd/jeffi-incremental-sync.timer (runs every 30min)

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
SECRETS_FILE="${SECRETS_FILE:-/etc/jeffistores-replication/.env}"
if [[ ! -r "$SECRETS_FILE" ]]; then
    echo "[sync] FATAL: cannot read $SECRETS_FILE" >&2; exit 2
fi
set -a; source "$SECRETS_FILE"; set +a

: "${RDS_HOST:?missing in $SECRETS_FILE}"
: "${RDS_PORT:=5432}"
: "${RDS_DATABASE:?missing}"
: "${RDS_MASTER_USER:=postgres}"
: "${RDS_MASTER_PASSWORD:?missing}"
: "${EC2_SSH_HOST:?missing}"
: "${EC2_SSH_USER:=ec2-user}"
: "${EC2_SSH_KEY:?missing}"
: "${LOCAL_DB:=jeffi_replica}"
: "${LOCAL_USER:=jeffi_replica}"
: "${LOCAL_HOST:=100.82.208.8}"
: "${LOCAL_PORT:=5432}"
: "${LOCAL_TUNNEL_PORT:=15433}"   # Different from full-replica's 15432 to avoid conflicts
: "${ADMIN_LOG_URL:=}"
: "${ADMIN_LOG_TOKEN:=}"
: "${RAZER_CLIENT_CERT:=}"
: "${RAZER_CLIENT_KEY:=}"
: "${OLLAMA_URL:=http://localhost:11434}"
: "${EMBED_MODEL:=nomic-embed-text}"
: "${STATE_DIR:=/var/lib/jeffistores-replication}"
: "${WATERMARK_FILE:=$STATE_DIR/watermarks.json}"
: "${LOOKBACK_BUFFER:=300}"       # Extra seconds of lookback to handle clock skew

mkdir -p "$STATE_DIR"

# ─── Helpers ─────────────────────────────────────────────────────────────────
START_EPOCH=$(date -u +%s)
RUN_ID="sync-$(date -u +%Y%m%dT%H%M%SZ)"
WORKDIR=$(mktemp -d -t jeffi-sync-XXXXXX)
TUNNEL_PID=""
TOTAL_UPSERTED=0
FAILED_TABLES=()

log()  { printf '[%s] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ID" "$*"; }
fail() { log "FAIL: $*"; post_admin_log "failed" "$*"; exit 1; }

cleanup() {
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
    rm -rf "$WORKDIR" 2>/dev/null || true
}
trap cleanup EXIT

# Read watermark for a table (epoch seconds, default 0 = first run = full copy)
get_watermark() {
    local tbl="$1"
    if [[ -f "$WATERMARK_FILE" ]]; then
        python3 -c "import json,sys; d=json.load(open('$WATERMARK_FILE')); print(d.get('$tbl', 0))" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

# Save watermark (only called on success)
set_watermark() {
    local tbl="$1" epoch="$2"
    python3 - <<PYEOF
import json, os
f = "$WATERMARK_FILE"
d = json.load(open(f)) if os.path.exists(f) else {}
d["$tbl"] = $epoch
json.dump(d, open(f, "w"))
PYEOF
}

# Admin log POST (same logic as replicate_jeffi.sh)
post_admin_log() {
    local status="$1" message="${2:-}"
    [[ -z "$ADMIN_LOG_URL" ]] && return 0
    local now duration
    now=$(date -u +%s); duration=$((now - START_EPOCH))
    local payload
    payload=$(printf '{"run_id":"%s","status":"%s","message":%s,"duration_seconds":%s,"row_count":%s,"source":"razer-incremental"}' \
        "$RUN_ID" "$status" "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$message")" \
        "$duration" "$TOTAL_UPSERTED")
    local curl_auth=()
    if [[ -n "$RAZER_CLIENT_CERT" && -n "$RAZER_CLIENT_KEY" ]]; then
        curl_auth=(--cert "$RAZER_CLIENT_CERT" --key "$RAZER_CLIENT_KEY")
    elif [[ -n "$ADMIN_LOG_TOKEN" ]]; then
        curl_auth=(-H "Authorization: Bearer $ADMIN_LOG_TOKEN")
    else
        return 0
    fi
    curl --silent --max-time 30 "${curl_auth[@]}" \
        -H "Content-Type: application/json" -X POST -d "$payload" \
        "$ADMIN_LOG_URL" >/dev/null 2>&1 || log "warn: admin log POST failed (non-fatal)"
}

# ─── SSH Tunnel ───────────────────────────────────────────────────────────────
log "Opening SSH tunnel: localhost:$LOCAL_TUNNEL_PORT -> $EC2_SSH_HOST -> $RDS_HOST:$RDS_PORT"
ssh -o StrictHostKeyChecking=accept-new \
    -o ServerAliveInterval=30 \
    -o ExitOnForwardFailure=yes \
    -i "$EC2_SSH_KEY" \
    -N -L "127.0.0.1:$LOCAL_TUNNEL_PORT:$RDS_HOST:$RDS_PORT" \
    "$EC2_SSH_USER@$EC2_SSH_HOST" &
TUNNEL_PID=$!

for _ in {1..15}; do
    if (echo > "/dev/tcp/127.0.0.1/$LOCAL_TUNNEL_PORT") 2>/dev/null; then
        log "Tunnel ready (pid $TUNNEL_PID)"; break
    fi
    sleep 1
done
(echo > "/dev/tcp/127.0.0.1/$LOCAL_TUNNEL_PORT") 2>/dev/null || fail "tunnel never opened"

# Shortcuts for psql
RDS_PSQL=(psql -h 127.0.0.1 -p "$LOCAL_TUNNEL_PORT" -U "$RDS_MASTER_USER" -d "$RDS_DATABASE")
LOCAL_PSQL=(psql -h "$LOCAL_HOST" -p "$LOCAL_PORT" -U "$LOCAL_USER" -d "$LOCAL_DB")
export PGPASSWORD_RDS="$RDS_MASTER_PASSWORD"
export PGPASSWORD_LOCAL
PGPASSWORD_LOCAL=$(awk -F: -v u="$LOCAL_USER" '$4==u {print $5; exit}' "$HOME/.pgpass" 2>/dev/null || echo "")

# ─── Table definitions ────────────────────────────────────────────────────────
# Format: "table_name|pk_column|track_column"
# track_column = updated_at (for mutable tables) or created_at (for append-only)
TABLES=(
    "products|id|updated_at"
    "product_variants|id|updated_at"
    "product_sub_variants|id|updated_at"
    "product_images|id|updated_at"
    "variant_images|id|updated_at"
    "categories|id|updated_at"
    "brands|id|created_at"
    "orders|id|updated_at"
    "order_items|id|created_at"
    "users|id|updated_at"
    "campaigns|kind|updated_at"
    "coupons|id|created_at"
    "quotations|id|updated_at"
    "quotation_items|id|created_at"
    "invoices|id|created_at"
    "cash_sales|id|updated_at"
    "customer_notes|id|created_at"
    "customer_health|user_id|updated_at"
    "customer_tags|id|created_at"
    "customer_tasks|id|updated_at"
    "customer_tag_definitions|id|created_at"
    "suppliers|id|created_at"
)

# Current time (used as watermark ceiling for this run — consistent across all tables)
RUN_EPOCH=$(date -u +%s)
RUN_TS=$(date -u -d "@$RUN_EPOCH" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || \
         date -u -r "$RUN_EPOCH" '+%Y-%m-%d %H:%M:%S')   # macOS fallback

# ─── Incremental upsert per table ────────────────────────────────────────────
upsert_table() {
    local tbl="$1" pk="$2" track_col="$3"

    local wm_epoch
    wm_epoch=$(get_watermark "$tbl")
    # Subtract buffer to handle clock skew / delayed writes
    local from_epoch=$(( wm_epoch - LOOKBACK_BUFFER ))
    local from_ts
    from_ts=$(date -u -d "@$from_epoch" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || \
              date -u -r "$from_epoch" '+%Y-%m-%d %H:%M:%S')

    # Get column list from source (RDS) — only columns that exist on both sides
    local cols
    cols=$(PGPASSWORD="$RDS_MASTER_PASSWORD" "${RDS_PSQL[@]}" -tAc "
        SELECT string_agg(column_name, ',' ORDER BY ordinal_position)
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name='$tbl';" 2>/dev/null)

    if [[ -z "$cols" ]]; then
        log "  SKIP $tbl — could not read column list"
        return 0
    fi

    # Dump changed rows from RDS to CSV
    local csv_file="$WORKDIR/${tbl}.csv"
    PGPASSWORD="$RDS_MASTER_PASSWORD" "${RDS_PSQL[@]}" -tAc "
        COPY (
            SELECT * FROM $tbl
            WHERE $track_col >= '$from_ts'::timestamptz
              AND $track_col <  '$RUN_TS'::timestamptz
        ) TO STDOUT WITH (FORMAT csv, HEADER true, NULL '');" > "$csv_file" 2>/dev/null

    local row_count
    row_count=$(( $(wc -l < "$csv_file") - 1 ))   # subtract header line
    if [[ $row_count -le 0 ]]; then
        log "  $tbl — no changes since $(date -u -d "@$wm_epoch" '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date -u -r "$wm_epoch" '+%Y-%m-%dT%H:%M:%SZ')"
        set_watermark "$tbl" "$RUN_EPOCH"
        return 0
    fi

    log "  $tbl — upserting $row_count rows (since ${from_ts}Z)"

    # Build SET clause for ON CONFLICT UPDATE (all columns except pk)
    local set_clause
    set_clause=$(echo "$cols" | tr ',' '\n' | grep -v "^${pk}$" | \
        awk '{printf "%s = EXCLUDED.%s, ", $1, $1}' | sed 's/, $//')

    # Load into temp table then upsert on razer
    PGPASSWORD="$PGPASSWORD_LOCAL" "${LOCAL_PSQL[@]}" -v ON_ERROR_STOP=1 <<SQLEOF 2>/dev/null
BEGIN;
CREATE TEMP TABLE _sync_${tbl} (LIKE ${tbl} INCLUDING DEFAULTS) ON COMMIT DROP;
\COPY _sync_${tbl} FROM '$csv_file' WITH (FORMAT csv, HEADER true, NULL '');
INSERT INTO ${tbl} SELECT * FROM _sync_${tbl}
  ON CONFLICT ($pk) DO UPDATE SET $set_clause;
COMMIT;
SQLEOF

    TOTAL_UPSERTED=$(( TOTAL_UPSERTED + row_count ))
    set_watermark "$tbl" "$RUN_EPOCH"
    log "  $tbl — done ($row_count upserted)"
}

for entry in "${TABLES[@]}"; do
    IFS='|' read -r tbl pk track <<< "$entry"
    if ! upsert_table "$tbl" "$pk" "$track"; then
        log "  WARN: $tbl failed — continuing"
        FAILED_TABLES+=("$tbl")
    fi
done

# ─── Embeddings refresh ───────────────────────────────────────────────────────
# Re-embed products and users that changed in this run.
# Requires pgvector installed on razer: CREATE EXTENSION IF NOT EXISTS vector;
log "Refreshing embeddings for changed rows..."

# Ensure embeddings table + pgvector exist on razer
PGPASSWORD="$PGPASSWORD_LOCAL" "${LOCAL_PSQL[@]}" -v ON_ERROR_STOP=1 <<'SQLEOF' 2>/dev/null
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS embeddings (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_table  TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    content       TEXT NOT NULL,
    embedding     VECTOR(768),
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_table, source_id)
);
CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
SQLEOF

# Python embedder — batches changed products/variants/users and upserts embeddings
python3 - <<PYEOF
import json, os, sys, time
import psycopg2
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LOCAL_HOST  = os.environ.get("LOCAL_HOST", "100.82.208.8")
LOCAL_PORT  = int(os.environ.get("LOCAL_PORT", "5432"))
LOCAL_USER  = os.environ.get("LOCAL_USER", "jeffi_replica")
LOCAL_DB    = os.environ.get("LOCAL_DB", "jeffi_replica")
RUN_TS      = "$RUN_TS"

# Read local password from .pgpass
pgpass = os.path.expanduser("~/.pgpass")
local_pass = ""
if os.path.exists(pgpass):
    for line in open(pgpass):
        parts = line.strip().split(":")
        if len(parts) == 5 and parts[3] == LOCAL_USER:
            local_pass = parts[4]; break

conn = psycopg2.connect(host=LOCAL_HOST, port=LOCAL_PORT, user=LOCAL_USER,
                        password=local_pass, dbname=LOCAL_DB)
conn.autocommit = False
cur = conn.cursor()

WATERMARK_FILE = "$WATERMARK_FILE"
watermarks = json.load(open(WATERMARK_FILE)) if os.path.exists(WATERMARK_FILE) else {}

def get_wm(tbl):
    # Use pre-run watermark (before this run updated it) = RUN_EPOCH - lookback
    wm = watermarks.get(tbl, 0)
    return wm - $LOOKBACK_BUFFER

def embed(text):
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/embeddings",
        data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["embedding"]

def upsert_embedding(source_table, source_id, content, metadata=None):
    try:
        vec = embed(content)
        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
        cur.execute("""
            INSERT INTO embeddings (source_table, source_id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s::vector, %s)
            ON CONFLICT (source_table, source_id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """, (source_table, source_id, content, vec_str, json.dumps(metadata or {})))
    except Exception as e:
        print(f"  embed warn: {source_table}:{source_id} — {e}", file=sys.stderr)

# ── Products ──
from_epoch = get_wm("products")
from_ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(from_epoch))
cur.execute("""
    SELECT p.id::text, p.name, p.sku, p.short_description, p.description,
           b.name AS brand, c.name AS category,
           p.base_price::text, p.is_featured, p.is_active
    FROM products p
    LEFT JOIN brands b ON b.id = p.brand_id
    LEFT JOIN categories c ON c.id = p.category_id
    WHERE p.updated_at >= %s::timestamptz AND p.updated_at < %s::timestamptz
""", (from_ts, RUN_TS))
products = cur.fetchall()
print(f"  embedding {len(products)} products...", flush=True)
for pid, name, sku, short_desc, desc, brand, cat, price, is_feat, is_active in products:
    parts = [f"Product: {name}"]
    if sku:       parts.append(f"SKU: {sku}")
    if brand:     parts.append(f"Brand: {brand}")
    if cat:       parts.append(f"Category: {cat}")
    if short_desc: parts.append(short_desc)
    if desc and len(desc) > len(short_desc or ""):
        parts.append(desc[:400])
    content = ". ".join(p for p in parts if p)
    upsert_embedding("products", pid, content, {
        "sku": sku, "price": price, "is_featured": is_feat, "is_active": is_active
    })
conn.commit()

# ── Product variants ──
from_epoch = get_wm("product_variants")
from_ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(from_epoch))
cur.execute("""
    SELECT pv.id::text, p.name, pv.name AS variant_name, pv.sku,
           pv.price::text, pv.inventory_quantity
    FROM product_variants pv
    JOIN products p ON p.id = pv.product_id
    WHERE pv.updated_at >= %s::timestamptz AND pv.updated_at < %s::timestamptz
""", (from_ts, RUN_TS))
variants = cur.fetchall()
print(f"  embedding {len(variants)} variants...", flush=True)
for vid, prod_name, var_name, sku, price, stock in variants:
    parts = [f"Product: {prod_name}"]
    if var_name: parts.append(f"Variant: {var_name}")
    if sku:      parts.append(f"SKU: {sku}")
    content = ". ".join(p for p in parts if p)
    upsert_embedding("product_variants", vid, content, {
        "sku": sku, "price": price, "stock": stock
    })
conn.commit()

# ── Users / customers ──
from_epoch = get_wm("users")
from_ts = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(from_epoch))
cur.execute("""
    SELECT id::text, email, first_name, last_name, phone,
           created_at::text
    FROM users
    WHERE updated_at >= %s::timestamptz AND updated_at < %s::timestamptz
      AND email NOT LIKE 'guest\_%%@temporary.local'
""", (from_ts, RUN_TS))
users = cur.fetchall()
print(f"  embedding {len(users)} users...", flush=True)
for uid, email, first_name, last_name, phone, created_at in users:
    name = " ".join(p for p in [first_name, last_name] if p) or email
    parts = [f"Customer: {name}"]
    if email:  parts.append(f"Email: {email}")
    if phone:  parts.append(f"Phone: {phone}")
    content = ". ".join(p for p in parts if p)
    upsert_embedding("users", uid, content, {
        "email": email, "name": name, "joined": created_at
    })
conn.commit()

cur.close()
conn.close()
print("  embeddings done", flush=True)
PYEOF

# ─── Done ────────────────────────────────────────────────────────────────────
DURATION=$(( $(date -u +%s) - START_EPOCH ))
if [[ ${#FAILED_TABLES[@]} -gt 0 ]]; then
    FAIL_MSG="tables failed: ${FAILED_TABLES[*]}"
    log "PARTIAL OK: upserted=$TOTAL_UPSERTED duration=${DURATION}s $FAIL_MSG"
    post_admin_log "partial" "$FAIL_MSG"
else
    log "OK: upserted=$TOTAL_UPSERTED duration=${DURATION}s"
    post_admin_log "ok" "upserted=$TOTAL_UPSERTED"
fi
