#!/usr/bin/env bash
# hpc-alloc test suite: unit tests + end-to-end scenarios against tests/shim/ssh.
# Usage: tests/run.sh   (from anywhere; no cluster or network needed)
set -u
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(dirname "$here")"
cli="$repo/hpc-alloc"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export HPCTEST_LOG="$work/scancel.log"

fails=0
check() {  # check <description> <expected> <actual>
  if [ "$2" = "$3" ]; then echo "  ok: $1"
  else echo "  FAIL: $1 (expected '$2', got '$3')"; fails=$((fails + 1)); fi
}
contains() {  # contains <description> <needle> <haystack-file>
  if grep -qF "$2" "$3"; then echo "  ok: $1"
  else echo "  FAIL: $1 (no '$2' in $3)"; fails=$((fails + 1)); fi
}

mkstate() {  # mkstate <node-json>
  rm -rf "$work/home" && mkdir -p "$work/home/.config/hpc-alloc" "$work/home/.ssh"
  cat > "$work/home/.config/hpc-alloc/state.json" <<EOF
{"netid":"ab1234","clusters":{"bouchet":{"host":"bouchet.ycrc.yale.edu"}},
 "allocs":{"h200":{"name":"h200","cluster":"bouchet","jobid":"9123","node":$1,
 "partition":"gpu_h200","time":"8:00:00","gpus":"h200:1","idle_timeout":30,"created":"x"}}}
EOF
}
hpc() {  # hpc <mode> <args...>
  local mode="$1"; shift
  HOME="$work/home" HPCTEST_MODE="$mode" PATH="$here/shim:$PATH" "$cli" "$@" </dev/null
}
allocs_left() {
  HOME="$work/home" python3 -c \
    "import json;print(len(json.load(open('$work/home/.config/hpc-alloc/state.json'))['allocs']))"
}

echo "== unit: python helpers =="
python3 - "$cli" <<'PY' || fails=$((fails + 1))
import sys
m = {}
exec(open(sys.argv[1]).read().split("if __name__")[0], m)
assert m["timeleft_minutes"]("3:59:00") == 239
assert m["timeleft_minutes"]("0:12:00") == 12
assert m["timeleft_minutes"]("1-00:00:00") == 1440
assert m["timeleft_minutes"]("45:30") == 45
assert m["timeleft_minutes"]("UNLIMITED") is None
assert m["parse_gres"]("gpu:h200:8(S:0-1),gpu:2") == [("h200", 8), ("gpu", 2)]
assert m["parse_gres"]("(null)") == []
assert m["classify_node_state"]("mix*") == "mix"
assert m["classify_node_state"]("drain") == "other"
s = {"allocs": {"h200": {}, "dev": {}}}
assert m["split_ssh_args"](s, ["h200", "--", "ls", "-la"]) == ("h200", ["ls", "-la"])
assert m["split_ssh_args"](s, ["--", "ls"]) == (None, ["ls"])
print("  ok: timeleft/gres/state/ssh-args helpers")
PY

echo "== unit: dry runs (no network) =="
mkstate null
out="$(hpc running up --dry-run -G h200:1 2>/dev/null)"
case "$out" in *"self-releases after 30 min"*) echo "  ok: GPU watchdog default 30min";;
  *) echo "  FAIL: watchdog missing"; fails=$((fails + 1));; esac
out="$(hpc running run --dry-run -G h200:1 -- python train.py --epochs 5 2>/dev/null)"
case "$out" in *"--wrap 'python train.py --epochs 5'"*) echo "  ok: run captures command";;
  *) echo "  FAIL: run command capture"; fails=$((fails + 1));; esac

echo "== scenario: network down (state must survive) =="
mkstate '"r806u23n04"'
hpc down status > "$work/out" 2>&1; check "exit code" 3 $?
check "alloc preserved" 1 "$(allocs_left)"

echo "== scenario: slurmctld error (state must survive) =="
mkstate '"r806u23n04"'
hpc squeue-err status > "$work/out" 2>&1; check "exit code" 3 $?
check "alloc preserved" 1 "$(allocs_left)"

echo "== scenario: job gone (reap + why forensics) =="
mkstate '"r806u23n04"'
hpc gone status > "$work/out" 2>&1; check "status exit" 0 $?
contains "shows ENDED" "ENDED" "$work/out"
check "alloc reaped" 0 "$(allocs_left)"
mkstate '"r806u23n04"'
hpc gone why 9123 > "$work/out" 2>&1; check "why exit" 0 $?
contains "diagnoses walltime" "walltime limit" "$work/out"

echo "== scenario: running (status table, runs section, gpu util) =="
mkstate null
hpc running status > "$work/out" 2>&1; check "status exit" 0 $?
contains "node discovered" "r806u23n04" "$work/out"
contains "gpu util shown" "42%" "$work/out"
contains "run job listed" "9200" "$work/out"
hpc running status --json > "$work/out" 2>&1
contains "json run job" '"jobid": "9200"' "$work/out"
python3 -c "
import json; d = json.load(open('$work/out'))
assert len(d['runs']) == 1, d['runs']         # foreign job 9300 excluded
assert d['allocs'][0]['gpu_util'] == 42
print('  ok: json runs filtered, gpu_util=42')" || fails=$((fails + 1))

echo "== scenario: cancel safety =="
mkstate '"r806u23n04"'; : > "$HPCTEST_LOG"
hpc running cancel 9200 > "$work/out" 2>&1; check "cancel own run job" 0 $?
contains "scancel invoked" "scancel 9200" "$HPCTEST_LOG"
hpc running cancel 9300 > "$work/out" 2>&1; check "refuses foreign job" 1 $?
hpc running cancel 8888 > "$work/out" 2>&1; check "unknown job" 1 $?

echo "== scenario: logs =="
mkstate '"r806u23n04"'
hpc running logs 9200 > "$work/out" 2>&1; check "logs exit" 0 $?
contains "log content" "epoch 1: loss 0.42" "$work/out"

echo "== scenario: pending on QOS cap (why) =="
mkstate null
hpc pending-qos why h200 > "$work/out" 2>&1; check "why exit" 0 $?
contains "cap diagnosis" "resource cap (QOSMaxGRESPerUser)" "$work/out"

echo "== scenario: expiring allocation =="
mkstate '"r806u23n04"'
hpc expiring status > "$work/out" 2>&1
contains "expiry marker" "0:12:00!" "$work/out"
mkstate '"r806u23n04"'
hpc expiring status --json > "$work/out" 2>&1
contains "expiring_soon flag" '"expiring_soon": true' "$work/out"

echo "== scenario: avail digest =="
mkstate null
hpc running avail > "$work/out" 2>&1; check "avail exit" 0 $?
contains "gpu free/total" "h200 10/16" "$work/out"
hpc running avail --json > "$work/out" 2>&1
python3 -c "
import json; d = json.load(open('$work/out'))['partitions']
assert d['gpu_h200']['gpus']['h200'] == {'total': 16, 'used': 6, 'free': 10}
assert d['gpu_h200']['nodes'] == {'idle': 1, 'mix': 1, 'alloc': 0, 'other': 1}
assert d['day']['cpus_idle'] == 64 and d['day']['cpus_total'] == 128
print('  ok: avail json aggregation')" || fails=$((fails + 1))

echo "== scenario: partitions with features =="
mkstate null
hpc running partitions --json > "$work/out" 2>&1
contains "feature tags" '"features": "h200"' "$work/out"

echo
if [ "$fails" -eq 0 ]; then echo "ALL TESTS PASSED"; else echo "$fails FAILURE(S)"; exit 1; fi
