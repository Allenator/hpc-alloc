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
HOME="$work" python3 - "$cli" <<'PY' || fails=$((fails + 1))
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

echo "== unit: toml fallback parser =="
HOME="$work" python3 - "$cli" <<'PY' || fails=$((fails + 1))
import sys
m = {}
exec(open(sys.argv[1]).read().split("if __name__")[0], m)
d = m["parse_toml_subset"]('''
# comment
[defaults]
partition = "week"   # trailing comment
cpus = 8
flag = true
[cluster.bouchet]
host = "bouchet.ycrc.yale.edu"
''')
assert d == {"defaults": {"partition": "week", "cpus": 8, "flag": True},
             "cluster": {"bouchet": {"host": "bouchet.ycrc.yale.edu"}}}, d
print("  ok: toml subset parser")
PY

echo "== config: precedence and identity pinning =="
mkstate null
cat > "$work/home/.config/hpc-alloc/config.toml" <<'EOF'
[defaults]
partition = "week"
cpus = 8
idle_timeout = 45
[ssh]
identity_file = "~/.ssh/id_test"
[cluster.bouchet]
gpu_partition = "gpu_h200"
EOF
out="$(hpc running up --dry-run 2>/dev/null)"
case "$out" in *"--partition=week"*) echo "  ok: config partition default";;
  *) echo "  FAIL: config partition ($out)"; fails=$((fails + 1));; esac
case "$out" in *"--cpus-per-task=8"*) echo "  ok: config cpus default";;
  *) echo "  FAIL: config cpus"; fails=$((fails + 1));; esac
out="$(hpc running up --dry-run -p day -c 2 2>/dev/null)"
case "$out" in *"--partition=day"*) echo "  ok: CLI flag beats config";;
  *) echo "  FAIL: flag precedence"; fails=$((fails + 1));; esac
out="$(hpc running up --dry-run -G 1 2>/dev/null)"
case "$out" in *"--partition=gpu_h200"*) echo "  ok: per-cluster gpu_partition";;
  *) echo "  FAIL: per-cluster gpu_partition ($out)"; fails=$((fails + 1));; esac
case "$out" in *"self-releases after 45 min"*) echo "  ok: config idle_timeout";;
  *) echo "  FAIL: config idle_timeout"; fails=$((fails + 1));; esac
HOME="$work/home" python3 -c "
m = {}
exec(open('$cli').read().split('if __name__')[0], m)
m['save_state'](m['load_state']())"
contains "IdentityFile in ssh_config" "IdentityFile ~/.ssh/id_test" \
  "$work/home/.config/hpc-alloc/ssh_config"
contains "IdentitiesOnly in ssh_config" "IdentitiesOnly yes" \
  "$work/home/.config/hpc-alloc/ssh_config"
hpc running config > "$work/out" 2>&1; check "config exit" 0 $?
contains "config shows configured value" "week" "$work/out"
contains "config shows provenance" "[config]" "$work/out"
hpc running config --json > "$work/out" 2>&1
python3 -c "
import json; d = json.load(open('$work/out'))
assert d['defaults']['partition'] == {'value': 'week', 'source': 'config'}
assert d['defaults']['time']['source'] == 'builtin'
assert d['cluster_overrides']['bouchet']['gpu_partition'] == 'gpu_h200'
assert d['ssh_identity_file'] == '~/.ssh/id_test'
print('  ok: config json provenance')" || fails=$((fails + 1))
mkstate null   # no config.toml
hpc running config > "$work/out" 2>&1
contains "absent config reported" "absent" "$work/out"
contains "builtins flagged" "[builtin]" "$work/out"

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
contains "orphan sleeper surfaced" "UNTRACKED" "$work/out"
contains "fresh sleeper not cancel-hinted" "just submitted" "$work/out"
contains "other machine's alloc labelled, no cancel hint" "created on 'othermachine'" "$work/out"
hpc running status --json 2>/dev/null > "$work/out"
python3 -c "
import json; d = json.load(open('$work/out'))    # stdout must be pure JSON
kinds = {r['jobid']: r['kind'] for r in d['runs']}
assert kinds == {'9200': 'run', '9400': 'orphan', '9500': 'recent',
                 '9600': 'other-machine'}, kinds
assert [r['orphan'] for r in d['runs']] == [r['kind'] == 'orphan' for r in d['runs']]
owners = {r['jobid']: r['owner'] for r in d['runs']}
assert owners['9600'] == 'othermachine' and owners['9300'] is None if '9300' in owners else True
assert d['allocs'][0]['gpu_util'] == 42
print('  ok: json kinds incl. other-machine ownership, stdout pure, gpu_util=42')" || fails=$((fails + 1))
hpc running up --dry-run --name run > "$work/out" 2>&1
check "name 'run' reserved" 1 $?

echo "== review fixes: regression checks =="
mkstate null
out="$(hpc running up --dry-run -G 1 2>/dev/null)"
case "$out" in *'case "$u" in'*) echo "  ok: watchdog validates nvidia-smi output";;
  *) echo "  FAIL: watchdog validation missing"; fails=$((fails + 1));; esac
out="$(hpc running run --dry-run -- python -c 'x = 1' 2>/dev/null)"
case "$out" in *"--wrap 'python -c x = 1'"*) echo "  FAIL: run argv boundaries lost"; fails=$((fails + 1));;
  *"x = 1"*) echo "  ok: run preserves argv quoting";;
  *) echo "  FAIL: run command missing"; fails=$((fails + 1));; esac
echo 'partition = week' > "$work/home/.config/hpc-alloc/config.toml"   # invalid TOML
hpc running up --dry-run > "$work/out" 2>&1; check "invalid config tolerated" 0 $?
contains "config warning printed" "ignoring config.toml" "$work/out"
mkstate '"r806u23n04"'
HPCTEST_SCANCEL_RC=1 hpc running down h200 > "$work/out" 2>&1
check "down signals scancel failure (exit 1)" 1 $?
contains "state kept on scancel failure" "keeping it in state" "$work/out"
check "alloc preserved after failed scancel" 1 "$(allocs_left)"
echo '{"netid":"ab1234","clusters":{"bouchet":{"host":"bouchet.ycrc.yale.edu"}},"allocs":{}}' \
  > "$work/home/.config/hpc-alloc/state.json"
hpc running down > "$work/out" 2>&1; check "down with zero allocs exits 1" 1 $?
contains "clear empty message" "no active allocations" "$work/out"
mkstate null
hpc gone run -- echo hi > "$work/out" 2>&1; check "run mirrors TIMEOUT as exit 1" 1 $?
contains "final state reported" "TIMEOUT" "$work/out"

echo "== stage-1: final-state verdict discipline =="
mkstate null
rm -f "$work/sacct.n"
HPCTEST_SACCT_LAG="$work/sacct.n" hpc running run -- echo hi > "$work/out" 2>&1
check "successful run exits 0 despite stale RUNNING records (dbd lag)" 0 $?
contains "verdict waits for final state" "COMPLETED" "$work/out"
rm -f "$work/sacct.n"
HPCTEST_SACCT_LAG="$work/sacct.n" hpc gone run -- echo hi > "$work/out" 2>&1
check "failed run still exits 1 after lag" 1 $?
mkstate '"r806u23n04"'
hpc gone logs 9123 -f > "$work/out" 2>&1
check "logs -f is a watcher: exit 0 even for non-COMPLETED job" 0 $?
contains "outcome still reported informationally" "TIMEOUT" "$work/out"

echo "== stage-5: P3 batch =="
mkstate '"oldnode"'
hpc running status > "$work/out" 2>&1; check "requeue detected" 0 $?
contains "node move reported" "moved: oldnode -> r806u23n04" "$work/out"
contains "alias points at new node" "r806u23n04" "$work/out"
mkstate null
hpc suspended why h200 > "$work/out" 2>&1; check "why on suspended job" 0 $?
contains "no false running claim" "SUSPENDED — not running" "$work/out"
out="$(hpc running run --dry-run -- echo hi 2>/dev/null)"
case "$out" in *'$HOME'*) echo "  FAIL: dry-run not pastable"; fails=$((fails + 1));;
  *".hpc-alloc/run-%j.log"*) echo "  ok: run dry-run pastable (relative log path)";;
  *) echo "  FAIL: log path missing"; fails=$((fails + 1));; esac
rm -rf "$work/home/.ssh" && mkdir -p "$work/home/.ssh"
printf '# Include ~/.config/hpc-alloc/ssh_config\n' > "$work/home/.ssh/config"
HOME="$work/home" python3 -c "
m = {}
exec(open('$cli').read().split('if __name__')[0], m)
m['ensure_include']()"
n=$(grep -c '^Include ~/.config/hpc-alloc/ssh_config' "$work/home/.ssh/config")
check "commented Include no longer satisfies the guard" 1 "$n"

echo "== stage-4: config input hardening =="
mkstate null
printf '[defaults]\nidle_timeout = true\n' > "$work/home/.config/hpc-alloc/config.toml"
hpc running up --dry-run -G 1 > "$work/out" 2>&1
check "bool idle_timeout rejected" 1 $?
contains "clear type message" "must be an integer, got true" "$work/out"
printf '[defaults]\ntime = 8\n' > "$work/home/.config/hpc-alloc/config.toml"
hpc running up --dry-run > "$work/out" 2>&1
check "integer time rejected" 1 $?
contains "duration guidance" "quoted Slurm duration" "$work/out"
printf '[cluster]\nbouchet = "bouchet.ycrc.yale.edu"\n' > "$work/home/.config/hpc-alloc/config.toml"
hpc running config > "$work/out" 2>&1
check "non-table [cluster] value survives" 0 $?
contains "misuse explained" "IGNORED" "$work/out"

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

echo "== stage-2: stdout purity and hostkey classification =="
mkstate null
out_only="$(hpc running up --dry-run -G 1 2>/dev/null)"
# note: the sbatch payload itself contains 'hpc-alloc:' inside the watchdog
# echo — only a LINE starting with the prefix is a leaked notice
if printf '%s\n' "$out_only" | grep -q '^hpc-alloc:'; then
  echo "  FAIL: info notice leaked to stdout"; fails=$((fails + 1))
else echo "  ok: stdout carries only payload"; fi
err_only="$(hpc running up --dry-run -G 1 2>&1 >/dev/null)"
case "$err_only" in *"--gpus given"*) echo "  ok: notices go to stderr";;
  *) echo "  FAIL: notice missing from stderr"; fails=$((fails + 1));; esac
mkstate '"r806u23n04"'
hpc hostkey status > "$work/out" 2>&1; check "hostkey probe exits 3" 3 $?
contains "hostkey surfaced, not masked as VPN" "HOST KEY VERIFICATION FAILED" "$work/out"
check "alloc preserved on hostkey failure" 1 "$(allocs_left)"
hpc hostkey connect --push > "$work/out" 2>&1; check "push refused on hostkey" 3 $?
contains "no push attempted" "HOST KEY VERIFICATION FAILED" "$work/out"
if grep -q "requesting Duo push" "$work/out"; then
  echo "  FAIL: push was sent despite hostkey failure"; fails=$((fails + 1))
else echo "  ok: no Duo push sent on hostkey failure"; fi

echo "== scenario: Duo push auth (connect --push) =="
mkstate null
rm -f "$work/duo.mark"
HPCTEST_MARK="$work/duo.mark" hpc duo connect > "$work/out" 2>&1
check "without --push: exit 3" 3 $?
rm -f "$work/duo.mark"
HPCTEST_MARK="$work/duo.mark" hpc duo connect --push > "$work/out" 2>&1
check "with --push: exit" 0 $?
contains "push approved" "Duo approved" "$work/out"
contains "master established" "login OK" "$work/out"

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
