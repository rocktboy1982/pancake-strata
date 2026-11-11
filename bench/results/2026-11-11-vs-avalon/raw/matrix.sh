#!/bin/bash
S=/tmp/claude-1000/-home-dan/2aa43621-cde7-4980-b127-c1c751efc501/scratchpad
R=/home/dan/clone-of-d/src/Strata
R3=/home/dan/clone-of-d/src/Strata-rdna3
A=/home/dan/clone-of-d/Avalon
AM=/home/dan/clone-of-d/Avalon-models
M=/home/dan/clone-of-d/Strata-data/models
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
log() { echo "[$(date +%T)] $*" >> $S/matrix.log; }
wait_free() {
  for i in $(seq 1 90); do
    if ! pgrep -x strata >/dev/null && ! pgrep -x llama-server >/dev/null && ! ss -ltn | grep -qE ':(8080|8081|8082) '; then break; fi; sleep 2; done
  for i in $(seq 1 90); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 48 ] && break; sleep 3; done
}
run() {
  tag=$1; port=$2; dir=$3; shift 3
  wait_free; log "START $tag"
  ( cd "$dir" && exec setsid "$@" ) > $S/srv-$tag.log 2>&1 &
  pid=$!
  ready=0
  for i in $(seq 1 180); do
    c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health); [ "$c" = 200 ] && { ready=1; break; }
    kill -0 $pid 2>/dev/null || break; sleep 5; done
  if [ $ready = 1 ]; then
    for p in 1 2; do python3 -I $S/cmp_bench.py $tag-p$p $port $S/m-$tag-p$p.json > $S/m-$tag-p$p.log 2>&1; done
    log "DONE $tag"
  else log "FAILED-START $tag"; fi
  kill -TERM -- -$pid 2>/dev/null; kill -TERM $pid 2>/dev/null
  for i in $(seq 1 45); do pgrep -x strata >/dev/null || pgrep -x llama-server >/dev/null || break; sleep 2; done
  pkill -x strata 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 3
}
pan() { # tag cfg
  run $1 8080 $R $R/tools/pancake/capped.sh $R/.venv/bin/python $R/serve/server.py --engine strata --config $S/cfg/$2 --port 8080
}
av() { # tag gguf
  run $1 8081 $A $A/build/bin/llama-server -m $2 -c 131072 -b 4096 -ub 4096 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 -md $AM/Qwen3.8-Flash-Next-Avalon-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type ngram-simple,draft-mtp --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 --chat-template-file $AM/avalon-chat-template.jinja --port 8081 --host 127.0.0.1
}
pan p-iq2xs-hip p-iq2xs-hip.json
pan p-q2-fast p-q2-fast.json
pan p-q2-turbo p-q2-turbo.json
pan p-q2-hip p-q2-hip.json
STRATA_RAM_KEEP_GIB=12 run r3-q2 8082 $R3 $R/.venv/bin/python serve/server.py --engine strata --config $S/cfg/r3-q2.json --port 8082
av av-q2 $M/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
av av-iq2xs $M/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf
log "MATRIX COMPLETE"
