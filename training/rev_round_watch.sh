#!/bin/bash
# Watch a rev_round.py run and print one line per event worth acting on, for a Claude Code
# Monitor (each line is a notification): stage changes, failures in the Modal output, the
# round process exiting without finishing, and silence (no output for STALL_MINUTES).
#
#   training/rev_round_watch.sh runs/rev/logs/round-s8.log runs/rev/rev-0.8b-s8.round.log
#
# The first log is rev_round.py's own output (its stage lines); the second is the Modal
# commands' output it appends to runs/rev/NAME.round.log.
set -u
stages=${1:?usage: rev_round_watch.sh ROUND_OUTPUT MODAL_LOG}
detail=${2:?usage: rev_round_watch.sh ROUND_OUTPUT MODAL_LOG}
stall_minutes=${STALL_MINUTES:-45}
name=$(basename "$detail" .round.log)   # runs/rev/NAME.round.log: several rounds can run at once
touch "$detail"
seen_stages=$(wc -l < "$stages"); seen_detail=$(wc -l < "$detail")
warned_stall=0

new_lines() {  # file, lines already seen
  local total; total=$(wc -l < "$1")
  (( total > $2 )) && tail -n +"$(($2 + 1))" "$1" | head -n "$((total - $2))"
}

while true; do
  new_lines "$stages" "$seen_stages" | grep -E '^\[[0-9:]+\]|failed|has [0-9]+ of|Traceback|Error' | cut -c1-200
  new_lines "$detail" "$seen_detail" \
    | grep -E 'Traceback|Error:|preemption|riding it again|REV COLLECT .*: (failed|[0-9]+ parts)|REV EVAL .*laps|saved /runs' \
    | grep -v 'GPG error\|signature verification' | cut -c1-200
  seen_stages=$(wc -l < "$stages"); seen_detail=$(wc -l < "$detail")
  if grep -q 'round .* took' "$stages"; then
    echo "ROUND FINISHED: $(grep 'round .* took' "$stages" | tail -n 1)"
    exit 0
  fi
  if ! pgrep -f "training/rev_round.py.*--name $name( |$)" > /dev/null; then
    echo "ROUND PROCESS EXITED WITHOUT FINISHING: $(tail -n 1 "$stages" | cut -c1-160)"
    exit 1
  fi
  idle=$(( ($(date +%s) - $(stat -c %Y "$detail")) / 60 ))
  if (( idle >= stall_minutes )); then
    (( warned_stall == 0 )) && echo "ROUND SILENT ${idle} min: $(tail -n 1 "$detail" | cut -c1-160)"
    warned_stall=1
  else
    warned_stall=0
  fi
  sleep 30
done
