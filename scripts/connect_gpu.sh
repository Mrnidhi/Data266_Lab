#!/usr/bin/env bash
# Direct TCP SSH only. Does not allocate/start a Pod or register an access key.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/connect_gpu.sh HOST PORT [shell|session|gpu]' >&2
  echo 'Copy HOST and PORT from the running RunPod Connect panel: SSH over exposed TCP.' >&2
  exit 2
fi
lab_host=$1
lab_port=$2
lab_mode=${3:-shell}
lab_key=${LAB1_SSH_KEY:-"$HOME/.ssh/data266_runpod_ed25519"}
[[ "$lab_host" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || { echo 'Invalid SSH host.' >&2; exit 2; }
[[ "$lab_port" =~ ^[0-9]{1,5}$ ]] && (( 10#$lab_port >= 1 && 10#$lab_port <= 65535 )) || { echo 'Invalid TCP port.' >&2; exit 2; }
[[ -f "$lab_key" ]] || { echo "Missing local SSH identity: $lab_key" >&2; exit 2; }
lab_ssh=(ssh -tt -i "$lab_key" -o IdentitiesOnly=yes -o ForwardAgent=no
  -o StrictHostKeyChecking=ask -o ServerAliveInterval=15 -o ServerAliveCountMax=3
  -o ConnectTimeout=15 -p "$lab_port" "root@$lab_host")
case "$lab_mode" in
  shell) lab_remote='exec bash -l' ;;
  session) lab_remote='command -v tmux >/dev/null || { echo "Install tmux in the Pod before starting persistent training." >&2; exit 1; }; exec tmux new-session -A -s data266' ;;
  gpu) lab_remote='exec nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv -l 2' ;;
  *) echo 'Mode must be shell, session, or gpu.' >&2; exit 2 ;;
esac
if [[ ${LAB1_SSH_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${lab_ssh[@]}" "$lab_remote"
  printf '\n'
  exit 0
fi
exec "${lab_ssh[@]}" "$lab_remote"
