#!/usr/bin/env bash
# Runs ON the Mac: copy this working tree to a pod, or pull results back.
#   runpod/sync.sh push <ip> <port>   rsync code to /workspace/jacobian-lens
#   runpod/sync.sh pull <ip> <port>   rsync slices/ and lenses/ back here
# <ip> and <port> come from the pod's direct SSH address (root@<ip> -p <port>).
set -euo pipefail
cd "$(dirname "$0")/.."
action=$1 ip=$2 port=$3
remote=/workspace/jacobian-lens
# /workspace is a network volume that refuses chown, so do not preserve owner/group.
ssh_cmd="ssh -p $port -o StrictHostKeyChecking=accept-new"

case $action in
  push)
    rsync -az --no-owner --no-group -e "$ssh_cmd" --exclude .venv --exclude .git --exclude slices --exclude lenses \
      --exclude __pycache__ ./ "root@$ip:$remote/"
    ;;
  pull)
    rsync -az -e "$ssh_cmd" "root@$ip:$remote/slices/" slices/ || true
    rsync -az -e "$ssh_cmd" "root@$ip:$remote/lenses/" lenses/ || true
    ;;
  *) echo "usage: $0 push|pull <ip> <port>" >&2; exit 2 ;;
esac
