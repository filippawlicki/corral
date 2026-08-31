#!/usr/bin/env bash
# Corral install helper. Installs the `corral` CLI for the current user and
# prints the remaining steps needed to turn it into a shared, multi-user queue.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Installing corral from $HERE ..."
pip install --user -e "$HERE"

echo
echo "corral CLI installed. Next steps for a shared, multi-user install:"
echo
echo "1. Pick a shared directory that every user on this server can read/write,"
echo "   e.g. somewhere in a group-writable project mount:"
echo "     mkdir -p /path/to/shared/corral_home"
echo "     chmod 2775 /path/to/shared/corral_home   # setgid so new files inherit the group"
echo
echo "2. Have every user export CORRAL_HOME to that path -- e.g. add this line to"
echo "   a shared shell-profile snippet (/etc/profile.d/corral.sh) or to each"
echo "   user's own ~/.bashrc:"
echo "     export CORRAL_HOME=/path/to/shared/corral_home"
echo
echo "3. Start the daemon ONCE -- it coordinates scheduling for everyone. tmux"
echo "   does not forward your shell's exported variables into a new session,"
echo "   so pass CORRAL_HOME explicitly with -e (or use"
echo "   scripts/corral-daemon.service for systemd, which doesn't need it):"
echo "     tmux new-session -d -s corral-daemon -e CORRAL_HOME=/path/to/shared/corral_home 'corral daemon'"
echo
echo "4. Anyone on the server can now run:"
echo "     corral gpus                         # see detected GPUs and who's using them"
echo "     corral submit --gpus 2 -- python train.py"
echo "     corral queue"
echo "   Each user's first 'corral submit' auto-starts their own 'corral launcher'"
echo "   (their own tmux session, their own OS account) -- nothing else to set up."
echo
echo "See README.md for full usage and design notes."
