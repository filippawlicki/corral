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
echo "2. (Recommended) Install into a shared venv instead, so users need nothing"
echo "   installed locally:"
echo "     python3 -m venv /path/to/shared/corral_venv"
echo "     /path/to/shared/corral_venv/bin/pip install -e $HERE"
echo
echo "3. Have every user add to their ~/.bashrc or ~/.zshrc:"
echo "     export CORRAL_HOME=/path/to/shared/corral_home"
echo "     corral() { /path/to/shared/corral_venv/bin/python -m corral.cli \"\$@\"; }"
echo "   (skip the function if they'd rather 'pip install --user -e' their own copy)"
echo
echo "4. Start the daemon ONCE -- it coordinates scheduling for everyone. tmux"
echo "   does not forward your shell's exported variables into a new session,"
echo "   so pass CORRAL_HOME explicitly with -e (or use"
echo "   scripts/corral-daemon.service for systemd, which doesn't need it):"
echo "     tmux new-session -d -s corral-daemon -e CORRAL_HOME=/path/to/shared/corral_home 'corral daemon'"
echo
echo "5. Anyone can now run 'corral submit --gpus 2 -- python train.py', 'corral"
echo "   queue', 'corral gpus', etc. -- their own launcher auto-starts on first"
echo "   submit, nothing else to set up. See 'corral --help' for the full command list."
echo
echo "See README.md for more."
