#!/bin/bash
set -e

viewer_dir="$(cd "$(dirname "$0")" && pwd)"
environment_dir="$viewer_dir/.venv"
python_candidate=""

show_error() {
  exit_status=$?
  if [ "$exit_status" -ne 0 ]; then
    echo
    echo "The viewer could not start. Keep this window open and copy the message above if you need help."
    read -r -p "Press Return to close…"
  fi
}
trap show_error EXIT

cd "$viewer_dir"

for candidate in \
  /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  /opt/homebrew/bin/python3 \
  /usr/local/bin/python3 \
  /usr/bin/python3
do
  if [ -x "$candidate" ] && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 9))'; then
    python_candidate="$candidate"
    break
  fi
done

if [ -z "$python_candidate" ]; then
  echo "Python 3.9 or newer is required."
  echo "Install Python from https://www.python.org/downloads/ and try again."
  read -r -p "Press Return to close…"
  exit 1
fi

if [ ! -x "$environment_dir/bin/python3" ]; then
  "$python_candidate" -m venv "$environment_dir"
  "$environment_dir/bin/python3" -m pip install --upgrade pip
fi

if ! "$environment_dir/bin/python3" -c 'import PySide6, h5py, numpy, vispy' 2>/dev/null; then
  "$environment_dir/bin/python3" -m pip install -r "$viewer_dir/requirements.txt"
fi

exec "$environment_dir/bin/python3" "$viewer_dir/nde_viewer.py" "$@"
