#!/usr/bin/env bash
# Data Foundry launcher for Linux and macOS.
cd "$(dirname "$0")" || exit 1

PY=""
for candidate in python3.14 python3.13 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "Data Foundry needs Python 3.13 or newer."
  echo "Download it from https://www.python.org/downloads/ and run this again."
  read -r -p "Press Enter to close. " _
  exit 1
fi

if [ ! -x venv/bin/python ]; then
  echo "First run: setting things up. This needs an internet connection and takes a minute or two."
  "$PY" -m venv venv || { read -r -p "Could not create the environment. Press Enter to close. " _; exit 1; }
  venv/bin/python -m pip install --quiet -r requirements.txt || { read -r -p "Install failed (see above). Press Enter to close. " _; exit 1; }
fi

PORT=$(venv/bin/python -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1]); s.close()")
URL="http://127.0.0.1:$PORT/"
echo "Data Foundry is starting at $URL"
echo "Leave this window open while you use it. Close it, or press Ctrl+C, to stop."
( sleep 2; if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; elif command -v open >/dev/null 2>&1; then open "$URL"; fi ) >/dev/null 2>&1 &
exec venv/bin/python -m flask --app app run --port "$PORT"
