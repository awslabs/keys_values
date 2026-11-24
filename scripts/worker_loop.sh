#!/bin/bash
# moved to scripts/rl/worker_loop.sh; shim kept for queued jobs that still call this path
exec bash "$(dirname "$0")/rl/worker_loop.sh" "$@"
