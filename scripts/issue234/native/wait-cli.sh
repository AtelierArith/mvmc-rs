#!/usr/bin/env bash
# Source-only candidate. Caught traps reset to default in executed children;
# unlike trap '', this does not make the CLI ignore SIGUSR1.
install_notification_handler() {
 notification_count=0
 notification_receipt=$1
 (set -o noclobber; : > "$notification_receipt") || return 2
 trap 'notification_count=$((notification_count + 1)); printf "SIGUSR1 notification=%s\n" "$notification_count" >> "$notification_receipt"' USR1
}

wait_cli() {
 local child=$1 before candidate
 while :; do
  before=$notification_count
  if wait "$child"; then candidate=0; else candidate=$?; fi
  # A caught signal can interrupt wait with 128+signal. Retry the SAME child,
  # including when it exited concurrently: Bash retains its wait status.
  if (( notification_count != before )); then continue; fi
  cli_status=$candidate
  return 0
 done
}
