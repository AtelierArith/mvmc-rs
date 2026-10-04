#!/usr/bin/env bash
set -euo pipefail
root=${1:?fresh exclusive control root}
here=$(cd "$(dirname "$0")" && pwd)
mkdir "$root"
for expected in 0 1 10 138; do
 for timing in before during after concurrent; do
  # Separate shell ensures genuine errexit behavior, independent of an outer
  # conditional test. No manual wait/retry of the CLI process is used.
  bash -euo pipefail -c '
   expected=$1; timing=$2; receipt=$3; helper=$4
   source "$helper"
   install_notification_handler "$receipt"
   parent=$BASHPID
   if [[ $timing = before ]]; then kill -USR1 "$parent"; fi
   notifier=
   if [[ $timing = during ]]; then
    (sleep .01; kill -USR1 "$parent") & notifier=$!
   fi
   if bash -c '\''
     sleep .03
     if [[ $2 = concurrent ]]; then kill -USR1 "$3"; fi
     if [[ $1 = 138 ]]; then kill -USR1 "$BASHPID"; else exit "$1"; fi
   '\'' _ "$expected" "$timing" "$parent"; then status=0; else status=$?; fi
   # Only notifier bookkeeping uses wait; it cannot determine CLI status.
   if [[ -n $notifier ]]; then if wait "$notifier"; then :; else exit 2; fi; fi
   if [[ $timing = after ]]; then kill -USR1 "$parent"; fi
   [[ $status = "$expected" && $notification_count = 1 ]]
   printf "FOREGROUND status=%s timing=%s notifications=%s PASS\n" "$status" "$timing" "$notification_count"
  ' _ "$expected" "$timing" "$root/$expected-$timing.notifications" "$here/wait-cli.sh"
 done
done
