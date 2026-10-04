#!/usr/bin/env bash
# Requested CI transport binding only, not a numerical or topology observer.
set -euo pipefail
[[ $# == 1 && -f $1 && ! -L $1 && ${UCX_TLS-} == self,sm,tcp ]]
awk '
 /^UCX_TLS=/ { n++; if ($0 != "UCX_TLS=self,sm,tcp") bad=1 }
 END { exit(bad || n!=1) }
' "$1"
