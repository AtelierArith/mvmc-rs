# Shared fail-closed preflight, sourced before builds or reference launches.
mpi179_validate_selection() {
    local file=$1 scope=$2 steps step regex selected ranks2=0 ranks4=0
    [[ -s $file ]] || { echo 'missing/empty matrix' >&2; return 2; }
    steps=${MPI179_STEPS_LIST-1 2 3}
    [[ $steps =~ ^[123]([[:space:]]+[123])*$ ]] || {
        echo 'MPI179_STEPS_LIST must be a nonempty list of prefixes 1/2/3' >&2; return 2;
    }
    local seen=' '
    for step in $steps; do
        [[ $seen != *" $step "* ]] || { echo 'duplicate prefix' >&2; return 2; }
        seen+="$step "
    done
    regex=${MPI179_CELL_REGEX-.*}
    [[ -n $regex ]] || { echo 'empty cell filter' >&2; return 2; }
    [[ probe =~ $regex ]] && : || {
        [[ $? != 2 ]] || { echo 'malformed cell filter' >&2; return 2; }
    }
    selected=0
    local id ranks mode split cg store projection expected status exitcode extra
    local -A ids=()
    while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode extra; do
        [[ $id == cell || $id == id ]] && continue
        [[ $id =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && ! ${ids[$id]+present} ]] || {
            echo 'invalid/duplicate cell identity' >&2; return 2;
        }
        ids[$id]=1
        [[ -z $extra && ( $ranks == 2 || $ranks == 4 ) &&
           ( $mode == real || $mode == cmp || $mode == fsz ) &&
           $split =~ ^[0-9]+$ && $cg =~ ^[0-9]+$ && $store =~ ^[0-9]+$ &&
           ( $projection == identity || $projection == standard || $projection == opttrans ) &&
           ( $expected == success || $expected == rejection || $expected == missing || $expected == output-failure ) && -n $status &&
           ( $exitcode == NA || $exitcode =~ ^[0-9]+$ ) ]] || {
            echo 'malformed matrix row' >&2; return 2;
        }
        [[ $expected == success ]] || continue
        (( split > 0 )) || { echo 'success cell has nonpositive group width' >&2; return 2; }
        [[ ( $cg == 0 || $cg == 1 ) && ( $store == 0 || $store == 1 ) ]] || {
            echo 'malformed success solver/storage axis' >&2; return 2;
        }
        [[ $ranks == 2 || $ranks == 4 ]] || { echo 'invalid rank axis' >&2; return 2; }
        case $scope in
            workers) [[ ( $projection == identity || $projection == standard ) && $id != *uneven* && $id != *empty* ]] || continue ;;
            rust) [[ $projection == identity && $id == *-cg* ]] || continue ;;
            states) [[ $id =~ $regex ]] || continue ;;
            *) return 2 ;;
        esac
        selected=$((selected + 1))
        if [[ $ranks == 2 ]]; then ranks2=1; else ranks4=1; fi
    done < "$file"
    (( selected > 0 )) || { echo 'zero eligible verification cells' >&2; return 2; }
    if [[ ${MPI179_EXPECTED_CELLS+x} ]]; then
        [[ $MPI179_EXPECTED_CELLS =~ ^[1-9][0-9]*$ ]] &&
            (( selected == MPI179_EXPECTED_CELLS )) || {
                echo 'selected cell count differs from declared coverage' >&2; return 2;
            }
    fi
    # A deliberate diagnostic filter is not a full matrix; unfiltered gates
    # must contain both MPI rank axes. Worker axes are fixed by the callers.
    if [[ $scope != states || ! ${MPI179_CELL_REGEX+x} ]]; then
        (( ranks2 && ranks4 )) || { echo 'missing required 2/4-rank axis' >&2; return 2; }
    fi
    printf 'Selected %s cells (%s); prefixes %s\n' "$selected" "$scope" "$steps"
    MPI179_SELECTED_CELLS=$selected
    MPI179_SELECTED_PREFIXES=0
    for step in $steps; do MPI179_SELECTED_PREFIXES=$((MPI179_SELECTED_PREFIXES + 1)); done
    if [[ ${MPI179_EXPECTED_TOTAL+x} ]]; then
        [[ $MPI179_EXPECTED_TOTAL =~ ^[1-9][0-9]*$ ]] &&
            (( selected * MPI179_SELECTED_PREFIXES * 3 == MPI179_EXPECTED_TOTAL )) || {
                echo 'prefix/worker product differs from declared coverage' >&2; return 2;
            }
    fi
}

mpi179_require_test() {
    local binary=$1 name=$2 listing
    listing=$("$binary" --list) || return
    [[ $listing == *"$name: test"* ]] || {
        echo "Unsupported: binary lacks required MPI test $name" >&2; return 2;
    }
}
