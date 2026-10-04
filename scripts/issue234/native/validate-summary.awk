function fail(message) { print message > "/dev/stderr"; bad=1 }
/ISSUE234/ {
    line=$0
    # libtest may prefix the first marker with the exact selected test name.
    sub(/^test public_rank_local_summary_and_root_readback_failure \.\.\. /,"",line)
    n=split(line, fields, " ")
    if(n!=6 || fields[1]!="ISSUE234_SUMMARY_RETURN") { fail("marker shape"); next }
    if(passed) fail("marker after terminal summary")
    if(fields[2]!="rank=" rank || fields[3]!="world=" world || fields[6]!="validated=true") fail("marker identity")
    width=(count<2 ? 1 : 2)
    faulty=(count%2==0 ? "false" : "true")
    if(fields[4]!="width=" width || fields[5]!="faulty=" faulty) fail("marker order")
    count++
}
 /test result:/ {
    if($0 !~ /^test result: ok[.] 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+([.][0-9]+)?s$/) fail("test summary shape")
    passed++
}
passed && NF && $0 !~ /^test result:/ { fail("unexpected text after terminal summary") }
END { if(count!=4 || passed!=1 || bad) exit 1; print "RANK_VALIDATED rank=" rank " world=" world }
