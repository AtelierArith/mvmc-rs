# Invocation: awk -v ranks=N -v width=W -v physcal=true/false -v bad=B -f this.awk cell.log
BEGIN { single=(expected_rank!=""); prefix="test mpi_phase2::actual_sampling_failure_reaches_comm1_then_all_global_ranks ... " }
{
 if(index($0,prefix)==1 && substr($0,length(prefix)+1,13)=="ISSUE178_MPI_") $0=substr($0,length(prefix)+1)
 if(split($0,markers,"ISSUE178_MPI_")>2)fail("multiple rank markers/interleaving on one line")
}
function fail(message) { print "FAIL: " message > "/dev/stderr"; failures++ }
function value(key, line,    fields,n,i,p) { n=split(line,fields," "); for(i=1;i<=n;i++) {p=index(fields[i],"="); if(substr(fields[i],1,p-1)==key) return substr(fields[i],p+1)} return "" }
/ISSUE178_MPI_RETURN/ {
 if ($1 != "ISSUE178_MPI_RETURN") { fail("interleaved RETURN"); next }
 r=value("rank",$0); t=value("trial",$0)
 if(r !~ /^[0-9]+$/ || r>=ranks || (single && r!=expected_rank) || t !~ /^[01]$/) {fail("invalid RETURN rank/trial");next}
 if(value("ranks",$0)!=ranks || value("width",$0)!=width || value("physcal",$0)!=physcal || value("bad_rank",$0)!=bad) fail("wrong RETURN cell")
 if(index($0,"result=Err(")==0 || value("callbacks",$0)!=0) fail("unexpected successful return/callback")
 if(++returns[r,t]!=1) fail("duplicate RETURN")
}
/ISSUE178_MPI_CHECKPOINT/ {
 if($1!="ISSUE178_MPI_CHECKPOINT") {fail("interleaved CHECKPOINT");next}
 r=value("rank",$0);t=value("trial",$0)
 if(r !~ /^[0-9]+$/ || r>=ranks || (single && r!=expected_rank) || t !~ /^[01]$/) {fail("invalid CHECKPOINT rank/trial");next}
 if(++checks[r,t]!=1) fail("duplicate CHECKPOINT")
 line=$0;sub(/trial=[01]/,"trial=REPEAT",line)
 if(t==0) first[r]=line;else if(first[r]!=line) fail("same-rank checkpoint repeat differs")
 if(index(line," raw=[")==0 || index(line," next624=[")==0 || index(line," configuration=")==0) fail("incomplete checkpoint")
 raw=line;sub(/^.* raw=\[/,"",raw);sub(/\].*$/,"",raw)
 nextwords=line;sub(/^.* next624=\[/,"",nextwords);sub(/\].*$/,"",nextwords)
 if(split(raw,rawwords,",")!=624 || split(nextwords,words,",")!=624)fail("checkpoint words must have624 entries")
}
/ISSUE178_MPI_DONE/ {
 if($1!="ISSUE178_MPI_DONE") {fail("interleaved DONE");next}
 r=value("rank",$0)
 if(r !~ /^[0-9]+$/ || r>=ranks || (single && r!=expected_rank) || ++done[r]!=1) fail("invalid/duplicate DONE")
 if(value("ranks",$0)!=ranks || value("width",$0)!=width || value("physcal",$0)!=physcal || value("bad_rank",$0)!=bad) fail("wrong DONE cell")
}
/test result: ok\. 1 passed; 0 failed; 0 ignored;/ {summaries++}
/FAILED|panicked at|test result: FAILED/ {fail("native rank failure")}
END {
 start=single?expected_rank:0;end=single?expected_rank+1:ranks
 for(r=start;r<end;r++){for(t=0;t<2;t++){if(returns[r,t]!=1 || checks[r,t]!=1) fail("missing rank/trial record")}if(done[r]!=1)fail("missing DONE")}
 if(summaries!=(single?1:ranks))fail("native pass summaries do not equal ranks")
 if(failures)exit 1
 print "VALIDATED ranks=" ranks " width=" width " physcal=" physcal " bad=" bad " trials=2"
}
