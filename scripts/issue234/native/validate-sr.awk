# Exact per-rank capture; source-only validator, not a numerical oracle.
function bad(s) { print "INVALID "s > "/dev/stderr"; failed=1 }
function expected(i) {
 width=(i<=5?1:2); slot=(i-1)%5
 faulty=(slot>0 && rank==(slot<=2?0:world-1))?"true":"false"
 failure=slot>0?"true":"false"
 callbacks=slot>0?0:1
}
function checkpoint(line, a, b, n, i, cursor, count) {
 n=split(line,a,"raw="); if(n!=2){bad("raw field");return ""}
 if(a[2]!~/^\[[0-9, ]+\] cursor=[0-9]+ count=[0-9]+ next624=\[[0-9, ]+\] config=ElectronConfiguration /)bad("checkpoint schema")
 split(a[2],b," cursor=");split(b[2],a," ");cursor=a[1]
 if(cursor!~/^(0|[1-9][0-9]*)$/ || cursor+0>624)bad("cursor grammar/range")
 split(line,b," count=");split(b[2],a," ");count=a[1]
 if(count!~/^(0|[1-9][0-9]*)$/)bad("count grammar")
 split(line,a,"raw=")
 n=split(a[2],b,"\\] cursor=");if(n!=2)bad("raw boundary")
 sub(/^\[/,"",b[1]);n=split(b[1],a,", ");if(n!=624)bad("raw length")
 for(i=1;i<=n;i++)if(a[i]!~/^(0|[1-9][0-9]*)$/ || a[i]+0>4294967295)bad("raw word")
 n=split(line,b,"next624=\\[");if(n!=2)bad("next boundary")
 sub(/\].*$/,"",b[2]);n=split(b[2],a,", ");if(n!=624)bad("next length")
 for(i=1;i<=n;i++)if(a[i]!~/^(0|[1-9][0-9]*)$/ || a[i]+0>4294967295)bad("next word")
 return substr(line,index(line,"raw="))
}
/ISSUE178_SR_/ {
 if($1=="ISSUE178_SR_RETURN") {
  if(pending || checks!=returns || done)bad("RETURN order")
  returns++;expected(returns)
  prefix="ISSUE178_SR_RETURN rank="rank" width="width" faulty="faulty" result="
  if(index($0,prefix)!=1)bad("RETURN identity")
  suffix=substr($0,length(prefix)+1)
  if(slot==0 && suffix!="Ok(()) callbacks=1")bad("healthy return")
  want="Err(\"vmc_para_opt: direct SR failed at step 0 (local status "(faulty=="true"?1:0)"); parameters were not updated\") callbacks=0"
  if(slot>0 && suffix!=want)bad("failure return literal/local status")
 } else if($1=="ISSUE178_SR_CHECK") {
  if(checks+1!=returns || pending || done)bad("CHECK order")
  checks++;expected(checks)
  prefix="ISSUE178_SR_CHECK rank="rank" width="width" faulty="faulty" collective_failure="failure" diagonal="
  if(index($0,prefix)!=1)bad("CHECK identity")
  suffix=substr($0,length(prefix)+1)
  split(suffix,tokens," ");diagonal=tokens[1]
  if(diagonal!~/^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?(0|[1-9][0-9]*))?$/)bad("finite diagonal grammar")
  if(faulty=="true" && diagonal+0>=0)bad("negative diagonal")
  if(faulty=="false" && diagonal+0<=0)bad("positive diagonal")
  if(faulty=="true") {
   if(suffix!~/^-[0-9.eE+-]+ factor=Some\([1-9][0-9]*\) solve=None callbacks=0 raw=/)bad("failed factor/solve")
  } else if(suffix!~/^[0-9][0-9.eE+-]* factor=Some\(0\) solve=Some\(0\) callbacks=[01] raw=/)bad("healthy factor/solve")
  if(index(suffix," callbacks="callbacks" raw=")==0)bad("callbacks")
  saved=checkpoint($0)
  if(slot==1 || slot==3)anchor=saved
  if((slot==2 || slot==4) && saved!=anchor)bad("repeat checkpoint differs")
  pending=slot>0
 } else if($1=="ISSUE178_SR_REPEAT") {
  expected(checks);bad_rank=slot<=2?0:world-1;trial=(slot==2 || slot==4)?1:0
  want="ISSUE178_SR_REPEAT rank="rank" world="world" width="width" bad_rank="bad_rank" trial="trial" exact=true"
  if(!pending || $0!=want || done)bad("REPEAT fields/order")
  pending=0;repeats++
 } else if($1=="ISSUE178_SR_DONE") {
  if($0!="ISSUE178_SR_DONE rank="rank" world="world || checks!=10 || pending || ++done!=1)bad("DONE identity/order")
 } else bad("unknown/interleaved marker")
}
/^test result:/ { if($0!~/^test result: ok\. 1 passed; 0 failed; 0 ignored;/)bad("libtest result"); summaries++ }
END {
 if(returns!=10 || checks!=10 || repeats!=8 || done!=1 || summaries!=1)bad("incomplete rank receipt")
 if(failed)exit 1
 print "VALIDATED SR rank="rank" world="world" returns=10 checks=10 repeats=8"
}
