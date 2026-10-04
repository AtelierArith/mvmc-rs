# SOURCE candidate only; strict per-rank externally owned capture, no runtime oracle.
function bad(message) { print "INVALID "message > "/dev/stderr"; failed=1 }
function value(key, line, fields, n, i) {
 n=split(line,fields," "); for(i=1;i<=n;i++)if(index(fields[i],key"=")==1)return substr(fields[i],length(key)+2)
 return ""
}
function shape(line, marker, fields, n, i, pair, key, allowed, seen, required, count) {
 n=split(line,fields," "); split("rank world trial width case",required," ")
 for(i=1;i<=5;i++)allowed[required[i]]=1
 count=5
 if(marker!="ISSUE178_CLI_START"){allowed["status"]=1;count++}
 if(marker=="ISSUE178_CLI_RETURN"){allowed["output_exists"]=1;count++}
 if(n!=count+1)bad("field count")
 for(i=2;i<=n;i++) {
  if(split(fields[i],pair,"=")!=2) {bad("malformed field");continue}
  key=pair[1];if(!(key in allowed) || seen[key]++)bad("duplicate/unknown field")
 }
 for(key in allowed)if(!(key in seen))bad("missing field")
 if(value("rank",line)!~/^(0|[1-9][0-9]*)$/)bad("noncanonical rank")
 if(value("world",line)!~/^(2|4)$/)bad("noncanonical world")
 if(value("width",line)!~/^(1|2)$/)bad("noncanonical width")
 if(value("trial",line)!~/^(0|1)$/)bad("noncanonical trial")
 if(value("case",line)!~/^[a-z][a-z0-9-]*$/)bad("noncanonical case")
 if(marker!="ISSUE178_CLI_START" && value("status",line)!~/^(0|1)$/)bad("noncanonical status")
 if(marker=="ISSUE178_CLI_RETURN" && value("output_exists",line)!~/^(true|false)$/)bad("output boolean")
}
/ISSUE178_CLI_/ {
 if($1!~/^ISSUE178_CLI_(START|RETURN|DONE)$/)bad("unexpected/interleaved marker")
 shape($0,$1)
 if("x"value("rank",$0)!="x"rank || "x"value("world",$0)!="x"world || "x"value("trial",$0)!="x"trial || "x"value("width",$0)!="x"width || "x"value("case",$0)!="x"case_name)bad("identity mismatch")
 if($1=="ISSUE178_CLI_START") {if(++starts!=1 || returns || dones)bad("START order/count")}
 if($1=="ISSUE178_CLI_RETURN") {
  if(++returns!=1 || starts!=1 || dones)bad("RETURN order/count")
  if("x"value("status",$0)!="x"expected_status)bad("actual CLI exit status")
  if(value("output_exists",$0)!=expected_output)bad("specified output boundary")
 }
 if($1=="ISSUE178_CLI_DONE") {
  if(++dones!=1 || starts!=1 || returns!=1)bad("DONE order/count")
  if("x"value("status",$0)!="x"expected_status)bad("DONE status")
 }
}
{if(index($0,diagnostic)>0 && diagnostic!="")diagnostic_seen=1}
END {
 if(starts!=1 || returns!=1 || dones!=1)bad("incomplete process markers")
 if(expected_status!=0 && !diagnostic_seen)bad("missing independently reviewed diagnostic")
 if(failed)exit 1
 print "VALIDATED CLI rank="rank" world="world" trial="trial" width="width" case="case_name
}
