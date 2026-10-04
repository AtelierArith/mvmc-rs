# Prospective additional checker; run alongside unchanged historical validator.
# Saved-plane semantics only. Scratch/burn storage contents are not interpreted.
function fail(s){print "INVALID_SEMANTIC "s > "/dev/stderr";failed=1}
function unsigned(s,max){return s~/^(0|[1-9][0-9]*)$/ && (length(s)<length(max)||(length(s)==length(max)&&("x"s)<= ("x"max)))}
function scalar(line,key, a,n){n=split(line,a,key);if(n!=2){fail("scalar "key);return ""}sub(/[, ].*$/,"",a[2]);return a[2]}
function array(text,key,out, a,n,i){
 delete out;n=split(text,a,key"=\\[");if(n!=2){fail("array "key);return -1}
 if((key=="raw" && a[2]!~/^[^]]*\] cursor=/)||(key=="next624" && a[2]!~/^[^]]*\] config=ElectronConfiguration /)){fail("array closing boundary "key);return -1}
 sub(/\].*$/,"",a[2]);if(a[2]=="")return 0
 n=split(a[2],out,", ");return n
}
function cfg_array(text,key,out, a,n){
 delete out;n=split(text,a,", "key": \\[");if(n!=2){fail("configuration array "key);return -1}
 if(a[2]!~/^[^]]*\], /){fail("configuration array closing boundary "key);return -1}
 sub(/\].*$/,"",a[2]);if(a[2]=="")return 0
 return split(a[2],out,", ")
}
/^ISSUE178_SR_CHECK /{
 checks++
 d=scalar($0,"diagonal=");formatted=sprintf("%.17g",d+0)
 if(d!~/^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?(0|[1-9][0-9]*))?$/ || formatted~/[iInN]/)fail("nonfinite diagonal")
 cursor=scalar($0,"cursor=");count=scalar($0,"count=")
 if(!unsigned(cursor,"624"))fail("cursor")
 if(!unsigned(count,"340282366920938463463374607431768211455"))fail("u128 count")
 for(k=1;k<=2;k++){
  key=k==1?"raw":"next624";n=array($0,key,words)
  if(n!=624)fail(key" length")
  for(i=1;i<=n;i++)if(!unsigned(words[i],"4294967295"))fail(key" word")
 }
 n=split($0,parts,"config=ElectronConfiguration ");if(n!=2){fail("config boundary");next}
 c=parts[2];if(c!~/^\{ / || c!~/ \}$/)fail("config termination")
 # Require every printed field in its declared order and the final closing brace.
 # Scratch arrays receive grammar/field-presence checks, not physical semantics.
 integer="(-1|0|[1-9][0-9]*)";list="\\[("integer"(, "integer")*)?\\]"
 shape="^\\{ n_sample: [0-9]+, n_size: [0-9]+, n_site2: [0-9]+, n_proj: [0-9]+"
 nf=split("ele_idx ele_cfg ele_num ele_proj_cnt ele_spn tmp_ele_idx tmp_ele_cfg tmp_ele_num tmp_ele_proj_cnt tmp_ele_spn burn_ele_idx burn_ele_cfg burn_ele_num burn_ele_proj_cnt burn_ele_spn counter",fields," ")
 for(f=1;f<=nf;f++)shape=shape", "fields[f]": "list
 shape=shape" \\}$"
 if(c!~shape){fail("full config field/closing schema");next}
 samples=scalar(c,"n_sample: ");size=scalar(c,"n_size: ");site2=scalar(c,"n_site2: ");proj=scalar(c,"n_proj: ")
 # Bounded supported capture schema avoids coercion/overflow in dimension math.
 if(!unsigned(samples,"1000000")||!unsigned(size,"1000000")||!unsigned(site2,"1000000")||!unsigned(proj,"1000000")||samples+0<1||size+0<2||size%2||site2+0<2||site2%2){fail("dimensions");next}
 sites=site2/2;half=size/2
 # Capture-only resource cap; never iterate huge dimension products after failure.
 if(samples*size>4096||samples*site2>4096||samples*proj>4096){fail("capture total cell bound");next}
 ni=cfg_array(c,"ele_idx",idx);nc=cfg_array(c,"ele_cfg",map);nn=cfg_array(c,"ele_num",occ);np=cfg_array(c,"ele_proj_cnt",projection)
 if(ni!=samples*size||nc!=samples*site2||nn!=nc||np!=samples*proj){fail("saved plane lengths");next}
 for(i=1;i<=ni;i++)if(!unsigned(idx[i],sprintf("%.0f",sites-1)))fail("site index")
 for(i=1;i<=nc;i++){
  if(map[i]!="-1"&&!unsigned(map[i],sprintf("%.0f",half-1)))fail("electron map")
  if(occ[i]!~/^[01]$/||(map[i]=="-1"?0:1)!=occ[i]+0)fail("occupancy map")
 }
 for(i=1;i<=np;i++)if(!unsigned(projection[i],"2147483647"))fail("projection count")
 for(s=0;s<samples;s++)for(sp=0;sp<2;sp++)for(e=0;e<half;e++){
  site=idx[s*size+sp*half+e+1]
  if(map[s*site2+sp*sites+site+1]!=e)fail("saved inverse identity")
 }
}
END{if(checks!=10)fail("CHECK count");if(failed)exit 1;print "VALIDATED_SEMANTIC checks="checks}
