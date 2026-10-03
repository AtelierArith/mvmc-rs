# Independent byte-line comparison of original and generated negative input.
# No execution yet. Require exactly one intended field change on first CHECK.
NR==FNR {
 original[FNR]=$0;original_count=FNR
 if(!first_check && $0~/^ISSUE178_SR_CHECK /)first_check=FNR
 next
}
{
 modified_count=FNR
 if(FNR>original_count){failed=1;next}
 if($0==original[FNR])next
 differences++
 if(FNR!=first_check){failed=1;next}
 expected=original[FNR]
 if(variant=="floatoverflow")n=sub(/diagonal=[^ ]+/,"diagonal=1e999",expected)
 else if(variant=="configtruncated")n=sub(/ }$/," ",expected)
 else if(variant=="cursoroverflow")n=sub(/cursor=[0-9]+/,"cursor=625",expected)
 else if(variant=="rawoverflow")n=sub(/raw=\[[0-9]+/,"raw=[4294967296",expected)
 else if(variant=="rawnegative")n=sub(/raw=\[[0-9]+/,"raw=[-1",expected)
 else if(variant=="rawshort")n=sub(/raw=\[[0-9]+, /,"raw=[",expected)
 else if(variant=="countoverflow")n=sub(/count=[0-9]+/,"count=340282366920938463463374607431768211456",expected)
 else if(variant=="configinconsistent")n=sub(/, ele_num: \[[01]/,", ele_num: [2",expected)
 else {failed=1;n=0}
 if(n!=1 || expected!=$0 || expected==original[FNR])failed=1
}
END {
 if(failed||!first_check||differences!=1||original_count!=modified_count){print "INVALID_MUTATION "variant > "/dev/stderr";exit 1}
 print "MUTATION_VERIFIED variant="variant" changed_lines=1 first_check="first_check" unaffected_lines="original_count-1
}
