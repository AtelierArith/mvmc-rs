#!/usr/bin/env perl
# Explicit developer command; never called by Cargo.
use strict;use warnings;use JSON::PP;use Digest::SHA qw(sha256_hex);
use FindBin;require "$FindBin::Bin/Native13InitParser.pm";use Fcntl qw(O_CREAT O_EXCL O_WRONLY);
my($first,$remaining,$inputs,$descriptor,$provenance,$dest)=@ARGV;
die 'args' unless defined $dest;die 'destination exists' if $dest ne '--verify-only' && -e $dest;
sub bytes {open my $f,'<:raw',$_[0] or die $!;my $b=do{local $/;<$f>};close $f or die $!;return $b}
sub manifest {
 my @rows=split /\n/,bytes($_[0]);die 'empty manifest' unless @rows;my %seen;
 for(@rows){die 'manifest schema' unless /\A([0-9a-f]{64})  (\/[^\n]+)\z/;my($h,$p)=($1,$2);
  # Independently pinned unified receipts concatenate constituent manifests.
  # Repeated identical path/hash rows are legitimate; conflicting hashes fail.
  die 'manifest conflicting duplicate' if exists($seen{$p}) && $seen{$p} ne $h;
  $seen{$p}=$h;die 'manifest SHA' unless sha256_hex(bytes($p)) eq $h
 }return \%seen;
}
sub archived {
 my($archive,$member)=@_;open my $f,'-|','tar','-xzOf',$archive,"./$member" or die $!;
 my $data=do{local $/;<$f>};close $f or die 'archive member missing/invalid';return $data;
}
my %pins=(native_binary=>'1cd80a8d3b168e447c2f7f3b33278c63e96f5c217c715396df382a923e827f4f',original_vmcmain=>'fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63',instrumented_vmcmain=>'c033a58fc62be7ec01696a0b1b99f2a53a145c6a42c2825729a9e363388bbc4c',readdef=>'6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9',sfmt_source=>'b61f0f0239193fd4c06f2879e6e8452225bec45b3bca426902c244fd31ad4e3a',capture_first_archive=>'9a9bad9ab3b0dc37bd18d9adf27185eb3139e5b582afbdf268ea42d7f8369761',capture_remaining_archive=>'c2c9b364fb594141af21ea80721ff45e15593a8d79bbb91539685c496d9bc158');
my $origin=decode_json(bytes($provenance));
my %closure_pins=(runtime_manifest=>'9d6f2457d9a51005b6cf24b154fe13be754cdf304ed1cf96aa299640a3127fb6',source_manifest=>'c9cebb26ece3ec437642b5d1dbf91b1df14063e98fc5ce92f62f49b8f01f108f',products_manifest=>'2c711329f22426efa9845f6e9a2642cd8fba0965fdab8fbce5489678737be1ef',tools_manifest=>'b7191a85d9256708337fec960c6744d7e56dad19b21120a2db1ba9b178c4473d',first_binding=>'066d0e7e90f340a856d6ab339bf914d911dbeeb033b63a3d56b0d8f83469abb6',remaining_binding=>'577f59cf9b09113eb51de34b69e900f091535d7161261f9ca41c8b5f6485b85b');
@pins{keys %closure_pins}=values %closure_pins;
for my $name(keys(%pins),qw(runtime_manifest source_manifest products_manifest tools_manifest first_binding remaining_binding)){
 my $r=$origin->{$name};die 'provenance schema' unless ref($r) eq 'HASH'&&keys(%$r)==2&&defined($r->{path})&&($r->{sha256}//'')=~/\A[0-9a-f]{64}\z/;
 die 'artifact bytes' unless sha256_hex(bytes($r->{path})) eq $r->{sha256};
 die 'independent pin' if exists($pins{$name})&&$r->{sha256} ne $pins{$name};
 manifest($r->{path}) if $name=~/manifest$|binding$/;
}
die 'descriptor pin' unless sha256_hex(bytes($descriptor)) eq '691d1716344c549f7e987a600f7f6015734ab399ed0ec5212170bf4f0daed10a';
my %defs;for(split /\n/,bytes($descriptor)){my($m,$kind,$v)=split /\t/,$_,3;die 'descriptor row' unless defined($v)&&$kind=~/\A(?:header|dims|segment)\z/;push @{$defs{$m}{$kind}},$v}
die '13 inventory' unless keys(%defs)==13;
my %first=map{$_=>1}qw(HeisenbergChain_cmp GeneralRBM_cmp HeisenbergChain_fsz);
my %result;
for my $m(sort keys %defs){
 my $root=$first{$m}?$first:$remaining;my $d=$defs{$m};
 die 'cardinality' unless @{$d->{header}}==1&&@{$d->{dims}}==1&&@{$d->{segment}}==13;
 die 'header schema' unless $d->{header}[0]=~/\Aseed=\d+ AllComplexFlag=\d+\z/;
 die 'dims schema' unless $d->{dims}[0]=~/\ANPara=(\d+) NProj=(\d+) NRBM=(\d+) FlagRBM=([01]) NSlater=(\d+) NOptTrans=0 NQPOptTrans=1 NProjBF=0 APCount=(\d+) ParallelHeaderCount=(\d+) OrbitalGeneral=([01])\z/;
 my($n,$proj,$rbm,$flag,$slater,$ap,$p,$general)=($1,$2,$3,$4,$5,$6,$7,$8);
 die 'layout' unless $n==$proj+$flag*$rbm+$slater&&$slater==$ap+2*$p&&($general?($p>0):($p==0));
 my @mask;my $offset=0;for(@{$d->{segment}}){die 'segment schema' unless /\Asegment (\S+) start=(\d+) count=(\d+) ComplexFlag=(\d+) reader=([12])\z/;my($start,$len,$complex,$reader)=($2,$3,$4,$5);die 'offset' unless $start==$offset;for(1..$len){push @mask,1,($reader==2||$complex)?1:0}$offset+=$len}die 'total' unless $offset==$n;
 my $archive=$origin->{$first{$m}?'capture_first_archive':'capture_remaining_archive'}{path};
 die 'aggregate' unless bytes("$root/aggregate.status") eq "0\n" && archived($archive,'aggregate.status') eq "0\n";
 for(qw(native comparison original-inputs.after)){my $name="$m.$_.status";die 'case status/archive' unless bytes("$root/$name") eq "0\n" && archived($archive,$name) eq "0\n"}
 my $input_manifest="$inputs/$m/collection-inputs.before.sha256";
 my $binding_text=bytes($origin->{$first{$m}?'first_binding':'remaining_binding'}{path});
 my $input_hash=sha256_hex(bytes($input_manifest));
 die 'original input manifest not anchored' unless $binding_text=~/^\Q$input_hash  $input_manifest\E$/m;
 my $input_bound=manifest($input_manifest);
 my @stages;
 for my $stage(qw(seeded workspace-query random-initialized loaded-inputs synchronized pre-sampling)){
  my $raw=bytes("$root/$m/checkpoints/$stage.txt");my @lines=split /\n/,$raw;
  die 'checkpoint not pinned archive' unless $raw eq archived($archive,"$m/checkpoints/$stage.txt");
  push @stages,Native13InitParser::stage($raw,$stage,$d,$n,\@mask);
 }
 my @original_defs=sort grep {/\Q$inputs\E\/\Q$m\E\/[^\/]+\.def\z/}keys %$input_bound;
 my @actual_defs=sort glob "$inputs/$m/*.def";
 die 'original definition exact set' unless @original_defs && join("\n",@original_defs) eq join("\n",@actual_defs);
 my @files;for my $f(@actual_defs){push @files,{name=>($f=~m{([^/]+)$})[0],sha256=>sha256_hex(bytes($f))}}
 $result{$m}={model=>$m,definitions=>\@files,stages=>\@stages};
}
# Post-validate every prospective source/runtime binding before any writes.
for(qw(runtime_manifest source_manifest products_manifest tools_manifest first_binding remaining_binding)){manifest($origin->{$_}{path})}
if($dest eq '--verify-only'){print "Verified13InitializationCases no fixture writes\n";exit 0}
mkdir($dest,0700) or die $!;my $json=JSON::PP->new->canonical->pretty;
sub exclusive_json {my($path,$object)=@_;sysopen my $out,$path,O_CREAT|O_EXCL|O_WRONLY or die $!;print {$out}$json->encode($object) or die $!;close $out or die $!}
for(sort keys %result){exclusive_json("$dest/$_.json",$result{$_})}
exclusive_json("$dest/provenance.json",{schema=>'native13-init-v2',scope=>'initialization-only; sampling MissingEvidence',origin=>$origin});
print "13 C-derived initialization fixtures written\n";
