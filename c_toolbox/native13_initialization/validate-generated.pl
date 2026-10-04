#!/usr/bin/env perl
# Independent output schema/serialization check, not Rust numerical comparison.
use strict;use warnings;use JSON::PP;use Digest::SHA qw(sha256_hex);
my($output,$first,$remaining,$descriptor,$inputs)=@ARGV;die 'args' unless defined $inputs;
sub bytes{open my $f,'<:raw',$_[0] or die $!;my $b=do{local $/;<$f>};close $f or die $!;return $b}
my %models;for(split /\n/,bytes($descriptor)){my($m,$kind,$v)=split /\t/,$_,3;push @{$models{$m}{$kind}},$v}
my @files=sort glob "$output/*.json";die 'exact JSON inventory' unless @files==14;
my %first=map{$_=>1}qw(HeisenbergChain_cmp GeneralRBM_cmp HeisenbergChain_fsz);
my @names=qw(seeded workspace-query random-initialized loaded-inputs synchronized pre-sampling);
for my $m(sort keys %models){
 my $j=decode_json(bytes("$output/$m.json"));die 'model/schema' unless $j->{model} eq $m&&keys(%$j)==3&&@{$j->{stages}}==6&&@{$j->{definitions}}>0;
 my @expected_defs;my %seen_defs;
 for(split /\n/,bytes("$inputs/$m/collection-inputs.before.sha256")){
  die 'definition manifest grammar' unless /\A([0-9a-f]{64})  (\/[^\n]+)\z/;my($hash,$path)=($1,$2);
  next unless $path=~m{\A\Q$inputs/$m/\E([^/]+\.def)\z};my $name=$1;
  die 'duplicate definition' if $seen_defs{$name}++;die 'definition bytes SHA' unless sha256_hex(bytes($path)) eq $hash;
  push @expected_defs,{name=>$name,sha256=>$hash};
 }
 @expected_defs=sort{$a->{name} cmp $b->{name}}@expected_defs;
 my $canonical=JSON::PP->new->canonical;
 die 'definition exact serialization' unless @expected_defs && $canonical->encode(\@expected_defs) eq $canonical->encode($j->{definitions});
 for my $i(0..5){my $stage=$j->{stages}[$i];die 'stage order' unless $stage->{stage} eq $names[$i];
  my $root=$first{$m}?$first:$remaining;my $c=bytes("$root/$m/checkpoints/$names[$i].txt");
  die 'stage SHA' unless sha256_hex($c) eq $stage->{source_sha256};
  die 'dimension/segments' unless $stage->{dimensions} eq $models{$m}{dims}[0]&&join("\n",@{$stage->{segments}}) eq join("\n",@{$models{$m}{segment}});
  my($captured,$cursor,$count)=$c=~/CapturedNPara=(\d+) cursor=(\d+) observed_gen_rand32=(\d+)/;
  die 'raw shapes/header' unless @{$stage->{raw624}}==624&&@{$stage->{next624}}==624&&$stage->{cursor}==$cursor&&$stage->{word_count} eq $count;
  my @lines=split /\n/,$c;
  die 'C QP boundary' unless $lines[15]=~/\AParaQPOptTrans 0 (\S+)\z/;
  die 'QP serialization' unless $stage->{qp_weight} eq $1;
  for my $pair(['raw624',16],['next624',17]){
   my @words=split /\s+/,$lines[$pair->[1]];die 'C word shape' unless @words==624;
   for my $word(0..623){die 'C u32 domain' unless $words[$word]=~/\A(?:0|[1-9][0-9]*)\z/&&$words[$word]<=4294967295;die "raw content/order $pair->[0] $word" unless $stage->{$pair->[0]}[$word] eq $words[$word]}
  }
  my @parameters;my @mask;my @flags;
  for(split /\n/,$c){if(/^parameter (\d+) (\S+) (\S+)$/){push @parameters,[$2,$3]}elsif(/^flag (\d+) mask=([01]) (.+)$/){my($axis,$bit,$value)=($1,$2,$3);push @mask,0+$bit;if($bit){die 'C defined flag' unless $value=~/^value=(-?\d+)$/;push @flags,[0+$axis,0+$1]}else{die 'C unwritten value' unless $value eq 'NOT_DEFINED'}}}
  die 'parameter serialization lost text/bits' unless encode_json(\@parameters) eq encode_json($stage->{parameters});
  die 'mask/defined values' unless encode_json(\@mask) eq encode_json($stage->{written_mask})&&encode_json(\@flags) eq encode_json($stage->{defined_flags});
  die 'captured shape' unless @parameters==$captured&&@mask==2*$captured;
 }
}
my $p=decode_json(bytes("$output/provenance.json"));die 'provenance schema' unless $p->{schema} eq 'native13-init-v2';
print "Independent13x6RawContentOrderQPDefinitionsAndParameterTextRoundtrip PASS\n";
