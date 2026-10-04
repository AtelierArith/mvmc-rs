package Native13InitParser;
use strict;use warnings;use POSIX qw(isfinite strtod);use Digest::SHA qw(sha256_hex);
sub number {
 my($s)=@_;die 'numeric grammar' unless $s=~/\A[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|0[xX](?:[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?|\.[0-9a-fA-F]+)[pP][+-]?[0-9]+)\z/;
 my($v,$tail)=strtod($s);die 'nonfinite' unless $tail==0&&isfinite($v);return $s;
}
sub stage {
 my($raw,$stage,$d,$n,$mask)=@_;my @mask=@$mask;my @lines=split /\n/,$raw;
 die 'stage whitelist' unless $stage=~/\A(?:seeded|workspace-query|random-initialized|loaded-inputs|synchronized|pre-sampling)\z/;
  my $captured=($stage eq 'seeded'||$stage eq 'workspace-query')?0:$n;
  die 'stage header' unless shift(@lines)=~/\Arank=0 group=0 \Q$d->{header}[0]\E CapturedNPara=$captured cursor=((?:0|[1-9][0-9]*)) observed_gen_rand32=((?:0|[1-9][0-9]*))\z/;
  my($cursor,$count)=($1,$2);die 'cursor' unless $cursor<=624;
 die 'u64 count domain' unless length($count)<20 || (length($count)==20 && $count le '18446744073709551615');
  die 'dims mismatch' unless shift(@lines) eq $d->{dims}[0];
  for(@{$d->{segment}}){die 'segment mismatch' unless shift(@lines) eq $_}
  die 'QP schema' unless shift(@lines)=~/\AParaQPOptTrans 0 (\S+)\z/;my $qp=number($1);
  my %e=(stage=>$stage,source_sha256=>sha256_hex($raw),cursor=>0+$cursor,word_count=>"$count",qp_weight=>$qp,dimensions=>$d->{dims}[0],segments=>$d->{segment});
  for my $field(qw(raw624 next624)){my @w=split /\s+/,shift(@lines);die 'raw shape' unless @w==624;for(@w){die 'u32' unless /\A(?:0|[1-9][0-9]*)\z/&&$_<=4294967295}$e{$field}=\@w}
  my(@params,@written,@flags);
  for(@lines){
   if(/\Aparameter (\d+) (\S+) (\S+)\z/){my($i,$re,$im)=($1,$2,$3);die 'parameter duplicate/order' unless $i==@params;push @params,[number($re),number($im)]}
   elsif(/\Aflag (\d+) mask=([01]) (.+)\z/){my($i,$bit,$v)=($1,$2,$3);die 'flag index/mask' unless $i==@written&&$i<2*$captured&&$bit==$mask[$i];push @written,0+$bit;
    if($bit){die 'int32' unless $v=~/\Avalue=(-?(?:0|[1-9][0-9]*))\z/&&$1>=-2147483648&&$1<=2147483647;push @flags,[0+$i,0+$1]}else{die 'unwritten value' unless $v eq 'NOT_DEFINED'}
   }else{die 'unexpected record'}
  }
  die 'stage shape' unless @params==$captured&&@written==2*$captured;
  $e{parameters}=\@params;$e{written_mask}=\@written;$e{defined_flags}=\@flags;return \%e;
}
1;
