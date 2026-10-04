#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use Fcntl qw(O_CREAT O_EXCL O_WRONLY);
my($path,$dest)=@ARGV;die "args\n" unless defined $dest;
open my $in,'<:raw',$path or die $!;local $/;my $original=<$in>;close $in;
die "source SHA\n" unless sha256_hex($original) eq 'fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63';
my $text=$original;
sub insert_after {my($anchor,$add)=@_;my $n=()=$text=~/\Q$anchor\E/g;die "ambiguous $anchor\n" unless $n==1;$text=~s/\Q$anchor\E/$anchor$add/}
insert_after('#include "physcal_lanczos.h"',"\n#include \"family_hook.h\"");
my @anchors=('  init_gen_rand(RndSeed+group1);','  LapackLWork = getLWork_fcmp(); //TBC',
    '  InitParameter(); /* Run parallelly for synchronization of random generator */',
    '  if(rank0==0) fprintf(stdout,"End  : Initialize parameters.\\n");',
    '  SyncModifiedParameter(comm0);','  InitQPWeight();');
my @stages=qw(seeded workspace-query random-initialized loaded-inputs synchronized pre-sampling);
for my $i(0..5){my $capture=$i<2?0:1;insert_after($anchors[$i],
    "\n  if (native13_capture_family(\"$stages[$i]\", rank0, group1, RndSeed+group1, $capture) != 0) {\n".
    "    fprintf(stderr, \"native13 family checkpoint failure: $stages[$i]\\n\");\n    MPI_Abort(comm0, 91);\n  }")}
sysopen my $out,$dest,O_CREAT|O_EXCL|O_WRONLY or die $!;binmode $out;print {$out}$text or die $!;close $out or die $!;
print "original=",sha256_hex($original),"\ninstrumented=",sha256_hex($text),"\nhooks=6\n";
