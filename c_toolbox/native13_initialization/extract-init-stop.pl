#!/usr/bin/env perl
# Private diagnostic variant only; not the completed WWRAd0 binary.
use strict;use warnings;use Digest::SHA qw(sha256_hex);
use Fcntl qw(O_CREAT O_EXCL O_WRONLY);
my($input,$output)=@ARGV;die 'args' unless defined $output;
open my $f,'<:raw',$input or die $!;my $body=do{local $/;<$f>};close $f;
die 'family source pin' unless sha256_hex($body) eq '4f40056ffa052db102c6bda6878b37fb498eccd2e8b8116ecfc8507eabb7907b';
my $anchor="    fprintf(stderr, \"native13 family checkpoint failure: pre-sampling\\n\");\n    MPI_Abort(comm0, 91);\n  }";
die 'stop boundary ambiguity' unless (()=$body=~/\Q$anchor\E/g)==1;
my $stop="\n  /* Diagnostic termination AFTER original InitQPWeight and six captures.\n   * No sampler/solver execution; this is not full executable parity. */\n  fprintf(stdout, \"NATIVE13_INITIALIZATION_ONLY_COMPLETE\\n\");\n  MPI_Finalize();\n  return 0;";
$body=~s/\Q$anchor\E/$anchor$stop/;
sysopen my $out,$output,O_CREAT|O_EXCL|O_WRONLY or die $!;print {$out}$body;close $out or die $!;
print 'initialization_only=',sha256_hex($body),"\n";
