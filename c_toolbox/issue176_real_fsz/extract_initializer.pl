#!/usr/bin/env perl
# Optional developer extraction only; never invoked/read by Cargo tests.
# Extracts the unmodified makeInitialSample_fsz_real from the pinned original
# vmcmake_fsz_real.c (the real-FSZ sampler's initializer, vmcmake_fsz_real.c:97-120).
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use Fcntl qw(O_CREAT O_EXCL O_WRONLY);
my ($in, $out) = @ARGV;
die "args vmcmake_fsz_real.c exclusive_output_directory\n" unless @ARGV == 2 && -d $out;
open my $f, '<:raw', $in or die "IO input: $!";
my $source = do { local $/; <$f> };
close $f;
my $start = 'int makeInitialSample_fsz_real(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,int *eleSpn,';
die "nonunique start\n" unless (() = $source =~ /\Q$start\E/g) == 1;
my $a = index($source, $start);
my $b = rindex($source, "\n#endif");
die "boundary order\n" unless $b > $a;
my $body = substr($source, $a, $b - $a);
die "closing body\n" unless $body =~ /return 0;\s*}\s*\z/;
my $hash = sha256_hex($source);
sysopen my $o, "$out/initializer-fsz-real.inc", O_CREAT | O_EXCL | O_WRONLY or die "IO output: $!";
print {$o} "/* Unmodified body from vmcmake_fsz_real.c, original SHA256 $hash;\n * GPL-3.0-or-later (see upstream licence header). */\n$body\n" or die;
close $o or die;
sysopen my $p, "$out/source-provenance.txt", O_CREAT | O_EXCL | O_WRONLY or die "IO output: $!";
print {$p} "vmcmake_fsz_real.c original_sha256=$hash\ninitializer-fsz-real.inc body_sha256=" . sha256_hex($body) . " body_bytes=" . length($body) . "\n";
close $p or die;
