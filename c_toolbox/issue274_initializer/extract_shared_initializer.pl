#!/usr/bin/env perl
# Optional developer extraction only; never invoked/read by Cargo tests.
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use Fcntl qw(O_CREAT O_EXCL O_WRONLY);
my ($input, $out) = @ARGV;
die "args vmcmake.c exclusive_output_directory\n" unless @ARGV == 2 && -d $out;
open my $f, '<:raw', $input or die "IO input: $!";
my $source = do { local $/; <$f> };
close $f or die "IO close: $!";
my $expected = '8431b58eaec53e5325c801b69aa8a56306ec6a154b54e099266f4e21e5dd42ed';
die "source pin mismatch\n" unless sha256_hex($source) eq $expected;
my $start = 'int makeInitialSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,';
my $end = 'void copyFromBurnSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt)';
die "nonunique full-body boundaries\n" unless (() = $source =~ /\Q$start\E/g) == 1
    && (() = $source =~ /\Q$end\E/g) == 1;
my $a = index($source, $start);
my $b = index($source, $end, $a);
die "boundary order\n" unless $b > $a;
my $body = substr($source, $a, $b - $a);
die "full closing body\n" unless $body =~ /return 0;\s*}\s*\z/;
my $license_end = index($source, '*/');
die "missing upstream license\n" if $license_end < 0;
my $license = substr($source, 0, $license_end + 2);
sub write_new {
    my ($name, $bytes) = @_;
    sysopen my $o, "$out/$name", O_CREAT | O_EXCL | O_WRONLY or die "IO exclusive output: $!";
    print {$o} $bytes or die "IO write: $!";
    close $o or die "IO output close: $!";
}
write_new('shared-initializer.inc', $license . "\n/* Unmodified body from vmcmake.c:359 through before copyFromBurnSample.\n * Original source SHA256 $expected; GPL-3.0-or-later. */\n" . $body);
write_new('source-provenance.txt', "original_sha256=$expected\nbody_sha256=" . sha256_hex($body)
    . "\nbody_bytes=" . length($body) . "\nboundaries=makeInitialSample signature -> copyFromBurnSample signature\n");
