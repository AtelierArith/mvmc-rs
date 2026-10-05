#!/usr/bin/env perl
# Optional developer extraction only; never invoked/read by Cargo tests.
# Extracts the unmodified shared initializer makeInitialSample from the pinned
# original vmcmake.c into an exclusive new file. vmcmake_real.c:69/109 calls this
# same complex function; makeInitialSample_real is BF-only and not extracted.
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use Fcntl qw(O_CREAT O_EXCL O_WRONLY);
my ($complex_in, $out) = @ARGV;
die "args vmcmake.c exclusive_output_directory\n" unless @ARGV == 2 && -d $out;

sub slurp {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "IO input $path: $!";
    my $s = do { local $/; <$f> };
    close $f or die "IO close: $!";
    return $s;
}
sub write_new {
    my ($name, $bytes) = @_;
    sysopen my $o, "$out/$name", O_CREAT | O_EXCL | O_WRONLY or die "IO exclusive output $name: $!";
    print {$o} $bytes or die "IO write: $!";
    close $o or die "IO output close: $!";
}
my @jobs = (
    [ $complex_in, '8431b58eaec53e5325c801b69aa8a56306ec6a154b54e099266f4e21e5dd42ed',
      'int makeInitialSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt,',
      'void copyFromBurnSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt)',
      'initializer-complex.inc', 'vmcmake.c' ],
);
my $prov = '';
for my $j (@jobs) {
    my ($path, $expected, $start, $end, $name, $label) = @$j;
    my $source = slurp($path);
    die "$label source pin mismatch\n" unless sha256_hex($source) eq $expected;
    die "$label nonunique boundaries\n" unless (() = $source =~ /\Q$start\E/g) == 1
        && (() = $source =~ /\Q$end\E/g) == 1;
    my $a = index($source, $start);
    my $b = index($source, $end, $a);
    die "$label boundary order\n" unless $b > $a;
    my $body = substr($source, $a, $b - $a);
    die "$label closing body\n" unless $body =~ /return 0;\s*}\s*\z/;
    write_new($name, "/* Unmodified body from $label, original SHA256 $expected;\n * GPL-3.0-or-later (see upstream licence header). */\n" . $body);
    $prov .= "$label original_sha256=$expected\n$name body_sha256=" . sha256_hex($body)
        . " body_bytes=" . length($body) . "\n";
}
write_new('source-provenance.txt', $prov);
