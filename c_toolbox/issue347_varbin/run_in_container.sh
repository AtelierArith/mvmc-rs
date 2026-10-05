set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
V=/work/build/src/mVMC/vmc.out
rm -rf /work/run
mkdir -p /work/run
cp -r /work/opt /work/run/opt_b
cp -r /work/opt /work/run/opt_t
cp -r /work/phys /work/run/phys_b
cp -r /work/phys /work/run/phys_t
# odd NPara: drop the Jastrow family
cp -r /work/opt /work/run/odd_b
cp -r /work/opt /work/run/odd_t
for d in odd_b odd_t; do
  grep -v "Jastrow" /work/run/$d/namelist.def > /work/run/$d/n2 && mv /work/run/$d/n2 /work/run/$d/namelist.def
done
(cd /work/run/opt_b && $V -b namelist.def > stdout.txt 2> stderr.txt; echo rc=$?; ls)
(cd /work/run/opt_t && $V namelist.def > stdout.txt 2> stderr.txt; echo rc=$?; ls)
(cd /work/run/phys_b && $V -b namelist.def zqp_opt.dat > stdout.txt 2> stderr.txt; echo rc=$?; ls)
(cd /work/run/phys_t && $V namelist.def zqp_opt.dat > stdout.txt 2> stderr.txt; echo rc=$?; ls)
(cd /work/run/phys_b && cat stderr.txt | head -5)
(cd /work/run/odd_b && $V -b namelist.def > stdout.txt 2> stderr.txt; echo rc=$?; ls; head -5 stderr.txt)
(cd /work/run/odd_t && $V namelist.def > stdout.txt 2> stderr.txt; echo rc=$?; ls)
$V -v || true
$V -h || true
# positional-parameter contract cases
cp -r /work/phys /work/run/phys_noinit_b
cp -r /work/opt /work/run/opt_init_b
cp /work/phys/zqp_opt.dat /work/run/opt_init_b/initpara.dat
(cd /work/run/phys_noinit_b && $V -b namelist.def > stdout.txt 2> stderr.txt; echo rc=$?; ls output; head -3 stderr.txt)
(cd /work/run/opt_init_b && $V -b namelist.def initpara.dat > stdout.txt 2> stderr.txt; echo rc=$?; ls output; head -3 stderr.txt)
