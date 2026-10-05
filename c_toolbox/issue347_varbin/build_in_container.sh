set -e
cd /work
rm -rf src build
cp -r /repo/extern/mVMC-1.3.0 src
mkdir build; cd build
export CC=gcc CXX=g++ FC=gfortran
cmake ../src -DCMAKE_BUILD_TYPE=Release -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF -DMPI_C_COMPILER=/opt/mpich/bin/mpicc >/dev/null
make -j8 vmc.out 2>&1 | tail -3
sha256sum src/mVMC/vmc.out
gcc --version | head -1
cd ../src
sha256sum src/mVMC/vmcmain.c src/mVMC/initfile.c
