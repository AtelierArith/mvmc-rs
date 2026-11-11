// Optional NVRTC/driver resource probe; no kernel is launched.
#include <cuda.h>
#include <nvrtc.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
static void driver(CUresult status) {
  if (status != CUDA_SUCCESS) { const char* error; cuGetErrorString(status,&error); std::fprintf(stderr,"%s\n",error); std::exit(1); }
}
int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 2;
  std::ifstream file(argv[1]);
  const std::string source((std::istreambuf_iterator<char>(file)), {});
  if (!file || source.empty()) return 2;
  nvrtcProgram program;
  if (nvrtcCreateProgram(&program,source.c_str(),"interval.cu",0,nullptr,nullptr) != NVRTC_SUCCESS) return 1;
  const char* options[]={"--gpu-architecture=compute_86","--std=c++14","--fmad=false"};
  // RawExec::module uses normal contraction; the optional extra argument
  // reproduces the C-order module_without_fmad diagnostic instead.
  if (argc == 3 && std::string(argv[2]) != "--no-fmad") return 2;
  const auto status=nvrtcCompileProgram(program,argc == 3 ? 3 : 2,options);
  if (status != NVRTC_SUCCESS) {
    size_t size; nvrtcGetProgramLogSize(program,&size); std::vector<char> log(size);
    nvrtcGetProgramLog(program,log.data()); std::fprintf(stderr,"%s\n",log.data()); return 1;
  }
  size_t size; nvrtcGetPTXSize(program,&size); std::vector<char> ptx(size);
  nvrtcGetPTX(program,ptx.data()); nvrtcDestroyProgram(&program);
  driver(cuInit(0)); CUdevice device; driver(cuDeviceGet(&device,0));
  CUcontext context; driver(cuCtxCreate(&context,0,device));
  CUmodule module; driver(cuModuleLoadData(&module,ptx.data()));
  std::printf("kernel,registers,local_bytes,static_shared_bytes,ptx_bytes\n");
  for (const char* name : {"pfinv_f64","interval_resident_factor","interval_resident_factor_panel",
                           "interval_paired_factor","k_interval_hop","interval_complex_factor"}) {
    CUfunction function;
    const auto found=cuModuleGetFunction(&function,module,name);
    if (found == CUDA_ERROR_NOT_FOUND) continue;
    driver(found);
    int registers,local,shared;
    driver(cuFuncGetAttribute(&registers,CU_FUNC_ATTRIBUTE_NUM_REGS,function));
    driver(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,function));
    driver(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,function));
    std::printf("%s,%d,%d,%d,%zu\n",name,registers,local,shared,size);
  }
  driver(cuModuleUnload(module)); driver(cuCtxDestroy(context));
}
