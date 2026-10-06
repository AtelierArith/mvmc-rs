//! `mvmc-cuda`: the normal `mvmc` command line with the CUDA provider registered (issue #464),
//! so `MVMC_RS_SR_BACKEND=cuda[:N]` works. Everything else is the unchanged `mvmc` driver.

fn main() {
    mvmc_gpu_cuda::install();
    mvmc_cli::run_cli();
}
