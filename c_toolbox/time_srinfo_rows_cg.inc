/* Verbatim lines from authoritative mVMC-1.4.0 (GPL-3.0-or-later, Copyright (C) 2016 The University of Tokyo).
 * Regenerate with scripts/check_time_srinfo_c_parity.py --write. Optional oracle, not a Rust build/test dependency. */
/* stcopt_cg_impl.c:285-286 sha256=84a1bb1867a957f25d8a822f67213ff044bf52448d96263c2efdf085bb815dd5 */
    fprintf(FileSRinfo, "%5d %5d %5d %5d % .5e % .5e % .5e %5d %d\n",NPara,nSmat,optNum,cutNum,
            sDiagMax,sDiagMin,rmax,smatToParaIdx[simax], info);
