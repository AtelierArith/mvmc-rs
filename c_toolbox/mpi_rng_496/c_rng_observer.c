#define _GNU_SOURCE
#include <mpi.h>
#include <link.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifndef RNG_BINARY_OFFSETS_VERIFIED
#error "Verify vmc.out SHA-256 6ae44625e227ccd6b9c38d2a7731fb445955108772e8741b452f70c6a641ac28 before compiling and launching"
#endif
/* GNU nm -S on that unchanged upstream executable:
   00000000008555c0 0000000000000004 b idx
   00000000008555e0 00000000000009c0 b sfmt
   The ELF main image base is read at runtime; ASLR remains enabled. */
static uintptr_t image_base;
static int found;
static int locate_main(struct dl_phdr_info *info, size_t size, void *unused) {
    (void)size; (void)unused;
    if (info->dlpi_name[0] != '\0') return 0;
    image_base = info->dlpi_addr;
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr) *h = &info->dlpi_phdr[i];
        if (h->p_type == PT_LOAD && (h->p_flags & PF_W) &&
            0x8555c0 >= h->p_vaddr && 0x8555e0 + 624 * sizeof(uint32_t) <= h->p_vaddr + h->p_memsz) found = 1;
    }
    return 1;
}
extern int openblas_get_num_threads(void);
int MPI_Init(int *argc, char ***argv) {
    int status = PMPI_Init(argc,argv), rank = -1, size = -1;
    if (status != MPI_SUCCESS) return status;
    PMPI_Comm_rank(MPI_COMM_WORLD,&rank); PMPI_Comm_size(MPI_COMM_WORLD,&size);
    printf("WORLD %d %d\nTHREADS %d %d\nBLAS_THREADS %d %d\n",rank,size,rank,omp_get_max_threads(),rank,openblas_get_num_threads());
    fflush(stdout);
    return status;
}
int MPI_Finalize(void) {
    const char *dir = getenv("MVMC_RNG_AUDIT_DIR");
    if (dir && dir[0]) {
        int rank = -1; PMPI_Comm_rank(MPI_COMM_WORLD,&rank);
        dl_iterate_phdr(locate_main,NULL);
        if (!found) { fprintf(stderr,"Cannot validate main ELF RNG addresses\n"); PMPI_Abort(MPI_COMM_WORLD,1); }
        const int *idx = (const int *)(image_base + 0x8555c0);
        const uint32_t *words = (const uint32_t *)(image_base + 0x8555e0);
        if (*idx < 0 || *idx > 624) { fprintf(stderr,"Invalid SFMT idx %d\n",*idx); PMPI_Abort(MPI_COMM_WORLD,1); }
        char path[4096];
        int needed=snprintf(path,sizeof(path),"%s/c-rng-rank-%d.txt",dir,rank);
        if (needed < 0 || (size_t)needed >= sizeof(path)) PMPI_Abort(MPI_COMM_WORLD,1);
        FILE *f=fopen(path,"w");
        if (!f) { perror(path); PMPI_Abort(MPI_COMM_WORLD,1); }
        fprintf(f,"binary_sha256=6ae44625e227ccd6b9c38d2a7731fb445955108772e8741b452f70c6a641ac28\n");
        fprintf(f,"idx=%d\n",*idx);
        for (int i=0;i<624;i++) fprintf(f,"%s%u",i ? "," : "",words[i]);
        fprintf(f,"\n");
        if (fclose(f)) PMPI_Abort(MPI_COMM_WORLD,1);
    }
    return PMPI_Finalize();
}
