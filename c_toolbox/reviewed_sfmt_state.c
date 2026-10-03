/* Optional SFMT diagnostics, compiled against the actual reference package's
 * unchanged SFMT.c. See reviewed_parameter_c_audit.md. No Cargo dependency. */
#include <stdint.h>
#include <string.h>
#ifndef REVIEWED_SFMT_BASELINE
#define gen_rand32 reviewed_original_gen_rand32
#define init_gen_rand reviewed_original_init_gen_rand
#endif
#include "SFMT.c"
#ifndef REVIEWED_SFMT_BASELINE
#undef gen_rand32
#undef init_gen_rand
static uint64_t reviewed_words;
uint32_t gen_rand32(void) {
    uint32_t result=reviewed_original_gen_rand32();
    ++reviewed_words;
    return result;
}
void init_gen_rand(uint32_t seed) {
    reviewed_original_init_gen_rand(seed);
    reviewed_words=0;
}
#endif
/* Read only: no draws, reconstruction, reseeding or state replacement. */
int reviewed_sfmt_state(uint32_t *words, int *index, uint64_t *count) {
    if(!words || !index || !count) return -1;
    if(!initialized) return -2; /* No fabricated pre-seed state. */
    memcpy(words,psfmt32,624*sizeof(*words));
    *index=idx;
#ifdef REVIEWED_SFMT_BASELINE
    *count=0;
#else
    *count=reviewed_words;
#endif
    return 0;
}
