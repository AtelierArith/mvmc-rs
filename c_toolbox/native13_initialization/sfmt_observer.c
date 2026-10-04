/* Developer-only native SFMT observation. Included SFMT.c retains its original
 * 2006/2007 Saito/Matsumoto/Hiroshima copyright and BSD license (LICENSE.txt).
 * Compile with the original native SFMT flags and exact pinned include path.
 * Initial real2 path counts only gen_rand32; other generator APIs fail snapshot.
 */
#include <stdint.h>
#include <stdlib.h>
#define gen_rand32 native13_original_word
#define gen_rand64 native13_original_wide
#define init_gen_rand native13_original_seed
#define init_by_array native13_original_array_seed
#define fill_array32 native13_original_fill32
#define fill_array64 native13_original_fill64
#include "SFMT.c"
#undef gen_rand32
#undef gen_rand64
#undef init_gen_rand
#undef init_by_array
#undef fill_array32
#undef fill_array64
#include "observer.h"
#if defined(BIG_ENDIAN64) || defined(ONLY64) || defined(HAVE_ALTIVEC)
#error "Observation requires little-endian SFMT19937"
#endif
_Static_assert(MEXP == 19937 && N32 == 624 && sizeof(sfmt) == 2496, "SFMT ABI");
static uint64_t observed_count;
static int unsupported;
uint32_t gen_rand32(void) {
    uint32_t result = native13_original_word();
    if (observed_count == UINT64_MAX) abort();
    ++observed_count;
    return result;
}
void init_gen_rand(uint32_t seed) {
    native13_original_seed(seed); observed_count = 0; unsupported = 0;
}
void init_by_array(uint32_t *keys, int length) {
    native13_original_array_seed(keys, length); observed_count = 0; unsupported = 0;
}
uint64_t gen_rand64(void) { unsupported = 1; return native13_original_wide(); }
void fill_array32(uint32_t *out, int size) { unsupported = 1; native13_original_fill32(out, size); }
void fill_array64(uint64_t *out, int size) { unsupported = 1; native13_original_fill64(out, size); }
/* Local copy recurrence exactly follows pinned SFMT.c gen_rand_all. No live
 * static state restoration, draws, reseeding or mutation is performed. */
static void refill_copy(w128_t copy[N]) {
    w128_t *r1 = &copy[N - 2], *r2 = &copy[N - 1];
    int i;
    for (i = 0; i < N - POS1; ++i) {
        do_recursion(&copy[i], &copy[i], &copy[i + POS1], r1, r2);
        r1 = r2; r2 = &copy[i];
    }
    for (; i < N; ++i) {
        do_recursion(&copy[i], &copy[i], &copy[i + POS1 - N], r1, r2);
        r1 = r2; r2 = &copy[i];
    }
}
int native13_snapshot(uint32_t raw[624], int *cursor, uint64_t *count,
                      uint32_t next[624]) {
    if (!raw || !cursor || !count || !next) return -1;
    if (!initialized || unsupported || idx < 0 || idx > N32) return -2;
    w128_t before[N], copy[N];
    int saved_idx = idx, saved_initialized = initialized;
    uint64_t saved_count = observed_count;
    memcpy(before, sfmt, sizeof(sfmt)); memcpy(copy, before, sizeof(copy));
    memcpy(raw, psfmt32, sizeof(sfmt));
    *cursor = idx; *count = observed_count;
    int local_idx = idx;
    for (int i = 0; i < 624; ++i) {
        if (local_idx >= N32) { refill_copy(copy); local_idx = 0; }
        next[i] = copy[local_idx / 4].u[local_idx % 4]; ++local_idx;
    }
    return memcmp(before, sfmt, sizeof(sfmt)) || idx != saved_idx ||
        initialized != saved_initialized || observed_count != saved_count ? -3 : 0;
}
