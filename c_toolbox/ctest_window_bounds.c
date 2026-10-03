/* Bounds-only C assignment/index probe; never reads uninitialized numbers. */
#include <stdio.h>
#include <stdlib.h>
enum { IdxSROptItrStep, IdxSROptItrSmp };
static int NSROptItrStep, NSROptItrSmp;
static unsigned char *written;
static void StoreOptData(int index) {
    if (index < 0 || index >= NSROptItrSmp) abort();
    written[index] = 1;
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    int bufInt[2] = {atoi(argv[1]), atoi(argv[2])};
    if (bufInt[0] < 1 || bufInt[1] < 1 || bufInt[0] > 10000 || bufInt[1] > 10000) return 2;
    /* Verbatim readdef.c:683-684, no clamp. */
    NSROptItrStep = bufInt[IdxSROptItrStep];
    NSROptItrSmp = bufInt[IdxSROptItrSmp];
    written = calloc((size_t)NSROptItrSmp, sizeof(*written));
    if (!written) return 3;
    int step;
    /* Verbatim vmcmain.c:339 loop header and :511-513 collection body. */
    for(step=0;step<NSROptItrStep;step++) {
        if(step >= NSROptItrStep-NSROptItrSmp) {
            StoreOptData(step-(NSROptItrStep-NSROptItrSmp));
        }
    }
    int count = 0, first = -1, last = -1;
    for (int i = 0; i < NSROptItrSmp; ++i) {
        if (written[i]) {
            ++count;
            if (first == -1) first = i;
            last = i;
        }
    }
    printf("steps=%d requested_window=%d effective_window=%d written=%d unwritten=%d first=%d last=%d\n",
        NSROptItrStep, bufInt[IdxSROptItrSmp], NSROptItrSmp, count,
        NSROptItrSmp - count, first, last);
    free(written);
    return 0;
}
