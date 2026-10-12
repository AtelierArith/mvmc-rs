// Native XORWOW recurrence; no cuRAND library or CUDA headers required.
extern "C" __global__ void xorwow_words(unsigned int* state, unsigned int* out,
                                        int streams, int words_per_stream) {
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= streams) return;
    unsigned int* s = state + lane * 6;
    unsigned int x=s[0], y=s[1], z=s[2], w=s[3], v=s[4], d=s[5];
    for (int i=0; i<words_per_stream; ++i) {
        unsigned int t=x^(x>>2);
        x=y; y=z; z=w; w=v;
        v=v^(v<<4)^t^(t<<1);
        d+=362437u;
        out[(size_t)i*streams+lane]=v+d;
    }
    s[0]=x; s[1]=y; s[2]=z; s[3]=w; s[4]=v; s[5]=d;
}
