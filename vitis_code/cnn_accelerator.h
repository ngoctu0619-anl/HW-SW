#ifndef CNN_ACCELERATOR_H
#define CNN_ACCELERATOR_H

#include <hls_stream.h>
#include <ap_int.h>

typedef ap_int<8> pixel_t;
typedef ap_int<32> acc_t;

struct axis_t {
    ap_int<32> data;
    ap_uint<1> last;
};

void cnn_accelerator(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream);

#endif