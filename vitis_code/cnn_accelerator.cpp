#include "cnn_accelerator.h"
#include "model_weights.h" 

void cnn_accelerator(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    pixel_t image_buffer[4096];
    #pragma HLS ARRAY_PARTITION variable=image_buffer cyclic factor=8 dim=1

    for(int i = 0; i < 4096; i++) {
        #pragma HLS PIPELINE II=1
        axis_t temp = in_stream.read();
        image_buffer[i] = (pixel_t)temp.data;
    }

    conv_loop_y: for(int y = 0; y < 62; y++) {
        conv_loop_x: for(int x = 0; x < 62; x++) {
            #pragma HLS PIPELINE II=1
            
            acc_t sum = 0;
            
            conv_kernel_y: for(int ky = 0; ky < 3; ky++) {
                conv_kernel_x: for(int kx = 0; kx < 3; kx++) {
                    pixel_t val = image_buffer[(y + ky) * 64 + (x + kx)];
                    int weight = conv1_weight[ky * 3 + kx];
                    
                    sum += (acc_t)(val * weight);
                }
            }
            
            acc_t result = sum + conv1_bias[0];
            if (result < 0) result = 0;
            
            axis_t out_pkt;
            out_pkt.data = (ap_int<32>)result;
            out_pkt.last = (y == 61 && x == 61) ? 1 : 0;
            out_stream.write(out_pkt);
        }
    }
}