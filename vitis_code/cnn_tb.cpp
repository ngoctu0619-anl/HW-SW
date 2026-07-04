#include <hls_stream.h>
#include <ap_int.h>

struct axis_t {
    ap_int<32> data;
    ap_uint<1> last;
};

extern void cnn_accelerator(hls::stream<axis_t>& in_stream, hls::stream<axis_t>& out_stream);
int main() {
    hls::stream<axis_t> in, out;
    for(int i = 0; i < 4096; i++) {
        axis_t pkt = {i, 0}; 
        in.write(pkt);
    }
    
    cnn_accelerator(in, out);
    
    axis_t res = out.read();
    return 0; 
}