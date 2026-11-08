// fused-sparse-attn-integration.cpp – Wrapper for sparse attention executor

#include "fused-sparse-attn.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace tenselerate {

class FusedSparseAttnExecutor {
public:
    enum DataType {
        FLOAT32,
        BFLOAT16
    };

    FusedSparseAttnExecutor(
        int n_heads,
        int d_k,
        int d_v,
        int local_window,
        int stride,
        DataType dtype = FLOAT32
    ) : n_heads_(n_heads), d_k_(d_k), d_v_(d_v),
        local_window_(local_window), stride_(stride), dtype_(dtype) {
        if (n_heads <= 0 || d_k <= 0 || d_v <= 0) {
            throw std::invalid_argument("Invalid attention dimensions");
        }
    }

    cudaError_t execute(
        const void* Q,
        const void* K,
        const void* V,
        void* output,
        int batch_seq,
        int n_heads,
        int d_k,
        int d_v,
        int local_window,
        int stride,
        cudaStream_t stream = nullptr
    ) {
        // Override if dimensions provided
        int actual_n_heads = (n_heads > 0) ? n_heads : n_heads_;
        int actual_d_k = (d_k > 0) ? d_k : d_k_;
        int actual_d_v = (d_v > 0) ? d_v : d_v_;
        int actual_window = (local_window > 0) ? local_window : local_window_;
        int actual_stride = (stride > 0) ? stride : stride_;

        if (!Q || !K || !V || !output) {
            throw std::invalid_argument("Null pointer passed");
        }

        if (stream == nullptr) {
            stream = 0;
        }

        cudaError_t err;

        switch (dtype_) {
            case FLOAT32:
                err = launch_fused_sparse_attn_f32(
                    reinterpret_cast<const float*>(Q),
                    reinterpret_cast<const float*>(K),
                    reinterpret_cast<const float*>(V),
                    reinterpret_cast<float*>(output),
                    batch_seq, actual_n_heads, actual_d_k, actual_d_v,
                    actual_window, actual_stride, stream
                );
                break;

            case BFLOAT16:
                err = launch_fused_sparse_attn_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(Q),
                    reinterpret_cast<const __nv_bfloat16*>(K),
                    reinterpret_cast<const __nv_bfloat16*>(V),
                    reinterpret_cast<__nv_bfloat16*>(output),
                    batch_seq, actual_n_heads, actual_d_k, actual_d_v,
                    actual_window, actual_stride, stream
                );
                break;

            default:
                throw std::invalid_argument("Unknown data type");
        }

        return err;
    }

    int n_heads() const { return n_heads_; }
    int d_k() const { return d_k_; }
    int d_v() const { return d_v_; }
    int local_window() const { return local_window_; }
    int stride() const { return stride_; }

private:
    int n_heads_;
    int d_k_;
    int d_v_;
    int local_window_;
    int stride_;
    DataType dtype_;
};

// C interface
extern "C" {
    FusedSparseAttnExecutor* create_fused_sparse_attn_executor(
        int n_heads,
        int d_k,
        int d_v,
        int local_window,
        int stride
    ) {
        try {
            return new FusedSparseAttnExecutor(n_heads, d_k, d_v, local_window, stride);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_fused_sparse_attn_executor(FusedSparseAttnExecutor* exec) {
        delete exec;
    }

    int fused_sparse_attn_execute(
        FusedSparseAttnExecutor* exec,
        const void* Q,
        const void* K,
        const void* V,
        void* output,
        int batch_seq,
        int n_heads,
        int d_k,
        int d_v,
        int local_window,
        int stride,
        cudaStream_t stream
    ) {
        if (!exec) return 1;
        try {
            cudaError_t err = exec->execute(Q, K, V, output, batch_seq,
                                           n_heads, d_k, d_v, local_window, stride, stream);
            return (err == cudaSuccess) ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }
}

} // namespace tenselerate
