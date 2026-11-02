// fused-ffn-integration.cpp – Wrapper for integrating fused FFN into inference loop
// Provides high-level API for model inference without exposing CUDA details

#include "fused-ffn.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace tenselerate {

class FusedFFNExecutor {
public:
    enum DataType {
        FLOAT32,
        BFLOAT16
    };

    FusedFFNExecutor(int hidden, int ffn_hidden, DataType dtype)
        : hidden_(hidden), ffn_hidden_(ffn_hidden), dtype_(dtype) {
        if (hidden <= 0 || ffn_hidden <= 0) {
            throw std::invalid_argument("Invalid dimensions");
        }
    }

    // Execute fused FFN
    cudaError_t execute(
        const void* input,
        const void* ffn_up,
        const void* ffn_down,
        void* output,
        int batch_seq,
        cudaStream_t stream = nullptr
    ) {
        if (!input || !ffn_up || !ffn_down || !output) {
            throw std::invalid_argument("Null pointer passed");
        }

        cudaError_t err;

        if (stream == nullptr) {
            stream = 0;  // Default stream
        }

        switch (dtype_) {
            case FLOAT32:
                err = launch_fused_ffn_f32(
                    reinterpret_cast<const float*>(input),
                    reinterpret_cast<const float*>(ffn_up),
                    reinterpret_cast<const float*>(ffn_down),
                    reinterpret_cast<float*>(output),
                    batch_seq, hidden_, ffn_hidden_, stream
                );
                break;

            case BFLOAT16:
                err = launch_fused_ffn_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(input),
                    reinterpret_cast<const __nv_bfloat16*>(ffn_up),
                    reinterpret_cast<const __nv_bfloat16*>(ffn_down),
                    reinterpret_cast<__nv_bfloat16*>(output),
                    batch_seq, hidden_, ffn_hidden_, stream
                );
                break;

            default:
                throw std::invalid_argument("Unknown data type");
        }

        return err;
    }

    // Benchmark this executor
    float benchmark(
        const void* input,
        const void* ffn_up,
        const void* ffn_down,
        void* output,
        int batch_seq,
        int iterations = 100
    ) {
        if (dtype_ != FLOAT32) {
            throw std::runtime_error("Benchmark only implemented for float32");
        }

        return benchmark_fused_ffn_f32(
            reinterpret_cast<const float*>(input),
            reinterpret_cast<const float*>(ffn_up),
            reinterpret_cast<const float*>(ffn_down),
            reinterpret_cast<float*>(output),
            batch_seq, hidden_, ffn_hidden_, iterations
        );
    }

    int hidden() const { return hidden_; }
    int ffn_hidden() const { return ffn_hidden_; }
    DataType dtype() const { return dtype_; }

    // Estimate memory requirements
    size_t estimate_memory_bytes(int batch_seq) const {
        size_t elem_size = (dtype_ == FLOAT32) ? 4 : 2;
        // input + ffn_up + ffn_down + output + intermediate
        return batch_seq * hidden_ * elem_size  // input
             + hidden_ * ffn_hidden_ * elem_size  // ffn_up
             + ffn_hidden_ * hidden_ * elem_size  // ffn_down
             + batch_seq * hidden_ * elem_size;   // output
    }

private:
    int hidden_;
    int ffn_hidden_;
    DataType dtype_;
};

// C interface for integration with C/Fortran code
extern "C" {
    FusedFFNExecutor* create_fused_ffn_executor(
        int hidden,
        int ffn_hidden,
        int dtype_enum  // 0=FLOAT32, 1=BFLOAT16
    ) {
        try {
            auto dtype = (dtype_enum == 1)
                ? FusedFFNExecutor::BFLOAT16
                : FusedFFNExecutor::FLOAT32;
            return new FusedFFNExecutor(hidden, ffn_hidden, dtype);
        } catch (...) {
            return nullptr;
        }
    }

    void destroy_fused_ffn_executor(FusedFFNExecutor* exec) {
        delete exec;
    }

    int fused_ffn_execute(
        FusedFFNExecutor* exec,
        const void* input,
        const void* ffn_up,
        const void* ffn_down,
        void* output,
        int batch_seq,
        cudaStream_t stream
    ) {
        if (!exec) return 1;
        try {
            cudaError_t err = exec->execute(input, ffn_up, ffn_down, output, batch_seq, stream);
            return (err == cudaSuccess) ? 0 : 1;
        } catch (...) {
            return 1;
        }
    }

    float fused_ffn_benchmark(
        FusedFFNExecutor* exec,
        const void* input,
        const void* ffn_up,
        const void* ffn_down,
        void* output,
        int batch_seq,
        int iterations
    ) {
        if (!exec) return -1.0f;
        try {
            return exec->benchmark(input, ffn_up, ffn_down, output, batch_seq, iterations);
        } catch (...) {
            return -1.0f;
        }
    }
}

} // namespace tenselerate
