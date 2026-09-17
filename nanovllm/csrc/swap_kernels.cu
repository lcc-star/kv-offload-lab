#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_MAPPING 4

__global__ void swap_blocks_kernel(
    char* __restrict__ src,
    char* __restrict__ dst,
    const int64_t* __restrict__ block_mapping,
    int num_mappings,
    int64_t plane_block_size_in_bytes,
    int64_t src_num_blocks,
    int64_t dst_num_blocks
) {
    int map_idx = blockIdx.x;
    if (map_idx >= num_mappings) return;

    int64_t plane_idx = blockIdx.z;
    int64_t src_block_idx = block_mapping[map_idx * 2];
    int64_t dst_block_idx = block_mapping[map_idx * 2 + 1];
    char* src_block = src +
        (plane_idx * src_num_blocks + src_block_idx) * plane_block_size_in_bytes;
    char* dst_block = dst +
        (plane_idx * dst_num_blocks + dst_block_idx) * plane_block_size_in_bytes;

    // blockIdx.y splits data within one mapping across BLOCKS_PER_MAPPING blocks
    int64_t chunk_size =
        (plane_block_size_in_bytes / 16 + gridDim.y - 1) / gridDim.y * 16;
    int64_t start = blockIdx.y * chunk_size;
    int64_t end = start + chunk_size;
    if (end > plane_block_size_in_bytes) end = plane_block_size_in_bytes;

    for (int64_t off = start + threadIdx.x * 16; off < end; off += blockDim.x * 16)
        *reinterpret_cast<int4*>(dst_block + off) =
            *reinterpret_cast<const int4*>(src_block + off);
}

void swap_blocks(
    torch::Tensor& src,
    torch::Tensor& dst,
    torch::Tensor& block_mapping,
    int64_t block_size_in_bytes
) {
    int num_mappings = block_mapping.size(0);
    if (num_mappings == 0) return;

    TORCH_CHECK(block_size_in_bytes % 16 == 0,
                "block_size_in_bytes must be aligned to 16 bytes");
    TORCH_CHECK(src.dim() >= 3 && dst.dim() == src.dim(),
                "src and dst must have matching KV cache dimensions");
    TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(),
                "src and dst must be contiguous");
    TORCH_CHECK(src.size(0) == dst.size(0) && src.size(1) == dst.size(1),
                "src and dst must have matching K/V and layer dimensions");
    for (int dim = 3; dim < src.dim(); ++dim) {
        TORCH_CHECK(src.size(dim) == dst.size(dim),
                    "src and dst must have matching block shapes");
    }

    int64_t num_planes = src.size(0) * src.size(1);
    int64_t plane_block_size_in_bytes = src.stride(2) * src.element_size();
    TORCH_CHECK(plane_block_size_in_bytes % 16 == 0,
                "each plane block must be aligned to 16 bytes");
    TORCH_CHECK(block_size_in_bytes == num_planes * plane_block_size_in_bytes,
                "block_size_in_bytes does not match the KV cache layout");

    char* src_ptr;
    char* dst_ptr;

    if (src.is_cuda() && !dst.is_cuda()) {
        src_ptr = static_cast<char*>(src.data_ptr());
        cudaHostGetDevicePointer((void**)&dst_ptr, dst.data_ptr(), 0);
    } else if (!src.is_cuda() && dst.is_cuda()) {
        cudaHostGetDevicePointer((void**)&src_ptr, src.data_ptr(), 0);
        dst_ptr = static_cast<char*>(dst.data_ptr());
    } else {
        src_ptr = static_cast<char*>(src.data_ptr());
        dst_ptr = static_cast<char*>(dst.data_ptr());
    }

    dim3 grid(num_mappings, BLOCKS_PER_MAPPING, num_planes);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    swap_blocks_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
        src_ptr, dst_ptr,
        block_mapping.data_ptr<int64_t>(),
        num_mappings, plane_block_size_in_bytes,
        src.size(2), dst.size(2)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swap_blocks", &swap_blocks,
          "Swap KV cache blocks between GPU and CPU via zero-copy");
}
