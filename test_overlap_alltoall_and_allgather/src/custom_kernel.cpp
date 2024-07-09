#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvrtc.h>
#include <torch/extension.h>

#include <regex>
#include <vector>
#if defined(__linux__)
#include <sys/wait.h>
#endif

#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_CPU
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS

#define CHECK_EQ(x, y) AT_ASSERTM((x) == (y), "CHECK_EQ fails.")
#define CHECK_NE(x, y) AT_ASSERTM((x) != (y), "CHECK_NE fails.")
#define CHECK_LE(x, y) AT_ASSERTM((x) <= (y), "CHECK_LE fails.")
#define CHECK_CPU(x) AT_ASSERTM(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
	AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

static std::vector<ncclComm_t> g_nccl_comm;
static at::cuda::CUDAEvent g_cuda_events;
static std::vector<at::cuda::CUDAStream> g_nccl_stream;
static int g_world_size = 0;
static int g_world_rank = 0;
static int g_local_size = 0;
static int g_local_rank = 0;

static size_t get_nccl_unique_id_size() { return sizeof(ncclUniqueId); }

static void get_nccl_unique_id(torch::Tensor &nccl_unique_id_tensor) {
	int num_stream = nccl_unique_id_tensor.size(0);
	CHECK_CPU(nccl_unique_id_tensor);
	CHECK_EQ(nccl_unique_id_tensor.nbytes(), num_stream * sizeof(ncclUniqueId));
	for (int i = 0; i < num_stream; ++i) {
		ncclUniqueId nccl_unique_id;
		CHECK_EQ(0, ncclGetUniqueId(&nccl_unique_id));
		memcpy(((char *)nccl_unique_id_tensor.data_ptr()) +
		           i * sizeof(ncclUniqueId),
		       &nccl_unique_id, sizeof(ncclUniqueId));
	}
}

static void init_nccl(const torch::Tensor &nccl_unique_id_tensor,
                      int world_size, int world_rank) {
	int num_stream = nccl_unique_id_tensor.size(0);
	g_nccl_comm.resize(num_stream);
	// printf("%d\n", num_stream);
	ncclUniqueId nccl_unique_id;
	CHECK_CPU(nccl_unique_id_tensor);
	CHECK_EQ(nccl_unique_id_tensor.nbytes(), num_stream * sizeof(ncclUniqueId));
	for (int i = 0; i < num_stream; ++i) {
		memcpy(&nccl_unique_id,
		       ((void *)nccl_unique_id_tensor.data_ptr()) +
		           i * sizeof(ncclUniqueId),
		       sizeof(ncclUniqueId));
		CHECK_EQ(0, ncclGroupStart());
		CHECK_EQ(0, ncclCommInitRank(&g_nccl_comm[i], world_size,
		                             nccl_unique_id, world_rank));
		CHECK_EQ(0, ncclGroupEnd());
		g_nccl_stream.emplace_back(at::cuda::getStreamFromPool());
	}
	g_world_size = world_size;
	g_world_rank = world_rank;

	if (const char *local_size = std::getenv("LOCAL_SIZE")) {
		g_local_size = std::atoi(local_size);
	} else {
		CHECK_EQ(0, cudaGetDeviceCount(&g_local_size));
	}
	CHECK_EQ(0, ncclCommCuDevice(g_nccl_comm[0], &g_local_rank));
}

// static void all_to_all_overlap_all_gather(const torch::Tensor &input,
//                                           const torch::Tensor &es_buffer,
//                                           const torch::Tensor &output,
//                                           const int &es_size,
//                                           const int &mp_size) {
// 	size_t length = input.nbytes() / g_world_size;
// 	int es_rank = g_world_rank % es_size;
// 	int es_group = g_world_rank / es_size;
// 	int S_es = g_world_size / es_size;

// 	int mp_rank = g_world_rank % mp_size;
// 	int mp_group = g_world_rank / mp_size;

// 	for (auto &nccl_stream : g_nccl_stream) {
// 		c10::cuda::CUDACachingAllocator::recordStream(
// 		    input.storage().data_ptr(), nccl_stream);
// 		c10::cuda::CUDACachingAllocator::recordStream(
// 		    es_buffer.storage().data_ptr(), nccl_stream);
// 		c10::cuda::CUDACachingAllocator::recordStream(
// 		    output.storage().data_ptr(), nccl_stream);
// 	}
// 	const at::cuda::CUDAStream &original_stream =
// 	    at::cuda::getCurrentCUDAStream();
// 	at::cuda::setCurrentCUDAStream(g_nccl_stream[0]);
// 	g_cuda_events.record(original_stream);
// 	g_cuda_events.block(g_nccl_stream[0]);
// 	for (int i = 0; i < S_es; ++i) {
// 		CHECK_EQ(0, ncclGroupStart());
// 		for (int j = 0; j < es_size; ++j) {
// 			int send =
// 			    ((i + es_group) % S_es * es_size + (es_rank + j) % es_size);
// 			int recv = ((es_group - i + S_es) % S_es * es_size +
// 			            (es_rank + es_size - j) % es_size);
// 			CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + send * length,
// 			                     length, ncclInt8, send, g_nccl_comm[0],
// 			                     g_nccl_stream[0].stream()));
// 			CHECK_EQ(0, ncclRecv(((char *)es_buffer.data_ptr()) + j * length,
// 			                     length, ncclInt8, recv, g_nccl_comm[0],
// 			                     g_nccl_stream[0].stream()));
// 		}
// 		CHECK_EQ(0, ncclGroupEnd());
// 		torch::Tensor mp_buff = es_buffer.sum(0);
// 		c10::cuda::CUDACachingAllocator::recordStream(
// 		    mp_buff.storage().data_ptr(), g_nccl_stream[1]);
// 		g_cuda_events.record(g_nccl_stream[0]);
// 		g_cuda_events.block(g_nccl_stream[1]);
// 		CHECK_EQ(0, ncclGroupStart());
// 		for (int j = 0; j < mp_size; ++j) {
// 			int send = mp_group * mp_size + j;
// 			int recv = send;
// 			CHECK_EQ(0,
// 			         ncclSend(((char *)mp_buff.data_ptr()), length, ncclInt8,
// 			                  send, g_nccl_comm[1], g_nccl_stream[1].stream()));
// 			CHECK_EQ(
// 			    0, ncclRecv(
// 			           ((char *)output.data_ptr()) +
// 			               ((send / es_size - i + S_es) % S_es * mp_size + j) *
// 			                   length,
// 			           length, ncclInt8, recv, g_nccl_comm[1],
// 			           g_nccl_stream[1].stream()));
// 		}
// 		CHECK_EQ(0, ncclGroupEnd());
// 	}
// 	g_cuda_events.record(g_nccl_stream[0]);
// 	g_cuda_events.block(original_stream);
// 	g_cuda_events.record(g_nccl_stream[1]);
// 	g_cuda_events.block(original_stream);
// 	at::cuda::setCurrentCUDAStream(original_stream);
// }

static void all_to_all_overlap_all_gather(const torch::Tensor &input,
                                          const torch::Tensor &output) {
	int mp_size = output.numel() / input.numel();
	int local_start = g_world_rank / mp_size * mp_size;
	int local_id = g_world_rank - local_start;
	int alltoall_bias = input.nbytes() * local_id;
	size_t length = input.nbytes() / g_world_size;
	for(auto &nccl_stream : g_nccl_stream){
	c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(),
	                                              nccl_stream);
	c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(),
	                                              nccl_stream);
	}
	g_cuda_events.record(at::cuda::getCurrentCUDAStream());
	g_cuda_events.block(g_nccl_stream[0]);

	// ========== 0 step ==========
	CHECK_EQ(0, ncclGroupStart());
	CHECK_EQ(0, ncclSend(((char *)input.data_ptr()) + g_world_rank * length,
	                     length, ncclInt8, g_world_rank, g_nccl_comm[0],
	                     g_nccl_stream[0].stream()));
	CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) + g_world_rank * length +
	                         alltoall_bias,
	                     length, ncclInt8, g_world_rank, g_nccl_comm[0],
	                     g_nccl_stream[0].stream()));
	CHECK_EQ(0, ncclGroupEnd());
	// ========== 1 to n - 1 step ==========

	for (int i = 1; i < g_world_size; ++i) {
		g_cuda_events.record(g_nccl_stream[0]);
		g_cuda_events.block(g_nccl_stream[1]);
		CHECK_EQ(0, ncclGroupStart());
		for (int j = 0; j < mp_size; ++j) {
			CHECK_EQ(0, ncclSend(((char *)output.data_ptr()) +
			                         (g_world_rank + g_world_size - i + 1) %
			                             g_world_size * length +
			                         alltoall_bias,
			                     length, ncclInt8, local_start + j,
g_nccl_comm[1], 			                     g_nccl_stream[1].stream()));

			CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) +
			                         (local_start + j + g_world_size - i + 1) %
			                             g_world_size * length +
			                         input.nbytes() * j,
			                     length, ncclInt8, local_start + j,
g_nccl_comm[1], 			                     g_nccl_stream[1].stream()));
		}
		CHECK_EQ(0, ncclGroupEnd());
		CHECK_EQ(0, ncclGroupStart());
		CHECK_EQ(0,
		         ncclSend(((char *)input.data_ptr()) +
		                      (g_world_rank + i) % g_world_size * length,
		                  length, ncclInt8, (g_world_rank + i) % g_world_size,
		                  g_nccl_comm[0], g_nccl_stream[0].stream()));
		CHECK_EQ(0, ncclRecv(((char *)output.data_ptr()) +
		                         (g_world_rank + g_world_size - i) %
		                             g_world_size * length +
		                         alltoall_bias,
		                     length, ncclInt8,
		                     (g_world_rank + g_world_size - i) % g_world_size,
		                     g_nccl_comm[0], g_nccl_stream[0].stream()));
		CHECK_EQ(0, ncclGroupEnd());
	}
	// =========== n step ==========
	CHECK_EQ(0, ncclGroupStart());
	for (int j = 0; j < mp_size; ++j) {
		CHECK_EQ(0,
					ncclSend(((char *)output.data_ptr()) +
								(g_world_rank + 1) % g_world_size * length +
								alltoall_bias,
							length, ncclInt8, local_start + j, g_nccl_comm[0],
							g_nccl_stream[0].stream()));

		CHECK_EQ(
			0, ncclRecv(((char *)output.data_ptr()) +
							(local_start + j + 1) % g_world_size * length +
							input.nbytes() * j,
						length, ncclInt8, local_start + j, g_nccl_comm[0],
						g_nccl_stream[0].stream()));
	}
	CHECK_EQ(0, ncclGroupEnd());

	g_cuda_events.record(g_nccl_stream[0]);
	g_cuda_events.block(at::cuda::getCurrentCUDAStream());
	g_cuda_events.record(g_nccl_stream[1]);
	g_cuda_events.block(at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("get_nccl_unique_id_size", &get_nccl_unique_id_size,
	      "Get size of ncclUniqueId in bytes");
	m.def("get_nccl_unique_id", &get_nccl_unique_id,
	      "Get ncclUniqueId for NCCL initialization");
	m.def("init_nccl", &init_nccl, "NCCL initialization");
	// m.def("all_to_all", &all_to_all, "all to all");
	m.def("all_to_all_overlap_all_gather", &all_to_all_overlap_all_gather,
	      "all_to_all overlap all_gather");
	// m.def("all_to_all_method2", &all_to_all_method2, "all_to_all method2");
	// m.def("all_to_all_method3", &all_to_all_method3, "all_to_all method3");
}