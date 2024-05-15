#include <concepts>

#include <external/OpticalFlow/nvOpticalFlowCuda.h>
#include <external/OpticalFlow/nvOpticalFlowCommon.h>
#include <external/OpticalFlow/NvOF.h>
#include <external/OpticalFlow/NvOFUtils.h>

#include <tl/expected.hpp>
#include <OpticalFlow.hpp>
#include <Types.hpp>

struct OpticalFlowCUDASDKDesc {
	Config config;
	uint32_t gridSize;
	uint32_t hintGridSize;
	bool enableExternalHints;
	uint32_t numInputBuffers;
};

template <std::integral T>
class OpticalFlowCUDASDK : public IOpticalFlow<T> {
	OpticalFlowCUDASDKDesc desc;
	int gpuId;
	int scaleFactor = 1;
	CUdevice device;
	CUcontext context;
	CUstream inputStream;
	CUstream outputStream;
	NV_OF_CUDA_BUFFER_TYPE inputBufferType = NV_OF_CUDA_BUFFER_TYPE_CUARRAY;
	NV_OF_CUDA_BUFFER_TYPE outputBufferType = NV_OF_CUDA_BUFFER_TYPE_CUARRAY;
	std::vector<NvOFBufferObj> externalHintBuffers;
	std::vector<NvOFBufferObj> upSampleBuffers;
	std::vector<NvOFBufferObj> inputBuffers;
	std::vector<NvOFBufferObj> outputBuffers;
	std::unique_ptr<NvOF> nvOpticalFlow;
	std::unique_ptr<NvOFFileWriter> flowFileWriter;
	std::unique_ptr<NV_OF_FLOW_VECTOR[]> flowVectors;

	tl::expected<void, std::string> initialise();
	OpticalFlowCUDASDK(OpticalFlowCUDASDKDesc desc);
public:
	static tl::expected<OpticalFlowCUDASDK, std::string> create(OpticalFlowCUDASDKDesc desc);
	tl::expected<void, std::string> run();
	~OpticalFlowCUDASDK();
	void calculateFlowVectors(const thrust::device_vector<u16>& input1, const thrust::device_vector<u16>& input2) override;
};