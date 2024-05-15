#include <OpticalFlowCUDASDK.hpp>
#include <external/OpticalFlow/NvOFCuda.h>
#include <format>

template <std::integral T>
OpticalFlowCUDASDK<T>::OpticalFlowCUDASDK(OpticalFlowCUDASDKDesc desc)
	: desc(desc) {}

template <std::integral T>
tl::expected<void, std::string> OpticalFlowCUDASDK<T>::initialise() {
	static_assert(std::is_same_v<T, uint8_t>, "OpticalFlowCUDASDK only supports 8-bit unsigned integer (uint8_t).");

	CUDA_DRVAPI_CALL(cuInit(0));
	CUDA_DRVAPI_CALL(cuDeviceGet(&device, gpuId));
	char szDeviceName[80];
	CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), device));
	std::cout << "GPU in use: " << szDeviceName << std::endl;
	CUDA_DRVAPI_CALL(cuCtxCreate(&context, 0, device));

	CUDA_DRVAPI_CALL(cuStreamCreate(&inputStream, CU_STREAM_DEFAULT));
	CUDA_DRVAPI_CALL(cuStreamCreate(&outputStream, CU_STREAM_DEFAULT));

	nvOpticalFlow = NvOFCuda::Create(
		context,
		desc.config.imageWidth,
		desc.config.imageHeight,
		NV_OF_BUFFER_FORMAT_GRAYSCALE8,
		inputBufferType,
		outputBufferType,
		NV_OF_MODE_OPTICALFLOW,
		NV_OF_PERF_LEVEL_MEDIUM,
		inputStream,
		outputStream
	);

	uint32_t hwGridSize;
	if (!nvOpticalFlow->CheckGridSize(desc.gridSize)) {
		if (!nvOpticalFlow->GetNextMinGridSize(desc.gridSize, hwGridSize)) {
			return std::string("Invalid grid size parameter");
		}
		else {
			scaleFactor = hwGridSize / desc.gridSize;
		}
	}
	else {
		hwGridSize = desc.gridSize;
	}

	if (desc.enableExternalHints && (desc.hintGridSize < hwGridSize)) {
		return "Hint grid size must be same or bigger than output grid size";
	}

	nvOpticalFlow->Init(desc.gridSize, desc.hintGridSize, desc.enableExternalHints, false);

	const uint32_t numOutputBuffers = desc.numInputBuffers - 1;
	inputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, desc.numInputBuffers);
	outputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, numOutputBuffers);

	if (desc.enableExternalHints) {
		externalHintBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_HINT, 1);
	}

	if (scaleFactor > 1) {
		uint32_t outWidth = (desc.config.width + desc.gridSize - 1) / desc.gridSize;
		uint32_t outHeight = (desc.config.height + desc.gridSize - 1) / desc.gridSize;

		upSampleBuffers = nvOpticalFlow->CreateBuffers(outWidth, outHeight, NV_OF_BUFFER_USAGE_OUTPUT, 16);

		uint32_t outSize = outWidth * outHeight;
		flowVectors = std::make_unique<NV_OF_FLOW_VECTOR[]>(new NV_OF_FLOW_VECTOR[outSize]);
		if (flowVectors == nullptr) {
			return tl::unexpected(std::format("Failed to allocate output host memory of size {} bytes", outSize * sizeof(NV_OF_FLOW_VECTOR)));
		}

		flowFileWriter = NvOFFileWriter::Create(outWidth,
			outHeight,
			NV_OF_MODE_OPTICALFLOW,
			32.0f);
	}
	else {
		uint32_t outSize = outputBuffers[0]->getWidth() * outputBuffers[0]->getHeight();
		flowVectors = std::make_unique<NV_OF_FLOW_VECTOR[]>(new NV_OF_FLOW_VECTOR[outSize]);
		if (flowVectors == nullptr) {
			return tl::unexpected(std::format("Failed to allocate output host memory of size {} bytes", outSize * sizeof(NV_OF_FLOW_VECTOR)));
		}

		flowFileWriter = NvOFFileWriter::Create(outputBuffers[0]->getWidth(),
			outputBuffers[0]->getHeight(),
			NV_OF_MODE_OPTICALFLOW,
			32.0f);
	}

	return {};
}

template <std::integral T>
tl::expected<OpticalFlowCUDASDK<T>, std::string> OpticalFlowCUDASDK<T>::create(OpticalFlowCUDASDKDesc desc) {
	OpticalFlowCUDASDK instance(desc);
	auto initResult = instance.initialise();
	if (!initResult)
		return tl::unexpected(initResult.error());
	return instance;
}

template <std::integral T>
tl::expected<void, std::string> OpticalFlowCUDASDK<T>::run() {
	uint32_t	curFrameIdx = 0;
	uint32_t	frameCount	= 0;

	while (true) {
		inputBuffers[curFrameIdx]->UploadData(nullptr);
		if (enableExternalHints && (curFrameIdx > 0)) {
			if (!dataLoaderFlo->IsDone())
			{
				exthintBuffers[curFrameIdx - 1]->UploadData(dataLoaderFlo->CurrentItem());
				dataLoaderFlo->Next();
			}
			else
			{
				throw std::runtime_error("no hint file!");
			}
		}
	}

	if (curFrameIdx)
}

template <std::integral T>
OpticalFlowCUDASDK<T>::~OpticalFlowCUDASDK() {
	CUDA_DRVAPI_CALL(cuStreamDestroy(outputStream));
	CUDA_DRVAPI_CALL(cuStreamDestroy(inputStream));
	CUDA_DRVAPI_CALL(cuCtxDestroy(context));
}

template <std::integral T>
void OpticalFlowCUDASDK<T>::calculateFlowVectors(const thrust::device_vector<u16>& input1, const thrust::device_vector<u16>& input2) {

}