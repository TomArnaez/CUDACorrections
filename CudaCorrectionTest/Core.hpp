#include <concepts>
#include <chrono>
#include <cuda.h>
#include <driver_types.h>
#include <expected>
#include <format>
#include <iostream>
#include <memory>

#include <SLImage.h>

#include <Acquistion.hpp>
#include <Buffer.hpp>
#include <Corrections.hpp>
#include <ErrorMacros.hpp>
#include <Types.hpp>

constexpr size_t BUFFER_SIZE = 16;

void executeGraph(cudaStream_t stream, cudaGraph_t graph) {
	cudaGraphExec_t instance;
	cudaErrorCheck(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
	cudaErrorCheck(cudaGraphLaunch(instance, stream));
	cudaErrorCheck(cudaStreamSynchronize(stream));
	cudaErrorCheck(cudaGraphExecDestroy(instance));
}

class CorrectionGraphWrapper {
	cudaGraph_t graph;
	std::unique_ptr<ICorrection> correction;

public:
	CorrectionGraphWrapper(ICorrection* correction, thrust::device_vector<u16>& input, cudaStream_t stream)
		: correction(std::unique_ptr<ICorrection>(correction)) {
		cudaErrorCheck(cudaGraphCreate(&graph, 0));

		cudaErrorCheck(cudaStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
		correction->run(input, stream);
		cudaErrorCheck(cudaStreamEndCapture(stream, &graph));
	}

	void run(thrust::device_vector<u16>& input, cudaStream_t stream) {
		correction->run(input, stream);
	}

	cudaGraph_t Graph() const {
		return graph;
	}

	~CorrectionGraphWrapper() {
		cudaErrorCheck(cudaGraphDestroy(graph));
	}
};

// For std::visit
template<class... Ts> struct overload : Ts... { using Ts::operator()...; };

template<typename T>
requires Detector<T>
class Core {
	Config config;
	T* detector;
	thrust::device_vector<u16> deviceInput;
	cudaStream_t stream;
	Buffer<u16, BUFFER_SIZE> buffer;
	std::vector<ICorrection*> corrections;

	void runCorrections(thrust::device_vector<u16>& input) {
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (auto& corr : corrections)
			corr->run(input, stream);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Time for operation: %f ms\n", milliseconds);
	}

public:
	Core(Config config, thrust::device_vector<u16> deviceInput, T* detector)
		: config(config), deviceInput(deviceInput), buffer(config.imageHeight * config.imageWidth), detector(detector) {
		cudaErrorCheck(cudaStreamCreate(&stream));
	}

	void addCorrection(ICorrection* correction) {
		cudaErrorCheck(cudaStreamSynchronize(stream));
		cudaErrorCheck(cudaDeviceSynchronize());

		corrections.push_back(correction);

		cudaErrorCheck(cudaStreamSynchronize(stream));
		cudaErrorCheck(cudaDeviceSynchronize());
	}

	void startCapture() {
		detector->startAcquisition(10, [this](std::variant<ushort*, SLError> var) {
			std::visit(overload{
				[](SLError err) { std::cout << "Got error " << SLErrorToString(err) << std::endl; },
				[this](ushort* data) {
					auto start = std::chrono::high_resolution_clock::now();
					buffer.insert(data);

					if (buffer.getFillerBuffersCount() == BUFFER_SIZE)
						buffer.readBuffer([this](auto& buffer) {
							this->runCorrections(buffer[0]);
					});

					auto end = std::chrono::high_resolution_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
					std::cout << "Total execution time: " << duration.count() << " ms\n";
				}
			}, var);		
		});
	}

	thrust::host_vector<u16> execute() {
		runCorrections(deviceInput);
		return deviceInput;
	}

	//void recordGraph() {
	//	std::vector<cudaGraphNode_t> childNodes(corrections.size());
	//	cudaGraphNode_t dependencies[1];

	//	for (int i = 0; i < corrections.size(); ++i) {
	//		cudaGraph_t childGraph = corrections[i]->Graph();
	//		cudaGraphAddChildGraphNode(&childNodes[i], graph, nullptr, 0, childGraph);
	//	}

	//	for (int i = 1; i < childNodes.size(); ++i) {
	//		dependencies[0] = childNodes[i - 1];
	//		cudaGraphAddDependencies(graph, dependencies, &childNodes[i], 1);
	//	}
	//}

	~Core() {
		cudaErrorCheck(cudaStreamDestroy(stream));
	}
};
