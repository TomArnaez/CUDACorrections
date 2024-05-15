#pragma once

#include <tl/expected.hpp>

// CUDA Headers
#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <cub/device/device_histogram.cuh>

#include <Corrections.hpp>
#include <ErrorMacros.hpp>
#include <Types.hpp>


DarkCorrection::DarkCorrection(thrust::host_vector<u16>& darkMap, u16 offset)
	: offset(offset), darkMap(darkMap) {
}

void DarkCorrection::run(thrust::device_vector<u16>& input, cudaStream_t stream) {
	u16 offset = this->offset;
	thrust::transform(
		input.begin(), input.end(),
		darkMap.begin(),
		input.begin(),
		[offset] __device__(u16 x, u16 y) {
		return (x - y) + offset;
	});
}

void GainCorrection::run(thrust::device_vector<u16>& input, cudaStream_t stream) {
	thrust::transform(
		input.begin(), input.end(),
		normedGainMap.begin(),
		input.begin(),
		[] __device__(u16 val, double normedVal) {
		return val * normedVal;
	}
	);
}

void GainCorrection::normaliseGainMap(thrust::device_vector<u16> gainMap) {
	double sum = thrust::reduce(gainMap.begin(), gainMap.end(), unsigned long long(0), thrust::plus<unsigned long long>());
	double mean = sum / gainMap.size();

	normedGainMap = thrust::device_vector<float>(gainMap.size());

	thrust::transform(
		gainMap.begin(), gainMap.end(),
		normedGainMap.begin(),
		[mean] __device__(u16 val) {
		return  double(mean) / double(val);
	});
}

GainCorrection::GainCorrection(thrust::device_vector<u16> gainMap) {
	normaliseGainMap(gainMap);
}

constexpr size_t DEFECT_CORRECTION_KERNEL_SIZE = 3;
__constant__ u16 defectCorrectionKernel[DEFECT_CORRECTION_KERNEL_SIZE * DEFECT_CORRECTION_KERNEL_SIZE];

__global__ static void averageNeighboursKernel(u16* input, const u16* defectMap, int width, int height, int kernelSize) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int kernelRadius = DEFECT_CORRECTION_KERNEL_SIZE / 2;

	if (x >= width || y >= height) return;

	int count = 0;
	int sum = 0;

	for (int dy = -kernelRadius; dy <= kernelRadius; dy++) {
		for (int dx = -kernelRadius; dx <= kernelRadius; dx++) {
			int nx = x + dx;
			int ny = y + dy;
			if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
				int idx = (dy + kernelRadius) * kernelSize + (dx + kernelRadius);
				int defectMapIdx = ny * width + nx;
				sum += input[ny * width + nx] * defectCorrectionKernel[idx] * (1 - defectMap[defectMapIdx]);
				count += defectCorrectionKernel[idx] * (1 - defectMap[defectMapIdx]);
				//if (!defectMap[ny * width + nx]) {
				//	sum += input[ny * width + nx] * defectCorrectionKernel[idx];
				//	count += defectCorrectionKernel[idx];
				//}
			}
		}
	}

	if (count > 0)
		input[y * width + x] = sum / count;
}

DefectCorrection::DefectCorrection(Config config, thrust::device_vector<u16> defectMap)
	: defectMap(defectMap), config(config) {
	std::vector<u16> kernelTemp = {
		1, 1, 1,
		1, 0, 1,
		1, 1, 1
	};

	cudaMemcpyToSymbol(defectCorrectionKernel, kernelTemp.data(), kernelTemp.size() * sizeof(u16));
}

void DefectCorrection::run(thrust::device_vector<u16>& input, cudaStream_t stream) {
	dim3 blockSize(16, 16);
	dim3 gridSize((config.imageWidth + blockSize.x - 1) / blockSize.x,
		(config.imageHeight + blockSize.y - 1) / blockSize.y);

	u16* rawInputData = thrust::raw_pointer_cast(input.data());
	u16* rawDefectData = thrust::raw_pointer_cast(defectMap.data());

	averageNeighboursKernel << <gridSize, blockSize, 0, stream >> > (
		rawInputData,
		rawDefectData,
		config.imageWidth,
		config.imageHeight,
		3
		);
}

constexpr u16 HISTOGRAM_EQ_RANGE = 256;

HistogramEquilisation::HistogramEquilisation(Config config, int numBins)
	: config(config), numBins(numBins), histogram(numBins), normedHistogram(numBins), LUT(numBins) {
	cub::DeviceHistogram::HistogramEven(
		tempStorage, tempStorageBytes,
		static_cast<unsigned short*>(nullptr), thrust::raw_pointer_cast(histogram.data()), numBins,
		0, numBins - 1, config.imageHeight * config.imageWidth
	);

	cudaMalloc(&tempStorage, tempStorageBytes);
}

void HistogramEquilisation::run(thrust::device_vector<u16>& input, cudaStream_t stream) {
	int totalPixels = config.imageHeight * config.imageWidth;
	// TODO: Bug where the sum of the histogram doesn't equal number of pixels
	cub::DeviceHistogram::HistogramEven(
		tempStorage, tempStorageBytes,
		thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(histogram.data()), numBins,
		0, numBins - 1, totalPixels);

	thrust::inclusive_scan(histogram.begin(), histogram.end(), histogram.begin());

	thrust::transform(histogram.begin(), histogram.end(), normedHistogram.begin(),
		[totalPixels] __device__(unsigned int x) -> float {
		return static_cast<float>(x) / totalPixels;
	});

	thrust::transform(normedHistogram.begin(), normedHistogram.end(), LUT.begin(),
		[] __device__(float x) -> u16 {
		return static_cast<u16>(HISTOGRAM_EQ_RANGE * x);
	});

	thrust::gather(input.begin(), input.end(), LUT.begin(), input.begin());
}

HistogramEquilisation::~HistogramEquilisation() {
	if (tempStorage)
		cudaFree(tempStorage);
}