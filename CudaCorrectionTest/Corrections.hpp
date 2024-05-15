#pragma once

#include <tl/expected.hpp>

// CUDA Headers
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "ErrorMacros.hpp"
#include "Types.hpp"

class ICorrection {
public:
	virtual ~ICorrection() {}
	virtual void run(thrust::device_vector<u16>& input, cudaStream_t stream) = 0;
};

class DarkCorrection: public ICorrection {
private:
	thrust::device_vector<u16> darkMap;
	u16 offset;
public:
	void run(thrust::device_vector<u16>& input, cudaStream_t stream) override;
	DarkCorrection(thrust::host_vector<u16>& darkMap, u16 offset);
};

class GainCorrection : public ICorrection {
private:
	thrust::device_vector<double> normedGainMap;
public:
	GainCorrection(thrust::device_vector<u16> gainMap);
	void run(thrust::device_vector<u16>& input, cudaStream_t stream) override;
	void normaliseGainMap(thrust::device_vector<u16> gainMap);
};

class DefectCorrection : public ICorrection {
private:
	thrust::device_vector<u16> defectMap;
	Config config;
public:
	DefectCorrection(Config config, thrust::device_vector<u16> defectMap);
	void run(thrust::device_vector<u16>& input, cudaStream_t stream) override;
};

class HistogramEquilisation : public ICorrection {
private:
	Config config;
	int numBins;
	void* tempStorage = nullptr;
	thrust::device_vector<int> histogram;
	thrust::device_vector<float> normedHistogram;
	thrust::device_vector<u16> LUT;
	size_t tempStorageBytes = 0;
public:
	HistogramEquilisation(Config config, int numBins);
	~HistogramEquilisation();
	void run(thrust::device_vector<u16>& input, cudaStream_t stream) override;
};