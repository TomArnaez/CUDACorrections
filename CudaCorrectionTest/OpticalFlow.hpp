#include <concepts>
#include <thrust/device_vector.h>
#include <cstdint>

#include <Types.hpp>

template <std::integral T>
class IOpticalFlow {
public:
	virtual void calculateFlowVectors(const thrust::device_vector<T>& input1, const thrust::device_vector<T>& input2) = 0;
};