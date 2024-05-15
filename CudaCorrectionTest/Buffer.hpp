#pragma once

#include <array>
#include <functional>
#include <mutex>

#include <thrust/device_vector.h>

#include "tl/expected.hpp"


template<typename T, size_t N>
class Buffer {
private:
	using BufferStorage = std::array<thrust::device_vector<T>, N>;

	BufferStorage buffer;
	size_t size;
	size_t head = 0;
	size_t filledBuffersCount = 0;
	std::mutex mtx;
public:
	Buffer(size_t size)
	: size(size) {
		for (auto& vec : buffer)
			vec = thrust::device_vector<T>(size);
	}

	tl::expected<void, std::string> insert(T* data) {
		std::lock_guard<std::mutex> lock(mtx);

		try {
			thrust::copy(data, data + size, buffer[head].begin());

			cudaDeviceSynchronize();

			if (filledBuffersCount < N)
				filledBuffersCount++;

			head = (head + 1) % N;
		}
		catch (thrust::system_error& e) {
			return tl::make_unexpected(std::string("Thrust error: ") + e.what());
		}

		return {};
	}

	void readBuffer(std::function<void(BufferStorage&)> func) {
		std::lock_guard<std::mutex> lock(mtx);
		func(buffer);
	}

	size_t getFillerBuffersCount() const {
		return filledBuffersCount;
	}
};