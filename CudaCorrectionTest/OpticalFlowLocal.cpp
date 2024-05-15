//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//#include <thrust/host_vector.h>
//
//#include <OpticalFlowLocal.hpp>
//#include <Types.hpp>
//
//constexpr size_t PREWITT_MASK_SIZE = 3;
//__constant__ float Gx[PREWITT_MASK_SIZE][PREWITT_MASK_SIZE];
//__constant__ float Gy[PREWITT_MASK_SIZE][PREWITT_MASK_SIZE];
//
///**
// * Calculate flow using (A^T A)^{-1} A^T b.
// * @param AtA_00 The upper left entry of A^T A, corresponding to sum of (fx)^2
// * @param AtA_01 The upper right/bottom left entry of A^T A, corresponding to sum of fx * fy
// * @param AtA_11 The bottom right entry of A^T A, corresponding to sum of (fy)^2
// * @param Atb_0 The top entry of A^T b, corresponding to sum of fx * ft
// * @param Atb_1 The bottom entry of A^T b, corresponding to sum of fy * ft
// * @return A float2 containing the flow
// */
////inline __host__ __device__ float2 calcFlowFromMatrix(
////	float AtA_00, float AtA_01, float AtA_11, float Atb_0, float Atb_1
////) {
////	// Calculate the determinant and make sure it's not too small to be invertible
////	float det = AtA_00 * AtA_11 - AtA_01 * AtA_01;
////	if (abs(det) <= 1.5e-5)
////		return make_float2(0.0f, 0.0f);
////
////	return make_float2(AtA_11 * Atb_0 - AtA_01 * Atb_1, -AtA_01 * Atb_0 + AtA_00 * Atb_1) / det;
////}
//
///**
// * Calculate the spatial derivative fx and fy and the temporal derivative ft
// * from two frames using the Prewitt operator
// *
// * @param input1 Frame 1
// * @param input2 Frame 2
// * @param fx Horizontal spatial derivative
// * @param fy Vertical spatial derivative
// * @param ft Temporal derivative
// * @param maskSize Size of prewitt mask operator
// * @param height, width Height and width of frame
// */
//__global__
//void calculateDerivatives(
//	const u16* input1, const u16* input2,
//	float* fx, float* fy, float* ft,
//	int maskSize, int width, int height
//) {
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//	int halfMask = maskSize / 2;
//	int idx = y * width + x;
//
//	if (x >= halfMask && x < (width - halfMask) && y >= halfMask && y < (height - halfMask)) {
//		float px = 0.0, py = 0.0;
//		for (int i = -halfMask; i <= halfMask; i++) {
//			for (int j = -halfMask; j <= halfMask; j++) {
//				u16 val = input1[(y + i) * width + (x + j)];
//				px += Gx[i + halfMask][j + halfMask] * val;
//				py += Gy[i + halfMask][j + halfMask] * val;
//			}
//		}
//		fx[idx] = px;
//		fy[idx] = py;
//		ft[idx] = input2[idx] - input1[idx];
//	}
//}
//
///**
//* A simple kernel for performing the Lucas Kanade method with no tiling
//*/
//__global__
//void simpleLucasKanade(
//	const float* fx, const float* fy, const float* ft, float* angle, float* mag,
//	int width, int height, int s
//) {
//	int x = blockDim.y * blockIdx.y + threadIdx.y;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (x < height - s && y < width - s) {
//		float fxSqrdSum, fxFySum, fySqrdSum, fxFtSum, fyFtSum;
//		fxSqrdSum = fxFySum = fySqrdSum = fxFtSum = fyFtSum = 0.0f;
//		float cur_fx, cur_fy, cur_ft;
//
//		for (int i = x - s; i <= x + s; ++i) {
//			for (int j = y - s; j <= y + s; ++j) {
//				int offset = i * width + j;
//				cur_fx = fx[offset];
//				cur_fy = fy[offset];
//				cur_ft = ft[offset];
//
//				fxSqrdSum += cur_fx * cur_fx;
//				fxFySum += cur_fx * cur_fy;
//				fySqrdSum += cur_fy * cur_fy;
//				fxFtSum += cur_fx * cur_ft;
//				fyFtSum += cur_fy * cur_ft;
//			}
//		}
//	}
//}
//
//OpticalFlowLocal::OpticalFlowLocal(Config config)
//	: config(config) {
//	size_t imageSize = config.imageHeight * config.imageWidth;
//	fx.resize(imageSize, 0.0f);
//	fy.resize(imageSize, 0.0f);
//	ft.resize(imageSize, 0.0f);
//
//	thrust::host_vector<float> host_Gx = {
//		-1.0, 0.0, 1.0,
//		-1.0, 0.0, 1.0,
//		-1.0, 0.0, 1.0
//	};
//	thrust::host_vector<float> host_Gy = {
//		-1.0,  -1.0, -1.0,
//		 0.0,   0.0,  0.0,
//		 1.0,   1.0,  1.0
//	};
//
//	cudaMemcpyToSymbol(Gx, host_Gx.data(), host_Gx.size() * sizeof(float));
//	cudaMemcpyToSymbol(Gy, host_Gy.data(), host_Gy.size() * sizeof(float));
//}
//
//void OpticalFlowLocal::calculateFlowVectors(const thrust::device_vector<u16>& input1, const thrust::device_vector<u16>& input2) {
//	const u16* input1Ptr = thrust::raw_pointer_cast(input1.data());
//	const u16* input2Ptr = thrust::raw_pointer_cast(input2.data());
//	float* fxPtr = thrust::raw_pointer_cast(fx.data());
//	float* fyPtr = thrust::raw_pointer_cast(fy.data());
//	float* ftPtr = thrust::raw_pointer_cast(ft.data());
//
//	// Define block and grid sizes
//	dim3 blockDim(16, 16);
//	dim3 gridDim((config.imageWidth + blockDim.x - 1) / blockDim.x,
//		(config.imageHeight + blockDim.y - 1) / blockDim.y);
//
//	calculateDerivatives << <gridDim, blockDim >> > (
//		input1Ptr, input2Ptr, fxPtr, fyPtr, ftPtr,
//		3,
//		config.imageWidth, config.imageHeight
//		);
//
//	cudaDeviceSynchronize();
//}
//
//OpticalFlowLocal::~OpticalFlowLocal() {
//}