//#include <OpticalFlow.hpp>
//
//class OpticalFlowLocal : public IOpticalFlow {
//	Config config;
//	thrust::device_vector<float> fx;
//	thrust::device_vector<float> fy;
//	thrust::device_vector<float> ft;
//public:
//	OpticalFlowLocal(Config config);
//	~OpticalFlowLocal();
//	void calculateFlowVectors(const thrust::device_vector<u16>& input1, const thrust::device_vector<u16>& input2) override;
//};