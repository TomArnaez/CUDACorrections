#include <iostream>
#include <fstream>

#include <SLImage.h>
#include <OpticalFlow.hpp>
#include <Types.hpp>
#include <Acquistion.hpp>
#include <Core.hpp>
#include <external/OpticalFlow/NvOFDataLoader.h>

constexpr std::string RESULT_IMAGE_DIR = "Result Images\\";
constexpr std::string DEMO_IMAGE_DIR = "Demo Images\\";

constexpr int HISTOGRAM_BINS = 16384;

using namespace SpectrumLogic;

thrust::host_vector<u16> SLImageToThrust(SLImage& image, size_t frame = 0) {
	size_t numPixels = image.GetWidth() * image.GetHeight();
	thrust::host_vector<u16> vec(numPixels);
	std::memcpy(thrust::raw_pointer_cast(vec.data()), image.GetDataPointer(frame), numPixels * sizeof(ushort));
	return vec;
}

SLImage SaveThrust(const thrust::host_vector<u16>& vec, Config config, std::string filename) {
	SLImage image(config.imageWidth, config.imageHeight);
	std::memcpy(image.GetDataPointer(0), thrust::raw_pointer_cast(vec.data()), config.imageByteSize);
	SLImage::WriteTiffImage(filename, image, 16);
	return image;
}

template<typename T>
void saveAsBinary(const thrust::host_vector<T> vec, Config config, std::string filename) {
	size_t dataSize = config.imageHeight * config.imageWidth * sizeof(T);

	// Open the file in binary mode
	std::ofstream outFile(filename, std::ios::binary);
	if (!outFile) {
		std::cerr << "Failed to open file for writing." << std::endl;
		return;
	}

	// Write data to the file
	outFile.write(reinterpret_cast<const char*>(vec.data()), dataSize);
	if (!outFile)
		std::cerr << "Failed to write data to file." << std::endl;

	outFile.close();
}

#include <opencv2/imgcodecs.hpp>

int main() {
	try {
		std::string clock_video_8bit_path = DEMO_IMAGE_DIR + "clock_video_8bit.tiff";
		std::vector<cv::Mat> mats;
		if (cv::imreadmulti(clock_video_8bit_path, mats)) {
			std::cout << "Yay" << std::endl;
		}

		//SLImage clockTiff(DEMO_IMAGE_DIR + "clock_video.tif");

		//auto frame1 = SLImageToThrust(clockTiff, 0);
		//auto frame2 = SLImageToThrust(clockTiff, 1);

		//SLImage input(DEMO_IMAGE_DIR + "AVG_PCB_2802_2400.tif");
		//SLImage dark(DEMO_IMAGE_DIR + "AVG_Dark_2802_2400.tif");
		//SLImage gain(DEMO_IMAGE_DIR + "AVG_Gain_2802_2400.tif");
		//SLImage defect(DEMO_IMAGE_DIR + "DefectMap.tif");

		//Config config = {
		//	.imageWidth = clockTiff.GetWidth(),
		//	.imageHeight = clockTiff.GetHeight(),
		//	.imageByteSize = clockTiff.GetWidth() * clockTiff.GetHeight() * sizeof(ushort)
		//};

		//OpticalFlowModule module(config);
		//auto [fx, fy] = module.run(frame1, frame2);
		//saveAsBinary(fx, config, RESULT_IMAGE_DIR + "fx.bin");
		//saveAsBinary(fy, config, RESULT_IMAGE_DIR + "fy.bin");

		//thrust::host_vector<u16> PCBInput = SLImageToThrust(input);
		//thrust::host_vector<u16> PCBDark = SLImageToThrust(dark);
		//thrust::host_vector<u16> PCBGain = SLImageToThrust(gain);
		//thrust::host_vector<u16> PCBDefect = SLImageToThrust(defect);

		//ImageAcquirer* acquirer = new ImageAcquirer(DEMO_IMAGE_DIR + "clock_video.tif");
		//Core<ImageAcquirer> core(config, PCBInput, acquirer);
		//core.startCapture();
		//core.addCorrection(new DarkCorrection(PCBDark, 300));
		//core.addCorrection(new GainCorrection(PCBGain));
		//core.addCorrection(new DefectCorrection(config, PCBDefect));
		//core.addCorrection(new HistogramEquilization(config, HISTOGRAM_BINS));
		//auto res = core.execute();
		//SaveThrust(res, config, RESULT_IMAGE_DIR + "Result.tiff");
		//res = core.execute();
		//res = core.execute();
		//res = core.execute();

		while (true) {

		}
	}
	catch (std::exception ex) {
		std::cout << "Got exception: " << ex.what() << std::endl;
	}
}