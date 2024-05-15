#pragma once

#include <concepts>
#include <functional> 
#include <variant>  

#include <tl/expected.hpp>
#include <SLDevice.h>

using namespace SpectrumLogic;

using ImageCallback = std::function<void(std::variant<ushort*, SLError>)>;

template<typename T>
concept Detector = requires(T a, int exposureTime, ImageCallback imageCallback) {
    { a.startAcquisition(exposureTime, imageCallback) } -> std::same_as<tl::expected<void, SLError>>;
    { a.stopAcquisition() } -> std::same_as<tl::expected<void, SLError>>;
};

void acquisitionCallback(ushort* buffer, SLBufferInfo bufferInfo, void* userArgs) {
    auto* func = static_cast<ImageCallback*>(userArgs);
    bufferInfo.error == SLError::SL_ERROR_SUCCESS ? (*func)(buffer) : (*func)(bufferInfo.error);
}

class SDKDetector {
private:
    SLDevice device;

    template<typename Func, typename... Args>
    tl::expected<void, SLError> callDeviceErrFunction(Func func, Args&&... args) {
        SLError err = std::invoke(func, &device, std::forward<Args>(args)...);
        return err == SLError::SL_ERROR_SUCCESS ? tl::expected<void, SLError>{} : tl::unexpected(err);
    }

public:
    SDKDetector(DeviceInterface deviceInterface)
        : device(SLDevice(deviceInterface)) {
        auto result = callDeviceErrFunction(&SLDevice::OpenCamera, 100);
        if (!result.has_value())
            throw std::runtime_error("Failed to open camera with error: " + SLErrorToString(result.error()));
    }

    tl::expected<void, SLError> startAcquisition(int exposureTime, ImageCallback imageCallback) {
        auto result = callDeviceErrFunction(static_cast<SLError(SLDevice::*)(int)>(&SLDevice::SetExposureTime), exposureTime);
        if (!result) return result;
        return callDeviceErrFunction(static_cast<SLError(SLDevice::*)(acquisition_cb, void*)>(&SLDevice::StartStream), acquisitionCallback, reinterpret_cast<void*>(&imageCallback));
    }

    tl::expected<void, SLError> stopAcquisition() {
        return callDeviceErrFunction(&SLDevice::StopStream);
    }
};

class ImageAcquirer {
    SLImage image;
    int exposureTime;
    std::thread acquisitionThread;
    bool isRunning;

    void ImageThreadProc(ImageCallback imageCallback) {
        int frameCount = 0;
        isRunning = true;
        while (isRunning && frameCount < image.GetDepth()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(exposureTime));
            imageCallback(image.GetDataPointer(0));
        }
    }

public:
    ImageAcquirer(std::string filename)
    : image(filename) {
    }

    tl::expected<void, SLError> startAcquisition(int exposureTime, ImageCallback imageCallback) {
        this->exposureTime = exposureTime;
        try {
            acquisitionThread = std::thread(&ImageAcquirer::ImageThreadProc, this, imageCallback);
        }
        catch (const std::system_error& e) {
            return tl::make_unexpected(SLError::SL_ERROR_CRITICAL);
        }

        return {};
    }

    tl::expected<void, SLError> stopAcquisition() {
        if (isRunning) {
            isRunning = false;
            if (acquisitionThread.joinable())
                acquisitionThread.join();
        }
        return {};
    }

    ~ImageAcquirer() {
        if (isRunning) { 
            isRunning = false;
            if (acquisitionThread.joinable())
                acquisitionThread.join();
        }
    }
};