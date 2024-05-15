#include <vulkan/vulkan.hpp>

class VkTestCore {
public:
    VkBuffer buffer;
    VkDeviceMemory extMemory;

	VkTestCore() {
        vk::ApplicationInfo appInfo(
            "Vulkan App", VK_MAKE_VERSION(1, 0, 0),
            "No Engine", VK_MAKE_VERSION(1, 0, 0),
            VK_API_VERSION_1_2);

        vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo);
        vk::UniqueInstance instance = vk::createInstanceUnique(instanceCreateInfo);

        std::vector<vk::PhysicalDevice> physicalDevices = instance->enumeratePhysicalDevices();
        if (physicalDevices.empty()) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }
        vk::PhysicalDevice physicalDevice = physicalDevices[0];
	}
};

void createExternalBuffer(
    vk::Device device,
    vk::PhysicalDevice physicalDevice,
    vk::Buffer buffer,
    vk::DeviceMemory memory
) {
    vk::BufferCreateInfo bufferCreateInfo(
        {},
        1024,
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive
    );
    buffer = device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    vk::ExportMemoryAllocateInfoKHR exportAllocInfo(
        vk::ExternalMemoryHandleTypeFlagBitsKHR::eOpaqueWin32
    );
    vk::MemoryAllocateInfo allocInfo(
        memRequirements.size,
        findMemoryType(physicalDevice, memRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eDeviceLocal)
    );
    allocInfo.pNext = &exportAllocInfo;

    memory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(buffer, memory, 0);
}


uint32_t findMemoryType(vk::PhysicalDevice physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
}