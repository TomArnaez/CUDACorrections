#include <driver_types.h>
#include "ErrorMacros.hpp"

#ifdef _WIN64
#include <windows.h>
#include "VersionHelpers.h"
#endif
#include <cuda.h>

typedef int LINUX_HANDLE;
typedef HANDLE WIN_HANDLE;

#ifdef _WIN64
void cudaVKImportSemaphore(WIN_HANDLE handle, cudaExternalSemaphore_t semaphore) {
	cudaExternalSemaphoreHandleDesc extSemaphoreHandleDesc;
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));

	extSemaphoreHandleDesc.type =
		IsWindows8OrGreater() ? cudaExternalSemaphoreHandleTypeOpaqueWin32
		: cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
	extSemaphoreHandleDesc.handle.win32.handle = handle;
	extSemaphoreHandleDesc.flags = 0;

	cudaErrorCheck(cudaImportExternalSemaphore(&semaphore,
		&extSemaphoreHandleDesc));
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));
}
#else
void cudaVKImportSemaphore(LINUXHANDLE handle, cudaExternalSemaphore_t extSemaphore) {
	cudaExternalSemaphoreHandleDesc extSemaphoreHandleDesc;
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));

	extSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
	extSemaphoreHandleDesc.handle.fd = handle;
	extSemaphoreHandleDesc.flags = 0;
	cudaErrorCheck(cudaImportExternalSemaphore(&semaphore,
		&extSemaphoreHandleDesc));
	memset(&extSemaphoreHandleDesc, 0, sizeof(extSemaphoreHandleDesc));
}
#endif


#ifdef _WIN64
void cudaVKImportImageMem(CUdeviceptr devPtr, WIN_HANDLE handle, cudaExternalMemory_t mem, size_t size) {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
	cudaExtMemHandleDesc.type =
		IsWindows8OrGreater() ? cudaExternalMemoryHandleTypeOpaqueWin32
		: cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	cudaExtMemHandleDesc.handle.win32.handle = handle;
	cudaExtMemHandleDesc.size = size;
	cudaErrorCheck(cudaImportExternalMemory(&mem,

		&cudaExtMemHandleDesc));

	cudaExternalMemoryBufferDesc extMemBufferDesc = {};
	extMemBufferDesc.offset = 0;
	extMemBufferDesc.size = size;
	extMemBufferDesc.flags = 0;

	// TODO: Proper casting safety
	cudaErrorCheck(cudaExternalMemoryGetMappedBuffer((void**)devPtr, mem,
		&extMemBufferDesc));
}
#else
void cudaVKImportImageMem(LINUXHANDLE handle, cudaExternalMemory_t extMemory) {
	cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
	memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
	cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	cudaExtMemHandleDesc.handle.fd = handle;
#endif

void cudaVKSemaphoreSignal(cudaExternalSemaphore_t extSemaphore, cudaStream_t stream) {
	cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
	memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));

	extSemaphoreSignalParams.params.fence.value = 0;
	extSemaphoreSignalParams.flags = 0;
	cudaErrorCheck(cudaSignalExternalSemaphoresAsync(
		&extSemaphore, &extSemaphoreSignalParams, 1, stream));
}

void cudaVKSemaphoreWait(cudaExternalSemaphore_t extSemaphore, cudaStream_t stream) {
	cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
	memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
	extSemaphoreWaitParams.params.fence.value = 0;
	extSemaphoreWaitParams.flags = 0;
	cudaErrorCheck(cudaWaitExternalSemaphoresAsync(
		&extSemaphore, &extSemaphoreWaitParams, 1, stream));
}