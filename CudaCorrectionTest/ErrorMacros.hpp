#pragma once

#include <nppdefs.h>
#include <stdio.h>

#define nppErrorCheck(ans) { nppAssert((ans), __FILE__, __LINE__); }
inline void nppAssert(NppStatus code, const char* file, int line, bool abort = true)
{
    if (code != NPP_NO_ERROR)
    {
        fprintf(stderr, "cudaAssert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}


#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}