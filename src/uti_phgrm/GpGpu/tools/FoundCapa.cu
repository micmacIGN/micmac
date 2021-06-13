#include <stdio.h>


int main()
{
    int NbDevice = 0;

    if (cudaSuccess != cudaGetDeviceCount(&NbDevice))

        return -1;


    if (!NbDevice)

        return -1;


    for (int device = 0; device < NbDevice; ++device)
    {
        cudaDeviceProp propri;

        if (cudaSuccess != cudaGetDeviceProperties(&propri, device))
        {
            continue;
        }
        printf("%d.%d ", propri.major, propri.minor);
    }

    return 0;

}
