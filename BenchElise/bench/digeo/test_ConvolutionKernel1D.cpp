#include "ConvolutionKernel1D.h"

#include "base_types.h"

#include <iostream>

using namespace std;

int main( int argc, char **argv )
{
	INT ak0[] = { 1, 2, 3, 2, 1 };
	INT ak1[] = { 0, 0, 1, 2, 3, 2, 1, 0, 0 };
	INT ak2[] = { 0, 0, 0, 0, 0 };
	INT ak3[] = { 0, 1, 2 };
	ConvolutionKernel1D<INT> kernel( ak0, 5, 2, 0 ); // 5 = size, 2 = iCenter, 0 = nbShift
	kernel.dump();
	kernel.set( ak1, 7, 4, 0 );
	kernel.dump();
	kernel.set( ak2, 5, 4, 0 );
	kernel.dump();
	kernel.set( ak3, 3, 1, 0 );
	kernel.dump();

	return EXIT_SUCCESS;
}
