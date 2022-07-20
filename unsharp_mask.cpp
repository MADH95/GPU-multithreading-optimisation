#include <chrono>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "unsharp_mask.hpp"

int main( int argc, char *argv[] )
{
	using Timer = std::chrono::steady_clock;
	using Duration = std::chrono::duration<double>;

	const char *ifilename = argc > 1 ? argv[ 1 ] : "../ghost-town-8k.ppm"; // "../goldhill.ppm"
	const char *ofilename = argc > 2 ? argv[ 2 ] : "../out.ppm";
	const int blur_radius = argc > 3 ? std::atoi( argv[ 3 ] ) : 5;

	ppm img;

	thrust::host_vector< unsigned char > inData, outData;
	thrust::device_vector< unsigned char > d_inData, d_outData;

	auto readStart = Timer::now();

	//Read the data from file
	img.read( ifilename, inData );

	auto readEnd = Timer::now();
	std::cout << "Reading took: " << Duration( readEnd - readStart ).count() << " seconds.\n";


	auto hostAllocationStart = Timer::now();

	//Resize device vectors and allocate memory for the blurBuffer pointer
	outData.resize( img.capacity );

	auto hostAllocationEnd = Timer::now();
	std::cout << "Host Memory Allocation took: " << Duration( hostAllocationEnd - hostAllocationStart ).count() << " seconds.\n";


	auto deviceAllocationStart = Timer::now();

	d_inData.resize( img.capacity );
	d_outData.resize( img.capacity );

	DevPtr blurBuffer = thrust::device_malloc( img.capacity * sizeof( unsigned char ) );

	cudaDeviceSynchronize();

	auto deviceAllocationEnd = Timer::now();
	std::cout << "Device Memory Allocation took: " << Duration( deviceAllocationEnd - deviceAllocationStart ).count() << " seconds.\n";


	auto sharpenStart = Timer::now();

	//copy the data from the host to the device
	thrust::copy( std::begin( inData ), std::end( inData ), std::begin( d_inData ) );

	//Sharpen the image
	unsharp_mask( d_inData.data(), d_outData.data(), blurBuffer, blur_radius, img.width, img.height, img.nChannels );

	//copy the sharpened image data from device to the host
	thrust::copy( std::begin( d_outData ), std::end( d_outData ), std::begin( outData ) );

	//Make sure the thrust kernals have all run for accurate timing (not required, thrust sychronises when data is copied in the above line)
	//cudaDeviceSynchronize();

	auto sharpenEnd = Timer::now();
	std::cout << "Sharpening took: " << Duration(sharpenEnd - sharpenStart ).count() << " seconds.\n";


	//Write data back to file
	auto writeStart = Timer::now();

	img.write( ofilename, outData );

	auto writeEnd = Timer::now();
	std::cout << "Writing took: " << Duration( writeEnd - writeStart ).count() << " seconds.\n";

	thrust::device_free( blurBuffer );

	return 0;
}

