#pragma once

#include <thrust/device_ptr.h>

#include "ppm.hpp"
#include "blur.hpp"
#include "add_weighted.hpp"

using IntItr = thrust::counting_iterator< unsigned >;

using DevPtr = thrust::device_ptr< unsigned char >;

void unsharp_mask( DevPtr &in,
				   DevPtr &out,
				   DevPtr &blurBuffer,
				   const int &blur_radius, const unsigned &width,
				   const unsigned &height, const unsigned &nChannels )
{
	const auto alpha = 1.5f, beta = -0.5f, gamma = 0.0f;
	const unsigned imgSize = width * height * nChannels;

	thrust::transform( thrust::device, IntItr( 0 ), IntItr( imgSize ), blurBuffer,
					   pixel_average( blur_radius, width, height, nChannels, in ) );

	thrust::transform( thrust::device, IntItr( 0 ), IntItr( imgSize ), out,
					   pixel_average( blur_radius, width, height, nChannels, blurBuffer ) );

	thrust::transform( thrust::device, IntItr( 0 ), IntItr( imgSize ), blurBuffer,
					   pixel_average( blur_radius, width, height, nChannels, out ) );

	thrust::transform( thrust::device, in, in + imgSize, blurBuffer, out,
					   add_weighted<float>( alpha, beta, gamma ) );
}
