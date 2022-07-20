#pragma once

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

using IntItr = thrust::counting_iterator< unsigned >;

using DevPtr = thrust::device_ptr< unsigned char >;

struct pixel_sum : thrust::unary_function< int, unsigned >
{
	const int width, height, channel, x_min, y_min;
	const unsigned nChannels, rad_range;
	DevPtr pixels;
	const thrust::minimum< int > min{};
	const thrust::maximum< int > max{};

	__device__
	pixel_sum( int width, int height, int channel, unsigned nChannels, int x_min, int y_min, unsigned rad_range, DevPtr pixels )
		: width( width ), height( height ), channel( channel ), nChannels( nChannels ),
		x_min( x_min ), y_min( y_min ), rad_range( rad_range ), pixels( pixels )
	{}

	__device__
	unsigned operator()( int index )
	{
		int x = x_min + ( index % rad_range );
		int y = y_min + ( index / rad_range );

		const unsigned _x = min( width, max( 0, x ) );
		const unsigned _y = min( height, max( 0, y ) );

		unsigned byte_offset = ( _y * width + _x ) * nChannels + channel;

		return pixels[ byte_offset ];
	}
};


struct pixel_average : thrust::unary_function< unsigned, unsigned char >
{
	const int blur_radius, width, height;
	const unsigned nChannels, nSamples;
	DevPtr pixels;
	thrust::minimum< int > min{};
	thrust::maximum< int > max{};

	
	pixel_average( const int &blur_radius, const unsigned &width, const unsigned &height, const unsigned &nChannels, DevPtr &in )
		: blur_radius( blur_radius ), width( width ), height( height ), nChannels( nChannels ), pixels( in ),
		nSamples( ( blur_radius * 2 - 1 ) * ( blur_radius * 2 - 1 ) )
	{}

	__device__
	unsigned char operator()( const unsigned &index )
	{
		int channel = index % nChannels;
		int x = ( index / nChannels ) % width;
		int y = index / ( nChannels * width );

		unsigned total = 0;

		//total = parallel_pixel_sum( x, y, channel);

		for ( int _y = y - blur_radius + 1; _y < y + blur_radius; ++_y )
		{
			for ( int _x = x - blur_radius + 1; _x < x + blur_radius; ++_x )
			{
				const unsigned rad_x = min( width, max( 0, _x ) );
				const unsigned rad_y = min( height, max( 0, _y ) );

				unsigned byte_offset = ( ( rad_y * width + rad_x ) * nChannels ) + channel;

				total += pixels[ byte_offset ];
			}
		}

		return total / nSamples;
	}

	__device__
		unsigned parallel_pixel_sum( int x, int y, int channel )
	{
		pixel_sum pix_sum( width, height, channel, nChannels, ( x - blur_radius + 1 ), ( y - blur_radius + 1 ), ( blur_radius * 2 - 1 ), pixels );

		return thrust::transform_reduce( thrust::device,
										 IntItr( 0 ),
										 IntItr( nSamples ),
										 pix_sum, 0,
										 thrust::plus< unsigned >() );
	}
};
