#pragma once

#include <fstream>
#include <vector>
#include <limits>
#include <cassert>
#include <cerrno>

#include <thrust/host_vector.h>

struct ppm
{
	void read( std::string filename, thrust::host_vector< unsigned char > &out )
	{
		std::ifstream file( filename, std::ios::in );

		if ( !file )
		{
			throw errno;
			return;
		}

		file >> identifier >> width >> height >> max;
		
		assert( max <= std::numeric_limits< unsigned char >::max() );

		capacity = width * height * nChannels;

		out.reserve( capacity );

		while ( out.size() < capacity )
		{
			unsigned value;

			file >> value;

			out.push_back( value );
		}

		file.close();
	}

	void write( std::string filename, const thrust::host_vector< unsigned char > &data )
	{
		std::ofstream file( filename, std::ios::out);

		file << identifier << '\n' << width << ' ' << height << '\n' << max << '\n';

		unsigned count = 0;

		for ( unsigned value : data )
		{
			file << value
				 << ( ++count == ( width * nChannels ) ? ( count = 0, '\n' ) : ' ' );
		}

		file.close();
	}

	std::string identifier;
	std::size_t capacity, width, height, max;
	const unsigned nChannels = 3;  // e.g. RGB; RGBA has 4 channels
};
