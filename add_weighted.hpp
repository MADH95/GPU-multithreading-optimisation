#pragma once

#include <thrust/functional.h>

template< typename T >
struct add_weighted : thrust::binary_function< unsigned char, unsigned char, unsigned char >
{
	const T alpha, beta, gamma;
	const thrust::minimum< int > min{};
	const thrust::maximum< int > max{};
	const unsigned char uchar_max = std::numeric_limits<unsigned char>::max();

	add_weighted(const T &alpha, const T &beta, const T &gamma )
		: alpha( alpha ), beta( beta ), gamma( gamma )
	{}

	__device__
	unsigned char operator()( const unsigned char &in1, const unsigned char &in2 )
	{
		T tmp = in1 * alpha + in2 * beta + gamma;
		return min( uchar_max, max( 0, tmp ) );
	}
};
