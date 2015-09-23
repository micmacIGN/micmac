#ifndef __DIAMOND_SQUARE__
#define __DIAMOND_SQUARE__

#include "debug.h"
#include "TypeTraits.h"

#include <algorithm>
#include <iostream>
#include <list>
#include <cmath>

class Pt2di
{
public:
	int x, y;
	Pt2di( int aX=0, int aY=0 ):x(aX),y(aY){}
	Pt2di operator +( const Pt2di &b ){ return Pt2di(x+b.x,y+b.y); }
	Pt2di operator -( const Pt2di &b ){ return Pt2di(x-b.x,y-b.y); }
	Pt2di operator /( int b ){ return Pt2di(x/b,y/b); }
};

class DiamondSquareRectangle
{
public:
	Pt2di p0, p1;
	U_INT2 v0, v1, v2, v3;

	DiamondSquareRectangle( const Pt2di &i_p0, const Pt2di &i_p1, INT i_v0, INT i_v1, INT i_v2, INT i_v3 ):
		p0(i_p0), p1(i_p1),
		v0(i_v0), v1(i_v1), v2(i_v2), v3(i_v3){}
};

INT add_error( INT i_v, double i_dimensionFactor )
{
	const int maxErrorPercentage_1 = 71; // actually max error percentage +1
	const INT dataMax = 65535;
	const double scale = i_dimensionFactor*( ( (double)dataMax )/100. );
	const double ek = 0.;

	INT e = (INT)round( ( (double)(rand()%maxErrorPercentage_1) )*scale + ek );
	if ( rand()%2==0 ) e = -e;

	INT res = i_v+e;
	if ( res<0 ) res = 0;
	if ( res>dataMax ) res = dataMax;

	return res;
}

template <class tDataSrc, class tDataDst>
void ramp( const tDataSrc *i_src, int i_width, int i_height, const tDataSrc i_srcMin, const tDataSrc i_srcMax, const tDataDst i_dstMin, const tDataDst i_dstMax, tDataDst *o_dst )
{
	//~ typedef typename TypeTraits<tDataSrc>::tBase tBaseSrc;
	typedef typename TypeTraits<tDataDst>::tBase tBaseDst;

	const double srcMin = (double)i_srcMin;
	const double dstMin = (double)i_dstMin;
	const double dstMax = (double)i_dstMax;
	const double scale = ( dstMax-dstMin )/( (double)i_srcMax-srcMin );
	int iPix = i_width*i_height;

	while ( iPix-- )
	{
		tBaseDst v = ( ( (double)(*i_src++) )-srcMin )*scale+dstMin;
		if ( v<i_dstMin ) v = i_dstMin;
		if ( v>i_dstMax ) v = i_dstMax;
		*o_dst++ = (tDataDst)v;
	}
}

template <class tData> 
static tData ** __new_data_lines( int i_width, int i_height )
{
	tData *data = new tData[i_width*i_height];
	tData **lines = new tData*[i_height];
	for ( int y=0; y<i_height; y++ )
	{
		lines[y] = data;
		data += i_width;
	}
	return lines;
}

// i_data must have at least one line
template <class tData> 
static void __delete_data_lines( tData **i_data )
{
	delete [] i_data[0];
	delete [] i_data;
}

template <class tData>
void diamond_square( tData **i_data, int i_width, int i_height, U_INT2 i_v0, U_INT2 i_v1, U_INT2 i_v2, U_INT2 i_v3 )
{
	if ( i_width==0 && i_height==0 ) return;

	U_INT2 **tmp = __new_data_lines<U_INT2>(i_width,i_height);
	diamond_square( tmp, i_width, i_height, i_v0, i_v1, i_v2, i_v3 );
	ramp<U_INT2,tData>( tmp[0], i_width, i_height, (U_INT2)0, (U_INT2)65535, TypeTraits<tData>::nmin(), TypeTraits<tData>::nmax(), i_data[0] );
	__delete_data_lines<U_INT2>(tmp);
}

template <> void diamond_square( U_INT2 **i_data, int i_width, int i_height, U_INT2 i_v0, U_INT2 i_v1, U_INT2 i_v2, U_INT2 i_v3 )
{
	std::list<DiamondSquareRectangle> rectangles;
	rectangles.push_back( DiamondSquareRectangle( Pt2di(0,0), Pt2di(i_width-1,i_height-1), i_v0, i_v1, i_v2, i_v3 ) );

	const double maxDimension = (double)std::min<int>( i_width, i_height );
	while ( rectangles.begin()!=rectangles.end() )
	{
		DiamondSquareRectangle rect = rectangles.front();
		rectangles.pop_front();

		#ifdef __DEBUG
			const INT vmax = 65535;
		#endif
		ELISE_DEBUG_ERROR( rect.v0<0 || rect.v0>vmax, "diamond_square", "rect.v0 = " << rect.v0 << " vmax = " << vmax );
		ELISE_DEBUG_ERROR( rect.v1<0 || rect.v1>vmax, "diamond_square", "rect.v1 = " << rect.v1 << " vmax = " << vmax );
		ELISE_DEBUG_ERROR( rect.v2<0 || rect.v2>vmax, "diamond_square", "rect.v2 = " << rect.v2 << " vmax = " << vmax );
		ELISE_DEBUG_ERROR( rect.v3<0 || rect.v3>vmax, "diamond_square", "rect.v3 = " << rect.v3 << " vmax = " << vmax );

		i_data[rect.p0.y][rect.p0.x] = (U_INT2)rect.v0;
		i_data[rect.p0.y][rect.p1.x] = (U_INT2)rect.v1;
		i_data[rect.p1.y][rect.p0.x] = (U_INT2)rect.v2;
		i_data[rect.p1.y][rect.p1.x] = (U_INT2)rect.v3;

		Pt2di rectSize = rect.p1-rect.p0+Pt2di(1,1);
		const double dimensionFactor = ( (double)std::min<int>( rectSize.x, rectSize.y ) )/maxDimension;
		INT iv0 = add_error( ( rect.v0+rect.v1 )/2, 0 ),
		    iv1 = add_error( ( rect.v0+rect.v2 )/2, 0 ),
		    iv2 = add_error( ( rect.v1+rect.v3 )/2, 0 ),
		    iv3 = add_error( ( rect.v2+rect.v3 )/2, 0 ),
		    iv4 = add_error( ( rect.v0+rect.v1+rect.v2+rect.v3 )/4, dimensionFactor );
		Pt2di ip = ( rect.p0+rect.p1 )/2;
		if ( rectSize.x>2 || rectSize.y>2 )
		{
			rectangles.push_back( DiamondSquareRectangle( rect.p0, ip, rect.v0, iv0, iv1, iv4 ) );
			rectangles.push_back( DiamondSquareRectangle( Pt2di(ip.x,rect.p0.y), Pt2di(rect.p1.x,ip.y), iv0, rect.v1, iv4, iv2 ) );
			rectangles.push_back( DiamondSquareRectangle( Pt2di(rect.p0.x,ip.y), Pt2di(ip.x,rect.p1.y), iv1, iv4, rect.v2, iv3 ) );
			rectangles.push_back( DiamondSquareRectangle( ip, rect.p1, iv4, iv2, iv3, rect.v3 ) );
		}
	}
}

#endif
