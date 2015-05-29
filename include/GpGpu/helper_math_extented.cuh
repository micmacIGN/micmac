#pragma once

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_math.h>


#ifdef __GNUC__
#define SUPPRESS_NOT_USED_WARN __attribute__ ((unused))
#else
#define SUPPRESS_NOT_USED_WARN
#endif

#define NEGARECT Rect(-1,-1,-1,-1)
#define MAXIRECT Rect((int)1e7,(int)1e7,(int)-1e7,(int)-1e7)


inline __host__ __device__ int2 inc( int2 &a)
{
    a.x++;
    a.y++;

    return a;
}

inline __host__ __device__ uint2 make_uint2(ushort2 a)
{
    return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ int2 make_int2(ushort2 a)
{
    return make_int2((int)a.x,(int)a.y);
}

inline __host__ __device__ ushort2 make_ushort2(uint3 a)
{
    return make_ushort2((ushort)a.x,(ushort)a.y);
}

inline __host__ __device__ ushort2 make_ushort2(uint2 a)
{
    return make_ushort2((ushort)a.x,(ushort)a.y);
}

inline __host__ __device__ short2 make_short2(ushort2 a)
{
    return make_short2((short)a.x,(short)a.y);
}


/// \struct Rect
/// \brief Cette structure represente un rectangle definie par deux points
struct Rect
{
    /// \brief point haut et gauche du rectangle
	int2 pt0;
    /// \brief point bas et droite du rectangle
	int2 pt1;

    __device__ __host__ Rect(){}

    /// \brief Definie les points du rectangle avec des points entiers
    /// \param p0 : point haut et gauche du rectangle
    /// \param p1 : point bas et droite du rectangle
	Rect(int2 p0, int2 p1)
	{
		pt0 = p0;
		pt1 = p1;
    }

    /// \brief Definie les points du rectangle avec des points entiers non signes
    /// \param p0 : point haut et gauche du rectangle
    /// \param p1 : point bas et droite du rectangle
	Rect(uint2 p0, uint2 p1)
	{

		pt0 = make_int2(p0);
		pt1 = make_int2(p1);
    }

    /// \brief Definie les points du rectangle avec des 4 entiers, fonction host et device
    /// \param p0x : abscisse du point haut et gauche du rectangle
    /// \param p0y : ordonnee du point haut et gauche du rectangle
    /// \param p1x : abscisse du point bas et droite du rectangle
    /// \param p1y : ordonnee du point bas et droite du rectangle
	__device__ __host__ Rect(int p0x,int p0y,int p1x,int p1y)
	{

		pt0 = make_int2(p0x,p0y);
		pt1 = make_int2(p1x,p1y);
    }

	__device__ __host__
	///
	/// \brief Rect operateur de copie
	/// \param rect rectangle à copier
	///
	Rect(const Rect& rect)
    {
        pt0 = rect.pt0;
        pt1 = rect.pt1;
    }

    /// \brief Renvoie la dimension du rectangle
	const uint2 dimension()  const
	{
        return make_uint2(abs(pt1-pt0));
    }

	///
	/// \brief operator ==
	/// \param other rectangle à comparer
	/// \return vraie si le rectangle est identique
	///
    bool operator==(const Rect &other) const {
        return ( this->pt0.x == other.pt0.x && this->pt0.y == other.pt0.y && this->pt1.x == other.pt1.x && this->pt1.y == other.pt1.y);
    }

	///
	/// \brief operator !=
	/// \param other rectangle à comparer
	/// \return vraie si le rectangle est différent
	///
    bool operator!=(const Rect &other) const {
        return !(*this == other);
    }

	///
	/// \brief erode Erode le rectangle par le paramètre a
	/// \param a
	/// \return
	///
    Rect erode(int a)
    {
        pt0 = pt0 + a;
        pt1 = pt1 - a;

        return *this;
    }

	///
	/// \brief SetMaxMin si le point de coordonnées x y est en dehors du rectangle, le rectangle est ajusté pour contenir ce point
	/// \param x
	/// \param y
	///
    void SetMaxMin(int x, int y)
    {

        if (x < pt0.x ) pt0.x = x;
        if (y < pt0.y ) pt0.y = y;

        if (pt1.x < x) pt1.x = x;
        if (pt1.y < y) pt1.y = y;
    }


	///
	/// \brief inside Vérifie si le point pt est à l'intérieure
	/// \param pt
	/// \return
	///
    bool inside(int2 pt)
    {
        return (pt.x>= pt0.x) && (pt.x < pt1.x ) && (pt.y>= pt0.y) && (pt.y < pt1.y);
    }

	///
	/// \brief si le rectangle rect est en dehors du rectangle, le rectangle est ajusté pour contenir ce rectangle
	/// \param rect
	///
    void SetMaxMin(Rect rect)
    {

        if (rect.pt0.x < pt0.x ) pt0.x = rect.pt0.x;
        if (rect.pt0.y < pt0.y ) pt0.y = rect.pt0.y;

        if (pt1.x < rect.pt1.x) pt1.x = rect.pt1.x;
        if (pt1.y < rect.pt1.y) pt1.y = rect.pt1.y;
    }


	///
	/// \brief SetMaxMinInc si le rectangle rect + 1 est en dehors du rectangle, le rectangle est ajusté pour contenir ce rectangle
	/// \param rect
	///
    void SetMaxMinInc(Rect rect)
    {

        inc(rect.pt1);

        if (rect.pt0.x < pt0.x ) pt0.x = rect.pt0.x;
        if (rect.pt0.y < pt0.y ) pt0.y = rect.pt0.y;

        if (pt1.x < rect.pt1.x) pt1.x = rect.pt1.x;
        if (pt1.y < rect.pt1.y) pt1.y = rect.pt1.y;
    }

	///
	/// \brief area
	/// \return la surface du rectangle
	///
    uint area()
    {
        uint2 dim =  dimension();
        return dim.x * dim.y;
    }

	///
	/// \brief operator =
	/// \param copy
	/// \return
	///
    Rect& operator=(const Rect &copy)
    {

        pt0 = copy.pt0;
        pt1 = copy.pt1;

        return *this;
    }

	__device__ __host__
	///
	/// \brief out Affichage console de membres du rectangle
	///
	void out()
	{
		printf("[(%d,%d)(%d,%d)]",pt0.x ,pt0.y,pt1.x,pt1.y);
	}

};

static int iDivUp(int a, int b)
{
    int div = a / b;
    return ((a - div * b) != 0) ? (div + 1) : (div);
    //return (a % b != 0) ? (a / b + 1) : (a / b);
}

SUPPRESS_NOT_USED_WARN static int iDivUp32(uint a)
{
    int div = a >> 5;
    return ((a - (div << 5)) != 0) ? (div + 1) : (div);
    //return (a % b != 0) ? (a / b + 1) : (a / b);
}

template<int val>
int __nBitRotation()
{
	printf("ERROR __nBitRotation no define for %d\n",val);
	return 0;
}

#define __div_mult(val,rot)												\
template<>																\
inline int __nBitRotation<val>()										\
{																		\
	return rot;															\
}																		\
template<typename T>													\
struct Bar<val, T>														\
{																		\
	inline __device__ __host__											\
	T __opDiv( T const& a) const										\
	{																	\
		return  a >> rot ;												\
	}																	\
	inline __device__ __host__											\
	T __opMult( T const& a) const										\
	{																	\
		return  a << rot ;												\
	}																	\
	inline __device__ __host__											\
	T __opMult2( T const& a) const										\
	{																	\
	   T tmp;															\
	   tmp.x = a.x >> rot;											\
	   tmp.y = a.y >> rot;											\
	   return  tmp ;													\
	}																	\
	inline __device__ __host__											\
	T _iDivUp( T const& a) const										\
	{																	\
		const T div = __opDiv(a);										\
		return ((a - (__opMult(div))) != 0) ? (div + 1) : (div);		\
	}																	\
	inline __device__ __host__											\
	T _modulo( T const& a) const										\
	{																	\
		const T div = __opDiv(a);										\
		return a - __opMult(div);										\
	}																	\
};

///
/// \cond
///
template<int val, typename T>
struct Bar
{
 inline __device__ __host__
 T __opDiv( T const& a) const
 {
	return a/val ;
 }
 inline __device__ __host__
 T __opMult( T const& a) const
 {
	return  a*val ;
 }

 inline __device__ __host__
 T __opMult2( T const& a) const
 {
	T tmp;
	tmp.x = val*a.x;
	tmp.y = val*a.y;
	return  tmp ;
 }

 inline __device__ __host__
 T _iDivUp( T const& a) const
 {
	 const T div = __opDiv(a);
	 return ((a - (__opMult(div))) != 0) ? (div + 1) : (div);
 }
 inline __device__ __host__
 T _modulo( T const& a) const
 {
	 const T div = __opDiv(a);
	 return a - __opMult(div);
 }
};
/// \endcond


__div_mult(1,0)
__div_mult(2,1)
__div_mult(4,2)
__div_mult(8,3)
__div_mult(16,4)
__div_mult(32,5)
__div_mult(64,6)
__div_mult(128,7)
__div_mult(256,8)
__div_mult(512,9)
__div_mult(1024,10)

namespace sgpu
{
	template<int fraction, typename T>
	inline __device__ __host__
	T __div(T const& a)
	{
		const Bar<fraction, T> b;
		return b.__opDiv(a);
	}

	template<int fraction, typename T>
	inline __device__ __host__
	T __mult(T const& a)
	{
		const Bar<fraction, T> b;
		return b.__opMult(a);
	}

	template<int fraction, typename T>
	inline __device__ __host__
	T __mult2(T const& a)
	{
		const Bar<fraction, T> b;
		return b.__opMult2(a);
	}

	template<int fraction, typename T>
	inline __device__ __host__
	T __iDivUp(T const& a)
	{
		const Bar<fraction, T> b;
		return b._iDivUp(a);
	}

	template<int fraction, typename T>
	inline __device__ __host__
	T __mod(T const& a)
	{
		const Bar<fraction, T> b;
		return b._modulo(a);
	}

	template<int fraction, typename T>
	inline __device__ __host__
	T __multipleSup(T const& a)
	{
		return sgpu::__mult<fraction>(sgpu::__iDivUp<fraction>(a));
	}
}

SUPPRESS_NOT_USED_WARN static uint2 iDivUp(uint2 a, uint b)
{
	return make_uint2(iDivUp(a.x,b),iDivUp(a.y,b));
}

SUPPRESS_NOT_USED_WARN static uint2 iDivUp(uint2 a, uint2 b)
{
	return make_uint2(iDivUp(a.x,b.x),iDivUp(a.y,b.y));
}

SUPPRESS_NOT_USED_WARN static int2 iDivUp(int2 a, uint b)
{
	return make_int2(iDivUp(a.x,b),iDivUp(a.y,b));
}

inline __device__ __host__ ushort size(ushort2 v)
{
    return v.x * v.y;
}

inline __device__ __host__ uint size(uint2 v)
{
	return v.x * v.y;
}

inline __device__ __host__ uint size(int2 v)
{
	return v.x * v.y;
}

inline __device__ __host__ uint size(uint3 v)
{
    return v.x * v.y * v.z;
}

inline __host__ __device__ uint2 operator/(uint2 a, uint2 b)
{
	return make_uint2(a.x / b.x, a.y / b.y);
}

inline __host__ __device__ short diffYX(short2 a)
{
    return a.y  - a.x;
}

inline __host__ __device__ uint2 operator/(uint2 a, int b)
{
	return make_uint2(a.x / b, a.y / b);
}

inline __host__ __device__ int2 operator*(int a, ushort2 b)
{
	return a * make_int2(b);
}

inline __host__ __device__ ushort2 operator*(ushort2 a, int b )
{
    return make_ushort2(a.x * b, a.y * b);
}

inline __host__ __device__ int2 operator/(int2 a, uint b)
{
	return make_int2(a.x / b, a.y / b);
}

inline __host__ __device__ int2 operator/(int2 a, uint2 b)
{
	return make_int2(a.x / ((int)(b.x)), a.y / ((int)(b.y)));
}

inline __host__ __device__ uint2 operator/(uint2 a, ushort2 b)
{
    return make_uint2(a.x / ((uint)(b.x)), a.y / ((uint)(b.y)));
}

inline __host__ __device__ float2 operator/(float2 a, uint2 b)
{
	return make_float2(a.x / ((float)(b.x)), a.y / ((float)(b.y)));
}

inline __host__ __device__ uint2 operator*(int2 a, uint2 b)
{
	return make_uint2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ uint2 operator*(uint2 a, ushort2 b)
{
    return make_uint2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ int2 operator+(int2 a, uint2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ uint2 operator+(uint2 a, int2 b)
{
    return make_uint2((int)a.x + b.x,(int)a.y + b.y);
}

inline __host__ __device__ uint2 operator+(uint2 a, ushort2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}


inline __host__ __device__ int2 operator+(int2 a, short2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ int2 operator+(int2 a, ushort2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ uint2 operator+(uint2 a, short2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ ushort2 operator+(ushort2 a, int b)
{
    return make_ushort2(a.x + b, a.y + b);
}

inline __host__ __device__ float2 operator+(float2 a, uint2 b)
{
    return make_float2(a.x + (float)b.x, a.y + (float)b.y);
}

inline __host__ __device__ float2 operator+(float2 a, int2 b)
{
    return make_float2(a.x + (float)b.x, a.y + (float)b.y);
}

inline __host__ __device__ uint2 make_uint2(dim3 a)
{
	return make_uint2((uint3)a);
}

inline __host__ __device__ int2 make_int2(dim3 a)
{
	return make_int2(a.x,a.y);
}

inline __host__ __device__ short2 make_short2(uint3 a)
{
	return make_short2((short)a.x,(short)a.y);
}

inline __host__ __device__ short2 make_short2(float a)
{
    return make_short2((short)a,(short)a);
}


inline __host__ __device__ ushort2 make_ushort2(float a)
{
    return make_ushort2((ushort)a,(ushort)a);
}


inline __host__ __device__ int2 operator-(const uint3 a, uint2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ short2 operator-(const short2 a, ushort2 b)
{
    return make_short2(a.x - b.x, a.y - b.y);
}


inline __host__ __device__ int2 operator-(const int2 a, uint2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ short2 operator-(short2 a, uint2 b)
{
	return make_short2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ uint2 operator-(uint2 a, int2 b)
{
    return make_uint2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ uint2 operator-(const uint2 &a, ushort2 b)
{
	return make_uint2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ short2 operator+(short2 a, uint2 b)
{
	return make_short2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ int2 operator+(const uint3 a, uint2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ short2 operator+(const short2 a, ushort2 b)
{
    return make_short2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ ushort2 operator+(const ushort2 a, ushort2 b)
{
    return make_ushort2(a.x + b.x, a.y + b.y);
}


inline __host__ __device__ float2 operator+(const float2 a, short2 b)
{

    return make_float2(a.x + b.x, a.y + b.y);

}



//      Calcul specifique

inline __host__ __device__ int mdlo(uint a, uint b)
{

	if (b == 0 ) return 0;
	return ((int)a) - (((int)a )/((int)b))*((int)b);

}

inline __host__ __device__ ushort lenght(short2 a)
{

    return abs(a.y - a.x);
}

inline __host__ __device__ uint lenght(uint2 a)
{
    return (uint)abs((int)a.y - (int)a.x);
}


inline __host__ __device__ ushort count(short2 a)
{
    return a.y - a.x;
}

inline __host__ __device__ uint count(int2 a)
{
    return a.y - a.x;
}

//      Test                            --------

inline __host__ __device__ bool aSE(int2 a, int b)
{
	return ((a.x >= b) && (a.y >= b));
}


inline __host__ __device__ bool oSE(uint2 a, uint2 b)
{
	return ((a.x >= b.x) || (a.y >= b.y));
}

inline __host__ __device__ bool oSE(int2 a, uint2 b)
{
	return ((a.x >= (int)b.x) || (a.y >= (int)b.y));
}

inline __host__ __device__ bool oSE(uint3 a, uint2 b)
{
	return ((a.x >= b.x) || (a.y >= b.y));
}

inline __host__ __device__ bool oEq(float2 a, float b)
{
	return ((a.x == b) || (a.y == b));
}

inline __host__ __device__ bool oI(float2 a, float b)
{
	return ((a.x < b) || (a.y < b));
}

inline __host__ __device__ bool oEq(uint2 a, int b)
{
	return ((a.x == (uint)b) || (a.y == (uint)b));
}


inline __host__ __device__ bool aEq(int2 a, int b)
{
	return ((a.x == b) && (a.y == b));
}

inline __host__ __device__ bool aEq(uint2 a, uint2 b)
{
	return ((a.x == b.x) && (a.y == b.y));
}

inline __host__ __device__ bool aEq(uint2 a, int b)
{
	return ((a.x == (uint)b) && (a.y == (uint)b));
}

inline __host__ __device__ bool oI(uint2 a, uint2 b)
{
	return ((a.x < b.x) || (a.y < b.y));
}

inline __host__ __device__ bool oI(int2 a, int2 b)
{
    return ((a.x < b.x) || (a.y < b.y));
}


inline __host__ __device__ bool oI(int2 a, int b)
{
	return ((a.x < b) || (a.y < b));
}

inline __host__ __device__ bool aI(int2 a, int2 b)
{
	return ((a.x < b.x) && (a.y < b.y));
}

inline __host__ __device__ bool aI(int2 a, uint2 b)
{
	return ((a.x < (int)b.x) && (a.y < (int)b.y));
}

inline __host__ __device__ bool aIE(int2 a, int2 b)
{
    return ((a.x <= b.x) && (a.y <= b.y));
}

inline __host__ __device__ bool oI(uint3 a, uint2 b)
{
	return ((a.x < b.x) || (a.y < b.y));
}

inline __host__ __device__ bool oI(uint3 a, ushort2 b)
{
    return ((a.x < b.x) || (a.y < b.y));
}

//      2D to 1D                                      ------

inline __host__ __device__ uint to1D( uint2 c2D, uint2 dim)
{
	return c2D.y * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( int2 c2D, uint2 dim)
{
	return c2D.y * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( uint3 c2D, uint3 dim)
{
    return (dim.y * c2D.z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( uint3 c2D, uint2 dim)
{
    return (dim.y * c2D.z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( int3 c2D, uint2 dim)
{
    return (dim.y * c2D.z + c2D.y) * dim.x + c2D.x;
}


inline __host__ __device__ uint to1D( int3 c2D, uint3 dim)
{
    return (dim.y * c2D.z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( uint3 c2D, int2 dim)
{
    return (dim.y * c2D.z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( uint2 c2D, uint z, uint2 dim)
{
    return (dim.y * z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( uint2 c2D, ushort z, uint2 dim)
{
    return (dim.y * z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( int2 c2D, ushort z, uint2 dim)
{
    return (dim.y * z + c2D.y) * dim.x + c2D.x;
}

inline __host__ __device__ uint to1D( int x, int y, uint2 dim)
{
    return y * dim.x + x;
}

inline __host__ __device__ float2 f2X( float x)
{
    return make_float2(x,0);
}

inline __host__ __device__ uint2 ui2X( int x)
{
    return make_uint2((uint)x,0);
}

inline __host__ __device__ int2 i2X( int x)
{
    return make_int2((int)x,0);
}




