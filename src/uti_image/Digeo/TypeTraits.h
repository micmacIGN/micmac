#ifndef __TYPE_TRAITS_DEFINITIONS__
#define __TYPE_TRAITS_DEFINITIONS__

#include "base_types.h"

#include <string>
#include <limits>

template <class tData>
class TypeTraits
{
public:
	typedef tData tBase;
	static std::string Name();
	static tData nmin(){ return std::numeric_limits<tData>::min(); } // normalized min
	static tData nmax(){ return std::numeric_limits<tData>::max(); } // normalized max
};

#define TBASE typename TypeTraits<tData>::tBase


//----------------------------------------------------------------------
// integer types
//----------------------------------------------------------------------

template <> class TypeTraits<U_INT1>
{
public:
	typedef INT tBase;
	static std::string Name(){ return "U_INT1"; }
	static U_INT1 nmin(){ return std::numeric_limits<U_INT1>::min(); }
	static U_INT1 nmax(){ return std::numeric_limits<U_INT1>::max(); }
};

template <> class TypeTraits<U_INT2>
{
public:
	typedef INT tBase;
	static std::string Name(){ return "U_INT2"; }
	static U_INT2 nmin(){ return std::numeric_limits<U_INT2>::min(); }
	static U_INT2 nmax(){ return std::numeric_limits<U_INT2>::max(); }
};

template <> class TypeTraits<INT1>
{
public:
	typedef INT tBase;
	static std::string Name() { return "INT1"; }
	static INT1 nmin(){ return std::numeric_limits<INT1>::min(); }
	static INT1 nmax(){ return std::numeric_limits<INT1>::max(); }
};

template <> class TypeTraits<INT2>
{
public:
	typedef INT tBase;
	static std::string Name() { return "INT2"; }
	static INT2 nmin(){ return std::numeric_limits<INT2>::min(); }
	static INT2 nmax(){ return std::numeric_limits<INT2>::max(); }
};

template <> class TypeTraits<INT>
{
public:
	typedef INT tBase;
	static std::string Name() { return "INT"; }
	static INT nmin(){ return std::numeric_limits<INT>::min(); }
	static INT nmax(){ return std::numeric_limits<INT>::max(); }
};


//----------------------------------------------------------------------
// floating point types
//----------------------------------------------------------------------

template <> class TypeTraits<REAL4>
{
public:
	typedef REAL8 tBase;
	static std::string Name() { return "REAL4"; }
	static REAL4 nmin(){ return (REAL4)0.; }
	static REAL4 nmax(){ return (REAL4)1.; }
};

template <> class TypeTraits<REAL8>
{
public:
	typedef REAL8 tBase;
	static std::string Name() { return "REAL8"; }
	static REAL8 nmin(){ return (REAL8)0.; }
	static REAL8 nmax(){ return (REAL8)1.; }
};

template <> class TypeTraits<REAL16>
{
public:
	typedef REAL16 tBase;
	static std::string Name() { return "REAL16"; }
	static REAL16 nmin(){ return (REAL16)0.; }
	static REAL16 nmax(){ return (REAL16)1.; }
};

#endif
