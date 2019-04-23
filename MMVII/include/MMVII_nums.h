#ifndef  _MMVII_nums_H_
#define  _MMVII_nums_H_

namespace MMVII
{

/** \file MMVII_nums.h
    \brief some numerical function

*/

/* ================= Random generator  ======================= */

    // === Basic interface, global function but use C++11 modern
    // === generator. By default will be deterministic, 


///  Uniform distribution in 0-1
double RandUnif_0_1();
/// Uniform disrtibution in [0,N[ 
double RandUnif_N(int aN);
/// Eventualy free memory allocated for random generation
void FreeRandom();

/* ============ Definition of numerical type ================*/

typedef float    tREAL4;
typedef double   tREAL8;

typedef signed char  tINT1;
typedef signed short tINT2;
typedef signed int   tINT4;
typedef long int     tINT8;



typedef unsigned char  tU_INT1;
typedef unsigned short tU_INT2;
typedef unsigned int   tU_INT4;


typedef int    tStdInt;  ///< "natural" int
typedef double tStdDouble;  ///< "natural" int


/*  ==============  Traits on numerical type, usable in template function ===================   */
/*   tNumTrait => class to be used                                                              */
/*  tBaseNumTrait, tElemNumTrait   : accessory classes                                          */

    // =========================================================
    //  tBaseNumTrait : Base trait, separate int and float
    // =========================================================

template <class Type> class tBaseNumTrait
{
    public :
        typedef tStdInt  tBase;
};
template <> class tBaseNumTrait<tStdInt>
{
    public :
        static bool IsInt() {return true;}
        typedef tStdInt  tBase;
};
template <> class tBaseNumTrait<tINT8>
{
    public :
        static bool IsInt() {return true;}
        typedef tINT8  tBase;
};
template <> class tBaseNumTrait<tStdDouble>
{
    public :
        static bool IsInt() {return false;}
        typedef tStdDouble  tBase;
};

    // ========================================================================
    //  tElemNumTrait : declare what must be specialized for each type
    // ========================================================================

template <class Type> class tElemNumTrait
{
    public :
};

      // Unsigned int 

template <> class tElemNumTrait<tU_INT1> : public tBaseNumTrait<tStdInt>
{
    public :
        static bool   Signed() {return false;}
        static eTyNums   TyNum() {return eTyNums::eTN_U_INT1;}
};
template <> class tElemNumTrait<tU_INT2> : public tBaseNumTrait<tStdInt>
{
    public :
        static bool   Signed() {return false;}
        static eTyNums   TyNum() {return eTyNums::eTN_U_INT2;}
};
template <> class tElemNumTrait<tU_INT4> : public tBaseNumTrait<tINT8>
{
    public :
        static bool   Signed() {return false;}
        static eTyNums   TyNum() {return eTyNums::eTN_U_INT4;}
};

      // Signed int 

template <> class tElemNumTrait<tINT1> : public tBaseNumTrait<tStdInt>
{
    public :
        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT1;}
};
template <> class tElemNumTrait<tINT2> : public tBaseNumTrait<tStdInt>
{
    public :
        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT2;}
};
template <> class tElemNumTrait<tINT4> : public tBaseNumTrait<tStdInt>
{
    public :
        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT4;}
};
template <> class tElemNumTrait<tINT8> : public tBaseNumTrait<tStdInt>
{
    public :
        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT8;}
};

      // Floating type

template <> class tElemNumTrait<tREAL4> : public tBaseNumTrait<tStdDouble>
{
    public :
        static eTyNums   TyNum() {return eTyNums::eTN_REAL4;}
};
template <> class tElemNumTrait<tREAL8> : public tBaseNumTrait<tStdDouble>
{
    public :
        static eTyNums   TyNum() {return eTyNums::eTN_REAL8;}
};

    // ========================================================================
    //  tNumTrait class to be used
    // ========================================================================

template <class Type> class tNumTrait : public tElemNumTrait<Type>
{
    public :
         typedef Type  tVal;
         typedef tElemNumTrait<Type>  tETrait;
         typedef typename  tETrait::tBase tBase;
         static const tBase MaxValue() {return  std::numeric_limits<tVal>::max();}
         static const tBase MinValue() {return  std::numeric_limits<tVal>::min();}

         static bool OkOverFlow(const tBase & aV)
         {
               if (tETrait::IsInt())
                  return (aV>=MinValue()) && (aV<=MaxValue());
               return true;
         }
         static Type Trunc(const tBase & aV)
         {
               if (tETrait::IsInt())
               {
                  if  (aV<MinValue()) return MinValue();
                  if  (aV>MaxValue()) return MaxValue();
               }
               return Type(aV);
         }
         static Type RandomValue()
         {
              if (tETrait::IsInt())
                 return MinValue() + RandUnif_0_1() * (int(MaxValue())-int(MinValue())) ;
              return RandUnif_0_1();
         }
};





// typedef unsigned char tU_INT1;
// typedef unsigned char tU_INT1;

/* ================= Modulo ======================= */

/// work only when b > 0
inline tINT4 mod(tINT4 a,tINT4 b)
{
    tINT4 r = a%b;
    return (r <0) ? (r+b) : r;
}

/// work only also when b < 0
inline tINT4 mod_gen(tINT4 a,tINT4 b)
{
    tINT4 r = a%b;
    return (r <0) ? (r+ ((b>0) ? b : -b)) : r;
}


/* ================= rounding  ======================= */

/// return the smallest integral value >= r
template<class Type> inline Type Tpl_round_up(tREAL8 r)
{
       Type i = (Type) r;
       return i + (i < r);
}
inline tINT4 round_up(tREAL8 r)  { return Tpl_round_up<tINT4>(r); }
inline tINT8 lround_up(tREAL8 r) { return Tpl_round_up<tINT8>(r); }


/// return the smallest integral value > r
template<class Type> inline Type Tpl_round_Uup(tREAL8 r)
{
       Type i = (Type) r;
       return i + (i <= r);
}
inline tINT4 round_Uup(tREAL8 r) { return Tpl_round_Uup<int>(r); }


/// return the highest integral value <= r
template<class Type> inline Type Tpl_round_down(tREAL8 r)
{
       Type i = (Type) r;
       return i - (i > r);
}
inline tINT4  round_down(tREAL8 r) { return Tpl_round_down<tINT4>(r); }
inline tINT8 lround_down(tREAL8 r) { return Tpl_round_down<tINT8>(r); }

/// return the highest integral value < r
template<class Type> inline Type Tpl_round_Ddown(tREAL8 r)
{
       Type i = (Type) r;
       return i - (i >= r);
}
inline tINT4 round_Ddown(tREAL8 r) { return Tpl_round_Ddown<tINT4>(r); }

/// return the integral value closest to r , if r = i +0.5 (i integer) return i+1
template<class Type> inline Type Tpl_round_ni(tREAL8 r)
{
       Type i = (Type) r;
       i -= (i > r);
       // return i+ ((i+0.5) <= r) ; =>  2i+1<2r  => i < 2*r-i-1
       return i+ ((i+0.5) <= r) ;
}

inline tINT4  round_ni(tREAL8 r) { return Tpl_round_ni<tINT4>(r); }
inline tINT8 lround_ni(tREAL8 r) { return Tpl_round_ni<tINT8>(r); }


inline tREAL8 FracPart(tREAL8 r) {return r - round_down(r);}





};

#endif  //  _MMVII_nums_H_
