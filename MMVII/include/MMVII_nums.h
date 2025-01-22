#ifndef  _MMVII_nums_H_
#define  _MMVII_nums_H_

#include "MMVII_Error.h"
#include "MMVII_memory.h"
#include <limits>
#include "MMVII_AllClassDeclare.h"

namespace MMVII
{

// Call V1 Fast kth value extraction
double NC_KthVal(std::vector<double> &, double aProportion);
double Cst_KthVal(const std::vector<double> &, double aProportion);
template <class Type> Type Average(const Type * aTab,size_t aNb);
template <class Type> Type Average(const std::vector<Type> &);


tREAL8 AngleInRad(eTyUnitAngle);
bool AssertRadAngleInOneRound(tREAL8 aAngleRad, bool makeError=true);

// some time needs a null val for any type with + (neutral for +)

template <class T> class cNV
{
    public :
        static T V0(){return T(0);}
};
template <class T,const int Dim>  class  cNV<cPtxd<T,Dim> >;

/*
template<class Type> inline Type NullVal() {return (Type)(0);}
template<> cPtxd<double,2>   NullVal<cPtxd<double,2>  >();// {return cPt2dr::PCste(0);}
template<> cPtxd<double,3>   NullVal<cPtxd<double,3>  >();// {return cPt3dr::PCste(0);}
*/


template <class Type> bool ValidFloatValue(const Type & aV)
{
   return (std::isfinite)(aV) ;
}
template <class Type> bool ValidInvertibleFloatValue(const Type & aV)
{
    return ValidFloatValue(aV) && ( aV != static_cast<Type>(0.0));
}
template <class Type> bool ValidStrictPosFloatValue(const Type & aV)
{
    return ValidFloatValue(aV) && (aV > static_cast<Type>(0.0));
}
template <class Type> bool ValidPosFloatValue(const Type & aV)
{
    return ValidFloatValue(aV) && (aV >= static_cast<Type>(0.0));
}


/** \file MMVII_nums.h
    \brief some numerical function

*/

/* ================= Random generator  ======================= */

    // === Basic interface, global function but use C++11 modern
    // === generator. By default will be deterministic, 


double RandUnif_0_1(); ///<  Uniform distribution in 0-1
std::vector<double> VRandUnif_0_1(int aNb); ///<  Uniform distribution in 0-1
double RandUnif_C();   ///<  Uniform distribution in  -1 1
bool   HeadOrTail();   ///< 1/2 , french 'Pile ou Face'
double RandUnif_N(int aN); ///< Uniform disrtibution in [0,N[ 
double RandUnif_C_NotNull(double aEps);   ///<  Uniform distribution in  -1 1, but abs > aEps
double RandUnif_NotNull(double aEps);   ///<  Uniform distribution in  0 1, but abs > aEps
double RandInInterval(double a,double b); ///<  Uniform distribution in [a,b]
double RandInInterval(const cPt2dr &interval); ///<  Uniform distribution in [interval.x,interval.y]
double RandInInterval_C(const cPt2dr &interval); ///<  Uniform distribution in [-interval.y,-interval.x]U[interval.x,interval.y]

int RandUnif_M_N(int aM,int aN); ///< Uniform disrtibution in [M,N] 

/** Class for mapping object R->R */
class cFctrRR
{  
   public :
      virtual  double F (double) const;  ///< Default return 1.0
      static cFctrRR  TheOne;  ///< the object return always 1
      virtual ~cFctrRR() = default;
};
/// Random permutation , Higer Bias => Higer average rank
std::vector<int> RandPerm(int aN,cFctrRR & aBias =cFctrRR::TheOne);
/// Randomly order a vector , used in bench to multiply some test of possible order dependance
template<class Type>  std::vector<Type>  RandomOrder(const std::vector<Type> & aV)
{
    std::vector<int> aPermut = RandPerm(aV.size());
    std::vector<Type> aRes;
    for (const auto & aI : aPermut)
        aRes.push_back(aV.at(aI));
    return aRes;
}

/// Random subset K among  N  !! Higher bias => lower proba of selection
std::vector<int> RandSet(int aK,int aN,cFctrRR & aBias =cFctrRR::TheOne);
///  Random modification of K Value in a set of N elem
std::vector<int> RandNeighSet(int aK,int aN,const std::vector<int> & aSet);
/// Complement of aSet in [0,1...., N[    ;  ]]
std::vector<int> ComplemSet(int aN,const std::vector<int> & aSet);


/** class to generate a random subset of  K among N, not veru efficent if K<<N because all [0,N] must be parsed
    on the other hand efficient in memory */


class cRandKAmongN
{
    public :
      cRandKAmongN(int aK,int aN);

      bool GetNext();
    private :
        int mK;
        int mN;
};

/// K is the numbre to select, it will be selected regularly with a proportion aProp
bool SelectWithProp(int aK,double aProp);
bool SelectQAmongN(int aK,int aQ,int aN);


/* ============ Definition of numerical type ================*/

/*
typedef float       tREAL4;
typedef double      tREAL8;
typedef long double tREAL16;

typedef signed char  tINT1;
typedef signed short tINT2;
typedef signed int   tINT4;
typedef long int     tINT8;



typedef unsigned char  tU_INT1;
typedef unsigned short tU_INT2;
typedef unsigned int   tU_INT4;
typedef unsigned long int tU_INT8;


typedef int    tStdInt;  ///< "natural" int
typedef unsigned int    tStdUInt;  ///< "natural" int
typedef double tStdDouble;  ///< "natural" int
*/

/* ================= rounding  ======================= */

/// return the smallest integral value >= r
template<class Type> inline Type Tpl_round_up(tREAL8 r)
{
       Type i = static_cast<Type> (r);
       return i + (i < r);
}
inline tINT4 round_up(tREAL8 r)  { return Tpl_round_up<tINT4>(r); }
inline tINT8 lround_up(tREAL8 r) { return Tpl_round_up<tINT8>(r); }


/// return the smallest integral value > r
template<class Type> inline Type Tpl_round_Uup(tREAL8 r)
{
       Type i = static_cast<Type> (r);
       return i + (i <= r);
}
inline tINT4 round_Uup(tREAL8 r) { return Tpl_round_Uup<int>(r); }


/// return the highest integral value <= r
template<class Type> inline Type Tpl_round_down(tREAL8 r)
{
       Type i = static_cast<Type> (r);
       return i - (i > r);
}
inline tINT4  round_down(tREAL8 r) { return Tpl_round_down<tINT4>(r); }
inline tINT8 lround_down(tREAL8 r) { return Tpl_round_down<tINT8>(r); }

/// return the highest integral value < r
template<class Type> inline Type Tpl_round_Ddown(tREAL8 r)
{
       Type i = static_cast<Type> (r);
       return i - (i >= r);
}
inline tINT4 round_Ddown(tREAL8 r) { return Tpl_round_Ddown<tINT4>(r); }

/// return the integral value closest to r , if r = i +0.5 (i integer) return i+1
template<class Type> inline Type Tpl_round_ni(tREAL8 r)
{
       Type i = static_cast<Type> (r);
       i -= (i > r);
       // return i+ ((i+0.5) <= r) ; =>  2i+1<2r  => i < 2*r-i-1
       return i+ ((i+0.5) <= r) ;
}

inline tINT4  round_ni(tREAL8 r) { return Tpl_round_ni<tINT4>(r); }
inline tINT8 lround_ni(tREAL8 r) { return Tpl_round_ni<tINT8>(r); }

tINT4 EmbeddedIntVal(tREAL8 r); ///< When a real value is used for embedding a int, check that value is really int and return it
bool  EmbeddedBoolVal(tREAL8 r); ///< When a real value is used for embedding a bool, check that value is really bool and return it
bool  EmbeddedBoolVal(int V); ///< When a int value is used for embedding a bool, check that value is 0 or 1

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
 
        // For these type rounding mean something
        static int RoundDownToType(const double & aV) {return round_down(aV);}
        static int RoundNearestToType(const double & aV) {return round_ni(aV);}

        static bool constexpr IsInt() {return true;}
        typedef tStdInt  tBase;
        typedef tINT8    tBig;
};
template <> class tBaseNumTrait<tINT8>
{
    public :
        // For these type rounding mean something
        static tINT8 RoundDownToType(const double & aV) {return lround_down(aV);}
        static tINT8 RoundNearestToType(const double & aV) {return lround_ni(aV);}

        static bool constexpr IsInt() {return true;}
        typedef tINT8  tBase;
        typedef tINT8    tBig;
};
template <> class tBaseNumTrait<tStdDouble>
{
    public :
        // By default rounding has no meaning
        static double RoundDownToType(const double & aV) {return aV;}
        static double RoundNearestToType(const double & aV) {return aV;}

        static bool constexpr IsInt() {return false;}
        typedef tStdDouble  tBase;
        typedef tStdDouble  tBig;
};
template <> class tBaseNumTrait<tREAL16>
{
    public :
        // By default rounding has no meaning
        static double RoundDownToType(const double & aV) {return aV;}
        static double RoundNearestToType(const double & aV) {return aV;}

        static bool constexpr IsInt() {return false;}
        typedef tREAL16  tBase;
        typedef tREAL16  tBig;
};
/// Not sure usable by itself but required in some systematic template instantiatio
template <> class tBaseNumTrait<tREAL4>
{
    public :
        // By default rounding has no meaning
        static tREAL4 RoundDownToType(const double & aV) {return static_cast<tREAL4>(aV);}
        static tREAL4 RoundNearestToType(const double & aV) {return static_cast<tREAL4>(aV);}
        static bool constexpr IsInt() {return false;}
        typedef tREAL4      tBase;
        typedef tStdDouble  tBig;
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
        static tU_INT1 DummyVal() {MMVII_INTERNAL_ERROR("No DummyVal for type");return 0;}
        static tU_INT1 MaxVal() {return  std::numeric_limits<tU_INT1>::max();}
        static bool   Signed() {return false;}
        static eTyNums   TyNum() {return eTyNums::eTN_U_INT1;}
        typedef tREAL4   tFloatAssoc;
};
template <> class tElemNumTrait<tU_INT2> : public tBaseNumTrait<tStdInt>
{
    public :
        static tU_INT2 DummyVal() {MMVII_INTERNAL_ERROR("No DummyVal for type");return 0;}
        static tU_INT2 MaxVal() {return  std::numeric_limits<tU_INT2>::max();}
        static bool   Signed() {return false;}
        static eTyNums   TyNum() {return eTyNums::eTN_U_INT2;}
        typedef tREAL4   tFloatAssoc;
};
template <> class tElemNumTrait<tU_INT4> : public tBaseNumTrait<tINT8>
{
    public :
        static tU_INT4 DummyVal() {return MaxVal();}
        static tU_INT4 MaxVal() {return  std::numeric_limits<tU_INT4>::max();}
        static bool   Signed() {return false;}
        static eTyNums   TyNum() {return eTyNums::eTN_U_INT4;}
        typedef tREAL8   tFloatAssoc;
};

      // Signed int 

template <> class tElemNumTrait<tINT1> : public tBaseNumTrait<tStdInt>
{
    public :
        static tINT1 DummyVal() {MMVII_INTERNAL_ERROR("No DummyVal for type");return 0;}
        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT1;}
        typedef tREAL4   tFloatAssoc;
};
template <> class tElemNumTrait<tINT2> : public tBaseNumTrait<tStdInt>
{
    public :
        static tINT2 DummyVal() {MMVII_INTERNAL_ERROR("No DummyVal for type");return 0;}
        static tINT2 MaxVal() {return  std::numeric_limits<tINT2>::max();}
        static tINT2 MinVal() {return  std::numeric_limits<tINT2>::min();}
        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT2;}
        typedef tREAL4   tFloatAssoc;
};
template <> class tElemNumTrait<tINT4> : public tBaseNumTrait<tStdInt>
{
    public :
        static tINT4 MaxVal() {return  std::numeric_limits<tINT4>::max();}
        static tINT4 DummyVal() {return MaxVal();}

        static bool   Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT4;}
        typedef tREAL8   tFloatAssoc;
};
template <> class tElemNumTrait<tINT8> : public tBaseNumTrait<tINT8>
{
    public :
        static tINT8 MaxVal() {return  std::numeric_limits<tINT8>::max();}
        static tINT8 DummyVal() {return MaxVal();}
        static bool      Signed() {return true;}
        static eTyNums   TyNum() {return eTyNums::eTN_INT8;}
        typedef tREAL8   tFloatAssoc;
};

      // Floating type

template <> class tElemNumTrait<tREAL4> : public tBaseNumTrait<tStdDouble>
{
    public :
        static tREAL4 MaxVal() {return  std::numeric_limits<tREAL4>::max();}
        static tREAL4 MinVal() {return  std::numeric_limits<tREAL4>::min();}
        static tREAL4 DummyVal() {return std::nanf("");}
        static tREAL4 Accuracy() {return 1e-2f;}
        static bool   Signed() {return true;} ///< Not usefull but have same interface
        static eTyNums   TyNum() {return eTyNums::eTN_REAL4;}
        typedef tREAL4   tFloatAssoc;
};
template <> class tElemNumTrait<tREAL8> : public tBaseNumTrait<tStdDouble>
{
    public :
        static tREAL8 DummyVal() {return std::nan("");}
        static tREAL8 Accuracy() {return 1e-4;} 
        static bool   Signed() {return true;} ///< Not usefull but have same interface
        static eTyNums   TyNum() {return eTyNums::eTN_REAL8;}
        typedef tREAL8   tFloatAssoc;
};
template <> class tElemNumTrait<tREAL16> : public tBaseNumTrait<tREAL16>
{
    public :
        static tREAL16 DummyVal() {return std::nanl("");}
        static tREAL16 Accuracy() {return static_cast<tREAL16>(1e-6);} 
        static bool      Signed() {return true;} ///< Not usefull but have same interface
        static eTyNums   TyNum() {return eTyNums::eTN_REAL16;}
        typedef tREAL16  tFloatAssoc;
};

    // ========================================================================
    //  tNumTrait class to be used
    // ========================================================================

/** Sometime it may be usefull to manipulate the caracteristic (size, int/float ...)  of an unknown
  numerical type (for example read in a tiff file spec), this can be done using cVirtualTypeNum */


class cVirtualTypeNum
{
    public :
       virtual ~cVirtualTypeNum() = default;
       virtual bool V_IsInt()  const = 0;
       virtual bool V_Signed() const = 0;
       virtual int  V_Size()   const = 0;
       virtual eTyNums  V_TyNum() const = 0; ///< Used to check FromEnum, else ??
  

       static const cVirtualTypeNum & FromEnum(eTyNums);
};

template <class Type> class tNumTrait : public tElemNumTrait<Type> ,
                                        public cVirtualTypeNum
{
    public :


 
      // ===========================
         typedef Type  tVal;
         typedef tElemNumTrait<Type>  tETrait;
         typedef typename  tETrait::tBase tBase;
         typedef typename  tETrait::tBig  tBig ;
         typedef typename  tETrait::tFloatAssoc  tFloatAssoc ;
         // typedef typename  tETrait::tFloatAssoc  tFloatAssoc ;
      // ===========================
         bool V_IsInt()  const override {return  tBaseNumTrait<tBase>::IsInt();}
         bool V_Signed() const override {return  tETrait::Signed();}
         int  V_Size()   const override {return  sizeof(Type);}
         eTyNums  V_TyNum() const override {return  tETrait::TyNum();}

        // For these type rounding mean something
        static int RoundDownToType(const double & aV) {return tBaseNumTrait<tBase>::RoundDownToType(aV);}
        static int RoundNearestToType(const double & aV) {return tBaseNumTrait<tBase>::RoundNearestToType(aV);}
      //==============
         static const tBase MaxValue() {return  std::numeric_limits<tVal>::max();}
         static const tBase MinValue() {return  std::numeric_limits<tVal>::min();}

         static bool ValueOk(const tBase & aV)
         {
               if constexpr (tETrait::IsInt())
                  return (aV>=MinValue()) && (aV<=MaxValue());
               else
                return ValidFloatValue(aV);
         }
	 static void AssertValueOk(const tBase & aV)
	 {
              MMVII_INTERNAL_ASSERT_tiny(ValueOk(aV),"Bad value");
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
         static Type RandomValueCenter()
         {
              if (tETrait::IsInt())
                 return MinValue() + RandUnif_0_1() * (int(MaxValue())-int(MinValue())) ;
              return RandUnif_C();
         }
         static Type AmplRandomValueCenter()
         {
              if (tETrait::IsInt())
                 return  (int(MaxValue())-int(MinValue())) ;
              return 2.0;
         }



         static Type Eps()
         {
              if (tETrait::IsInt())
                 return 1;
              return  std::numeric_limits<Type>::epsilon();
         }

         static const tNumTrait<Type>  TheOnlyOne;
         static const std::string & NameType() {return E2Str(TheOnlyOne.V_TyNum());}
};

// Definition of tNumTrait<Type>::TheOnlyOne; needed in BenchMatrix.cpp
template <class Type> const tNumTrait<Type>   tNumTrait<Type>::TheOnlyOne;

// This traits type allow to comppute a temporary variable having the max
// precision between 2 floating types

template <class T1,class T2> class tMergeF { public : typedef tREAL16  tMax; };
template <> class tMergeF<tREAL4,tREAL4> { public : typedef tREAL4  tMax; };
template <> class tMergeF<tREAL8,tREAL4> { public : typedef tREAL8  tMax; };
template <> class tMergeF<tREAL4,tREAL8> { public : typedef tREAL8  tMax; };
template <> class tMergeF<tREAL8,tREAL8> { public : typedef tREAL8  tMax; };

template <class Type>  void AssertTabValueOk(const Type * aTab,size_t aNb)
{
    for (size_t aK=0 ; aK<aNb ; aK++)
        tNumTrait<Type>::AssertValueOk(aTab[aK]);
}

template <class Type> void  AssertTabValueOk(const std::vector<Type> & aVec)
{
    AssertTabValueOk(aVec.data(),aVec.size());
} 



// typedef unsigned char tU_INT1;
// typedef unsigned char tU_INT1;

/* ================= Modulo ======================= */

/// work only when b > 0
inline tINT4 round_to(tINT4 a,tINT4 b)
{
   return (a/b) * b;
}

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

///  Modulo with real value, same def as with int but build-in C support
inline tREAL8 mod_real(tREAL8 a,tREAL8 b)
{
   MMVII_INTERNAL_ASSERT_tiny(b>0,"modreal");
   tREAL8 aRes = a - b * round_ni(a/b);
   return (aRes<0) ? aRes+b : aRes;
}

template <class Type> Type diff_circ(const Type & a,const Type & b,const Type & aPer);

///  Return division superior : a <= d*b < a+b
template<class Type> Type DivSup(const Type & a,const Type & b) 
{
    MMVII_INTERNAL_ASSERT_tiny(b>0,"DivSup");
    return (a+b-1)/b; 
}
/// a/b but upper valuer  6/3=> 2 7/3 => 3
#define DIV_SUP(a,b) ((a+b-1)/b)  // macro version usefull for constexpr

/// Return a value depending only of ratio, in [-1,1], eq 0 if I1=I2, and invert sign when swap I1,I2
double NormalisedRatio(double aI1,double aI2);
double NormalisedRatioPos(double aI1,double aI2);
double Der_NormalisedRatio_I1(double aI1,double aI2);
double Der_NormalisedRatio_I2(double aI1,double aI2);
double Der_NormalisedRatio_I1Pos(double aI1,double aI2);
double Der_NormalisedRatio_I2Pos(double aI1,double aI2);


tINT4 HCF(tINT4 a,tINT4 b); ///< = PGCD = Highest Common Factor
tREAL8   rBinomialCoeff(int aK,int aN);
tU_INT8  liBinomialCoeff(int aK,int aN);
tU_INT4  iBinomialCoeff(int aK,int aN);
/* ****************  cDecomposPAdikVar *************  */

//  P-adik decomposition
//  given a b c ...
//     x y z   ->   x + a * y +  a * b *z
//     M  -> M%a (M/a)%b ...
//
class cDecomposPAdikVar
{
     public :
       typedef std::vector<int> tVI;
       cDecomposPAdikVar(const tVI &);  // Constructot from set of bases

       const tVI &  Decompos(int) const; // P-Adik decomposition return internal buffer
       const tVI &  DecomposSizeBase(int) const; // Make a decomposition using same size (push 0 is need), requires < mMumBase
       int          FromDecompos(const tVI &) const; // P-Adik recomposition
       static void Bench();  // Make the test on correctness of implantation
       const int&  MulBase() const;
     private:
       static void Bench(const std::vector<int> & aVB);
       void Bench(int aValue) const;
       const int & BaseOfK(int aK) const {return mVBases.at(aK%mNbBase);}

       tVI          mVBases;
       int          mNbBase;
       int          mMulBase;
       mutable tVI  mRes;
};

double  RelativeDifference(const double & aV1,const double & aV2,bool * Ok=nullptr);
double RelativeSafeDifference(const double & aV1,const double & aV2);

template <class Type> int SignSupEq0(const Type & aV) {return (aV>=0) ? 1 : -1;}

inline tREAL8 FracPart(tREAL8 r) {return r - round_down(r);}


template <class Type> Type Square(const Type & aV) {return aV*aV;}
template <class Type> Type Cube(const Type & aV) {return aV*aV*aV;}
template <class Type,class TCast> TCast TSquare(const Type & aV) {return aV* TCast(aV);}
template <class Type> tREAL8  R8Square(const Type & aV) {return TSquare<Type,tREAL8>(aV);} ///< To avoid oveflow with int type

template <class Type> Type Sqrt(const Type & aV) 
{
    MMVII_ASSERT_POS_VALUE(aV);
    return std::sqrt(aV);
}

template <class Type> void OrderMinMax(Type & aV1,Type & aV2)
{
   if (aV1>aV2)
      std::swap(aV1,aV2);
}

// 4 now use sort, will enhance with home made
template <class Type> Type NonConstMediane(std::vector<Type> & aV);
template <class Type> Type ConstMediane(const std::vector<Type> & aV);


/*  ******************************************* */
/*   Some basic operation, tested in debug mode */
/*  ******************************************* */

template<class Type> Type SafeDiv(const Type & aNumerator,const Type & aDenominator)
{
    MMVII_INTERNAL_ASSERT_NotNul(aDenominator);
    return aNumerator / aDenominator;
}


/*  ********************************* */
/*       Kernels                      */
/* ********************************** */

/// A kernel, approximating "gauss"

/**  a quick kernel, derivable, with support in [-1,1], coinciding with bicub in [-1,1] 
     not really gauss but has a "bell shape"
     1 +2X^3 -3X^2  , it's particular of bicub with Deriv(1) = 0
*/
/*
class cCubAppGauss
{
     public :
         tREAL8 Value(const tREAL8);
     private :
};
*/

/// If we dont need any kernel interface keep it simple 
// tREAL8 CubAppGaussVal(const tREAL8&);   

/*  ********************************* */
/*       Witch Min and Max            */
/* ********************************** */

template <class TypeIndex,class TypeVal,const bool IsMin> class cWhichExtrem
{
     public :
         cWhichExtrem(const TypeIndex & anIndex,const TypeVal & aVal) :
             mIsInit     (true),
             mIndexExtre (anIndex),
             mValExtre   (aVal)
         {
         }
         cWhichExtrem() :
             mIsInit   (false),
             mIndexExtre (cNV<TypeIndex>::V0()),  // required else compiler complains for possible use of un-initialised
             // mIndexExtre (NullVal<TypeIndex>()),  // required else compiler complains for possible use of un-initialised
             mValExtre   (0)
	 {
	 }
	 bool IsInit() const {return mIsInit;}

	 // return value indicate if modif was done
         bool Add(const TypeIndex & anIndex,const TypeVal & aNewVal)
         {
              if ( (IsMin?(aNewVal<mValExtre):(aNewVal>=mValExtre)) || (!mIsInit))
              {     
                    mValExtre   = aNewVal;
                    mIndexExtre = anIndex;
                    mIsInit = true;
		    return true;
              }
	      return false;
         }
         const TypeIndex & IndexExtre() const {AssertIsInit();return mIndexExtre;}
         const TypeVal   & ValExtre  () const {AssertIsInit();return mValExtre;}
     private :
	 void  AssertIsInit() const 
	 {
              MMVII_INTERNAL_ASSERT_tiny(mIsInit,"Exrem not init");
	 }
         bool      mIsInit;
         TypeIndex mIndexExtre;
         TypeVal   mValExtre;
};

template <class TypeIndex,class TypeVal> class cWhichMin : public cWhichExtrem<TypeIndex,TypeVal,true>
{
     public :
         typedef  cWhichExtrem<TypeIndex,TypeVal,true> tExrem;

         cWhichMin(const TypeIndex & anIndex,const TypeVal & aVal) :
            tExrem (anIndex,aVal) 
         {
         }
         cWhichMin() : tExrem () {}
     private :
};
template <class TypeIndex,class TypeVal> class cWhichMax : public cWhichExtrem<TypeIndex,TypeVal,false>
{
     public :
         typedef  cWhichExtrem<TypeIndex,TypeVal,false> tExrem;

         cWhichMax(const TypeIndex & anIndex,const TypeVal & aVal) :
            tExrem (anIndex,aVal) 
         {
         }
         cWhichMax() : tExrem () {}
     private :
};


template <class TypeIndex,class TypeVal> class cWhichMinMax
{
     public  :
         cWhichMinMax(const TypeIndex & anIndex,const TypeVal & aVal) :
             mMin(anIndex,aVal),
             mMax(anIndex,aVal)
         {
         }
         cWhichMinMax() { }

         void Add(const TypeIndex & anIndex,const TypeVal & aVal)
         {
             mMin.Add(anIndex,aVal);
             mMax.Add(anIndex,aVal);
         }
         const cWhichMin<TypeIndex,TypeVal> & Min() const {return  mMin;}
         const cWhichMax<TypeIndex,TypeVal> & Max() const {return  mMax;}

         const TypeIndex &  IndMin() const {return  mMin.IndexExtre();}
         const TypeIndex &  IndMax() const {return  mMax.IndexExtre();}

     private :
         cWhichMin<TypeIndex,TypeVal> mMin;
         cWhichMax<TypeIndex,TypeVal> mMax;
};

template <class TypeVal> void UpdateMin(TypeVal & aVar,const TypeVal & aValue) {if (aValue<aVar) aVar = aValue;}
template <class TypeVal> void UpdateMax(TypeVal & aVar,const TypeVal & aValue) {if (aValue>aVar) aVar = aValue;}

template <class TypeVal> void UpdateMinMax(TypeVal & aVarMin,TypeVal & aVarMax,const TypeVal & aValue) 
{
    // The two test are required (No else if ...) because initially we may have VarMin>VarMax
    if (aValue<aVarMin) aVarMin = aValue;
    if (aValue>aVarMax) aVarMax = aValue;
}

template <class TVal> TVal MinTab(TVal * Data,int aNb)
{
    MMVII_INTERNAL_ASSERT_tiny(aNb!=0,"No values in MinTab");
    TVal aMin=Data[0];
    for (int aK=1 ; aK<aNb ; aK++)
        if (Data[aK]< aMin)
           aMin = Data[aK];

    return aMin;
}




/// Class to store min and max values
template <class TypeVal> class cBoundVals
{
	public :
            cBoundVals() :
                   mVMin ( std::numeric_limits<TypeVal>::max()),
		   //mVMax (-std::numeric_limits<TypeVal>::max())  MPD : strange why not min() ??
		   mVMax (std::numeric_limits<TypeVal>::min())
	    {
            }
            void Add(const TypeVal & aVal)
            {
                 UpdateMinMax(mVMin,mVMax,aVal);
            }

	    const TypeVal  &  VMin () const {return mVMin;}
	    const TypeVal  &  VMax () const {return mVMax;}
	private :
            TypeVal  mVMin;
            TypeVal  mVMax;
};
/// Class to store min and max values AND average
template <class TypeVal> class cAvgAndBoundVals  : public cBoundVals<TypeVal>
{
	public :
		cAvgAndBoundVals() :
			cBoundVals<TypeVal>(),
			mSomVal  (0),
			mNbVals  (0)
	        {
	        }
                void Add(const TypeVal & aVal)
		{
			cBoundVals<TypeVal>::Add(aVal);
			mSomVal += aVal;
			mNbVals++;
		}
		TypeVal  Avg() const {return SafeDiv(mSomVal,mNbVals); }
	private :
           TypeVal  mSomVal;
           TypeVal  mNbVals;
};

// This rather "strange" function returns a value true at frequence  as close as possible
// to aFreq, and with the warantee that it is true for last index

bool SignalAtFrequence(tREAL8 anIndex,tREAL8 aFreq,tREAL8  aCenterPhase);

/*  ****************************************** */
/*       Analytical function used with fisheye */
/* ******************************************* */

/// Sinus cardinal with caution on tiny values < aEps
template <typename Type> Type sinC(const Type & aTeta,const Type & aEps);
/// Sinus cardinal default with epsilon of type, to have interface of unitary function
template <typename Type> Type sinC(const Type & aTeta);
/// Derivative Sinus cardinal with caution on tiny values < aEps
template <typename Type> Type DerSinC(const Type & aTeta,const Type & aEps);
/// Derivative Sinus cardinal with default epsilon of type, to have interface of unitary function
template <typename Type> Type DerSinC(const Type & aTeta);

/// to have it in good namespace in code gen
template <typename Type> Type ASin(const Type & aSin);
/// to have it as operator in code gen
template <typename Type> Type DerASin(const Type & aSin);

/// as sqrt but check value
template <typename Type> Type Sqrt(const Type & aX);
///  as 1/(2 sqrt) but check value
template <typename Type> Type DerSqrt(const Type & aX);

/// to have it in good namespace in code gen
template <typename Type> Type ATan2(const Type & aX,const Type & aY);
/// to have it d/dx in code gen
template <typename Type> Type DerX_ATan2(const Type & aX,const Type & aY);
/// to have it d/dy in code gen
template <typename Type> Type DerY_ATan2(const Type & aX,const Type & aY);


/// to have it in good namespace in code gen
template <typename Type> Type DiffAngMod(const Type & aA,const Type & aB);
/// to have it d/dx in code gen
template <typename Type> Type DerA_DiffAngMod(const Type & aA,const Type & aB);
/// to have it d/dy in code gen
template <typename Type> Type DerB_DiffAngMod(const Type & aA,const Type & aB);


/// Sinus hyperbolic
template <typename Type> Type sinH(const Type & aTeta);
/// CoSinus hyperbolic
template <typename Type> Type cosH(const Type & aTeta);




  //  ----- Function used for equilinear fisheye ----

   /// Arctan(x,y)/x  but stable when x->0,  !!! NOT C++ CONVENTION WHICH ARE atan2(y,x)
template <typename Type> Type AtanXsY_sX(const Type & X,const Type & Y);
  /// Derivate upon x of AtanXY_sX(x,y),  stable when x->0
template <typename Type> Type DerXAtanXsY_sX(const Type & X,const Type & Y);
  /// Derivate upon y of AtanXY_sX(x,y),  noting to care  when x->0
template <typename Type> Type DerYAtanXsY_sX(const Type & X,const Type & Y);

   /// Same as AtanXY_sX but user fix the "tiny" value, used for bench
template <typename Type> Type AtanXsY_sX(const Type & X,const Type & Y,const Type & aEps);
   /// Same as DerXAtanXY_sX ...  ... bench
template <typename Type> Type DerXAtanXsY_sX(const Type & X,const Type & Y,const Type & aEps);

      //   -------------- miscelaneaous functions ------------------------
/// Reciprocal function of X-> X|X|
template <typename Type> Type SignedSqrt(const Type & aTeta); 



/*  ****************************************** */
/*     REPRESENTATION of num on a base         */
/* ******************************************* */
  
///  Number minimal of digit for representing a number in a given base
size_t GetNDigit_OfBase(size_t aNum,size_t aBase);
///  Representation of number in a given base, can force minimal number of digit
std::string  NameOfNum_InBase(size_t aNum,size_t aBase,size_t aNbDigit=0);

/*  ****************************************** */
/*       BIT MANIPULATION FUNCTIONS            */
/* ******************************************* */


///  Number of bits to 1
size_t NbBits(tU_INT4 aVal);
///  Hamming distance (number of bit different)
size_t HammingDist(tU_INT4 aV1,tU_INT4 aV2);
/// make a circular permutation of bits, assuming a size NbIt, with  aPow2= NbBit^2
size_t  LeftBitsCircPerm(size_t aSetFlag,size_t aPow2);
/// make N iteratuio of LeftBitsCircPerm
size_t  N_LeftBitsCircPerm(size_t aSetFlag,size_t aPow2,size_t N);

/// make a symetry bits, assuming a size NbIt, with  aPow2= NbBit^2
size_t  BitMirror(size_t aSetFlag,size_t aPow2);
/// make a visualisation of bit flag as  (5,256) -> "10100000"
std::string  StrOfBitFlag(size_t aSetFlag,size_t aPow2);
/// Transformate a string-Visu in flag bits "10100000" -> 5
size_t  Str2BitFlag(const std::string & aStr);
/// Transormate a bit flage in vect of int, for easier manip
void  BitsToVect(std::vector<int> & aVBits,tU_INT4 aVal,size_t aPow2);
///  return the maximal length of consecutive 0 & 1, interpreted circularly    (94="01111010", 256=2^8)  =>  (3,2)
cPt2di MaxRunLength(tU_INT4 aVal,size_t aPow2);
///  idem above + memo the intervals
cPt2di MaxRunLength(tU_INT4 aVal,size_t aPow2,std::vector<cPt2di> & aV0,std::vector<cPt2di> & aV1);
/// Max of both run (0 and 1)
size_t MaxRun2Length(tU_INT4 aVal,size_t aPow2);

/// Low level function, read the pair Num->Code in a file
void  ReadCodesTarget(std::vector<cPt2di> & aVCode,const std::string & aNameFile);

/**  Helper class for cCompEquiCodes, store on set of code equivalent */
class cCelCC : public cMemCheck
{
     public :
        std::vector<size_t>  mEquivCode;  /// all codes equivalent
        size_t               mLowCode;    ///< lower representant
        bool                 mTmp;        ///< some marker to use when convenient
        int                  mNum;        ///< Num used so that names is alway the same whatever maybe the selection

	size_t HammingDist(const cCelCC &) const;

        cCelCC(size_t aLowestCode);
     public :
        cCelCC(const cCelCC &) = delete;
};

/** Class for computing equivalent code, typicall code that are equal up to a circular permutation */

class cCompEquiCodes : public cMemCheck
{
   public :
       typedef std::pair<cCelCC*,std::vector<cPt2di> > tAmbigPair;  // to represent possible ambiguity

       static std::string NameCERNLookUpTable(size_t aNbBits); ///< name of file where are stored CERN'S   LUT
       static std::string NameCERNPannel(size_t aNbBits); ///< name of file where are stored CERN'S   3D target
       ///  allocate & compute code , return the same adress if param eq
       static cCompEquiCodes * Alloc(size_t aNbBits,size_t aPerAmbig=1,bool WithMirror=false);

       /// For a set code (p.y()) return the cell containing them (or not contatining them)
       std::vector<cCelCC*>  VecOfUsedCode(const std::vector<cPt2di> &,bool Used);
       /// For a set of code return the ambiguity (code beloning to same class)
       std::list<tAmbigPair>  AmbiguousCode(const std::vector<cPt2di> &);
       const std::vector<cCelCC*>  & VecOfCells() const; ///< Accessor
       const cCelCC &  CellOfCodeOK(size_t aCode) const;  ///< Error if null
       const cCelCC *  CellOfCode(size_t) const;  ///< nullptr if bad range or no cell
       cCelCC *  CellOfCode(size_t) ;  ///< nullptr if bad range or no cell

       ~cCompEquiCodes();
       static void Bench(size_t aNBB,size_t aPer,bool Miror);

   private :
       static std::string NameCERStuff(const std::string & aPrefix,size_t aNbBits); ///< name of file where are stored CERN'S   3D target

       cCompEquiCodes(size_t aNbBits,size_t aPerdAmbig,bool WithMirror);
       /// put all the code identic, up to a circular permutation, in the same cellu
       void AddCodeWithPermCirc(size_t aCode,cCelCC *);

       size_t                   mNbBits;      ///< Number of bits
       size_t                   mPeriod;      ///< Period for equiv circ,
       size_t                   mNbCodeUC;    ///<  Number of code uncircullar i.e. 2 ^NbBits
       std::vector<cCelCC*>     mVCodes2Cell; ///< Code->Cell  vector of all code for sharing equivalence
       std::vector<cCelCC*>     mVecOfCells;  ///< vector of all different cells

       std::vector<int>         mHistoNbBit;

       // static std::map<std::string,cCompEquiCodes*> TheMapCodes;
};


class  cHamingCoder
{
    public :
         /// Constructor , indicate the number of bit of information
         cHamingCoder(int aNbBitsIn);

         /// Different of default, here we indicate the total number of bits, last indicate if require even number 
         static cHamingCoder HCOfBitTot(int aNbBitsTot,bool WithParity=false);

         int NbBitsOut() const; ///< Number of bit of coded messages
         int NbBitsIn() const;  ///< Number of bits of information
         int NbBitsRed() const; ///< Number of bits of redundancy

         tU_INT4  Coding(tU_INT4) const;  ///< From raw to encoded message
         /// Return initial message IFF no alteration, else return -1
         int  UnCodeWhenCorrect(tU_INT4);

    private :
        int mNbBitsIn;
        int mNbBitsRed;
        int mNbBitsOut;

        std::vector<bool>  mIsBitRed;
        std::vector<int>   mNumI2O;
        std::vector<int>   mNumO2I;
};


template <class Type> class  cPolynom
{
        public :
           typedef std::vector<Type>  tCoeffs;
           typedef cPtxd<Type,2>      tCompl;
           cPolynom(const tCoeffs &);
           cPolynom(const cPolynom &);
           cPolynom(size_t aDegre);
           size_t  Degree() const;

           static cPolynom<Type>  D0(const Type &aCste);      ///< constant polynom degre 0
           static cPolynom<Type>  D1FromRoot(const Type &aRoot);    ///< degre 1 polynom with aRoot
           static cPolynom<Type>  D2NoRoot(const Type & aVMin,const Type &aArgmin);  ///< +- (|V| + (x-a) ^2) ,

           static cPolynom<Type>  RandomPolyg(int aDegree,Type & anAmpl);
           ///  Generate random polygo from its randomly generated roots => test for
           static cPolynom<Type>  RandomPolyg(std::vector<Type> & aVRoots,int aNbRoot,int aNbNoRoot,Type Interv,Type MinDist);


           Type    Value(const Type & aVal) const;
           tCompl  Value(const tCompl & aVal) const;
           /// return som(|a_k x^k|) , used for some bounding stuffs
           Type  AbsValue(const Type & aVal) const;


           cPolynom<Type> operator * (const cPolynom<Type> & aP2) const;
           cPolynom<Type> operator + (const cPolynom<Type> & aP2) const;
           cPolynom<Type> operator - (const cPolynom<Type> & aP2) const;
           cPolynom<Type> operator * (const  Type & aVal) const;
           cPolynom<Type> Deriv() const;

           std::vector<Type> RealRoots(const Type & aTol,int ItMax) const;


           Type&   operator [] (size_t aK) {return mVCoeffs[aK];}
           const Type&   operator [] (size_t aK) const {return mVCoeffs[aK];}
           Type  KthDef(size_t aK) const {return (aK<mVCoeffs.size()) ? mVCoeffs[aK] : static_cast<Type>(0.0) ;}

           const tCoeffs &  VCoeffs() const;

        private :
           tCoeffs  mVCoeffs;
};

template <class Type,const int Dim>  cPolynom<Type> operator * (const  Type & aVal,const cPolynom<Type>  & aPol)  {return aPol*aVal;}
/// return polynom of (Cste + X Lin)^2
template <class Type,const int Dim> cPolynom<Type> PolSqN(const cPtxd<Type,Dim>& aVecCste,const cPtxd<Type,Dim>& aVecLin);

// Rank of values
template <class TCont,class TVal> double Rank(const TCont &, const TVal&);

/// Low level read of file containing nums in fixed format   F->double   S->string (skipped)
void  ReadFilesNum(const std::string & aNameFile,const std::string & aFormat,std::vector<std::vector<double>> & aVRes,int aComment=-1);

void  ReadFilesStruct
      (
          const std::string & aNameFile,  // name of file ...
	  const std::string & aFormat,    // format of each line like "NXYZ"
          int aL0,    // Num first line
	  int aLastL,  // Num last line, if <0 => infty
	  int aComment,  // Car for begining a comment, for ex -1 if no comment
          std::vector<std::vector<std::string>> & aVNames,  // "N" 
          std::vector<cPt3dr> & aVXYZ,   //  "XYZ"
          std::vector<cPt3dr>  & aVWKP,  //   "WPK" 
          std::vector<std::vector<double>>  & aVNums,  //  get other double "FF*F"
          bool CheckFormat= true  // if true check :  XYZN have same count ... and more 2 com
      );

class cReadFilesStruct
{
     public :

       cReadFilesStruct( const std::string &  aNameFile,const std::string & aFormat,
                         int aL0,int aLastL, int  aComment);

       void Read();

       const std::vector<std::string>               & VNameIm () const; ///< Accessor + Check init
       const std::vector<std::string>               & VNamePt () const; ///< Accessor + Check init  "N"
       const std::vector<std::vector<std::string>>  & VStrings () const; ///< Accessor + Check init  "S"
       const std::vector<cPt3dr>                    & VXYZ () const; ///< Accessor + Check init    "XYZ"
       const std::vector<cPt2dr>                    & Vij () const; ///< Accessor + Check init      "ij"
       const std::vector<cPt3dr>                    & VWPK () const; ///< Accessor + Check init    "WPK"
       const std::vector<std::vector<double>>       & VNums () const; ///< Accessor + Check init   "FF*F"
       const std::vector<std::vector<int>>          & VInts () const; ///< Accessor + Check init     "EE*E"
       const std::vector<std::string>               & VLinesInit () const; ///< Accessor + Check init
       int NbRead() const;  ///< Number of line read
       void SetMemoLinesInit() ;  ///< Activate the memo of initial lines (false by default)


     private :
         template <class Type> inline const std::vector<Type> & GetVect(const std::vector<Type> & aV) const
         {
                 MMVII_INTERNAL_ASSERT_tiny((int)aV.size()==mNbLineRead,"cReadFilesStruct::GetV");
                 return aV;
         }
         // ============== copy of  constructor parameters ===================

         std::string     mNameFile; ///< name of file
         std::string     mFormat;   ///< format of each line
         int             mL0;       ///< num of first line
         int             mLastL;    ///< num of last line
         int             mComment;  ///< carac used for comment if any

         int             mNbLineRead;  ///< count number of line
	 bool            mMemoLinesInt;   ///< Do we maintains a memory of initial line (w/o supressed one so that it match data)

         std::vector<std::string>               mVNameIm;
         std::vector<std::string>               mVNamePt;
         std::vector<cPt3dr>                    mVXYZ;
         std::vector<cPt2dr>                    mVij;
         std::vector<cPt3dr>                    mVWPK;
         std::vector<std::vector<double>>       mVNums;
         std::vector<std::vector<int>>          mVInts;
         std::vector<std::string>               mVLinesInit;
         std::vector<std::vector<std::string>>  mVStrings;
};

/// nuber of occurence of aC0 in aStr
int CptOccur(const std::string & aStr,char aC0);
/// Check same number of occurence of Str0 in aStr, and return it, Str0 cannot be empty
int CptSameOccur(const std::string & aStr,const std::string & aStr0);





/**  Class for implementing proba law whith given average & standard deviation*/

class cAvgDevLaw : public cMemCheck
{
    public  :
	    /// Value of the law
            tREAL8  NormalizedValue(tREAL8 aVal) const ;
            /// Allocator of a cubic law approx gaussian
            static cAvgDevLaw * CubAppGaussLaw(const tREAL8& aAvg,const  tREAL8& aStdDev);  
            /// Allocator of a gaussian law
            static cAvgDevLaw * GaussLaw(const tREAL8& aAvg,const  tREAL8& aStdDev);
            virtual ~cAvgDevLaw();  ///< virtual destructor because virtual method

            static void Bench();  ///<  Test of Integral/Average/Deviation for diff law
    protected :
            static void BenchOneLow(cAvgDevLaw *);  ///< Test of Integral/Average/Deviation for a given law

            cAvgDevLaw(const tREAL8& aAvg,const  tREAL8& aStdDev);
            virtual tREAL8  RawValue(tREAL8 aVal) const =0 ; ///< fundamental method to override

            tREAL8 mAvg;     ///<  Average of the law
            tREAL8 mStdDev;  ///<  Standard deviation of the law
};



};

#endif  //  _MMVII_nums_H_
