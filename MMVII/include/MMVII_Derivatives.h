#ifndef  _MMVII_Derivatives_H_
#define  _MMVII_Derivatives_H_

#include "ExternalInclude/Eigen/Dense"

namespace MMVII
{

/** \file  MMVII_Derivatives.h
    \brief Contains stuff necessary for computing derivatives, dont know if MMVII
     will use jets or generate code, for now it's interface for test
*/

class cInterfaceTestCam; // Interface for tested derivative 
struct cVarEpsNum;   //  More or less like Ceres
template<int N> struct cEpsNum;  // Sparse implementation of jets


/* *********************************************** */
/*                                                 */
/*              ::                                 */
/*                                                 */
/* *********************************************** */


/** Class that is can interface for computing derivative of a camera (projection fonction) 
    derived are typically jets or generated code */

class cInterfaceTestCam
{
    public :

       virtual ~cInterfaceTestCam() {}

       /// initialize parameteres from "raw" vector
       virtual void  InitFromParams(const std::vector<double> &) = 0;
       /// compute values and derivatives  and fill in  VVals and VDer
       virtual void  Compute(std::vector<double> & VVals,std::vector<std::vector<double> > & VDer)=0;
       /// Makes Nb computation (time bench ...)
       virtual void  Compute(int aNb) =0;
       /// Allocate a MMV1 object
       static cInterfaceTestCam * AllocMMV1();
};

/* ************************************** */
/*                                        */
/*           cVarEpsNum                   */
/*                                        */
/* ************************************** */

/**  cVarEpsNum functionnaly equivalent to cEpsNum, but are a tentative of optimization
    taking into account the sparseness. The implementation use a vector of non null indexes.
*/

struct cVarEpsNum
{
     // ====================== Data ===========
     private :
        // ----- static data for buffering
        static const int SzBuf = 1000;
        static double BufEps[SzBuf];
        static bool   IsInit;
     public  :
        // ----- member data
        double               mNum;  ///< Real value
        std::vector<int>     mVInd;  ///< Index of non 0 epsilon value
        std::vector<double>  mVEps;  ///< Value of non 0
     // ====================== Methods ===========

        /// Constructor for a "pure" number
        inline cVarEpsNum(double aNum) :
            mNum  (aNum)
        {
           Init();
        }
        /// Default Constructor convenient 
        inline cVarEpsNum() :
            cVarEpsNum  (0.0)
        {
        }
        /// Constructor for number + dXk
        inline cVarEpsNum(double aNum,int aK) :
            cVarEpsNum  (aNum)
        {
              AddVal(1.0,aK);
        }

        /// Add aVal * dXk
        inline void AddVal(double aVal,int aK)
        {
             mVEps.push_back(aVal);
             mVInd.push_back(aK);
        }
        /**  This constructor can be used in any "functionnal" composition 
             for exemple cos(g) =>  (g,cos(g.mNum),-sin(g.mNum)) */
        inline cVarEpsNum(const cVarEpsNum & aVEN,double aNum,double aMul) :
            mNum  (aNum),
            mVInd (aVEN.mVInd),
            mVEps (aVEN.mVEps)
        {
             for (auto & aEps : mVEps)
                 aEps *= aMul;
        }

        inline cVarEpsNum(const double & aNum,const std::vector<int>&aVInd,const std::vector<double>&aVEps) :
            mNum  (aNum),
            mVInd (aVInd),
            mVEps (aVEps)
        {
        }
        /// som
        inline cVarEpsNum operator+(const cVarEpsNum& g)  const
        {
           cVarEpsNum aRes(mNum+g.mNum);

           SetInBuf();
           g.AddToBuf();

           BuffAdd(aRes);
           g.BuffAdd(aRes);
   
           return aRes;
        }

        /// products
        inline cVarEpsNum operator*(const cVarEpsNum& g)  const
        {
           cVarEpsNum aRes(mNum*g.mNum);

           AddToBuf(g.mNum);
           g.AddToBuf(mNum);

           BuffAdd(aRes);
           g.BuffAdd(aRes);

           return aRes;
        }
        /// Diff
        cVarEpsNum operator-(const cVarEpsNum& g)  const
        {
           cVarEpsNum aRes(mNum-g.mNum);

           SetInBuf();
           g.SubToBuf();

           BuffAdd(aRes);
           g.BuffAdd(aRes);
   
           return aRes;
        }
        /// Div
        cVarEpsNum operator/(const cVarEpsNum& g)  const
        {
           cVarEpsNum aRes(mNum/g.mNum);

           AddToBuf(1.0/g.mNum);
           g.AddToBuf(-mNum/Square(g.mNum));

           BuffAdd(aRes);
           g.BuffAdd(aRes);
   
           return aRes;
        }


        ///  Ensure that BufEps is clear
        static void Init()
        {
            if (IsInit) return;  // If not first time, job already done
            IsInit = true;  //  next will not be first
            for (int aK=0 ; aK<SzBuf ; aK++)  // clear buf now
            {
                BufEps[aK] = 0.0;
            }
        }

        ///  Set the data in buf, only parse non 0,
        void SetInBuf() const
        {
           for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
           {
               BufEps[mVInd[aK] ] = mVEps[aK];
           }
        };
        ///  Add the data in buf, only parse non 0,
        void AddToBuf() const
        {
           for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
           {
               BufEps[mVInd[aK] ]    += mVEps[aK];
           }
        };
        ///  Sub the data in buf, only parse non 0,
        void SubToBuf() const
        {
           for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
           {
               BufEps[mVInd[aK] ]    -= mVEps[aK];
           }
        };
        ///  Add the data in buf with a multiplier, only parse non 0
        void AddToBuf(const double & aMul) const
        {
           for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
           {
               BufEps[mVInd[aK]] += mVEps[aK] * aMul;
           }
        };
        /**   Parse the index of non 0 value and :
                 - put the non 0 corresponding value in Res
                 - clear them in Buf
        */
        void BuffAdd(cVarEpsNum & aRes) const
        {
           // Parse index of non 0 value
           for (unsigned int aK=0 ; aK<mVEps.size(); aK++)
           {
               int IndF = mVInd[aK];
               double & aVal = BufEps[IndF];
               if ( aVal)  // if Buf is non 0
               {
                   aRes.AddVal(aVal,IndF);  // Add it in res
                   aVal = 0;  // erase it
               }
           }
        };
};

    // =========  Operation between constants and cVarEpsNum =====================

       // +++++++++++++++++++++++++++++++++++

inline cVarEpsNum operator+(const cVarEpsNum& f,const  double &  aV)
{
    cVarEpsNum aRes(f);
    aRes.mNum += aV;
    return aRes;
}
inline cVarEpsNum operator+(const double & aV,const cVarEpsNum& f) {return f+aV;}

       // -------------------------------------

inline cVarEpsNum operator-(const cVarEpsNum& f,const  double &  aV)
{
    return cVarEpsNum (f.mNum-aV,f.mVInd,f.mVEps);
}
inline cVarEpsNum operator-(const double & aV,const cVarEpsNum& f)
{
    return cVarEpsNum (aV-f.mNum,f.mVInd,f.mVEps);
}

       // *************************************

inline cVarEpsNum operator*(const cVarEpsNum& f,const  double &  aV)
{
    cVarEpsNum aRes(f);
    aRes.mNum *= aV; 
    for (auto & aEps : aRes.mVEps)
       aEps *= aV;
    return aRes;
}
inline cVarEpsNum operator*(const double & aV,const cVarEpsNum& f) {return f*aV;}

       // /////////////////////////////////////
inline cVarEpsNum operator/(const cVarEpsNum& f,const  double &  aV)
{
    return f * (1.0/aV);
}

inline cVarEpsNum operator/(const double & aV,const cVarEpsNum& f) 
{
    cVarEpsNum aRes(f);
    aRes.mNum = aV/f.mNum; 
    double aMul = - 1.0/ Square(f.mNum);
    for (auto & aEps : aRes.mVEps)
       aEps *= aMul;
    return aRes;
}

       // UNARY
inline cVarEpsNum COS(const cVarEpsNum& g)
{
    return cVarEpsNum(g,cos(g.mNum),-sin(g.mNum));
}
   // ==========


/* ************************************** */
/*                                        */
/*           cEpsNum<int N>               */
/*                                        */
/* ************************************** */

/**  cEpsNum are more or less equivalent to Ceres jets :
     it's a number + infininetely small in R^N

     Just wanted to be independant of Ceres during tests.
*/
 

template<int N> struct cEpsNum {




  // ====================== Data ===========

  double mNum;  ///< The number
  Eigen::Matrix<double, 1, N> mEps; ///< The infinitely small part

  // ====================== Methods ===========

  /// Full constructor from Num + small
  cEpsNum(const double  & aNum,const Eigen::Matrix<double, 1, N> & aEps) :
     mNum (aNum),
     mEps (aEps)
  {
  }

  /// constructor only numeric
  cEpsNum(double aNum) :
      cEpsNum (aNum,Eigen::Matrix<double, 1, N>::Zero())
  {
  }

  /// constructor Num + dXk 
  cEpsNum(const double  & aNum,int aK) :
     cEpsNum (aNum)
  {
      mEps(aK) = 1.0;
  }
  /// sometime need a def constructor
  cEpsNum() :
     cEpsNum(0.0)
  {
  }
  cVarEpsNum  ToVEN() const;
  static cEpsNum<N>  Random(double Densite)
  {
     cEpsNum<N> aRes(N*RandUnif_C());
     for (int aK=0 ; aK<N ; aK++)
     {
        if (RandUnif_0_1() < Densite)
        {
           aRes.mEps[aK] = RandUnif_C();
        }
     }
     return aRes;
  }

};

     // ====== operator + =============

template<int N> cEpsNum<N> operator+(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum + g.mNum, f.mEps + g.mEps);
}

template<int N> cEpsNum<N> operator+(const double & f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f + g.mNum, g.mEps);
}
template<int N> cEpsNum<N> operator+(const cEpsNum<N>& g,const double & f) {
  return f+g;
}
     // ====== operator - =============

template<int N> cEpsNum<N> operator-(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum - g.mNum, f.mEps - g.mEps);
}

template<int N> cEpsNum<N> operator-(const double & f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f - g.mNum, -g.mEps);
}
template<int N> cEpsNum<N> operator-(const cEpsNum<N>& g,const double & f) {
  return cEpsNum<N>(g.mNum -f, g.mEps);
}

     // ====== operator * =============

template<int N> cEpsNum<N> operator*(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum * g.mNum,  g.mNum * f.mEps + f.mNum * g.mEps);
}
template<int N> cEpsNum<N> operator*(const double & f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f*g.mNum,f*g.mEps);
}
template<int N> cEpsNum<N> operator*(const cEpsNum<N>& g,const double & f) {
  return f*g;
}

     // ====== operator / =============

template<int N> cEpsNum<N> operator/(const cEpsNum<N>& f, const cEpsNum<N>& g) {
  return cEpsNum<N>(f.mNum / g.mNum,  (f.mEps / g.mNum) - g.mEps *(f.mNum/Square(g.mNum)));
}

template<int N> cEpsNum<N> square(const cEpsNum<N>& f) {
  return cEpsNum<N>(Square(f.mNum) , 2* f.mNum* f.mEps  );
}

template<int N> cEpsNum<N> cube(const cEpsNum<N>& f) {
  return cEpsNum<N>(Cube(f.mNum) , (3*Square(f.mNum)) * f.mEps );
}

    // = conversion to Sparse

template<int N> cVarEpsNum   cEpsNum<N>::ToVEN() const
{
    cVarEpsNum aRes(mNum);
    for (int aK=0 ; aK<N ; aK++)
        if (mEps[aK])
           aRes.AddVal(mEps[aK],aK);
    return aRes;
}

       // UNARY

template<int N> cEpsNum<N> COS(const cEpsNum<N>& f) {
  return cEpsNum<N>(cos(f.mNum),-sin(f.mNum)*f.mEps);
}

// VERY VERY BORDER LINE, but we need this partial template specialization
// because cPtx<Eps> requires the tBig/tBase 
template <int Nb> class tElemNumTrait<cEpsNum<Nb> >  
{
    public :
         typedef cEpsNum<Nb> tBase;
         typedef cEpsNum<Nb> tBig ;
         typedef cEpsNum<Nb> tFloatAssoc ;
};


/* ************************************** */
/*                                        */
/*                ::                      */
/*                                        */
/* ************************************** */

inline double COS(const double & v) {return cos(v);}

/** Compute de difference between a sparse jet and a standard jet, used
    to check the consistency of the jets */

template<int N> double EpsDifference(const cEpsNum<N> & aEps,const cVarEpsNum & aVarEps)
{
    cVarEpsNum::Init();
    // take into account the standard value
    double aRes= std::abs(aEps.mNum - aVarEps.mNum);
    cEpsNum<N>  aEps2; // will be used to convert aVarEps

    for (unsigned int aK=0 ; aK<aVarEps.mVEps.size() ; aK++)
    {
         int  aInd = aVarEps.mVInd[aK];
         double aVal = aVarEps.mVEps[aK];
         if (aInd>=N)  // Is over size, do as if value was 0
            aRes += std::abs(aVal);
         else   // else put it in the non sparse representation
            aEps2.mEps[aInd] += aVal;  
    }
    // now add the difference of value under N
    for (int aK=0 ; aK<N ; aK++)
       aRes += std::abs(aEps2.mEps[aK]-aEps.mEps[aK]);

    return aRes;
}





};
#endif  //  _MMVII_Derivatives_H_
