#ifndef  _MMVII_Tpl_Images_H_
#define  _MMVII_Tpl_Images_H_

#include "MMVII_Matrix.h"

namespace MMVII
{


/** \file MMVII_Tpl_Images.h
    \brief  Implemantation of operators on images, matrix.  This is rather basic,
     more general/functionall are (will be) in  "ToCome.h" ;-)

*/

/*
    (Add-Diff)ImageInPlace    I1 = I2  (+-) I1  
*/


    // -------------------------- Soustraction -------------------------
template<class T1,class T2,class T3,int Dim>   // I1 = I2-I1
   void DiffImageInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const cDataTypedIm<T3,Dim> & aI3);
template<class T1,class T2,class T3>   // return I2-I3
   cIm2D<T1> DiffImage(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const cIm2D<T3> & aI3);
template<class T2,class T3>   cIm2D<T2> operator - (const cIm2D<T2> & aI2,const cIm2D<T3> & aI3); // return I2-I3
template<class T2>   cDenseMatrix<T2> operator - (const cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3) ; // return I2-I3
template<class T2>   cDenseVect<T2> operator - (const cDenseVect<T2> & aI2,const cDenseVect<T2> & aI3) ; // return I2-I3

    // -------------------------- Addition -------------------------
template<class T1,class T2,class T3,int Dim>   // I1 = I2 +I3
   void AddImageInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const cDataTypedIm<T3,Dim> & aI3);
template<class T1,class T2,int Dim>   // I1 += I2
   void AddIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2);
template<class T1,class T2,class T3>  // return I2 + I3
   cIm2D<T1> AddImage(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const cIm2D<T3> & aI3);
template<class T2,class T3>   cIm2D<T2> operator + (const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)  ; // return I2 + I3
template<class T2>   cDenseMatrix<T2> operator + (const cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3) ; // return I2 + I3
template<class T2>   cDenseVect<T2> operator + (const cDenseVect<T2> & aI2,const cDenseVect<T2> & aI3) ; // return I2-I3

    // -------------------------- Mul Cste -------------------------
template<class T1,class T2,class T3,int Dim>     // I1 = I2 * V3 
   void MulImageCsteInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const T3 & aV3);
template<class T2,class T3,int Dim>     //  I2 *= V3 
   void SelfMulImageCsteInPlace(cDataTypedIm<T2,Dim> & aI2,const T3 & aV3);
template<class T1,class T2,class T3>  
   cIm2D<T1> MulImageCste(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const  T3 & aV3); // return I2 * V3
template<class T2,class T3>   cIm2D<T2> operator * (const cIm2D<T2> & aI2,const  T3 & aV3)  ;
template<class T2,class T3>   cDenseMatrix<T2> operator * (const cDenseMatrix<T2> & aI2,const  T3 & aV3)   ;
template<class T2,class T3>   cDenseMatrix<T2> operator * (const  T3 & aV3,const cDenseMatrix<T2> & aI2) ;


    // -------------------------- Div Cste -------------------------
template<class T1,class T2,int Dim>  void DivCsteIn(cDataTypedIm<T1,Dim> & aI1,const T2 & aV2); // I1 /= V2
    // -------------------------- Copy -------------------------
template<class T1,class T2,int Dim>  void CopyIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2); // I1 = I2
    // -------------------------- Convert -------------------------

template<class T1,class T2>  cIm2D<T1>  Convert(T1*,const cDataIm2D<T2>&);  // T1 => trick to easy force type


       //===========   Substraction ===========

template<class T1,class T2,class T3,int Dim>  
   void DiffImageInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const cDataTypedIm<T3,Dim> & aI3)
{
    aI1.AssertSameArea(aI2); 
    aI1.AssertSameArea(aI3);

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) - aI3.GetRDL(aK) ;
}

template<class T1,class T2,class T3>  
   cIm2D<T1> DiffImage(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)
{
     cIm2D<T1>  aI1(aI2.DIm().P0(),aI2.DIm().P1());
     DiffImageInPlace(aI1.DIm(),aI2.DIm(),aI3.DIm());
     return aI1;
}

template<class T1,class T2,class T3>  
   cIm1D<T1> DiffImage(T1* /*Type specifier*/ ,const cIm1D<T2> & aI2,const cIm1D<T3> & aI3)
{
     cIm1D<T1>  aI1(aI2.DIm().X0(),aI2.DIm().X1());
     DiffImageInPlace(aI1.DIm(),aI2.DIm(),aI3.DIm());
     return aI1;
}

template<class T1,class T2,int Dim>  
   void DiffIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) -= aI2.GetRDL(aK) ;
}


template<class T2,class T3>   cIm2D<T2> operator - (const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)  
{
   return DiffImage((T2 *)nullptr,aI2,aI3);
}
template<class T2,class T3>   cIm1D<T2> operator - (const cIm1D<T2> & aI2,const cIm1D<T3> & aI3)  
{
   return DiffImage((T2 *)nullptr,aI2,aI3);
}


template<class T2>   cDenseMatrix<T2> operator - (const cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3)  
{
    return cDenseMatrix<T2>(aI2.Im()-aI3.Im());
}
template<class T2>   cDenseVect<T2> operator - (const cDenseVect<T2> & aI2,const cDenseVect<T2> & aI3)  
{
    return cDenseVect<T2>(aI2.Im()-aI3.Im());
}

template<class T2>   cDenseMatrix<T2> operator -= (cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3) 
{
	DiffIn(aI2.DIm(),aI3.DIm());
	return aI2;
}
template<class T2>   cDenseVect<T2> operator -= (cDenseVect<T2> & aI2,const cDenseVect<T2> & aI3) 
{
	DiffIn(aI2.DIm(),aI3.DIm());
	return aI2;
}

       //===========   Addition ===========

template<class T1,class T2,class T3,int Dim>  
   void AddImageInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const cDataTypedIm<T3,Dim> & aI3)
{
    aI1.AssertSameArea(aI2); 
    aI1.AssertSameArea(aI3);

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) + aI3.GetRDL(aK) ;
}

template<class T1,class T2,int Dim>  
   void AddIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) += aI2.GetRDL(aK) ;
}

template<class T1,class T2,class T3>  
   cIm2D<T1> AddImage(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)
{
     cIm2D<T1>  aI1(aI2.DIm().P0(),aI2.DIm().P1());
     AddImageInPlace(aI1.DIm(),aI2.DIm(),aI3.DIm());
     return aI1;
}

template<class T1,class T2,class T3>  
   cIm1D<T1> AddImage(T1* /*Type specifier*/ ,const cIm1D<T2> & aI2,const cIm1D<T3> & aI3)
{
     cIm1D<T1>  aI1(aI2.DIm().X0(),aI2.DIm().X1());
     AddImageInPlace(aI1.DIm(),aI2.DIm(),aI3.DIm());
     return aI1;
}

template<class T2,class T3>   cIm2D<T2> operator + (const cIm2D<T2> & aI2,const cIm2D<T3> & aI3)  
{
   return AddImage((T2 *)nullptr,aI2,aI3);
}
template<class T2,class T3>   cIm1D<T2> operator + (const cIm1D<T2> & aI2,const cIm1D<T3> & aI3)  
{
   return AddImage((T2 *)nullptr,aI2,aI3);
}

template<class T2>   cDenseMatrix<T2> operator + (const cDenseMatrix<T2> & aI2,const cDenseMatrix<T2> & aI3)  
{
    return cDenseMatrix<T2>(aI2.Im()+aI3.Im());
}
template<class T2>   cDenseVect<T2> operator + (const cDenseVect<T2> & aI2,const cDenseVect<T2> & aI3)  
{
    return cDenseVect<T2>(aI2.Im()+aI3.Im());
}

template<class T2>   cDenseVect<T2> & operator += (cDenseVect<T2> & aI2,const cDenseVect<T2> & aI3)  
{
      AddIn(aI2.DIm(),aI3.DIm());
      return aI2;
}

       //===========   MulCste ===========

template<class T1,class T2,class T3,int Dim>  
   void MulImageCsteInPlace(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2,const T3 & aV3)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) * aV3;
}

template<class T2,class T3,int Dim>  
   void SelfMulImageCsteInPlace(cDataTypedIm<T2,Dim> & aI2,const T3 & aV3)
{
    for (int aK=0 ; aK<aI2.NbElem() ; aK++)
        aI2.GetRDL(aK) *=  aV3;
}
template<class T2,class T3,int Dim>  
   void operator *=(cDataTypedIm<T2,Dim> & aI2,const T3 & aV3) {SelfMulImageCsteInPlace(aI2,aV3); }

/*
template<class T2,class T3> void operator *= (cDenseVect<T2> & aV2,const T3 & aV3) { aV2.DIm() *= aV3; }
template<class T2,class T3> void operator *= (cDenseMatrix<T2> & aV2,const T3 & aV3) { aV2.DIm() *= aV3; }
*/


template<class T1,class T2,class T3>  
   cIm2D<T1> MulImageCste(T1* /*Type specifier*/ ,const cIm2D<T2> & aI2,const  T3 & aV3)
{
     cIm2D<T1>  aI1(aI2.DIm().P0(),aI2.DIm().P1());
     MulImageCsteInPlace(aI1.DIm(),aI2.DIm(),aV3);
     return aI1;
}

template<class T2,class T3>   cIm2D<T2> operator * (const cIm2D<T2> & aI2,const  T3 & aV3)  
{
   return MulImageCste((T2 *)nullptr,aI2,aV3);
}

template<class T2,class T3>   cDenseMatrix<T2> operator * (const cDenseMatrix<T2> & aI2,const  T3 & aV3)  
{
    return cDenseMatrix<T2>(aI2.Im()*aV3);
}
template<class T2,class T3>   cDenseMatrix<T2> operator * (const  T3 & aV3,const cDenseMatrix<T2> & aI2)
{
    return cDenseMatrix<T2>(aI2.Im()*aV3);
}
       //===========   DivCste ===========

template<class T1,class T2,int Dim>  
   void DivCsteIn(cDataTypedIm<T1,Dim> & aI1,const T2 & aV2)
{
    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) /= aV2;
}

       //===========   Copy, +=  ===========

template<class T1,class T2,int Dim>  
   void CopyIn(cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) = aI2.GetRDL(aK) ;
}

template<class T1,class T2>  cIm2D<T1>  Convert(T1*,const cDataIm2D<T2>& aDIm2)
{
     cIm2D<T1> aIm1(aDIm2.Sz());
     CopyIn(aIm1.DIm(),aDIm2);
     return aIm1;
}


template<class T1,class T2,int Dim>  
   void WeightedAddIn(cDataTypedIm<T1,Dim> & aI1,const T2 & aV,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 

    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aI1.GetRDL(aK) += aV*aI2.GetRDL(aK) ;
}


/*****************************************************/
/*                                                   */
/*          AGREGATION                               */
/*                                                   */
/*****************************************************/

template<class T1,class T2,int Dim>  
   double DotProduct(const cDataTypedIm<T1,Dim> & aI1,const cDataTypedIm<T2,Dim> & aI2)
{
    aI1.AssertSameArea(aI2); 
    double aSom = 0.0;
    for (int aK=0 ; aK<aI1.NbElem() ; aK++)
        aSom += aI1.GetRDL(aK) * aI2.GetRDL(aK) ;

    return aSom;
}

template<class T1,int Dim>
   void GetBounds(T1 & aVMin,T1& aVMax,const cDataTypedIm<T1,Dim> & aIm)
{
    aVMin = aVMax = aIm.GetRDL(0);
    for (int aK=1 ; aK<aIm.NbElem() ; aK++)
    {
        const T1 & aV = aIm.GetRDL(aK);
        aVMin = std::min(aV,aVMin);
        aVMax = std::max(aV,aVMax);
    }
}

template<class TOper,class T1,int Dim>
   cPtxd<int,Dim> WhichMinMax(const TOper & Op, const cDataTypedIm<T1,Dim> & aIm)
{
    int aKMax = 0;
    T1 aVMax = aIm.GetRDL(aKMax);
    for (int aK=1 ; aK<aIm.NbElem() ; aK++)
    {
        const T1 & aV = aIm.GetRDL(aK);
        // if (aV > aVMax)
        if (Op(aV,aVMax))
        {
            aVMax = aV;
            aKMax = aK;
        }
    }
    return aIm.FromIndexeLinear(aKMax);
}

template<class T1,int Dim>
   cPtxd<int,Dim> WhichMax(const cDataTypedIm<T1,Dim> & aIm)
{
     return WhichMinMax([](const T1&aV1,const T1&aV2){return aV1>aV2;},aIm);
}


template<class T1> typename cDataIm1D<T1>::tBase cDataIm1D<T1>::SomInterv(int aX0,int aX1) const
{
    tPB::AssertInside(aX0);
    tPB::AssertInside(aX1-1);
    tBase aRes = 0;
    for (int aX=aX0; aX<aX1; aX++)
        aRes += mRawData1D[aX];
    return aRes;
}

template<class T1> tREAL8  cDataIm1D<T1>::AvgInterv(int aX0,int aX1) const
{
   MMVII_INTERNAL_ASSERT_medium((aX1>aX0),"Bad lim in Avg");
   return SomInterv(aX0,aX1) / double(aX1-aX0);
}

/*****************************************************/
/*                                                   */
/*          NORMALIZATION                            */
/*                                                   */
/*****************************************************/

template<class T,int Dim>
   void NormalizedAvgDev(cDataTypedIm<T,Dim> & aIm,tREAL8 aEpsilon)
{
    cComputeStdDev<tREAL8>  aCSD;
    for (int aK=0 ; aK<aIm.NbElem() ; aK++)
    {
        aCSD.Add(aIm.GetRDL(aK),1.0);
    }
    aCSD = aCSD.Normalize(aEpsilon);
    for (int aK=0 ; aK<aIm.NbElem() ; aK++)
    {
        T& aV = aIm.GetRDL(aK);
        aV = aCSD.NormalizedVal(aV);
    }
}

template<class T>  cIm2D<T> NormalizedAvgDev(const cIm2D<T> & aIm,tREAL8 aEpsilon)
{
    cIm2D<T> aRes = aIm.Dup();
    NormalizedAvgDev(aRes.DIm(),aEpsilon);
    return aRes;
}


template <class TFonc,class TMasq>
         bool BornesFonc
              (
                   int & aPxMin,
                   int & aPxMax,
                   cIm2D<TFonc> aIFonc,
                   cIm2D<TMasq> * aPtrIM,
                   int aNbDecim,
                   double aProp,
                   double aRatio
              )
{
      aIFonc =  aIFonc.Decimate(aNbDecim);
      cIm2D<TMasq> *  aPtrIMR = nullptr;

      cIm2D<TMasq>    aIMR(cPt2di(1,1));
      if (aPtrIM)
      {
           aIMR = aPtrIM->Decimate(aNbDecim);
           aPtrIM = & aIMR;
      }

      std::vector<double> aVPx;
      for (const auto & aP : aIFonc.DIm())
      {
          if ((aPtrIMR ==nullptr) || (aPtrIMR->DIm().GetV(aP) != 0))
          {
             aVPx.push_back(aIFonc.DIm().GetV(aP));
          }
      }
      if (aVPx.size() > 0)
      {
         aPxMin = round_ni(aRatio * NC_KthVal(aVPx,  aProp));
         aPxMax = round_ni(aRatio * NC_KthVal(aVPx,1-aProp));
         return true;
      }
      else
      {
         aPxMin = 0;
         aPxMax = 0;
         return false;
      }
}

/*
template <const int Dim> std::vector< cPtxd<tREAL8,Dim> > ToR(const std::vector< cPtxd<int,Dim> > &aVPtI)
{
	StdOut() << "KKKKKKKKKKKKKKKKKKKKKKk\n";
    std::vector< cPtxd<tREAL8,Dim> > aRes;
    std::transform(aVPtI.begin(),aVPtI.end(),aRes.begin(),[](auto aPtI){return ToR<int>(aPtI);});

	StdOut() << "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n";
    return aRes;
}
*/



};

#endif  //  _MMVII_Tpl_Images_H_
