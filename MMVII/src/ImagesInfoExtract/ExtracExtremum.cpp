#include "include/MMVII_all.h"


namespace MMVII
{


/*
template <class Type,class FDist>  std::vector<Type>  SparseOrder(const std::vector<Type> & aV,const std::vector<Type>& aRepuls0)
{
     int aNb=aV.size();
     std::vector<Type> aRes = aRepuls0;
     std::vector<bool> aVSel(aNb,false);

     for (int aNbIter=0 ; aNbIter<aNb ; aNbIter++)
     {
         double aDMax = 0.0;
         for (
         
     }
} 
*/

// std::vector<cPt2di> SparsedVectOfRadius(const double & aR0,const double & aR1) // > R0 et <= R1

std::vector<cPt2di> SortedVectOfRadius(const double & aR0,const double & aR1) // > R0 et <= R1
{
    std::vector<cPt2di> aRes;

    double aR02 = Square(aR0);
    double aR12 = Square(aR1);
    
    for (const auto & aP : cRect2::BoxWindow(round_up(aR1)))
    {
        double aR2 = SqN2(aP);
        if ((aR2>aR02) && (aR2<=aR12))
           aRes.push_back(aP);
    }
    std::sort(aRes.begin(),aRes.end(),CmpN2<int,2>);

    return aRes;
}


/// Compute etremum point of an image
/** 
     This class computes extremum of an image, also it may seems
     quite complex for a basic definition, it deals with three specification :
        * makes comparison using a strict order (cmp Val, then x, then y )
        * be efficient using a maximum of "cuts", so order the comparison to maximize
          the probability that we know asap that we are not a max/min
        * be relarively easy to read
*/

template <class Type> class cComputeExtrem1Im
{
     public :

         cComputeExtrem1Im(const cDataIm2D<Type>  &anIm,cResultExtremum &,double aRadius);
         void ComputeRes();
     protected :
        void TestIsExtre1();

        inline bool IsImCSupCurP(const cPt2di & aDP) const
        {
             Type aV = mDIM.GetV(mPCur+aDP);
             // Compare values
             if (aV>mVCur) return true;
             if (aV<mVCur) return false;
             // if equals compare x
             if (aDP.x()>0) return true;
             if (aDP.x()<0) return false;
             // if equals compare y
             return aDP.y() >  0;
        }

        const cDataIm2D<Type>  & mDIM;  ///< Image analysed
        cResultExtremum &    mRes;      ///< Storing of results
        double               mRadius;   ///< Size of neighboorhood
        std::vector<cPt2di>  mSortedNeigh;  ///< Neighooring point sorted by growing distance
        cPt2di               mPCur;      ///< Current point explored
        Type                 mVCur;      ///< Value of current point
};

template <class Type> class cComputeExtrem3Im : public cComputeExtrem1Im<Type>
{
     public :
         cComputeExtrem3Im
         (
               const cDataIm2D<Type>  &anIUp,
               const cDataIm2D<Type>  &anImC,
               const cDataIm2D<Type>  &anIBottom,
               cResultExtremum &,
               double aRadius
         );
        void ComputeRes();
     protected :
        typedef cComputeExtrem1Im<Type> t1Im;
        void TestIsExtre3();

        ///< Test for decentred images
         inline bool IsImUpSupCurP(const cPt2di & aDP) const
         {
             return  (mDIMUp.GetV(t1Im::mPCur+aDP)>=t1Im::mVCur) ; 
         }
         inline bool IsImBotSupCurP(const cPt2di & aDP) const
         {
             return  (mDIMBot.GetV(t1Im::mPCur+aDP)>t1Im::mVCur) ; 
         }
         
         const cDataIm2D<Type>  & mDIMUp; ///< "Up" Image in the pyramid
         const cDataIm2D<Type>  & mDIMBot; ///< "Bottom" Image in the pyramid
};


/*   ================================= */
/*         cResultExtremum             */
/*   ================================= */

void cResultExtremum::Clear()
{
    mPtsMin.clear();
    mPtsMax.clear();
}

/*   ================================= */
/*         cComputeExtrem1Im           */
/*   ================================= */

template <class Type> void cComputeExtrem1Im<Type>::TestIsExtre1()
{
     mVCur = mDIM.GetV(mPCur);
     // Compare with left neighboor ,  after we know if it has to be a min or a max
     bool IsMin = IsImCSupCurP(cPt2di(-1,0));

     //   Now we know that if any comparison with a neighboor is not coherent with
     // the first one, it cannot be an extremum

     if (IsImCSupCurP(cPt2di(1,0)) != IsMin) return;
     if (IsImCSupCurP(cPt2di(0,1)) != IsMin) return;
     if (IsImCSupCurP(cPt2di(0,-1)) != IsMin) return;
 
     for (const auto & aDP : mSortedNeigh)
         if (IsImCSupCurP(aDP) != IsMin) 
            return;
    if (IsMin)
       mRes.mPtsMin.push_back(mPCur);
    else
       mRes.mPtsMax.push_back(mPCur);
}



template <class Type> 
    cComputeExtrem1Im<Type>::cComputeExtrem1Im(const cDataIm2D<Type>  &anIm,cResultExtremum & aRes,double aRadius) :
       mDIM  (anIm),
       mRes  (aRes),
       mRadius       (aRadius),
       mSortedNeigh  (SortedVectOfRadius(1.01,mRadius))
{
}

template <class Type> void cComputeExtrem1Im<Type>::ComputeRes()
{
    mRes.Clear();
    cPt2di aSzW = cPt2di::PCste(round_up(mRadius));
    cRect2 aRectInt (mDIM.Dilate(-aSzW));
    
    for (const auto & aPCur : aRectInt)
    {
         mPCur = aPCur;
         TestIsExtre1();
    }
}

template <class Type> void ExtractExtremum1(const cDataIm2D<Type>  &anIm,cResultExtremum & aRes,double aRadius)
{
    cComputeExtrem1Im<Type> aCEI(anIm,aRes,aRadius);
    aCEI.ComputeRes();
}


/*   ================================= */
/*         cComputeExtrem1Im           */
/*   ================================= */

template <class Type> void cComputeExtrem3Im<Type>::TestIsExtre3()
{
     t1Im::mVCur = t1Im::mDIM.GetV(t1Im::mPCur);
     // Compare with left neighboor ,  after we know if it has to be a min or a max
     bool IsMin = t1Im::IsImCSupCurP(cPt2di(-1,0));

     //   Now we know that if any comparison with a neighboor is not coherent with
     // the first one, it cannot be an extremum

     if (t1Im::IsImCSupCurP(cPt2di(1,0)) != IsMin) return;
     if (t1Im::IsImCSupCurP(cPt2di(0,1)) != IsMin) return;
     if (t1Im::IsImCSupCurP(cPt2di(0,-1)) != IsMin) return;
   
     // Test vertical 
     if (IsImUpSupCurP (cPt2di(0,0)) != IsMin) return;
     if (IsImBotSupCurP(cPt2di(0,0)) != IsMin) return;

     // Test first neighboor
     if (IsImUpSupCurP (cPt2di(-1,0)) != IsMin) return;
     if (IsImBotSupCurP(cPt2di(-1,0)) != IsMin) return;

     // Test 3 neigh
     if (IsImUpSupCurP (cPt2di( 1, 0)) != IsMin) return;
     if (IsImBotSupCurP(cPt2di( 1, 0)) != IsMin) return;
     if (IsImUpSupCurP (cPt2di( 0, 1)) != IsMin) return;
     if (IsImBotSupCurP(cPt2di( 0, 1)) != IsMin) return;
     if (IsImUpSupCurP (cPt2di( 0,-1)) != IsMin) return;
     if (IsImBotSupCurP(cPt2di( 0,-1)) != IsMin) return;

 
     for (const auto & aDP : t1Im::mSortedNeigh)
     {
         if (t1Im::IsImCSupCurP  (aDP) != IsMin) return;
         if (IsImUpSupCurP (aDP) != IsMin) return;
         if (IsImBotSupCurP(aDP) != IsMin) return;
     }
     if (IsMin)
        t1Im::mRes.mPtsMin.push_back(t1Im::mPCur);
     else
        t1Im::mRes.mPtsMax.push_back(t1Im::mPCur);
}

template <class Type> 
    cComputeExtrem3Im<Type>::cComputeExtrem3Im
    (
               const cDataIm2D<Type>  &anIUp,
               const cDataIm2D<Type>  &anImC,
               const cDataIm2D<Type>  &anIBottom,
               cResultExtremum & aRes,
               double aRadius
    ) :
      cComputeExtrem1Im<Type>(anImC,aRes,aRadius),
      mDIMUp   (anIUp),
      mDIMBot  (anIBottom)
       
{
    anImC.AssertSameArea(anIUp);
    anImC.AssertSameArea(anIBottom);
}

template <class Type> void cComputeExtrem3Im<Type>::ComputeRes()
{
    t1Im::mRes.Clear();
    cPt2di aSzW = cPt2di::PCste(round_up(t1Im::mRadius));
    cRect2 aRectInt (t1Im::mDIM.Dilate(-aSzW));
    
    for (const auto & aPCur : aRectInt)
    {
         t1Im::mPCur = aPCur;
         TestIsExtre3();
    }
}

template <class Type> void ExtractExtremum3
                           (
                                const cDataIm2D<Type>  &anImUp,
                                const cDataIm2D<Type>  &anImC,
                                const cDataIm2D<Type>  &anImBot,
                                cResultExtremum & aRes,
                                double aRadius
                           )
{
    cComputeExtrem3Im<Type> aCEI(anImUp,anImC,anImBot,aRes,aRadius);
    aCEI.ComputeRes();
}


/* ========================== */
/*     cDataGenUnTypedIm      */
/* ========================== */


#define MACRO_INSTANTIATE_ExtractExtremum(Type)\
template  class cComputeExtrem1Im<Type>;\
template  class cComputeExtrem3Im<Type>;\
template void ExtractExtremum1(const cDataIm2D<Type>  &anIm,cResultExtremum & aRes,double aRadius);\
template void ExtractExtremum3(const cDataIm2D<Type>  &anImUp,const cDataIm2D<Type>  &anIm, const cDataIm2D<Type>  &anImDown,cResultExtremum & aRes,double aRadius);\




MACRO_INSTANTIATE_ExtractExtremum(tREAL4);
/*
MACRO_INSTANTIATE_ExpoFilter(tREAL8);
MACRO_INSTANTIATE_ExpoFilter(tREAL16);
MACRO_INSTANTIATE_ExpoFilter(tINT4);
MACRO_INSTANTIATE_ExpoFilter(tINT2);
*/


};
