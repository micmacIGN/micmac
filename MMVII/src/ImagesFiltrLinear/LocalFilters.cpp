#include "MMVII_Matrix.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_ImageInfoExtract.h"


/**  \file : contain implemenation for filter that are :
         -  invariant by translation
         - defined on bounded neighboorhoud

*/



namespace MMVII
{

void MakeStdIm8BIts(const cIm2D<tREAL4>& aImIn,const std::string& aName)
{
    cComputeStdDev<tREAL8>   aStdDev;
    const cDataIm2D<tREAL4> & aDIm = aImIn.DIm();

    for (const auto & aPt : aDIm)
        aStdDev.Add(aDIm.GetV(aPt));

    aStdDev = aStdDev.Normalize();

    cIm2D<tU_INT1> aImINorm(aDIm.Sz());
    for (const auto & aPt : aDIm)
    {
        tREAL8 aNormV = aStdDev.NormalizedVal(aDIm.GetV(aPt));
        aNormV = 128 + 128 * std::erf(aNormV);
        aImINorm.DIm().SetVTrunc(aPt,round_ni(aNormV));
    }
    aImINorm.DIm().ToFile(aName);
}



static const cPt2di TheP00(0,0);
static const cPt2di ThePXp(1,0);
static const cPt2di ThePXm(-1,0);
static const cPt2di ThePYp(0,1);
static const cPt2di ThePYm(0,-1);

static const cPt2di ThePXpYp( 1, 1);
static const cPt2di ThePXmYm(-1,-1);
static const cPt2di ThePXpYm( 1,-1);
static const cPt2di ThePXmYp(-1, 1);

/*  ********************************************************** *
*                                                              *
*   Functers class for image access                            *
*                                                              *
************************************************************** */

/*  To limit code duplication , the tem
*/

/// Idem cImAccesStd but add a central point P0
template<class Type> class cImAccesStd_CenteredPt
{
     public :
        cImAccesStd_CenteredPt(const cDataIm2D<Type> & aIm,const cPt2di & aP0) : mIm (aIm), mP0(aP0)  {}
	inline Type operator() (const cPt2di & aP) const {return mIm.GetV(aP+mP0);}
	typedef Type  tVal;
	typedef typename tBaseNumTrait<Type>::tBase tBase;
   
     private :
        const cDataIm2D<Type>&  mIm;
        cPt2di                  mP0;
};



/// Idem cImAccesProj but add a central point P0
template<class Type> class cImAccesProj_CenteredPt
{
     public :
        cImAccesProj_CenteredPt(const cDataIm2D<Type> & aIm,const cPt2di& aP0) : mIm (aIm) , mP0 (aP0) {}
	inline Type operator() (const cPt2di & aP) const {return mIm.GetV(mIm.Proj(aP+mP0));}
	typedef Type  tVal;
	typedef typename tBaseNumTrait<Type>::tBase tBase;
     private :
        const cDataIm2D<Type>&  mIm;
        cPt2di                  mP0;
};

template<class Type,class tFctrInside,class tFctrBorder> 
   cIm2D<Type> GenLocal_Filter
               (
                   const cIm2D<Type>&   anImIn,
                   const  tFctrInside & aFctrI,
                   const  tFctrBorder & aFctBrd,
                   int aSzBrd
               )
{
    const cDataIm2D<Type> & aDImIn(anImIn.DIm());
    cIm2D<Type> aRes(aDImIn.Sz());

    for (const auto & aPt : aDImIn.Interior(aSzBrd))
    {
        typename tFctrInside::tFuncter   aImAccessStd (aDImIn,aPt);
        aRes.DIm().SetV(aPt,aFctrI(aImAccessStd));
    }

    if (aSzBrd)
    {
       for (const auto & aPt : aDImIn.Border(aSzBrd) )
       {
           typename tFctrBorder::tFuncter   aImAccessBrd (aDImIn,aPt);
           aRes.DIm().SetV(aPt,aFctBrd(aImAccessBrd));
       }
    }

    return aRes;
}


/*   Not used for now
/// Make a functor for an image, giving standard access to its value
template<class Type> class cImAccesStd
{
     public :
        cImAccesStd(const cDataIm2D<Type> & aIm) : mIm (aIm) {}
	inline Type operator() (const cPt2di & aP) const {return mIm.GetV(aP);}
	typedef Type  tVal;
	typedef typename tBaseNumTrait<Type>::tBase tBase;

     private :
        const cDataIm2D<Type>&  mIm;
};
/// Make a "safe" functor for an image, giving access to its  value of projected point
template<class Type> class cImAccesProj
{
     public :
        cImAccesProj(const cDataIm2D<Type> & aIm) : mIm (aIm) {}
	inline Type operator() (const cPt2di & aP) const {return mIm.GetV(mIm.Proj(aP));}
	typedef Type  tVal;
	typedef typename tBaseNumTrait<Type>::tBase tBase;
     private :
        const cDataIm2D<Type>&  mIm;
};
*/


/* ****************************************************** */
/*                                                        */
/*                     CourbTgt                           */
/*                                                        */
/* ****************************************************** */

/*
*/

/**  Fct for computing the CourbTgt filter */

template <class tFunc>  class  cCourbTgtOfFctr
{
    public :
      typedef typename tFunc::tBase tBase;
      typedef typename tFunc::tVal  tVal ;
      typedef tFunc    tFuncter;
  
      cCourbTgtOfFctr(tREAL8 aExp) : mExp(aExp) {}

      tREAL8 operator() (const tFunc & aFunc) const
      {
         //  compute values that will be used several times
         tVal  a2V00  = aFunc(TheP00) * 2;
         tVal  aVxP1 = aFunc(ThePXp);
         tVal  aVxM1 = aFunc(ThePXm);
         tVal  aVyP1 = aFunc(ThePYp);
         tVal  aVyM1 = aFunc(ThePYm);

         //  compute gradient and its norm
         tBase  aGx = (aVxP1 - aVxM1) / 2;
         tBase  aGy = (aVyP1 - aVyM1) / 2;
         tREAL8  aG2 = Square(aGx)+Square(aGy);

         if (aG2 == 0)
         {
            return 0;
         }

         // compute hessian  
         tVal  aD2xx     = (aVxP1+aVxM1- a2V00);
         tVal  aD2yy     = (aVyP1+aVyM1- a2V00);
         tVal  aD2xy     = ( aFunc(ThePXpYp) +  aFunc(ThePXmYm) - aFunc(ThePXmYp)-aFunc(ThePXpYm)) / 4;

         //  apply the hessian to direction orthonal to gradient (= direction of the level curve)
         return   ( (aD2xx * aGy) * aGy - 2 * (aD2xy * aGx) * aGy + (aD2yy * aGx) * aGx) / std::pow(aG2,mExp);
      }
    private :
      tREAL8 mExp;
};



template<class Type> cIm2D<Type> CourbTgt(cIm2D<Type>  aImIn)
{
   return GenLocal_Filter
          (
              aImIn,
              cCourbTgtOfFctr<cImAccesStd_CenteredPt<Type>>(0.5),
              cCourbTgtOfFctr<cImAccesProj_CenteredPt<Type>>(0.5),
              1
          );
}
template<class Type> void SelfCourbTgt(cIm2D<Type> anIm)
{
   cIm2D<Type> aRes = CourbTgt(anIm);
   aRes.DIm().DupIn(anIm.DIm());
}

/* ****************************************************** */
/*                                                        */
/*                        Lapl                            */
/*                                                        */
/* ****************************************************** */

template <class tFunc>  class  cLaplOfFctr
{
    public :
      typedef typename tFunc::tBase tBase;
      typedef typename tFunc::tVal  tVal ;
      typedef tFunc    tFuncter;

      tREAL8 operator() (const tFunc & aFunc) const
      {
         return 4 * aFunc(TheP00) - (aFunc(ThePXp) + aFunc(ThePXm) + aFunc(ThePYp) + aFunc(ThePYm));
      }
};

template<class Type> cIm2D<Type> Lapl(cIm2D<Type>  aImIn)
{
   return GenLocal_Filter
          (
              aImIn,
              cLaplOfFctr<cImAccesStd_CenteredPt<Type>>(),
              cLaplOfFctr<cImAccesProj_CenteredPt<Type>>(),
              1
          );
}

/* ****************************************************** */
/*                                                        */
/*                     MajLab                             */
/*                                                        */
/* ****************************************************** */

template <class tFunc>  class  cMajLabOfFctr
{
    public :
      typedef typename tFunc::tBase tBase;
      typedef typename tFunc::tVal  tVal ;
      typedef tFunc    tFuncter;
      
      cMajLabOfFctr(const cBox2di & aBox,int aVMin,int aVMax)  :
           mBox   (aBox),
           mVMin  (aVMin),
           mVMax  (aVMax),
           mHisto (1+mVMax-mVMin)
      {
      }

      tREAL8 operator() (const tFunc & aFunc) const
      {
         // reset histogramm to 0
         for (auto & aH : mHisto)
             aH=0;
         // increment histogramm for each label met
         for (const auto & aP : cRect2(mBox)) //  cRect2::BoxWindow(mSz))
             mHisto.at(aFunc(aP)-mVMin) ++;

         cWhichMax<int,int> aWMax;
         for (size_t aK=0 ; aK<mHisto.size() ; aK++)
             aWMax.Add(aK,mHisto.at(aK));

         return aWMax.IndexExtre()+mVMin;
      }
    private :
      cBox2di           mBox;
      int               mVMin;
      int               mVMax;
      mutable std::vector<int>  mHisto;
};

template<class Type> cIm2D<Type> LabMaj(cIm2D<Type>  aImIn,const cBox2di & aBox)
{
   auto [aMin,aMax] = ValExtre(aImIn);

   return GenLocal_Filter
          (
              aImIn,
              cMajLabOfFctr<cImAccesStd_CenteredPt<Type>>(aBox,aMin,aMax),
              cMajLabOfFctr<cImAccesProj_CenteredPt<Type>>(aBox,aMin,aMax),
              std::max(NormInf(aBox.P0()),NormInf(aBox.P1()))
          );
}
template <class Type> void SelfLabMaj(cIm2D<Type> anImIn,const cBox2di &  aBox)
{
   cIm2D<Type> aRes = LabMaj(anImIn,aBox);
   aRes.DIm().DupIn(anImIn.DIm());
}


template cIm2D<tINT2> LabMaj(cIm2D<tINT2>  aImIn,const cBox2di &);
template void SelfLabMaj(cIm2D<tINT2> anImIn,const cBox2di &  aBox);


/*
*/

    //=================  Global Solution ==================


/*  Apparently no longer used,  maintain it it if some regret ....
cIm2D<tREAL4> Lapl(cIm2D<tREAL4> aImIn)
{
    cIm2D<tREAL4> aRes(aImIn.DIm().Sz());

    Im2D<tREAL4,tREAL8>  aV1In  = cMMV1_Conv<tREAL4>::ImToMMV1(aImIn.DIm());
    Im2D<tREAL4,tREAL8>  aV1Res = cMMV1_Conv<tREAL4>::ImToMMV1( aRes.DIm());

    ELISE-COPY(aV1In.all_pts(),Laplacien(aV1In.in_proj()),aV1Res.out());

    return aRes;
}
*/



#define INSTANTIATE_TRAIT_AIME(TYPE)\
template cIm2D<TYPE> CourbTgt(cIm2D<TYPE> aImIn);\
template cIm2D<TYPE> Lapl(cIm2D<TYPE> aImIn);\
template void SelfCourbTgt(cIm2D<TYPE> aImIn);

INSTANTIATE_TRAIT_AIME(tREAL4)
INSTANTIATE_TRAIT_AIME(tINT2)



};
