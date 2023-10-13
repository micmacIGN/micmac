#include "MMVII_2Include_Tiling.h"

namespace MMVII
{



/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */


template <const int Dim> tREAL8  cTilingIndex<Dim>::ComputeStep(const tRBox & aBox, int aNbCase)
{
    const tRPt &  aSz = aBox.Sz();

    tREAL8 aElemVol = MulCoord(aSz)/aNbCase;    // elementary volume of each elem box 
    tREAL8 aStep = pow(aElemVol,1/tREAL8(Dim));

    tIPt aPNb =  Pt_round_up(aSz/aStep);  // How many shoudl be required for each dim

    tRPt aPStep = DivCByC(aSz,ToR(aPNb)); // Step for each dim

    return MinAbsCoord(aPStep);
}


template <const int Dim>  cTilingIndex<Dim>::cTilingIndex(const tRBox & aBox,bool OkOut, int aNbCase) :
	mRBoxIn  (aBox),
	mOkOut   (OkOut),
        mNbCase  (aNbCase),
	mStep    (ComputeStep(aBox,aNbCase)),
	mSzI     (Pt_round_up(aBox.Sz()/mStep)),
        mIBoxIn  (  tIBox(tIPt::PCste(0),mSzI+tIPt::PCste(2)))
{
    // StdOut()  <<  " -- SSSS=" << mStep << " " << mSzI << std::endl;
    // StdOut()  <<  mIBoxIn.P0()  << mIBoxIn.P1() << std::endl;
    // StdOut()  <<  mIBoxIn.Proj(cPt2di(1,1))   << mIBoxIn.Proj(cPt2di(-3,-3))  << mIBoxIn.Proj(cPt2di(10,10)) << std::endl;
/*
*/
}

template <const int Dim> void cTilingIndex<Dim>::AssertInside(const tRPt & aPt) const
{
    MMVII_INTERNAL_ASSERT_tiny(mOkOut || mRBoxIn.Inside(aPt),"cTilingIndex point out");
}

template <const int Dim>  typename cTilingIndex<Dim>::tIPt  cTilingIndex<Dim>::PtIndex(const tRPt & aPt) const
{
     return  mIBoxIn.Proj( Pt_round_down((aPt-mRBoxIn.P0())/mStep) + tIPt::PCste(1)   );
}

template <const int Dim>  int  cTilingIndex<Dim>::IIndex(const tRPt & aPt) const
{
	return mIBoxIn.IndexeLinear(PtIndex(aPt));
}

template <const int Dim> typename cTilingIndex<Dim>::tLIInd cTilingIndex<Dim>::GetCrossingIndexes(const tRPt & aPt)  const
{
      return tLIInd({IIndex(aPt)});
}

template <const int Dim> typename cTilingIndex<Dim>::tLIInd cTilingIndex<Dim>::GetCrossingIndexes(const tRBox & aBox)  const
{
      tLIInd aRes;
      tIPt  aPI0 = PtIndex(aBox.P0());
      tIPt  aPI1 = PtIndex(aBox.P1()) + tIPt::PCste(1);

      cPixBox<Dim> aBoxI(aPI0,aPI1);

      for (const auto & aPInt : aBoxI)
      {
          aRes.push_back(mIBoxIn.IndexeLinear(aPInt));
      }
      return aRes;
}



template <const int Dim>  size_t  cTilingIndex<Dim>::NbElem() const
{
       return mIBoxIn.NbElem();
}

template <const int Dim> bool  cTilingIndex<Dim>::OkOut() const {return mOkOut;}

template <const int Dim>  const typename cTilingIndex<Dim>::tRBox &  cTilingIndex<Dim>::Box() const {return mRBoxIn;}


template  class cTilingIndex<1>;
template  class cTilingIndex<2>;
template  class cTilingIndex<3>;




/* **************************************** */
/*                                          */
/*              cVerifSpatial               */
/*                                          */
/* **************************************** */

/** Testing that point extracted at given distance are excatly was is expected,
 * may be too restrictive due to rounding error, so we make an average, weighted by a function of distance
 * to have a robust control
 */

template <const int Dim> struct cVerifSpatial
{
    public :
       typedef cPtxd<tREAL8,Dim>     tRPt;

       cVerifSpatial(const tRPt & aC ,tREAL8 aD) : mC  (aC), mD  (aD) { }

       void Add(const tRPt & aP)
       {
           tREAL8 aW = std::pow(std::max(0.0,mD-Norm2(aP-mC)),0.5);
           mWAvg.Add(aW,aP);
       }

       cWeightAv<tREAL8,tRPt>  mWAvg;
       tRPt                    mC;
       tREAL8                  mD;
};


void OneBenchSpatialIndex()
{
    tREAL8 aMul = RandInInterval(0.1,10);
    cPt2dr aSz = cPt2dr(0.01,0.01) + cPt2dr::PRand() * aMul;  // generate size >0 
    cPt2dr aSzMargin = aSz * RandInInterval(0.01,0.1);   //  generate some margin to have point outside

    cPt2dr aP0 = cPt2dr::PRandC() * aMul; // random origin
    cPt2dr aP1 = aP0 + aSz;

    cBox2dr aBox(aP0,aP1); // Box of the tiling
    cBox2dr aBoxMargin(aP0-aSzMargin,aP1+aSzMargin); // box slightly bigger 

    cTiling<cPointSpInd<2>> aSI(aBox,true,1000,-1); // The tiling we want to check

    // Test the function GetObjAtPos
    std::list<cPt2dr>  aLPt;
    for (int aK=0 ; aK<100 ; aK++)
    {
       cPt2dr aPt = aBoxMargin.GeneratePointInside();
       aLPt.push_back(aPt);
       cPointSpInd<2> * aExtr = aSI.GetObjAtPos(aPt);
       MMVII_INTERNAL_ASSERT_bench(aExtr==nullptr,"Spat index, got unexpected");
       cPointSpInd<2> aObj(aPt);
       aSI.Add(aObj);
       aExtr = aSI.GetObjAtPos(aPt);
       MMVII_INTERNAL_ASSERT_bench(aExtr!=nullptr,"Spat index, ungot unexpected");
    }

    for (int aK=0 ; aK<100 ; aK++)
    {
         cPt2dr aP0 = aBoxMargin.GeneratePointInside();
	 tREAL8 aDist =  aMul * std::max(1e-5,std::pow(RandUnif_0_1(),3)); // max -> else bug in dilate
         std::list<cPointSpInd<2>*> aL = aSI.GetObjAtDist(aP0,aDist);


	 cVerifSpatial<2>  aVerif1(aP0,aDist);
	 for (const auto & aObj : aL)
	 {
              aVerif1.Add(aObj->GetPrimGeom());
	 }

	 cVerifSpatial<2>  aVerif2(aP0,aDist);
	 for (const auto & aPt : aLPt)
	 {
              aVerif2.Add(aPt);
	 }

	 MMVII_INTERNAL_ASSERT_bench(Norm2(aVerif1.mWAvg.SVW()-aVerif2.mWAvg.SVW())<1e-5,"GetObjAtDist");
	 //StdOut() << "Llllllll " << Norm2(aVerif1.mWAvg.SVW()-aVerif2.mWAvg.SVW()) << std::endl;
    }
    //getchar();
}



void Bench_SpatialIndex(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SpatialIndex")) return;

     /*cTilingIndex<2>  aTI(cBox2dr(cPt2dr(0,0),cPt2dr(2,2)),true,5);
     FakeUseIt(aTI); */

     for (int aK=0 ; aK<50 ; aK++)
     {
         OneBenchSpatialIndex();
     }


     aParam.EndBench();
}

};
