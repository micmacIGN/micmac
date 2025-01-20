#include "MMVII_2Include_Tiling.h"

namespace MMVII
{


/* **************************************** */
/*                                          */
/*              cGeneratePointDiff          */
/*                                          */
/* **************************************** */

template <const int TheDim> 
   cGeneratePointDiff<TheDim>:: cGeneratePointDiff(const tRBox & aBox,tREAL8 aDistMin,int aNbMax) :
      mNbMax           (std::min(aNbMax,(round_down(aBox.NbElem()/pow(aDistMin,TheDim))))),
      mTiling          (aBox,true,mNbMax,-1),
      mDistMin         (aDistMin)
{
}
           
                      ///  generate a new point
template <const int TheDim> 
    typename cGeneratePointDiff<TheDim>::tRPt cGeneratePointDiff<TheDim>::GetNewPoint(int aNbTest)
{
               for (int aK=0 ; aK<aNbTest ; aK++)
               {
                    tRPt aRes =  mTiling.Box().GeneratePointInside();  // generate a new random point
                    auto aL = mTiling.GetObjAtDist(aRes,mDistMin);  // extract all point at a given distance
                    if (aL.empty())   // if empty OK
                    {
                            mTiling.Add(tPSI(aRes));  // memo in the box
                            return aRes;  // return value
                    }
               }
               MMVII_INTERNAL_ERROR("Could not GetNewPoint in cGeneratePointDiff");
               return tRPt::PCste(0);
}
/*	 
	   */

template class cGeneratePointDiff<2>;

/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */

class cPoint2DValuated
{
    public :
        static constexpr int Dim = 2;
        typedef cPt2dr  tPrimGeom;
        typedef int     tArgPG;  /// unused here

        tPrimGeom  GetPrimGeom(int Arg=-1) const {return Proj(mPt);}

        cPoint2DValuated(const cPt3dr & aPt) :
           mPt (aPt)
        {
        }

    private :
         cPt3dr  mPt;
};


std::vector<cPt3dr>  FilterMaxLoc(std::vector<cPt3dr> & aVPt,tREAL8 aDist)
{
    SortOnCriteria(  aVPt,[](const auto & aPt){return - aPt.z();} );

    return FilterMaxLoc((cPt2dr*)nullptr,aVPt,[](const auto & aP) {return Proj(aP);}, aDist);
}




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

    return MinAbsCoord(aPStep); // smallest of each size
}


template <const int Dim>  cTilingIndex<Dim>::cTilingIndex(const tRBox & aBox,bool OkOut, int aNbCase) :
	mRBoxIn      (aBox),
	mOkOut       (OkOut),
        mNbCase      (aNbCase),
	mStep        (ComputeStep(aBox,aNbCase)),
	mSzI         (Pt_round_up(aBox.Sz()/mStep)),
        mIBoxIn      (  tIBox(tIPt::PCste(0),mSzI+tIPt::PCste(2))),
        mIndIsBorder (NbElem(),false)
{
    cBorderPixBox<Dim> aBorder(mIBoxIn,1);

    for (const auto& aPix : aBorder)
        mIndIsBorder.at(PInd2II(aPix)) = true;
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

template <const int Dim>  typename cTilingIndex<Dim>::tIPt  cTilingIndex<Dim>::RPt2PIndex(const tRPt & aPt) const
{
     return  mIBoxIn.Proj( Pt_round_down((aPt-mRBoxIn.P0())/mStep) + tIPt::PCste(1)   );
}

template <const int Dim>  typename cTilingIndex<Dim>::tRPt  cTilingIndex<Dim>::PIndex2RPt(const tRPt & aPt) const
{
	return    (aPt-tRPt::PCste(1.0))*mStep + mRBoxIn.P0();
}

template <const int Dim>  typename cTilingIndex<Dim>::tRPt  cTilingIndex<Dim>::PIndex2MidleBox(const tIPt & aPt) const
{
	return PIndex2RPt(ToR(aPt) + tRPt::PCste(0.5));
}



template <const int Dim>  int  cTilingIndex<Dim>::IIndex(const tRPt & aPt) const
{
	return mIBoxIn.IndexeLinear(RPt2PIndex(aPt));
}

template <const int Dim> typename cTilingIndex<Dim>::tLIInd cTilingIndex<Dim>::GetCrossingIndexes(const tRPt & aPt)  const
{
      return tLIInd({IIndex(aPt)});
}


template <const int Dim> typename cTilingIndex<Dim>::tLIInd cTilingIndex<Dim>::GetCrossingIndexes(const tRBox & aBox)  const
{
      tLIInd aRes;

      tIPt  aPI0 = RPt2PIndex(aBox.P0());
      tIPt  aPI1 = RPt2PIndex(aBox.P1()) + tIPt::PCste(1);

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

       void Add(const tRPt & aP,tREAL8 aD)
       {
           tREAL8 aW = std::pow(std::max(0.0,mD-aD),0.5);
           mWAvg.Add(aW,aP);
       }
       void Add(const tRPt & aP)
       {
	       Add(aP,Norm2(aP-mC));
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

    cTiling<cPointSpInd<2>> aSI(aBox,true,1000,-1); // The tiling we want to check , -1 is the fake arge

    // Test the function GetObjAtPos
    std::list<cPt2dr>  aLPt;
    for (int aK=0 ; aK<  100 ; aK++)
    {
       cPt2dr aPt = aBoxMargin.GeneratePointInside();

       {
            cPt2di aP0 = aSI.RPt2PIndex(aPt);
	    cPt2dr aP1 = aSI.PIndex2MidleBox(aP0);
	    cPt2di aP2 = aSI.RPt2PIndex(aP1);
	    // StdOut() << "OneBenchSpatialIndexOneBenchSpatialIndex : " << aP1 << aP0 << aP2 << "\n";  //  getchar();
            MMVII_INTERNAL_ASSERT_bench(aP0==aP2,"Spat index RPt2PIndex/PIndex2MidleBox");
       }
       
       aLPt.push_back(aPt);
       cPointSpInd<2> * aExtr = aSI.GetObjAtPos(aPt);
       MMVII_INTERNAL_ASSERT_bench(aExtr==nullptr,"Spat index, got unexpected");
       cPointSpInd<2> aObj(aPt);
       aSI.Add(aObj);
       aExtr = aSI.GetObjAtPos(aPt);
       MMVII_INTERNAL_ASSERT_bench(aExtr!=nullptr,"Spat index, ungot unexpected");
    }

    // Test etObjAtDist
    for (int aK=0 ; aK<100 ; aK++)
    {
	 tREAL8 aDist =  aMul * std::max(1e-5,std::pow(RandUnif_0_1(),3)); // max -> else bug in dilate
         cPt2dr aP0 = aBoxMargin.GeneratePointInside();

         // Test tiling with  point query
         {

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
	 }

         // Test tiling with  segment query
	 if (aK%10==0)
	 {
	    cVerifSpatial<2>  aVerif1(aP0,aDist);
	    cVerifSpatial<2>  aVerif2(aP0,aDist);

            cPt2dr aP1 = aP0;
	    while (Norm2(aP1-aP0)<1e-5)
                  aP1 = aBoxMargin.GeneratePointInside();

	    cClosedSeg2D  aSeg(aP0,aP1);

	    std::list<cPointSpInd<2>*> aL = aSI.GetObjAtDist(aSeg,aDist);

	    for (const auto & aPt : aL)
	    {
                 tREAL8 aDistSegPt = aSeg.Seg().DistClosedSeg(aPt->GetPrimGeom()) ;
	         // StdOut() << " DDDD "  <<  aSeg.Seg().DistClosedSeg(aPt->GetPrimGeom()) << " " << aDist << "\n";
                 aVerif2.Add(aPt->GetPrimGeom(),aDistSegPt);
	    }

        [[maybe_unused]] int aNbIn=0;
	    for (const auto & aPt : aLPt)
	    {
                tREAL8 aDistSegPt = aSeg.Seg().DistClosedSeg(aPt) ;
                aVerif1.Add(aPt,aDistSegPt);
		if (aDistSegPt<aDist)
                   aNbIn++;
	    }

	    MMVII_INTERNAL_ASSERT_bench(Norm2(aVerif1.mWAvg.SVW()-aVerif2.mWAvg.SVW())<1e-5,"GetObjAtDist");
	 }
    }
    //getchar();
}



void Bench_SpatialIndex(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SpatialIndex")) return;

     for (int aK=0 ; aK<50 ; aK++)
     {
         OneBenchSpatialIndex();
     }

     aParam.EndBench();
}

};
