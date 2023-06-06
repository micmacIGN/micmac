/*
 */
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"

namespace MMVII
{

/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */

template <const int Dim> class cTilingIndex
{
       public :
           typedef cPtxd<int,Dim>        tIPt;
           typedef cPtxd<tREAL8,Dim>     tRPt;
           typedef cPixBox<Dim>          tIBox;
           typedef cTplBox<tREAL8,Dim>   tRBox;

	   cTilingIndex(const tRBox &,bool WithBoxOut, int aNbCase);

        protected :
	   tIPt  Index(const tRPt &) const;
        private :

	   static tREAL8  ComputeStep(const tRBox &, int aNbCase);
	   void AssertInside(const tRPt &) const;
	  
	   // tIPt  Index(const tRPt &) const;
	   tRBox    mRBoxIn;
	   bool     mOkOut;
	   int      mNbCase;

	   tREAL8   mStep;
	   tIPt     mSzI;
	   tIBox    mIBoxIn;
};

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
    StdOut()  <<  " -- SSSS=" << mStep << " " << mSzI << "\n";
    StdOut()  <<  mIBoxIn.P0()  << mIBoxIn.P1() << "\n";
    // StdOut()  <<  mIBoxIn.Proj(cPt2di(1,1))   << mIBoxIn.Proj(cPt2di(-3,-3))  << mIBoxIn.Proj(cPt2di(10,10)) << "\n";
/*
*/
}

template <const int Dim> void cTilingIndex<Dim>::AssertInside(const tRPt & aPt) const
{
    MMVII_INTERNAL_ASSERT_tiny(mOkOut || mRBoxIn.Inside(aPt),"cTilingIndex point out");
}

template <const int Dim>  typename cTilingIndex<Dim>::tIPt  cTilingIndex<Dim>::Index(const tRPt & aPt) const
{
     return  mIBoxIn.Proj( Pt_round_down((aPt-mRBoxIn.P0())/mStep) + tIPt::PCste(1)   );
}


template  class cTilingIndex<1>;
template  class cTilingIndex<2>;
template  class cTilingIndex<3>;



/*

void Bench_SpatialIndex(cParamExeBench & aParam)
{
}
*/


/*
 *    Sz / S 
 
     Xn /s

pow(MulCoord(aBox.Sz())/aNbCase,1/double(Dim))



template <const int Dim>  typename  cTilingIndex<Dim>::tIPt cTilingIndex<Dim>::Index(const tRPt & aP) const
{
	return  Pt_round_down(MulCByC(aP-mRP0,mMulSz));
}

 */
/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */

class cTestSpatialIndex
{
    public :
        static constexpr int Dim = 2;
        typedef cPt2dr  tPrimGeom;

	const tPrimGeom & PrimGeom() const {return mPt;}
         
    private :
	cPt2dr  mPt;
};


template <class Type>  class  cTiling : public cTilingIndex<Type::Dim>
{
     public :
           typedef cTilingIndex<Type::Dim> tTI;
           typedef typename tTI::tRBox     tRBox;


	   cTiling(const tRBox &,bool WithBoxOut, int aNbCase);
     private :
};

template <class Type> 
    cTiling<Type>::cTiling(const tRBox & aBox,bool WithBoxOut, int aNbCase)  :
	   tTI(aBox,WithBoxOut,aNbCase)
{
}


template class cTiling<cTestSpatialIndex>;


void Bench_SpatialIndex(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SpatialIndex")) return;

     cTilingIndex<2>  aTI(cBox2dr(cPt2dr(0,0),cPt2dr(2,2)),true,5);
     FakeUseIt(aTI);



     aParam.EndBench();
}



};
