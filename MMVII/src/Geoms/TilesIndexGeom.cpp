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
           typedef cSegment<tREAL8,Dim>  tSeg;

           typedef std::list<int>        tLIInd;

	   cTilingIndex(const tRBox &,bool WithBoxOut, int aNbCase);

	   size_t  NbElem() const;

        protected :
	   tIPt  PtIndex(const tRPt &) const;
	   int   IIndex(const tRPt &) const;

	   tLIInd  GetCrossingIndexes(const tRPt &)  const;
	   // tLInd  GetCrossingIndexes(const tRBox &) const;  2 define
	   // tLInd  GetCrossingIndexes(const tSeg &) const;   2 define

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
/*
*/

template <const int Dim>  size_t  cTilingIndex<Dim>::NbElem() const
{
       return mIBoxIn.NbElem();
}


// template <const int Dim> std::list<tIPt>  GetCrossingIndexes(const tRPt &) const;

template  class cTilingIndex<1>;
template  class cTilingIndex<2>;
template  class cTilingIndex<3>;



/* **************************************** */
/*                                          */
/*              cTilingIndex                */
/*                                          */
/* **************************************** */

/*  For fast retrieving of object in tiling at given point position we test equality with a
 *  point;  but this equality can be called between, for ex, point and segemnt, so we define this special
 *  equality function that behave as an equality for 2 points, and generate error else (because exact
 *  retrieving from a point cannot be uses)
 */
template <class Type,const int Dim>  bool EqualPt(const Type &,const cPtxd<tREAL8,Dim> & aPt)
{
  MMVII_INTERNAL_ERROR("Called EqualPt with bad primitive");

  return false;
}

template <const int Dim>  bool EqualPt(const cPtxd<tREAL8,Dim> & aP1,const cPtxd<tREAL8,Dim> & aP2)
{
      return (aP1==aP2);
}


template <class Type,const int Dim> cTplBox<Type,Dim> GetBoxEnglob(const cPtxd<tREAL8,Dim> & aP1)
{
       return cTplBox<Type,Dim>(aP1,aP1,true);
}

template <class Type,const int Dim> cTplBox<Type,Dim> GetBoxEnglob(const cSegment<tREAL8,Dim> & aSeg)
{
       return cTplBox<Type,Dim>(aSeg.P1(),aSeg.P2(),true);
}



template <class Type>  class  cTiling : public cTilingIndex<Type::Dim>
{
     public :
           typedef cTilingIndex<Type::Dim>  tTI;
           typedef typename tTI::tRBox      tRBox;
           typedef typename tTI::tRPt       tRPt;
           typedef typename Type::tPrimGeom tPrimGeom;
	   static constexpr int Dim = Type::Dim;

           typedef std::list<Type>          tCont1Tile;
           typedef std::vector<tCont1Tile>  tVectTiles;

	   cTiling(const tRBox &,bool WithBoxOut, int aNbCase);
	   void Add(const Type & anObj);

	   /// this method can be used only when tPrimGeom is a point
	   void GetObjAtPos(std::list<Type*>&,const tRPt &);

     private :
	   tVectTiles  mVTiles;
};

template <class Type> 
    cTiling<Type>::cTiling(const tRBox & aBox,bool WithBoxOut, int aNbCase)  :
	   tTI     (aBox,WithBoxOut,aNbCase),
	   mVTiles (this->NbElem())
{
}

template <class Type> void cTiling<Type>::Add(const Type & anObj)
{
     for (const auto &  aInd :  tTI::GetCrossingIndexes(anObj.PrimGeom())  )
     {
         mVTiles.at(aInd).push_back(anObj);
     }
}

/*
template <class Type> void cTiling<Type>::GetObjAtPos(std::list<Type*>&,const tRPt & )
{
}
*/

/*=========== cTestSpatialIndex ===============*/
 
class cTestSpatialIndex
{
    public :
        static constexpr int Dim = 2;
        typedef cPt2dr  tPrimGeom;

	const tPrimGeom & PrimGeom() const {return mPt;}
         
    private :
	cPt2dr  mPt;
};
template class cTiling<cTestSpatialIndex>;




void Bench_SpatialIndex(cParamExeBench & aParam)
{
     if (! aParam.NewBench("SpatialIndex")) return;

     cTilingIndex<2>  aTI(cBox2dr(cPt2dr(0,0),cPt2dr(2,2)),true,5);
     FakeUseIt(aTI);



     aParam.EndBench();
}



};
