#include "MMVII_Mappings.h"
#include "MMVII_Geom2D.h"


namespace MMVII
{

/**   
    DimM = Dim of Manifold, for ex 1 for curve
    DimE = Dim of embedding space, for ex 2 for a curve in plane
*/
template <const int DimM,const int DimE> class cManifold
{
     public :
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>    tPtM;
        typedef cPtxd<tREAL8,DimE>    tPtE;
        typedef std::vector<tPtE>     tTgSp;


	cManifold(int aNbMap=1,const tREAL8 aEpsDeriv = 1e-4);

	/// return the num map of a point, defaults assume mNbMap==1
	virtual int  GetNumMap(const tPtE &) const ;
        ///  Retun a point of manifold in embedding space kowning a parameter, default error
        virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        ///  "Inverse" of PtOfParam, Assuming tPtE is on manifold
        virtual  std::pair<int,tPtM>   ParamOfPt(const tPtE &) const ;

        /// return TgSpace
        virtual tTgSp  TgSpace(const tPtE &) const ;
        /// Given a point, gives an approximate value of the projection
        virtual tPtE  ApproxProj(const tPtE &) const  = 0 ;
	///  Return the projection on the manifold, default iterative method
        virtual tPtE  Proj(const tPtE &) const  ;
     private :
	int     mNbMap;
	tREAL8  mEpsDeriv;
};

template <const int DimM,const int DimE>  
   cManifold<DimM,DimE>::cManifold(int aNbMap,const tREAL8 aEpsDeriv) :
      mNbMap    (aNbMap),
      mEpsDeriv (aEpsDeriv)
{
}

template <const int DimM,const int DimE>  int  cManifold<DimM,DimE>::GetNumMap(const tPtE &) const 
{
    MMVII_INTERNAL_ASSERT_tiny(mNbMap==1,"cCurveFromMapping->aNumMap");
    return 0;
}

template <const int DimM,const int DimE> cPtxd<tREAL8,DimE>  cManifold<DimM,DimE>::PtOfParam(const tPtM&,int)  const
{
   MMVII_INTERNAL_ERROR("No def cManifold<DimM,DimE>::PtOfParam");
   return tPtE();
}

template <const int DimM,const int DimE> std::pair<int,cPtxd<tREAL8,DimM>>  cManifold<DimM,DimE>::ParamOfPt(const tPtE&)  const
{
   MMVII_INTERNAL_ERROR("No def cManifold<DimM,DimE>::ParamOfPt");
   return std::pair<int,tPtM>(0,tPtM());
}


template <const int DimM,const int DimE> 
  typename cManifold<DimM,DimE>::tTgSp  cManifold<DimM,DimE>::TgSpace(const tPtE & aPE)const 
{
   tTgSp aRes;
   auto [aNum,aPParam] = ParamOfPt(aPE);

   for (int aKM=0 ; aKM<DimM ; aKM++)
   {
       tPtM aDelta = tPtM::P1Coord(aKM,mEpsDeriv);
       tPtE aPPlus = PtOfParam(aPParam+aDelta,aNum);
       tPtE aPMinus = PtOfParam(aPParam-aDelta,aNum);
       tPtE aTgt = (aPPlus-aPMinus) / (2*mEpsDeriv) ;

       // make some "on the fly" orthogonalization
       for (const auto & aPrec : aRes)
       {
           aTgt = aTgt - aPrec* Scal(aPrec,aTgt);
       }

       aRes.push_back(VUnit(aTgt));
   }

   return aRes;
}

template class cManifold<1,2>; // curve 2d
template class cManifold<1,3>; // curve 3d


template<int DimE> class cCurveFromMapping : public cManifold<1,DimE>
{
     public :
        // virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        static const int DimM = 1;
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>                  tPtM;
        typedef cPtxd<tREAL8,DimE>                  tPtE;
        typedef cSegmentCompiled<tREAL8,DimE>       tSeg;
        typedef cDataInvertibleMapping<tREAL8,DimE> tMap;

        tPtE   PtOfParam(const tPtM &,int aNumMap) const override;
        std::pair<int,tPtM>   ParamOfPt(const tPtE &) const override;
        tPtE  ApproxProj(const tPtE &) const  override ;
     private :
	tSeg   mSeg;
	tMap   *mMap;

};

template<int DimE>  cPtxd<tREAL8,DimE> cCurveFromMapping<DimE>::PtOfParam(const tPtM & aP1,int aNumMap) const 
{
    MMVII_INTERNAL_ASSERT_tiny(aNumMap==0,"cCurveFromMapping->aNumMap");

    tPtE aPtOnSeg = mSeg.PtOfAbscissa(aP1.x());

    return mMap->Value(aPtOnSeg);
}

template<int DimE>  std::pair<int,cPtxd<tREAL8,1>> cCurveFromMapping<DimE>::ParamOfPt(const tPtE & aPE) const 
{
    tPtE aPOnSeg = mMap->Inverse(aPE);
    return std::pair<int,tPtM>(0,tPtM(mSeg.Abscissa(aPOnSeg)));
}


template<int DimE>  cPtxd<tREAL8,DimE> cCurveFromMapping<DimE>::ApproxProj(const tPtE & aP1) const 
{
    tPtE aP2 = mMap->Value(aP1);
    tPtE aP3 = mSeg.Proj(aP2);
    return mMap->Inverse(aP3);
}


template class cCurveFromMapping<2>; // curve 2d
template class cCurveFromMapping<3>; // curve 2d
};



