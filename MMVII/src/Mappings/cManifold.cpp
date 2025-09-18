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
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        cManifold(int aNbMap=1,const tREAL8 aEpsDeriv = 1e-4);

        /// return the num map of a point, defaults assume mNbMap==1
        virtual int  GetNumMap(const tPtE &) const ;
        ///  Retun a point of manifold in embedding space kowning a parameter, default error
        virtual  tPtE   PtOfParam(const tResPOP &) const ;
        ///  "Inverse" of PtOfParam, Assuming tPtE is on manifold
        virtual  tResPOP   ParamOfPt(const tPtE &) const ;

        /// return TgSpace, assume aPtE is on the manifold
        virtual tTgSp  TgSpace(const tPtE & aPtE) const ;
        /// Given any point, gives an approximate value of the projection
        virtual tPtE  ApproxProj(const tPtE &) const  = 0 ;
        ///  Return the projection on the manifold, default iterative method
        virtual tPtE  Proj(const tPtE &) const  ;
        ///  Return the projection on the tangent space to a Manifolfd
     private :
        tPtE  OneIterProjByTgt(const tPtE & aP0,const tPtE & aP2Proj) const  ;

         int     mNbMap;
         tREAL8  mEpsDeriv;
};

template<int DimE> class cLineManifold : public cManifold<1,DimE>
{
     public :
        // virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        static const int DimM = 1;
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>                  tPtM;
        typedef cPtxd<tREAL8,DimE>                  tPtE;
        typedef cSegmentCompiled<tREAL8,DimE>       tSeg;
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        tPtE   PtOfParam(const tResPOP &) const override;
        tResPOP   ParamOfPt(const tPtE &) const override;
        tPtE  ApproxProj(const tPtE &) const  override ;

   private :
        tSeg   mSeg;
};


template <const int DimM,const int DimE> class cManifoldFromMapping : public  cManifold<DimM,DimE>
{
     public :
        typedef cDataInvertibleMapping<tREAL8,DimE> tMap;
        typedef cManifold<DimM,DimE>                tMan;
        // virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>                  tPtM;
        typedef cPtxd<tREAL8,DimE>                  tPtE;
        typedef cSegmentCompiled<tREAL8,DimE>       tSeg;
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        tPtE   PtOfParam(const tResPOP&) const override;
        tResPOP   ParamOfPt(const tPtE &) const override;
        tPtE  ApproxProj(const tPtE &) const  override ;

   private :
        tMan   *mMan;
        tMap   *mMap;

};



   /* *********************************************************** */
   /*                                                             */
   /*                        cManifold                            */
   /*                                                             */
   /* *********************************************************** */


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

template <const int DimM,const int DimE>
   typename cManifold<DimM,DimE>::tPtE  cManifold<DimM,DimE>::PtOfParam(const tResPOP&)  const
{
   MMVII_INTERNAL_ERROR("No def cManifold<DimM,DimE>::PtOfParam");
   return tPtE();
}

template <const int DimM,const int DimE>
   typename cManifold<DimM,DimE>::tResPOP  cManifold<DimM,DimE>::ParamOfPt(const tPtE&)  const
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
       tPtE aPPlus = PtOfParam(tResPOP(aNum,aPParam+aDelta));
       tPtE aPMinus = PtOfParam(tResPOP(aNum,aPParam-aDelta));
       tPtE aTgt = (aPPlus-aPMinus) / (2*mEpsDeriv) ;

       // make some "on the fly" orthogonalization
       for (const auto & aPrec : aRes)
       {
           aTgt += aPrec* (-Scal(aPrec,aTgt));
       }
       aRes.push_back(VUnit(aTgt));
   }

   return aRes;
}


template<const int DimM, const int DimE>
      typename cManifold<DimM,DimE>::tPtE
           cManifold<DimM, DimE>::OneIterProjByTgt(const tPtE& aPtApprox,const tPtE & aPt2Proj) const
{
   tTgSp aTgtS = TgSpace(aPtApprox);

   tPtE aCorrec = tPtE::PCste(0.0);
   tPtE aDelta = aPt2Proj-aPtApprox;
   for (const auto & aVec : aTgtS)
   {
       aCorrec += aVec * Scal(aVec,aDelta);
   }

   return ApproxProj(aPtApprox+aCorrec);
}


template<const int DimM, const int DimE>
      typename cManifold<DimM,DimE>::tPtE cManifold<DimM, DimE>::Proj(const tPtE & aPtE) const
{
   tREAL8 aEpsilon = 1e-6;
   int aNbIterMax = 10;
   tPtE aPt0 = ApproxProj(aPtE);
   bool GoOn = true;
   int aNbIter = 0;
   while (GoOn)
   {
       aNbIter++;
       tPtE aPt1 = OneIterProjByTgt(aPt0,aPtE);
       GoOn = (aNbIter< aNbIterMax) && (Norm2(aPt0-aPt1)>aEpsilon);
       aPt0 = aPt1;
   }
   return aPt0;
}

   /* *********************************************************** */
   /*                                                             */
   /*                        cLineManifol                         */
   /*                                                             */
   /* *********************************************************** */


template<int DimE>  typename cLineManifold<DimE>::tPtE cLineManifold<DimE>::PtOfParam(const tResPOP & aPair) const
{
     MMVII_INTERNAL_ASSERT_tiny(aPair.first==0,"cLineManifold->aNumMap");
     return mSeg.PtOfAbscissa(aPair.second.x());
}

template<int DimE>
     typename cLineManifold<DimE>::tResPOP
         cLineManifold<DimE>::ParamOfPt(const tPtE & aPE) const
{
    return std::pair<int,tPtM>(0,tPtM(mSeg.Abscissa(aPE)));
}

template<int DimE>  typename cLineManifold<DimE>::tPtE  cLineManifold<DimE>::ApproxProj(const tPtE & aPt) const
{
    return mSeg.Proj(aPt);
}

   /* *********************************************************** */
   /*                                                             */
   /*                    cManifoldFromMapping                     */
   /*                                                             */
   /* *********************************************************** */


template <const int DimM,const int DimE>
   typename cManifoldFromMapping<DimM,DimE>::tPtE
        cManifoldFromMapping<DimM,DimE>::PtOfParam(const tResPOP & aPair) const
{
   return mMap->Value(mMan->PtOfParam(aPair));
}


 template <const int DimM,const int DimE>
    typename cManifoldFromMapping<DimM,DimE>::tResPOP
       cManifoldFromMapping<DimM,DimE>::ParamOfPt(const tPtE & aPE) const
{
    return mMan->ParamOfPt(mMap->Inverse(aPE));
}

template <const int DimM,const int DimE>
   typename cManifoldFromMapping<DimM,DimE>::tPtE
      cManifoldFromMapping<DimM,DimE>::ApproxProj(const tPtE & aPt) const
{
    return mMap->Value(mMan->ApproxProj(mMap->Inverse(aPt)));
}



   /* *********************************************************** */
   /*                                                             */
   /*                        cSphereManifold                      */
   /*                                                             */
   /* *********************************************************** */

template<int DimE> class cSphereManifold : public cManifold<DimE-1,DimE>
{
     public :
        // virtual  tPtE   PtOfParam(const tPtM &,int aNumMap) const ;
        static const int DimM = DimE-1;
        static const int DimC = DimE - DimM;

        typedef cPtxd<tREAL8,DimM>                  tPtM;
        typedef cPtxd<tREAL8,DimE>                  tPtE;
        typedef cSegmentCompiled<tREAL8,DimE>       tSeg;
        typedef  std::pair<int,tPtM>                tResPOP; // Type result Param of Pt

        cSphereManifold(const tPtE & aPPerturb = tPtE::PCste(0.0) );
        tPtE   PtOfParam(const tResPOP&) const override;
        tResPOP   ParamOfPt(const tPtE &) const override;
        tPtE  ApproxProj(const tPtE &) const  override ;

   private :
        bool mIsPerturb; //< do use artificial perturbation, just 4 bench
        tPtE mPPerturb;  //< Value to use 4 articial perturbation
};

template<int DimE>  cSphereManifold<DimE>::cSphereManifold(const tPtE & aPPerturb) :
    cManifold<DimE-1,DimE>(2*DimE),
     mIsPerturb (!IsNull(aPPerturb)),
     mPPerturb  (aPPerturb)
{
}


template<int DimE>  typename cSphereManifold<DimE>::tPtE cSphereManifold<DimE>::PtOfParam(const tResPOP & aPair) const
{
    int aNumMap = aPair.first;
    const tPtM aProj = aPair.second;
    tREAL8 aCoord = std::sqrt(1.0-SqN2(aProj));

    if (aNumMap>=DimE)
    {
        aCoord = - aCoord;
        aNumMap -= DimE;
    }

    tPtE aRes;
    int aCpt=0;
    for (int aDim=0 ; aDim<DimE ; aDim++)
    {
        aRes[aDim] =  (aDim!= aNumMap) ? aProj[aCpt++] : aCoord;
    }
    return aRes;
}


template<int DimE>
     typename cSphereManifold<DimE>::tResPOP
         cSphereManifold<DimE>::ParamOfPt(const tPtE & aPE) const
{
    cWhichMax<int,tREAL8> aMaxC;
    for (int aDim=0 ; aDim<DimE ; aDim++)
        aMaxC.Add(aDim,std::abs(aPE[aDim]));

    int aDimMax = aMaxC.IndexExtre();
    int aCpt=0;
    tPtM aProj;

    for (int aDim=0 ; aDim<DimE ; aDim++)
    {
        if (aDim!= aDimMax)
            aProj[aCpt++] = aPE[aDim];
    }
    if (aPE[aDimMax]<0)
        aDimMax += DimE;

    return tResPOP(aDimMax,aProj);
}




template<int DimE>  typename cSphereManifold<DimE>::tPtE  cSphereManifold<DimE>::ApproxProj(const tPtE & aP2P) const
{
    if (IsNull(aP2P))  // in this case any point is closest ..
        return tPtE::P1Coord(0,1.0);
    tPtE aPProj =  VUnit(aP2P);

    if (!mIsPerturb)
        return aPProj;

    aPProj = aPProj + mPPerturb*Norm2(aP2P-aPProj);
    return VUnit(aPProj);
}

   /* ********************************************************** */
   /*                                                            */
   /*                         MMVII::                            */
   /*                                                            */
   /* ********************************************************** */

template<int DimE> void BenchManifold_Sphere()
{
     typedef cPtxd<tREAL8,DimE> tPtE;
     cSphereManifold<DimE> aSph(tPtE::PRandUnit()*1e-1);

int aCpt=0;
     for (int aKTest=0 ; aKTest<10; aKTest++)
     {
aCpt++;
        // Generate a random point on the sphere/manifold
        tPtE aPtS = tPtE::PRandUnit();
        // Extract its parametrization
        auto aProj = aSph.ParamOfPt(aPtS);
        // Extract the point from the parameters
        tPtE aPtS2 = aSph.PtOfParam(aProj);

        // check we go back to initial point
        MMVII_INTERNAL_ASSERT_bench(Norm2(aPtS-aPtS2)<1e-8,"BenchManifold_Sphere ParamOfPt/PtOfParam");

        // check tgt space
        auto aVTgt =aSph.TgSpace(aPtS);
        MMVII_INTERNAL_ASSERT_bench(aVTgt.size()==DimE-1,"Dim Tgt Space");  // good dim !

        for (size_t aKT1=0 ; aKT1<aVTgt.size(); aKT1++)
        {
            MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aVTgt.at(aKT1),aPtS))<1e-6,"Scal Tgt in BenchManifold_Sphere");
            MMVII_INTERNAL_ASSERT_bench(std::abs(Norm2(aVTgt.at(aKT1))-1)<1e-6,"Norm Tgt in BenchManifold_Sphere");
            for (size_t aKT2=aKT1+1 ; aKT2<aVTgt.size(); aKT2++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aVTgt.at(aKT1),aVTgt.at(aKT2)))<1e-6,"Scal Tgt in BenchManifold_Sphere");
            }
        }

        // Check proj
        aPtS2 = aPtS * RandInInterval(0.8,1.2);

        tPtE aPtProj = aSph.Proj(aPtS2) ;
        StdOut() << " Prrrojj= " << Norm2(aPtS - aPtProj)  << " " << aPtS - aSph.ApproxProj(aPtS2)<< "\n";
        if (0) // (Norm2(aPtS-aPtS2)>1e-8)
        {
        StdOut() << " BenchManifold_Sphere "
                 << " Cpt=" << aCpt
                  <<  " Dif=" << aPtS-aPtS2
                   << " Num=" << aProj.first
                   << " Norm=" << Norm2(aPtS2)
                   << "\n";
        getchar();
        }
     }

//StdOut() << " ENNNd-BenchManifold_Sphere \n";
//getchar();
}


void BenchManifold(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Manifold")) return;

    for (int aKT=0 ; aKT<100 ; aKT++)
    {
        BenchManifold_Sphere<2>();
        BenchManifold_Sphere<3>();
        BenchManifold_Sphere<4>();
    }

     aParam.EndBench();

}

   /* ********************************************************** */
   /*                                                            */
   /*                         INSTANCIATION                      */
   /*                                                            */
   /* ********************************************************** */


template class cManifold<1,2>; // like curve 2d
template class cManifold<1,3>; // like curve 3d
template class cManifold<2,3>; // like surf 3d


template class cLineManifold<2>; // Seg  2d
template class cLineManifold<3>; // Seg 3d

template class cSphereManifold<2>; // Seg  2d
template class cSphereManifold<3>; // Seg  2d
template class cSphereManifold<4>; // Seg  2d


template class cManifoldFromMapping<1,2>; // like curve 2d
template class cManifoldFromMapping<1,3>; // like curve 3d
template class cManifoldFromMapping<2,3>; // like surf 3d


};



