#include "MMVII_Mappings.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Manifolds.h"


namespace MMVII
{

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

template <const int DimM,const int DimE>
    cManifold<DimM,DimE>::~cManifold()
{
}

template <const int DimM,const int DimE>  int cManifold<DimM,DimE>::NbMap() const {return mNbMap;}

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

template<int DimE>   cLineManifold<DimE>::cLineManifold(const tSeg & aSeg) :
   mSeg  (aSeg)
{
}


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
    cManifoldFromMapping<DimM,DimE>::cManifoldFromMapping(tMan* aMan,tMap* aMap) :
        cManifold<DimM,DimE>(aMan->NbMap()),
        mMan (aMan),
        mMap (aMap)
{

}

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


    /* *********************************************************** */
    /*                                                             */
    /*                        cLineDist_Manifold                   */
    /*                                                             */
    /* *********************************************************** */

               /*  ------------------------  cAux_LineDist_Manifold  ------------------------ */

cAux_LineDist_Manifold::cAux_LineDist_Manifold(const tSeg2dr & aSeg,cPerspCamIntrCalib * aCalib ) :
    mMap_Ud_Red (aCalib),
    mMap_Red_Ud (&mMap_Ud_Red,false), // false -> does not adopt
    mLineMan    (tSeg2dr(aCalib->Undist(aSeg.P1()),aCalib->Undist(aSeg.P2())))
{
}

              /*  ------------------------  cLineDist_Manifold  ------------------------ */

cLineDist_Manifold::cLineDist_Manifold(const tSeg2dr & aSeg,cPerspCamIntrCalib * aCalib ) :
    cAux_LineDist_Manifold(aSeg,aCalib),
    cManifoldFromMapping<1,2>(&mLineMan,&mMap_Red_Ud)
     // cManifoldFromMapping<1,2>(&mLineMan,&mMap_Ud_Red)
{
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

     for (int aKTest=0 ; aKTest<10; aKTest++)
     {
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
            // vector are unitary and orthogonal to vector 0->PS (=normal here)
            MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aVTgt.at(aKT1),aPtS))<1e-6,"Scal Tgt in BenchManifold_Sphere");
            MMVII_INTERNAL_ASSERT_bench(std::abs(Norm2(aVTgt.at(aKT1))-1)<1e-6,"Norm Tgt in BenchManifold_Sphere");

            //  vector are orthogonal between them
            for (size_t aKT2=aKT1+1 ; aKT2<aVTgt.size(); aKT2++)
            {
                MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aVTgt.at(aKT1),aVTgt.at(aKT2)))<1e-6,"Scal Tgt in BenchManifold_Sphere");
            }
        }

        // Check proj
        aPtS2 = aPtS * RandInInterval(0.8,1.2);

        tPtE aPtProj = aSph.Proj(aPtS2) ;
        MMVII_INTERNAL_ASSERT_bench(Norm2(aPtS - aPtProj)<1e-6,"Proj in BenchManifold_Sphere");

        if (NeverHappens()) // (Norm2(aPtS-aPtS2)>1e-8)
        {
            static int aCpt=0; aCpt++;
            StdOut() << " Prrrojj= " << Norm2(aPtS - aPtProj)  << " " << aPtS - aSph.ApproxProj(aPtS2)<< "\n";
            StdOut() << " BenchManifold_Sphere "
                 << " Cpt=" << aCpt
                  <<  " Dif=" << aPtS-aPtS2
                   << " Num=" << aProj.first
                   << " Norm=" << Norm2(aPtS2)
                   << "\n";
             getchar();
        }
     }
}

template<int DimE> void BenchManifold_MapSphere(int aNbTest)
{
   // static int aCpt=0; aCpt++;

   typedef cPtxd<tREAL8,DimE> tPtE;
   typedef cPtxd<tREAL8,DimE-1> tPtM;


     // generate random symetrix matrix, close to identity
    cDenseMatrix<tREAL8> aMat(DimE);
    for (int aKX=0 ; aKX<DimE ; aKX++)
    {
         aMat.SetElem(aKX,aKX,1.0);
         for (int aKY=aKX+1 ; aKY<DimE ; aKY++)
         {
             aMat.SetElem(aKX,aKY,0.1* RandUnif_C());
             aMat.SetElem(aKY,aKX,aMat(aKX,aKY));
         }
    }
    tPtE aC = tPtE::PRandC();

    cBijAffMapElem<tREAL8,DimE> aMapElem(aMat,aC);
    cInvertMappingFromElem< cBijAffMapElem<tREAL8,DimE>> aMap(aMapElem);

    cSphereManifold<DimE> aSph;
    cManifoldFromMapping<DimE-1,DimE>  aMM(&aSph,&aMap);

     for (int aKPt=0 ; aKPt<aNbTest ; aKPt++)
     {
         tPtE aPt =  tPtE::PRandUnit()* RandInInterval(0.8,1.2);
         aPt = aMapElem.Value(aPt);

       //  StdOut() << "NNNN=" << Norm2()

         tPtE aPProj = aMM.Proj(aPt);
        // If Proj is on the ellipsoide Map-1(Proj) must be on unitary sphere
         MMVII_INTERNAL_ASSERT_bench( std::abs(Norm2(aMapElem.Inverse(aPProj))-1)<1e-6,"cManifoldFromMapping not on sphere");
        // compute Tgt Space and check ortoogonality
         auto aTgS = aMM.TgSpace(aPProj);
         for (const auto & aV : aTgS)
         {
             // StdOut() << " SccC=" << Scal(aV,aPt-aPProj) << "\n";
             MMVII_INTERNAL_ASSERT_bench(std::abs(Scal(aV,aPt-aPProj))<1e-5,"Tgt space in ManMap");
         }
         // Check minimality of dist
         auto aParam = aMM.ParamOfPt(aPProj);
         for (int aKNeigh=0 ; aKNeigh<20 ; aKNeigh++)
         {
             auto aParamN = aParam;
             aParamN.second += tPtM::PRandInSphere() * 1e-2;
             tPtE aNeighP = aMM.PtOfParam(aParamN);
             tREAL8 aDelta = Norm2(aNeighP-aPt) - Norm2(aPProj-aPt);

             // Theretically Delta>=, but due to epsilon-machine and approx, we tolerate a small negativity
             MMVII_INTERNAL_ASSERT_bench(aDelta>=-1e10,"Minimality in Manifold proj; D="+ToStr(aDelta));
         }
     }
     // BREAK_POINT("MANIF-MAP");
}

void BenchManifold(cParamExeBench & aParam)
{
    if (! aParam.NewBench("Manifold")) return;

    for (int aK=0 ; aK<1000; aK++)
    {
         BenchManifold_MapSphere<2>(1);
    }
    for (int aKT=0 ; aKT<100 ; aKT++)
    {
        BenchManifold_Sphere<2>();
        BenchManifold_Sphere<3>();
        BenchManifold_Sphere<4>();

        BenchManifold_MapSphere<2>(10);
        BenchManifold_MapSphere<3>(10);
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



