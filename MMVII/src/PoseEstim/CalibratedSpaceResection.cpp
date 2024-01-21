#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_BundleAdj.h"


/**
   \file CalibratedSpaceResection.cpp

   \brief file for implementing the space resection algorithm in calibrated case

 */

namespace MMVII
{
   // cWhichMin<cIsometry3D<tREAL8>,tREAL8>  aWMin(cIsometry3D<tREAL8>::Identity(),1e10);

template <class Type>  class cElemSpaceResection
{
      public :
           typedef cPtxd<Type,3>     tP3;
           typedef cPolynom<Type>    tPol;
	   typedef cTriangle<Type,3> tTri;
           typedef tP3               tResBC;

	   static tP3 ToPt(const cPt3dr & aP) {return tP3::FromPtR(aP);}

           cElemSpaceResection
           (
	        const tTri & aTriBundles,
	        const tTri & aTriGround
	   );

	       // intermediar called used in test, or usable in tutorials

	   std::list<tResBC>  ComputeBC() const;
           cTriangle<Type,3>  BC2LocCoord(const tResBC &) const ;
           cIsometry3D<Type>  BC2Pose(const tResBC &) const ;

           static std::list<cIsometry3D<Type> > ListPoseBySR
                                     (
                                          cPerspCamIntrCalib&,
                                          const cPair2D3D&,
                                          const cPair2D3D&,
                                          const cPair2D3D&
                                     );

	   /** For final result we ewport to the desored type for camera, theimportan is thae
	       eventualy the computation has been made with REAL16 is high accuracy was required */

           static  cWhichMin<tPoseR,tREAL8>  RansacPoseBySR
                                             (
                                                  cPerspCamIntrCalib&,
                                                  const cSet2D3D &,
					          size_t aNbTest,
					          int    aNbPtsMeasures = -1,
	                                          cTimerSegm * aTS = nullptr
                                             );



	   static void OneTestCorrectness();
       private :
           Type nNormA;
           Type nNormB;
           Type nNormC;
	   // copy bundles, are unitary
	   tP3  A;
	   tP3  B;
	   tP3  C;

	   tP3  AB;      ///<  Vector  A -> B
	   tP3  AC;      ///<  Vector  A -> C
	   tP3  BC;      ///<  Vector  A -> C
	   Type abb;     ///<  (A->B).B

	   // copy of ground point coordinates, local precision
	   tTri mTriG;
	   tP3  gA;
	   tP3  gB;
	   tP3  gC;


	   //  Square Distance between  ground points 
	   Type  gD2AB;
	   Type  gD2AC;
	   Type  gD2BC;

           Type  mSqPerimG;  ///<  Square Ground perimeter 
	   //  ratio of dist
	   Type  rABC;  ///<   mD2AB / mD2AC
	   Type  rCBA;  ///<   mD2CB / mD2CA

	   // std::list<tPairBC>  mListPair;
};

template <class Type> 
   cElemSpaceResection<Type>::cElemSpaceResection
   (
       const tTri & aTriB,
       const tTri & aTriG
   ) :
        nNormA  (Norm2(aTriB.Pt(0))),
        nNormB  (Norm2(aTriB.Pt(1))),
        nNormC  (Norm2(aTriB.Pt(2))),
        A       (aTriB.Pt(0) / nNormA),
        B       (aTriB.Pt(1) / nNormB),
        C       (aTriB.Pt(2) / nNormC),

        AB      (B - A),
        AC      (C - A),
        BC      (C - B),
	abb     (Scal(AB,B)),

	mTriG   (aTriG),
        gA (aTriG.Pt(0)),
        gB (aTriG.Pt(1)),
        gC (aTriG.Pt(2)),

	gD2AB (SqN2(mTriG.KVect(0))),
	gD2AC (SqN2(mTriG.KVect(2))),
	gD2BC (SqN2(mTriG.KVect(1))),
        mSqPerimG ( gD2AB + gD2AC + gD2BC),

	rABC  (gD2AB/gD2AC),
	rCBA  (gD2BC/gD2AC)
{
}



template <class Type> std::list<cPtxd<Type,3>>  cElemSpaceResection<Type>::ComputeBC() const
{
/*
      3 direction  of bundles  A,B,C   we have made ||A|| = ||B|| = ||C|| = 1
      We parametrize 3 point on the bundle by 2 parameters b & c:
           PA  = A  (arbitrarily we fix on this bundle)
	   PB  = B(1+b)
	   PC  = C(1+c)
      This parametrization is made to be more numerically stable when 
     the bundle are close to each others which is a current case (b & c small)
*/


/*  ===============  (1) eliminate b  =====================
     We have a conservation of ratio of distance :

     |PA-PB|^2    |GA-GB|^2
      -----   =  ---------  = rABC
     |PA-PC|^2    |GA-GC|^2


    ((1+b)B-A)^2 = rABC ((1+c)C-A)^2
     (bB + AB) ^2 = rABC (cC + AC) ^2

     b^2 + 2AB.B b  + (AB^2 -rABC(cC + AC)^2  ) =0
   # P(c) =  (AB^2 -rABC (cC + AC)^2)
     b^2 + 2AB.B b + P(c) =0   =  (B+AB.B)^2 - (AB.B^2 -P(c))
     2nd degre equation in b 
     b =  - AB.B +E SQRT(AB.B^2  -P(c))   E in {-1,+1}
   # Q(c) = AB.B^2 -P(c)
     b =  - AB.B + E S(Q(c))
*/


     // P(c) =  (AB^2 -rABC (cC + AC)^2)
    tPol  aPol_AC_C =  PolSqN(AC,C);
    cPolynom<Type> aPc =  tPol::D0(SqN2(AB)) -  aPol_AC_C *rABC ;
    // Q(c) = AB.B^2 -P(c)
    cPolynom<Type> aQc =  tPol::D0(Square(abb)) - aPc;


  //  Now we can eliminate b using :   b =  - AB.B + E S(Q(c))   E in {-1,1} 
/* ======================== (2) resolve c =====================
    2nd conservation  of ratio
     |PC-PB|^2    |GC-GB|^2
      -----   =  ---------  = rCBA
     |PA-PC|^2    |GC-GA|^2


     ((1+c)C - (1+b)B)^2 = ((1+c)C -A)^2 rCBA
     (BC + cC -bB) ^2 = rCBA (AC + cC) ^2
     rCBA (AC + cC) ^2 = (BC +cC - (-AB.B + E *  S(Q)) B)^2 =   ((BC + AB.B B +c C)  - E S(Q) B) ^2

     rCBA (AC + cC) ^2 = (BC + AB.B B +c C)^2 -2 (BC + AB.B B +c C) .B E S(Q) + Q B^2
     B^2 = 1
     rCBA (AC + cC) ^2 - (BC + AB.B B +c C)^2  - Q  =  -2 (BC.B + AB.B  +c C.B) E S(Q)

                      
                       R(c) = -2E L(c) S(Q)
		       R^2(c) = 4 L(c)^2 Q(c)

*/
       
    tPol  aRc =   aPol_AC_C *rCBA  -  aQc  -  PolSqN(BC +  abb*B  ,C);
    tPol  aLc ({Scal(BC,B)+abb,Scal(B,C)});
    tPol aSolver = Square(aRc) - aQc * Square(aLc) * 4;
    std::vector<Type> aVRoots = aSolver.RealRoots (1e-30,60);

    std::list<tResBC> aRes;

    for (Type c : aVRoots)
    {
        for (Type E : {-1.0,1.0})
        {
	    Type Q =  aQc.Value(c);
	    if (Q>=0)
	    {
	        Type b =  -abb + E * std::sqrt(Q);

		tP3 PA = A;
		tP3 PB = (1+b)  * B;
		tP3 PC = (1+c)  * C;

                Type aD2AB =  SqN2(PA-PB);
                Type aD2AC =  SqN2(PA-PC);
                Type aD2BC =  SqN2(PB-PC);

		// Due to squaring sign of E is not always consistant, so now we check if ratio are really found
		Type aCheckABC =  aD2AB/aD2AC - rABC;
		Type aCheckCBA =  aD2BC/aD2AC - rCBA;

		//  test with 1e-5  generate bench problem ...
		if (  (std::abs(aCheckABC)< 1e-3)  && (std::abs(aCheckCBA)< 1e-3) )
		{
                   Type aSqPerim = aD2AB + aD2AC + aD2BC;
                   aRes.push_back(tResBC((1+b),(1+c),std::sqrt(mSqPerimG/aSqPerim)));
		   // StdOut()  << " E " << E <<  " bc " << b << " " << c << " " << aCheckABC << " " << aCheckCBA << std::endl;
		}
	    }
        }
    }
    return aRes;
}

template <class Type> cTriangle<Type,3>  cElemSpaceResection<Type>::BC2LocCoord(const tResBC & aRBC) const 
{
     const Type & b =  aRBC.x();
     const Type & c =  aRBC.y();
     const Type & aMul = aRBC.z();

     return  cTriangle<Type,3>(aMul*A,(aMul*b)*B,(aMul*c)*C);
}

template <class Type> cIsometry3D<Type> cElemSpaceResection<Type>::BC2Pose(const tResBC & aRBC) const 
{
     cTriangle<Type,3> aTri = BC2LocCoord(aRBC);

     return cIsometry3D<Type>::FromTriInAndOut(0,aTri,0,mTriG);
}


template <class Type> void  cElemSpaceResection<Type>::OneTestCorrectness()
{
   static int aCpt=0; aCpt++;
   {
       // generate 3 bundle not too degenared => 0,P0,P1,P2 cot coplanar
       cTriangle<Type,3> aTriBund = RandomTetraTriangRegul<Type>(1e-2,1e2);

       //   Generate b &c ;  Too extrem value =>  unaccuracyy bench ; not : RandUnif_C_NotNull(1e-2) * 10
       Type b = pow(2.0,RandUnif_C());
       Type c = pow(2.0,RandUnif_C());

       // comput A,B,C  with  ratio given by b,c and A unitary
       cPtxd<Type,3> A = VUnit(aTriBund.Pt(0));
       cPtxd<Type,3> B = VUnit(aTriBund.Pt(1))*b;
       cPtxd<Type,3> C = VUnit(aTriBund.Pt(2))*c;

       //  put them anywhere and with any ratio using a random similitud
       cSimilitud3D<Type> aSim(
		               static_cast<Type>(RandUnif_C_NotNull(1e-2)*10.0),
			       cPtxd<Type,3>::PRandC()*static_cast<Type>(100.0),
			       cRotation3D<Type>::RandomRot()
                         );
       cTriangle<Type,3> aTriG(aSim.Value(A),aSim.Value(B),aSim.Value(C));

       //  Now see that we can recover b & c
       cElemSpaceResection<Type> anESR(aTriBund,aTriG);
       auto aLBC = anESR.ComputeBC();  //list of b,c,Perimeter

       cWhichMin<cPtxd<Type,3>,Type> aWMin(cPtxd<Type,3>(0,0,0),1e10);  // will extract b,c closest to ours
       for (auto & aTripl : aLBC)
       {
           aWMin.Add(aTripl,std::abs(aTripl.x()-b)+std::abs(aTripl.y()-c));
       }
       if (aWMin.ValExtre()>=1e-4)
       {
	    StdOut() << "cElemSpaceResection::BC " << aWMin.ValExtre() << std::endl;
            MMVII_INTERNAL_ASSERT_bench(aWMin.ValExtre()<1e-4,"2 value in OneTestCorrectness");  // is it close enough
       }


       //  Now see that if can recover local coord from b,c
       cTriangle<Type,3>  aTriComp = anESR.BC2LocCoord(aWMin.IndexExtre());
       for (auto aK : {0,1,2})
       {
              //  Test the triangle Local and Ground are isometric
             double aDif = RelativeSafeDifference(Norm2(aTriG.KVect(aK)),Norm2(aTriComp.KVect(aK))) ;
             MMVII_INTERNAL_ASSERT_bench(aDif<1e-4,"Local coord in OneTestCorrectness");
              //  Test the Local coordinate are aligned on bundles
             double aAngle = AbsAngleTrnk(aTriBund.Pt(aK),aTriComp.Pt(aK))  ;
             MMVII_INTERNAL_ASSERT_bench(aAngle<1e-4,"Local coord in OneTestCorrectness");
       }
       //  Now see that if can recover local pose from b,c
       cIsometry3D<Type>  aPose = anESR.BC2Pose(aWMin.IndexExtre());
       for (auto aK : {0,1,2})
       {
           // check that  Bundle is colinear to Pose^-1 (PGround)
           cPtxd<Type,3>  aPLoc=  aPose.Inverse(aTriG.Pt(aK));
           Type aAngle = AbsAngleTrnk(aPLoc,aTriBund.Pt(aK));
           MMVII_INTERNAL_ASSERT_bench(aAngle<1e-4,"Pose in OneTestCorrectness");
       }
   }
}

template <class Type>  
   std::list<cIsometry3D<Type>> cElemSpaceResection<Type>::ListPoseBySR
                                 (
                                       cPerspCamIntrCalib& aCalib,
                                       const cPair2D3D&          aPair1,
                                       const cPair2D3D&          aPair2,
                                       const cPair2D3D&          aPair3
                                 )
{
   tTri aTriB
        (
	     ToPt(aCalib.DirBundle(aPair1.mP2)),
	     ToPt(aCalib.DirBundle(aPair2.mP2)),
	     ToPt(aCalib.DirBundle(aPair3.mP2))
	);

   tTri aTriG(ToPt(aPair1.mP3),ToPt(aPair2.mP3),ToPt(aPair3.mP3));
   cElemSpaceResection<Type> anESR(aTriB,aTriG);


   std::list<cIsometry3D<Type> > aLPose;

   for (const auto & aBC : anESR.ComputeBC())
       aLPose.push_back(anESR.BC2Pose(aBC));

   return aLPose;
}

template <class Type>  
   cWhichMin<tPoseR,tREAL8>  cElemSpaceResection<Type>::RansacPoseBySR
                      (
                          cPerspCamIntrCalib& aCalib,
                          const  cSet2D3D & aSet0,
                          size_t aNbTriplet,
                          int    aNbPtsMeasures,
	                  cTimerSegm * aTS
                      )
{
   cAutoTimerSegm  anATS2 (aTS,"CreateTriplet");
   cWhichMin<tPoseR,tREAL8>  aWMin(cIsometry3D<tREAL8>::Identity(),1e10);

   const cSet2D3D * aSetTest = & aSet0;
   cSet2D3D aBufSetTest;  // will have the space to store locally the test set
   int aNbTot = aSet0.NbPair();

   MMVII_INTERNAL_ASSERT_tiny(aNbTot>3,"Not enough 2-3 corresp in RansacPoseBySR");


   //  is we require less test that total of point we must create the subset
   if ( (aNbPtsMeasures>0)  && (aNbPtsMeasures<aNbTot)  )  
   {
      aSetTest = & aBufSetTest;

      // class for selection aNbPtsMeasures  among total
      cRandKAmongN aSub(aNbPtsMeasures,aNbTot);
      for (int aKPair=0 ; aKPair<aNbTot ; aKPair++)
          if (aSub.GetNext())
             aBufSetTest.AddPair(aSet0.KthPair(aKPair));
   }

   std::vector<cSetIExtension>  aVecTriplet;
   GenRanQsubCardKAmongN(aVecTriplet,aNbTriplet,3,aNbTot);

   cAutoTimerSegm  anATS3 (aTS,"ResolveEqResec");
   for (const auto &aTriplet : aVecTriplet)
   {
       size_t aK1 = aTriplet.mElems.at(0);
       size_t aK2 = aTriplet.mElems.at(1);
       size_t aK3 = aTriplet.mElems.at(2);

       std::list<cIsometry3D<Type> >  aLIsom 
	       = cElemSpaceResection<Type>::ListPoseBySR
                 (
                    aCalib,
		    aSet0.KthPair(aK1),
		    aSet0.KthPair(aK2),
		    aSet0.KthPair(aK3)
		 );

       for (const auto & anIsom : aLIsom)
       {
           cIsometry3D<tREAL8>  aIsomR8 = ToReal8(anIsom);
           cSensorCamPC aCam("RansacSpaceResection",aIsomR8,&aCalib);

           aWMin.Add(aIsomR8,aCam.AvgAngularProjResiudal(*aSetTest));
       }
   }

   return aWMin;
}

template class cElemSpaceResection<tREAL8>;
template class cElemSpaceResection<tREAL16>;


/* ==================================================== */
/*                                                      */
/*                 cPerspCamIntrCalib                   */
/*                                                      */
/* ==================================================== */


cWhichMin<tPoseR,tREAL8> cPerspCamIntrCalib::RansacPoseEstimSpaceResection
       (
            const cSet2D3D & aSet0,
            size_t aNbTriplet,
            bool Real8,
            int aNbPtsMeasures,
	    cTimerSegm * aTS
        )
{
    if (Real8)
       return cElemSpaceResection<tREAL8>::RansacPoseBySR(*this,aSet0,aNbTriplet,aNbPtsMeasures,aTS);
    else 
       return cElemSpaceResection<tREAL16>::RansacPoseBySR(*this,aSet0,aNbTriplet,aNbPtsMeasures,aTS);

}

std::list<tPoseR >  cPerspCamIntrCalib::ElemPoseEstimSpaceResection(const cPair2D3D& aP1,const cPair2D3D& aP2,const cPair2D3D& aP3)
{
	return cElemSpaceResection<tREAL8>::ListPoseBySR(*this,aP1,aP2,aP3);
}



/* ==================================================== */
/*                                                      */
/*                 MMVII                                */
/*                                                      */
/* ==================================================== */

/**  For a given camera generate correspondance 3D/2D , certain perfect
 * and ceratin gross error,  check that with ransac we are able to recover
 * the real pose*/


void BenchCalibResection(cSensorCamPC & aCam,cTimerSegm * aTimeSeg)
{
    cAutoTimerSegm  anATS (aTimeSeg,"CreateSetResec");

    cSet2D3D aSet;  // set of pair used for testing 

    double aPropOk = 0.7;  // proporation of good point
    int aNbPts = 50;         // total number of point
    std::vector<bool> IsOk;  // memorize if point is  "perfect/error" 

    // structure to generate the specified nuber "perfect/error" in a random order
    cRandKAmongN aSelOk(round_ni(aNbPts*aPropOk),aNbPts);

    //  loop to generate the point
    for (int aK=0 ; aK<aNbPts ; aK++)
    {
        bool Ok = aSelOk.GetNext();
        IsOk.push_back(Ok);
	cPt2dr aPIm = aCam.RandomVisiblePIm();  // random point on the sensor

	//  --  generate  a point that project on PIm, at random depth
	cPt3dr aPGround = aCam.ImageAndDepth2Ground(cPt3dr(aPIm.x(),aPIm.y(),RandInInterval(1,2)));

	// if "gross" error add pertubation
	if (!Ok)
            aPGround = aPGround + cPt3dr::PRandC() * 0.1;

	aSet.AddPair(aPIm,aPGround,1.0);  // memorize pair
    }

    //  estimate pose by ransac
    cPerspCamIntrCalib * aCal = aCam.InternalCalib();
    cIsometry3D<tREAL8> aPose = aCal->RansacPoseEstimSpaceResection(aSet,100,true,-1,aTimeSeg).IndexExtre();

    // test with ground truth
    MMVII_INTERNAL_ASSERT_bench(Norm2(aPose.Tr() - aCam.Pose().Tr())<1e-4,"Translation in space resection");
    MMVII_INTERNAL_ASSERT_bench(aPose.Rot().Mat().L2Dist(aCam.Pose().Rot().Mat())<1e-4,"Matrix in space resection");
}

/** Genereate different calibration model (proj & degree) to test that space resection
 * works with all
 */

void BenchCalibResection(cParamExeBench & aParam)
{
   for (int aK=0 ; aK< 1000 ; aK++)
   {
      cElemSpaceResection<tREAL8>::OneTestCorrectness();
      cElemSpaceResection<tREAL16>::OneTestCorrectness();
   }

   cTimerSegm* aTimeSeg = aParam.Show()                                       ?
	                  new cTimerSegm  (&(cMMVII_Appli::CurrentAppli()))   :
	  		  nullptr                                             ;

   for (int aKCal=0 ; aKCal<4 ; aKCal++)  // Test different degree
   {
       for (int aKEnum=0 ; aKEnum<int(eProjPC::eNbVals) ; aKEnum++)  // Test all projections
       {
            cAutoTimerSegm * anATS = new cAutoTimerSegm(aTimeSeg,"CreateCalib");
            eProjPC aTypeProj = eProjPC(aKEnum);
	    //  K%3  =>  3 option for degree of dist in random calib
            cPerspCamIntrCalib *  aCalib = cPerspCamIntrCalib::RandomCalib(aTypeProj,aKCal%4);

	    delete anATS;

	    for (int aKPose=0 ; aKPose<3 ; aKPose++)  // Test different random poses
	    {
                cIsometry3D<tREAL8> aPose =  cIsometry3D<tREAL8>::RandomIsom3D(10.0);
                cSensorCamPC aCam("TestSR",aPose,aCalib);
		BenchCalibResection(aCam,aTimeSeg);
	    }
            delete aCalib;
       }

   }

   delete aTimeSeg;
}

bool BUGCAL = false;

void BenchPoseEstim(cParamExeBench & aParam)
{
   if (! aParam.NewBench("PoseEstim")) return;


   BenchUnCalibResection();  // test 11 parameters
   BenchCalibResection(aParam);  // test space resection
   aParam.EndBench();
}

/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_CalibratedSpaceResection : public cMMVII_Appli
{
     public :
        typedef std::vector<cPerspCamIntrCalib *> tVCal;

        cAppli_CalibratedSpaceResection(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :

        /// In case multiple pose for same camera try a robust compromise for each value
        void  DoMedianCalib();

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;
        cSetMesImGCP             mSetMes;
        cSet2D3D                 mSet23 ;

	int                      mNbTriplets;
	int                      mNbIterBundle;
	bool                     mShowBundle;
	tREAL8                   mThrsReject;
	tREAL8                   mMaxErrOK;
        std::string              mDirFilter;
        std::string              mNameReport;
        std::string              mNameIm;
        // bool                     mShow;
        // bool                     mReal16;

};

cAppli_CalibratedSpaceResection::cAppli_CalibratedSpaceResection
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mNbTriplets   (500),
     mNbIterBundle (10),
     mShowBundle   (false),
     mThrsReject   (10000.0),
     mMaxErrOK     (20.0)
{
}



cCollecSpecArg2007 & cAppli_CalibratedSpaceResection::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_CalibratedSpaceResection::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	   << AOpt2007(mNbTriplets,"NbTriplets","Number max of triplet tested in Ransac",{eTA2007::HDV})
	   << AOpt2007(mNbIterBundle,"NbIterBund","Number of bundle iteration, after ransac init",{eTA2007::HDV})
	   << AOpt2007(mShowBundle,"ShowBundle","Show detail of bundle results",{eTA2007::HDV})
	   << AOpt2007(mThrsReject,"ThrRej","Threshold for rejection of outlayer, in pixel",{eTA2007::HDV})
	   << AOpt2007(mMaxErrOK,"MaxErr","Max error acceptable for initial resection",{eTA2007::HDV})
	   <<  mPhProj.DPPointsMeasures().ArgDirOutOpt("DirFiltered","Directory for filtered point")
    ;
}

//================================================

int cAppli_CalibratedSpaceResection::Exe()
{
    mPhProj.FinishInit();

    mNameReport = "Rejected_Ori-" +   mPhProj.DPOrient().DirIn() + "_Mes-" + mPhProj.DPPointsMeasures().DirIn() ;

    InitReport(mNameReport,"csv",true);

    bool  aExpFilt = mPhProj.DPPointsMeasures().DirOutIsInit();

    if (RunMultiSet(0,0))
    {
        AddOneReportCSV(mNameReport,{"Image","GCP","Residual"});

        int aResult = ResultMultiSet();

        if (aResult != EXIT_SUCCESS)
           return aResult;

	if (aExpFilt)
	{
           mPhProj.CpGCP(); // Save GCP from StdIn to StdOut
        }
        mPhProj.CpSysIn2Out(false,true);

        return EXIT_SUCCESS;
    }


    // By default print detail if we are not in //
    SetIfNotInit(mShowBundle,LevelCall()==0);

    mNameIm =FileOfPath(mSpecImIn);

    mPhProj.LoadGCP(mSetMes);
    mPhProj.LoadIm(mSetMes,mNameIm);
    mSetMes.ExtractMes1Im(mSet23,mNameIm);

    MMVII_INTERNAL_ASSERT_User(mSet23.NbPair()>3,eTyUEr::eUnClassedError,"Not enouh 3-2 pair for space resection");

    cPerspCamIntrCalib *   aCal = mPhProj.InternalCalibFromStdName(mNameIm);

    // Pose estimation with ransac using 3 point method
    cWhichMin<tPoseR,tREAL8>  aWMin = aCal->RansacPoseEstimSpaceResection(mSet23,mNbTriplets);
    tPoseR   aPose = aWMin.IndexExtre();
    cSensorCamPC  aCam(FileOfPath(mNameIm,false),aPose,aCal);

    std::vector<double> aVRes;
    {
       for (const auto & aPair : mSet23.Pairs())
       {
            tREAL8 aRes = aCam.AngularProjResiudal(aPair) * aCal->F();
	    aVRes.push_back(aRes); 
       }
       std::sort(aVRes.begin(),aVRes.end());
       std::reverse(aVRes.begin(),aVRes.end()); // reverse to supress 3 lowest val => RANSAC
       for (int aK=0 ; aK<3 ; aK++)
          aVRes.pop_back();
       std::reverse(aVRes.begin(),aVRes.end());

       tREAL8 aMedErr = Cst_KthVal(aVRes,0.5);
       if (aMedErr> mMaxErrOK)
       {
           StdOut() << " ============================================================" << std::endl;
           StdOut() << " median error on residual seems too High " << aMedErr << std::endl;
           StdOut() << " check data or eventutally change value of [MaxErr] (now =" << mMaxErrOK << ")" << std::endl;
           MMVII_INTERNAL_ASSERT_User(false,eTyUEr::eUnClassedError,"Space resection probably failed due to bad data");
       }
    }
     

    // If we want to filter on residual 
    if (aExpFilt )
    {
         cFilterMesIm aFMIM(mPhProj,mNameIm);
         StdOut() <<   " =====  WORST RESIDUAL ============= " << std::endl;

         tREAL8 aThShow = aVRes.at(std::max(0,int(aVRes.size()-5)));  // arbitray threshols for worst points
	 for (const auto & aMes : mSetMes.MesImOfPt())
	 {
             if (! aMes.VMeasures().empty())
	     {
                 cPt2dr aPtIm  = aMes.VMeasures().at(0);
	         cMes1GCP      aGCP =  mSetMes.MesGCP().at(aMes.NumPt());

	         tREAL8 aRes = aCam.AngularProjResiudal(cPair2D3D(aPtIm,aGCP.mPt)) * aCal->F();

	         if (aRes>=aThShow)
                 {			  
                     StdOut() <<   " * Name=" << aGCP.mNamePt << " " << aRes << std::endl;
		 }
		 if (aRes>mThrsReject)
                     AddOneReportCSV(mNameReport,{mNameIm,aGCP.mNamePt,ToStr(aRes)});

                 aFMIM.AddInOrOut(aPtIm,aGCP.mNamePt,aRes < mThrsReject);
	     }
	 }
         aFMIM.SetFinished();

	 aFMIM.Save();
	 aFMIM.SetMesImGCP().ExtractMes1Im(mSet23,mNameIm);
    }

    if (mNbIterBundle)
    {
         tREAL8 aF0 = aCal->F();
         tREAL8 aRes0 = aCam.AvgSqResidual(mSet23);

         cCorresp32_BA aBA32(&aCam, mSet23);
         aBA32.Sys().SetFrozenFromPat(*aCal,".*",true);

	 for (int aK=0 ; aK<mNbIterBundle  ; aK++)
             aBA32.OneIteration();

         tREAL8 aF1 = aCal->F();
         tREAL8 aRes1 = aCam.AvgSqResidual(mSet23);
         tPoseR   aPose1 = aCam.Pose();

	 if (mShowBundle)
	 {
	    StdOut() <<  "DFoc : " <<  aF1-aF0 << std::endl;
	    StdOut() <<  "Pose DC=" <<  Norm2(aPose.Tr()-aPose1.Tr()) 
	              <<   " DMat=" <<  aPose.Rot().Mat().L2Dist(aPose1.Rot().Mat()) << "\n";
	    // StdOut() <<  "DPose : " <<  Norm2(aCam.Center()-aPose1.Tr()) << std::endl; // check conv, should be 0
	    StdOut() <<  "Sq Residual : " << aRes0 << " => " << aRes1 << std::endl;
	 }

    }

    mPhProj.SaveCamPC(aCam);


    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_CalibratedSpaceResection(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CalibratedSpaceResection(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriCalibratedSpaceResection
(
     "OriPoseEstimSpaceResection",
      Alloc_CalibratedSpaceResection,
      "Pose estimation from GCP, calibrated case",
      {eApF::Ori},
      {eApDT::GCP},
      {eApDT::Orient},
      __FILE__
);





}; // MMVII

