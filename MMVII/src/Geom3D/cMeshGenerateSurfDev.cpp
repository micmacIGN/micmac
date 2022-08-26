#include "include/MMVII_all.h"

#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_GenNameAlloc.h"


using namespace NS_SymbolicDerivative;
using namespace MMVII;

namespace MMVII
{



   /* ======  header of header  ====== */
typedef tREAL8  tCoordDevTri;
typedef  cTriangulation3D<tCoordDevTri> tTriangulation3D;
typedef  cRotation3D<tCoordDevTri>      tRot;
typedef  cPtxd<tCoordDevTri,3>          tPt3D;
typedef  cPtxd<tCoordDevTri,2>          tPt2D;

class cGenerateSurfDevOri;

   /* ======================= */
   /* ======  header   ====== */
   /* ======================= */

/**  Class for generating a syntetic developpable 3D surface, this class can be used
     for testing the devlopment algorithms,

     The surface is a generalized cylindre =>  curve logarithmic spiral in XZ , generatrice  Y-axes
 */

class cGenerateSurfDevOri
{
     public :
         
         cGenerateSurfDevOri
         (
             const cPt2di & aNb ,  // size of the grid Nb.x=> discreta of the curve, 
             double  aFactNonDev=0
         );

	 std::vector<tPt3D> VPts() const;
	 std::vector<cPt3di>   VFaces() const;

	 tPt3D PCenter() const;

     private :
	 int  NumOfPix(const cPt2di & aKpt) const; ///< Basic numerotation of points using video sens, create index for mesh
	 tPt3D  PlaneToSurf(const tPt2D & aKpt) const; ///< Map the plane to the 3D-surface
         tPt2D  AbsiceToCurve(const tCoordDevTri & aXCoord) const;

         // =========  parameters of the surface
	 cPt2di    mNb;
	 double    mFactNonDev;
	 bool      mPlaneCart; ///< if true generate a plane cartesian mesh
         double    mNbTour; // number of tour of the spiral
         double    mFactRhoByTour ; // growing exponential factor on each tour
         tPt3D     mAxeCyl ;

         // =========   value computed from parameters
         double    mTotalAngle;  // total variation of angle on the curve
         tRot      mRToAxe;  // transforme a cylindre with generatrix OZ in cylindre with  geneatrix AxeCyl

};


/* ******************************************************* */
/*                                                         */
/*               cGenerateSurfDevOri                       */
/*                                                         */
/* ******************************************************* */

int  cGenerateSurfDevOri::NumOfPix(const cPt2di & aKpt)  const
{
     return aKpt.x() + aKpt.y() *  (mNb.x());
}

tPt2D  cGenerateSurfDevOri::AbsiceToCurve(const tCoordDevTri & aXCoord) const
{
     double aTeta =  mTotalAngle * ( aXCoord  / (mNb.x()-1) -0.5);
     double aRho = pow(mFactRhoByTour,aTeta/(2*M_PI));
     return FromPolar(aRho,aTeta); // make them cartesian
}

tPt3D  cGenerateSurfDevOri::PlaneToSurf(const tPt2D & aKpt) const
{
     if (mPlaneCart)  // if planar stop here
     {
        return tPt3D(aKpt.x(),aKpt.y(),0.0);
     }



     // compute polar coordinates
/*
     double aTeta =  mTotalAngle * ((double(aKpt.x())  / (mNb.x()-1) -0.5));
     double aRho = pow(mFactRhoByTour,aTeta/(2*M_PI));
     aRho = aRho * (1 + (round_ni(aKpt.x()+aKpt.y())%2)* mFactNonDev);
     tPt2D  aPPlan = FromPolar(aRho,aTeta); // make them cartesian
*/
     tPt2D  aPPlan = AbsiceToCurve(aKpt.x()) *  (1 + (round_ni(aKpt.x()+aKpt.y())%2)* mFactNonDev);
     double  aZCyl = (aKpt.y() * mTotalAngle) / (mNb.x()-1);

     tPt3D  aPCyl(aPPlan.x(),aPPlan.y(),aZCyl);

     //tPt3D aAxe(1,1,1);
     //tRot aR3 =  tRot::CompleteRON(mAxeCyl);  // create a RON with Axe as  AxeI
     //aR3 =  tRot::CompleteRON(aR3.AxeJ(),aR3.AxeK()); // create a RON with Axe as AxeK
      

     return mRToAxe.Value(aPCyl); // tPt3D(aPCyl.y(),aPCyl.z(),aPCyl.x());

}

std::vector<tPt3D> cGenerateSurfDevOri::VPts() const
{
    std::vector<tPt3D> aRes(mNb.x()*mNb.y());

    for (const auto & aPix : cRect2(cPt2di(0,0),mNb))
    {
         aRes.at(NumOfPix(aPix)) = PlaneToSurf(ToR(aPix));
    }

    return aRes;
}

std::vector<cPt3di> cGenerateSurfDevOri::VFaces() const
{
    std::vector<cPt3di> aRes;
    // parse rectangle into each pixel
    for (const auto & aPix00 : cRect2(cPt2di(0,0),mNb-cPt2di(1,1)))
    {
         // split the pixel in two tri
          // const std::vector<cTriangle<int,2> > &   aVTri = SplitPixIn2<int>(HeadOrTail());
	  for (const auto & aTri : SplitPixIn2<int>(HeadOrTail()))
	  {
              cPt3di aFace;
              for (int aK=0 ; aK<3 ; aK++)
              {
                   cPt2di aPix = aPix00 + aTri.Pt(aK);
		   aFace[aK] = NumOfPix(aPix);
              }
	      aRes.push_back(aFace);
	  }
    }
    return aRes;
}

tPt3D  cGenerateSurfDevOri::PCenter() const
{
     return PlaneToSurf(ToR(mNb)/2.0);
}

cGenerateSurfDevOri::cGenerateSurfDevOri(const cPt2di & aNb,double aFactNonDev) :
     // ---  parameters 
     mNb             (aNb),
     mFactNonDev     (aFactNonDev),
     mPlaneCart      (false),
     mNbTour         (1.5),
     mFactRhoByTour  (2.0),
     mAxeCyl         (1,1,1),
     // ---  Computed values
     mTotalAngle     (mNbTour * 2.0 * M_PI),
     mRToAxe         (  tRot::CompleteRON(mAxeCyl))  // create a RON with Axe as  AxeI
{
     mRToAxe =  tRot::CompleteRON(mRToAxe.AxeJ(),mRToAxe.AxeK()); // create a RON with Axe as AxeK
}

/* ******************************************************* */
/*                                                         */
/*                 cAppliGenMeshDev                        */
/*                                                         */
/* ******************************************************* */

class cAppliGenMeshDev : public cMMVII_Appli
{
     public :

        cAppliGenMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

           // --- Mandatory ----
	      std::string mNameCloudOut;
           // --- Optionnal ----
	      bool        mBinOut;
	      double      mFactNonDev;  // Make the surface non devlopable
           // --- Internal ----

};

cAppliGenMeshDev::cAppliGenMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mBinOut          (true)
{
}


cCollecSpecArg2007 & cAppliGenMeshDev::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudOut,"Name of output cloud/mesh", {eTA2007::FileDirProj})
   ;
}

cCollecSpecArg2007 & cAppliGenMeshDev::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
            << AOpt2007(mFactNonDev,"NonDevFact","make the surface more or less devlopable ",{eTA2007::HDV})
            << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
           // << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file")
   ;
}



int  cAppliGenMeshDev::Exe()
{
   // was used to check that each instance create a new object, 
   if (0)
   {
      auto aPtr = EqConsDist(true,100);
      auto aPtr2 = EqConsRatioDist(true,100);
      StdOut() << "DIFPTR "  << ((void*) aPtr2) << " " <<  ((void *) aPtr) << "\n";
      delete aPtr;
      delete aPtr2;
   }


   // generate synthetic mesh
   cGenerateSurfDevOri aGenSD (cPt2di(15,5),mFactNonDev);
   tTriangulation3D  aTri(aGenSD.VPts(),aGenSD.VFaces());
   aTri.WriteFile(mNameCloudOut,mBinOut);
   aTri.MakeTopo();

   //  devlop it
/*
   cDevTriangu3d aDev(aTri);
   aDev.SetFaceC(aTri.IndexClosestFace(aGenSD.PCenter()));
   aDev.DoDevlpt();
   aDev.ExportDev("Devlp_"+mNameCloudOut);

   aDev.ShowQualityStat();
*/
   return EXIT_SUCCESS;
}



/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */


tMMVII_UnikPApli Alloc_GenMeshDev(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliGenMeshDev(aVArgs,aSpec));
}
cSpecMMVII_Appli  TheSpecGenMeshDev
(
     "MeshDevGen",
      Alloc_GenMeshDev,
      "Generate artificial(synthetic) devlopable surface",
      {eApF::Cloud},
      {eApDT::Console},
      {eApDT::Ply},
      __FILE__
);


};
