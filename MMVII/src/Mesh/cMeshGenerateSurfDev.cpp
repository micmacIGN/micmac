#include "SymbDer/SymbolicDerivatives.h"
#include "SymbDer/SymbDer_GenNameAlloc.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "cMMVII_Appli.h"
#include "MMVII_PhgrDist.h"
#include "MMVII_DeclareCste.h"

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

	 std::vector<tPt3D> VPts(bool Plane) const;  ///< return Pt3d of surface
	 std::vector<cPt3di>   VFaces() const;

	 tPt3D PCenter() const;

     private :
	 int  NumOfPix(const cPt2di & aKpt) const; ///< Basic numerotation of points using video sens, create index for mesh
	 tPt3D  PlaneToSurf(const tPt2D & aKpt) const; ///< Map the plane to the 3D-surface
         tPt3D  IndexToDev(const cPt2di & aKpt) const; ///< return a devlpt of the surface
         tPt2D  AbsiceToCurve(const tCoordDevTri & aXCoord) const; ///< compute point on the spiral
         double  ZOfY(const tCoordDevTri & aYCoord) const; ///< Compute Z from Y index (before rotation by RToAxe)
         double  FactNDev(const cPt2di & ) const; ///< Factor used to make surface non dev/non planar

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

         std::vector<double>  mIntegAbs;  // integral of curvil abs on spiral -> to generate GT for plane tri

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

double  cGenerateSurfDevOri::ZOfY(const tCoordDevTri & aYCoord) const
{
     return  (aYCoord * mTotalAngle) / (mNb.x()-1);
}

double  cGenerateSurfDevOri::FactNDev(const cPt2di & aInd) const
{
    return  ((aInd.x()+aInd.y())%2) * mFactNonDev;
}

tPt3D  cGenerateSurfDevOri::PlaneToSurf(const tPt2D & aKpt) const
{
     if (mPlaneCart)  // if planar stop here
     {
        return tPt3D(aKpt.x(),aKpt.y(),0.0);
     }

     // compute polar coordinates
     tPt2D  aPPlan = AbsiceToCurve(aKpt.x()) *  (1 + FactNDev(ToI(aKpt)));
     double  aZCyl = ZOfY(aKpt.y()); //  (aKpt.y() * mTotalAngle) / (mNb.x()-1);

     tPt3D  aPCyl(aPPlan.x(),aPPlan.y(),aZCyl);


     return mRToAxe.Value(aPCyl); // tPt3D(aPCyl.y(),aPCyl.z(),aPCyl.x());

}

tPt3D  cGenerateSurfDevOri::IndexToDev(const cPt2di & aKpt) const
{
    double aX =  mIntegAbs.at(aKpt.x());
    double aY =  ZOfY(aKpt.y());
    double aZ =  FactNDev(aKpt);

    return tPt3D(aX,aY,aZ);
}

std::vector<tPt3D> cGenerateSurfDevOri::VPts(bool PlaneDevlpt) const
{
    std::vector<tPt3D> aRes(mNb.x()*mNb.y());

    for (const auto & aPix : cRect2(cPt2di(0,0),mNb))
    {
         aRes.at(NumOfPix(aPix)) =  PlaneDevlpt ? IndexToDev(aPix) : PlaneToSurf(ToR(aPix));
    }

    return aRes;
}

std::vector<cPt3di> cGenerateSurfDevOri::VFaces() const
{
    std::vector<cPt3di> aRes;
    // parse rectangle into each pixel
    for (const auto & aPix00 : cRect2(cPt2di(0,0),mNb-cPt2di(1,1)))
    {
         // split the pixel in two tri, generate randomy the two possible decomposistion
	  for (const auto & aTri : SplitPixIn2<int>(HeadOrTail())) // parse the two triangle
	  {
              cPt3di aFace;
              for (int aK=0 ; aK<3 ; aK++) // parse 3 submit of elementary tri
              {
                   cPt2di aPix = aPix00 + aTri.Pt(aK); // add coordinat to have global index
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
      
     // compute integral of abscisse  along the discrete curve
     mIntegAbs.push_back(0);
     for (int aKx=1 ; aKx<=mNb.x() ; aKx++)
     {
          tPt2D  aP1 = AbsiceToCurve(aKx-1);
          tPt2D  aP2 = AbsiceToCurve(aKx);
          mIntegAbs.push_back(mIntegAbs.back() + Norm2(aP1-aP2));
     }
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
	      bool        mPlanDevSurf; // generate the developped surf
	      bool        mBinOut;
	      double      mFactNonDev;  // Make the surface non devlopable
           // --- Internal ----

};

cAppliGenMeshDev::cAppliGenMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPlanDevSurf     (false),
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
            << AOpt2007(mPlanDevSurf,"DevPlane","Generate the devloped planar surf",{eTA2007::HDV})
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
   tTriangulation3D  aTri(aGenSD.VPts(mPlanDevSurf),aGenSD.VFaces());
   aTri.WriteFile(mNameCloudOut,mBinOut);


   // aTri.MakeTopo();

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
