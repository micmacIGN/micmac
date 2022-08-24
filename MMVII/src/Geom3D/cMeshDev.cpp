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
typedef  cTriangle<tCoordDevTri,3>      tTri3D;
typedef  cIsometry3D<tCoordDevTri>      tIsom3D;
typedef  cSimilitud3D<tCoordDevTri>     tSim3D;
typedef  cPtxd<tCoordDevTri,3>          tPt3D;
typedef  cPtxd<tCoordDevTri,2>          tPt2D;
typedef cTriangle<int,2> tTriPix;

class cGenerateSurfDevOri;
class cDevTriangu3d;

   /* ======================= */
   /* ======  header   ====== */
   /* ======================= */

/*
class cGenerateSurfDevOri
{
     public :
         cGenerateSurfDevOri(const cPt2di & aNb = cPt2di(15,5),double  aFactNonDev=0);

	 // tTriangulation3D cTriangulation(const tVPt& =tVPt(),const tVFace & =tVFace());
	 //
	 std::vector<tPt3D> VPts() const;
	 std::vector<cPt3di>   VFaces() const;

	 tPt3D PCenter() const;

     private :
	 int  NumOfPix(const cPt2di & aKpt) const; ///< Basic numerotation of points using video sens
	 tPt3D  Pt3OfPix(const tPt2D & aKpt) const; ///< Basic numerotation of points using video sens

	 cPt2di    mNb;
	 double    mFactNonDev;
	 bool      mPlaneCart; ///< if true generate a plane cartesian mesh

};
*/

/** Class that effectively compute the "optimal" devlopment of a surface
 * Separate from cAppliMeshDev to be eventually reusable
 */

class cDevTriangu3d
{
      public :
          typedef typename cTriangulation<tCoordDevTri,3>::tFace tFace;

          static constexpr int NO_STEP = -1;

          cDevTriangu3d(const  tTriangulation3D &);
          /// Generate the 2D devlopment of 3D surface
          void DoDevlpt ();
	  ///  Defautl face is not ok for real 3d surface
	  void SetFaceC(int aNumF);
          /// Export devloped surface as ply file
	  void ExportDev(const std::string &aName) const;
	  
          /// Show statistics on geometry preservation : distance & angles
	  void ShowQualityStat() const;

	  /// Given a face with a least 2 points devlopped and a num in [0,2], return a 2-D point for a conformal dev, if SetVP memo in mVPtsDev
	  tPt2D  DevConform(int aKFace,int aNumInTri,bool SetVP) ;

      private :
	  cDevTriangu3d(const cDevTriangu3d &) = delete;

	  void AddOneFace(int aKFace); // Mark the face and its sums as reached when not
	  std::vector<int>  VNumsUnreached(int aKFace) const; // subset of the 3 vertices not reached

	  // tPt3D

	  int MakeNewFace();

	  int               mNumCurStep;
          int               mNbFaceReached;
	  const tTriangulation3D & mTri;
	  std::vector<int>  mStepReach_S;  ///< indicate if a submit is selected and at which step
	  std::vector<tPt2D>  mVPtsDev; ///< Vector of devloped 2D points
	  // size_t                mLastNbSel;  
	  std::vector<int>  mStepReach_F;  ///< indicate if a face and at which step
          int               mIndexFC; ///< Index of centerface
	  cPt3di            mFaceC;
};



/* ******************************************************* */
/*                                                         */
/*               cGenerateSurfDevOri                       */
/*                                                         */
/* ******************************************************* */

/*
int  cGenerateSurfDevOri::NumOfPix(const cPt2di & aKpt)  const
{
     return aKpt.x() + aKpt.y() *  (mNb.x());
}


tPt3D  cGenerateSurfDevOri::Pt3OfPix(const tPt2D & aKpt) const
{
     if (mPlaneCart)
     {
        return tPt3D(aKpt.x(),aKpt.y(),0.0);
     }
     double aNbTour =  1.5;
     double aRatiobByTour = 1.5;

     double aAmpleTeta =  aNbTour * 2 * M_PI;

     double aTeta =  aAmpleTeta * ((double(aKpt.x())  / (mNb.x()-1) -0.5));
     double aRho = pow(aRatiobByTour,aTeta/(2*M_PI));
     aRho = aRho * (1 + (round_ni(aKpt.x()+aKpt.y())%2)* mFactNonDev);
     tPt2D  aPPlan = FromPolar(aRho,aTeta);
     double  aZCyl = (aKpt.y() * aAmpleTeta) / (mNb.x()-1);

     tPt3D  aPCyl(aPPlan.x(),aPPlan.y(),aZCyl);


     return tPt3D(aPCyl.y(),aPCyl.z(),aPCyl.x());

}

std::vector<tPt3D> cGenerateSurfDevOri::VPts() const
{
    std::vector<tPt3D> aRes(mNb.x()*mNb.y());

    for (const auto & aPix : cRect2(cPt2di(0,0),mNb))
    {
         aRes.at(NumOfPix(aPix)) = Pt3OfPix(ToR(aPix));
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
     return Pt3OfPix(ToR(mNb)/2.0);
}

cGenerateSurfDevOri::cGenerateSurfDevOri(const cPt2di & aNb,double aFactNonDev) :
     mNb          (aNb),
     mFactNonDev  (aFactNonDev),
     mPlaneCart   (false)
{
}
*/

/* ******************************************************* */
/*                                                         */
/*                 cAppliGenMeshDev                        */
/*                                                         */
/* ******************************************************* */

/*
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
   {
      auto aPtr = EqConsDist(true,100);
      auto aPtr2 = EqConsRatioDist(true,100);
      StdOut() << "DIFPTR "  << ((void*) aPtr2) <<  ((void *) aPtr) << "\n";
      delete aPtr;
      delete aPtr2;
   }


   // generate synthetic mesh
   cGenerateSurfDevOri aGenSD (cPt2di(15,5),mFactNonDev);
   tTriangulation3D  aTri(aGenSD.VPts(),aGenSD.VFaces());
   aTri.WriteFile(mNameCloudOut,mBinOut);
   aTri.MakeTopo();

   //  devlop it
   cDevTriangu3d aDev(aTri);
   aDev.SetFaceC(aTri.IndexClosestFace(aGenSD.PCenter()));
   aDev.DoDevlpt();
   aDev.ExportDev("Devlp_"+mNameCloudOut);

   aDev.ShowQualityStat();
   return EXIT_SUCCESS;
}
*/




/* ******************************************************* */
/*                                                         */
/*                    cDevTriangu3d                        */
/*                                                         */
/* ******************************************************* */

cDevTriangu3d::cDevTriangu3d(const tTriangulation3D & aTri) :
     mNumCurStep  (0),
     mNbFaceReached (0),
     mTri         (aTri),
     mStepReach_S (mTri.NbPts() ,NO_STEP),
     mVPtsDev     (mTri.NbPts()),
     mStepReach_F (mTri.NbFace(),NO_STEP),
     mIndexFC     (-1)
{
}

void cDevTriangu3d::AddOneFace(int aKFace)
{
     if (mStepReach_F.at(aKFace) != NO_STEP) return; // if face already reached nothing to do

     mNbFaceReached++;
     mStepReach_F.at(aKFace) = mNumCurStep; // mark it reached at current step

     // Mark all som of face that are not marked
     const tFace & aFace = mTri.KthFace(aKFace);
     for (int aNumV=0 ; aNumV<3 ; aNumV++)
     {
         int aKS= aFace[aNumV];
         if (mStepReach_S.at(aKS) == NO_STEP)
            mStepReach_S.at(aKS) = mNumCurStep;
     }
}

std::vector<int>  cDevTriangu3d::VNumsUnreached(int aKFace) const
{
     const tFace & aFace = mTri.KthFace(aKFace);

     std::vector<int>  aRes;
     for (int aNumV=0 ; aNumV<3 ; aNumV++)
     {
         int aKS= aFace[aNumV];
         if (mStepReach_S.at(aKS) == NO_STEP)
	 {
            aRes.push_back(aNumV);
	 }
     }
     return aRes;
}

int cDevTriangu3d::MakeNewFace()
{
     mNumCurStep++;
     // A Face is adjacent to reached iff it contains exactly 2 reached 
     std::vector<int> aVFNeigh;  // put first in vect to avoir recursive add
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         std::vector<int> aVU=VNumsUnreached(aKF);
         if (aVU.size()==1)
	 {
            aVFNeigh.push_back(aKF);  // memo it
            DevConform(aKF,aVU.at(0),true);  // mark the initial dev of Pts
	 }
     }
     // Now mark the faces 
     for (const auto & aKF : aVFNeigh)
         AddOneFace(aKF);

     // Mark face that containt 3 reaches vertices (may have been created
     // by previous step)
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         if (VNumsUnreached(aKF).size()==0)
            AddOneFace(aKF);
     }

     return aVFNeigh.size();
}


void cDevTriangu3d::SetFaceC(int aNumF)
{
    mIndexFC = aNumF;
    mFaceC   = mTri.KthFace(mIndexFC);
}

tPt2D   cDevTriangu3d::DevConform(int aKFace,int aNum0,bool SetIt)
{
    tTri3D  aTri = mTri.KthTri(aKFace);  // 3D triangle
    cPt3di aFace   = mTri.KthFace(aKFace);

    int aNum1 = (aNum0+1)%3;
    int aNum2 = (aNum0+2)%3;

    int aK0 = aFace[aNum0];
    int aK1 = aFace[aNum1];
    int aK2 = aFace[aNum2];

    tSim3D  aSim = tSim3D::FromTriInAndSeg(mVPtsDev[aK1],mVPtsDev[aK2],aNum1,aTri);
    tPt2D aPDev =  Proj(aSim.Value(mTri.KthPts(aK0)));
		    
    if (SetIt)
       mVPtsDev[aK0] = aPDev;

    return aPDev;
}


void  cDevTriangu3d::DoDevlpt ()
{
    // =======  1   DEVELOP FIRST FACE ===========
    
    //  1.1 get it : if initial face was not set, init it with IndexCenter (good for Z=F(x,y))
    if (mIndexFC==-1)
        SetFaceC(mTri.IndexCenterFace());

    // 1.2  Make face and soms marked
    AddOneFace(mIndexFC); 

    // 1.3 computes its geometry
    {
       tTri3D  aTriC = mTri.KthTri(mIndexFC);  // 3D triangle
       tIsom3D  anIsom =  tIsom3D::FromTriOut(0,aTriC).MapInverse();  // Isometry plane TriC-> plane Z=0

       for (int aK=0 ; aK<3 ;aK++)
       {
            mVPtsDev.at(mFaceC[aK]) =  Proj(anIsom.Value(aTriC.Pt(aK)));
	    // StdOut() << " Pt= " << mFaceC[aK] << " " << mTri.NbPts() << "\n";
	    // StdOut() << " Pt= " << anIsom.Value(aTriC.Pt(aK)) << "\n";
       }
       // getchar();
    }

    while (int aNbF=MakeNewFace())
    {
        StdOut() << "NnFF " << aNbF << " Step="  << mNumCurStep<< "\n";
    }


    StdOut() << " FRR " << mNbFaceReached << " " << mTri.NbFace() << "\n";
}

void  cDevTriangu3d::ExportDev(const std::string &aName) const
{
     std::vector<tPt3D>  aVPlan3;
     for (const auto & aP2 :  mVPtsDev)
        aVPlan3.push_back(TP3z0(aP2));

     tTriangulation3D aTriPlane(aVPlan3,mTri.VFaces());
     aTriPlane.WriteFile(aName,true);
}

void cDevTriangu3d::ShowQualityStat() const
{
     tCoordDevTri aSomDist=0;
     tCoordDevTri aSomEcDist=0;
     int   aNbEdge = 0;
     tCoordDevTri aSomAng=0;

     for (const auto & aFace : mTri.VFaces())
     {
         for (int aNum=0; aNum<3 ; aNum++)
         {
              int aKa = aFace[aNum];
              int aKb = aFace[(aNum+1)%3];
              int aKc = aFace[(aNum+2)%3];

	      tPt3D aP3a = mTri.KthPts(aKa);
	      tPt3D aP3b = mTri.KthPts(aKb);
	      tPt3D aP3c = mTri.KthPts(aKc);
              tPt2D aP2a = mVPtsDev.at(aKa);
              tPt2D aP2b = mVPtsDev.at(aKb);
              tPt2D aP2c = mVPtsDev.at(aKc);

	      tCoordDevTri aD3 = Norm2(aP3a-aP3b);
	      tCoordDevTri aD2 = Norm2(aP2a-aP2b);

	      aSomDist += aD3;
	      aSomEcDist += std::abs(aD3-aD2);

	      tCoordDevTri aA2 = AbsAngle(aP2a-aP2b,aP2a-aP2c);
	      tCoordDevTri aA3 = AbsAngle(aP3a-aP3b,aP3a-aP3c);

	      aSomAng += std::abs(aA2-aA3);

	      aNbEdge++;
         }
     }

     StdOut()  <<  "DIST : " << aSomEcDist / aSomDist  << " Angle " << aSomAng/aNbEdge   << "\n";

}

/* ******************************************************* */
/*                                                         */
/*                    cAppliMeshDev                        */
/*                                                         */
/* ******************************************************* */


/**  A basic application for clipping 3d data ,  almost all the job is done in
 * libraries so it essentially interface to command line */

class cAppliMeshDev : public cMMVII_Appli
{
     public :

        cAppliMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

           // --- Mandatory ----
	      std::string mNameCloudIn;
           // --- Optionnal ----
	      std::string mNameCloudOut;
	      bool        mBinOut;
           // --- Internal ----

};

cAppliMeshDev::cAppliMeshDev(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cAppliMeshDev::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileCloud})
   ;
}

cCollecSpecArg2007 & cAppliMeshDev::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file")
           << AOpt2007(mBinOut,CurOP_OutBin,"Generate out in binary format",{eTA2007::HDV})
   ;
}


int  cAppliMeshDev::Exe()
{
   InitOutFromIn(mNameCloudOut,"Clip_"+mNameCloudIn);

   tTriangulation3D  aTri(mNameCloudIn);
   // aTri.WriteFile(DirProject()+mNameCloudOut,mBinOut);

   cDevTriangu3d aDev(aTri);
   aDev.DoDevlpt();

   return EXIT_SUCCESS;
}



/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_MeshDev(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliMeshDev(aVArgs,aSpec));
}
cSpecMMVII_Appli  TheSpecMeshDev
(
     "MeshDev",
      Alloc_MeshDev,
      "Clip a point cloud/mesh  using a region",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);

/*
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
*/


};
