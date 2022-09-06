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

class cSomDevT3D;
class cFaceDevT3D;
class cDevTriangu3d;

   /* ======================= */
   /* ======  header   ====== */
   /* ======================= */

class cSomFace3D
{
     public :
        void SetReached(int aNumStepReach) { mNumStepReach=aNumStepReach;}
        bool IsReached()  const {return mNumStepReach >= 0;}

        cSomFace3D (cDevTriangu3d * aDevTri,int aNum) : 
             mDevTri       (aDevTri),
             mNumStepReach (-1) ,
             mNumObj       (aNum)
        {
        }
        int  NumObj() const {return mNumObj;} ///< accessor
     protected :
        cDevTriangu3d  *   mDevTri;
        int                mNumStepReach;
        int                mNumObj;
};

class cSomDevT3D : public cSomFace3D
{
    public :
        cSomDevT3D(cDevTriangu3d * aDevTri,int aNum,const tPt3D & aP3);
	void  AddPt2(const tPt2D & aPt,tCoordDevTri aWeight);
	const tPt2D & Pt2() const;
	const tPt3D & Pt3() const;
    private :
        tPt3D         mPt3;
        tPt2D         mPt2;
	tCoordDevTri  mSomWInit;  ///< sum of weight on initialization

};

class cFaceDevT3D : public cSomFace3D
{
   public :
      cFaceDevT3D (cDevTriangu3d * aDevTri,int aNumF,cPt3di aIndSom);
      int IndKthSom(int aK) const;

      void InitGeomAsFaceCenter();

      tCoordDevTri  DistortionDist() const;
      
   private :
      cPt3di           mIndSoms;
};


/** Class that effectively compute the "optimal" devlopment of a surface
 * Separate from cAppliMeshDev to be eventually reusable
 */

class cDevTriangu3d
{
      public :
          typedef typename cTriangulation<tCoordDevTri,3>::tFace tFace;

          static constexpr int NO_STEP = -1;

          cDevTriangu3d(tTriangulation3D &);
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

	  const tTriangulation3D & Tri() const;  ///< accessor
          cSomDevT3D&   KthSom(size_t aK) ; ///< access to mVSoms


      private :
	  cDevTriangu3d(const cDevTriangu3d &) = delete;
	  void  AddOneFace(int aKFace,bool IsFaceC);

	  void OldAddOneFace(int aKFace); ///< Mark the face and its sums as reached when not
	  std::vector<int>  VNumsUnreached(int aKFace) const; ///< subset of the 3 vertices not reached

	  // tPt3D

	  int MakeNewFace();

	  int               mNumCurStep;       ///< num of iteration in devlopment, +- dist to center face
          int               mNbFaceReached;    ///< number of face reached untill now
	  const tTriangulation3D & mTri;       ///< reference to the 3D-triangulation to devlop
          const cGraphDual &       mDualGr;    ///< reference to the dual graph
	  std::vector<int>  mStepReach_S;      ///< indicate if a submit is selected and at which step
	  std::vector<tPt2D>  mVPtsDev;        ///< Vector of devloped 2D points
	  std::vector<int>  mStepReach_F;      ///< indicate at which step a face is reached (NO_STEP while unreached)
          int               mOldIndexFC;          ///< Index of centerface
	  cPt3di            mFaceC;            ///< Center face
          int               mNumGen;           ///< Num Gen


          
          std::vector<cSomDevT3D>   mVSoms;
          std::vector<cFaceDevT3D>  mVFaces;
          std::vector<size_t>       mVReachedFaces;
          std::vector<size_t>       mVReachedSoms;
};


/* ******************************************************* */
/*                                                         */
/*                    cSomDevT3D                           */
/*                                                         */
/* ******************************************************* */

cSomDevT3D::cSomDevT3D(cDevTriangu3d * aDevTri,int aNum,const tPt3D & aP3):
    cSomFace3D    (aDevTri,aNum),
    mPt3          (aP3),
    mPt2          (0.0,0.0),  
    mSomWInit     (0.0)
{
}

void  cSomDevT3D::AddPt2(const tPt2D & aPt,tCoordDevTri aWeight)
{
     mPt2 = (mPt2*mSomWInit + aPt*aWeight) / (mSomWInit+aWeight);
     mSomWInit += aWeight;
}

const tPt2D & cSomDevT3D::Pt2() const
{
   MMVII_INTERNAL_ASSERT_tiny(mSomWInit>0,"Bad  cSomFace3D::Pt2");  
   return mPt2;
}
const tPt3D & cSomDevT3D::Pt3() const { return mPt3; }


/* ******************************************************* */
/*                                                         */
/*                    cFaceDevT3D                          */
/*                                                         */
/* ******************************************************* */

cFaceDevT3D::cFaceDevT3D(cDevTriangu3d * aDevTri,int aNumF,cPt3di aIndSom) :
    cSomFace3D    (aDevTri,aNumF),
    mIndSoms      (aIndSom)
{
}

int cFaceDevT3D::IndKthSom(int aK) const {return mIndSoms[aK];}

void cFaceDevT3D::InitGeomAsFaceCenter()
{
    tTri3D  aTriC = mDevTri->Tri().KthTri(mNumObj);  // 3D triangle

    int aInd = aTriC.IndexLongestSeg(); // init on longer side for better stability on fix-var
    tIsom3D  aIsometry =  tIsom3D::FromTriOut(aInd,aTriC).MapInverse();   // get rotation Tri-> Plane 0XY
    for (int aK=0; aK<3 ; aK++)
    {
	tPt3D aP1 = aTriC.PtCirc(aK);
	tPt3D aQ1 = aIsometry.Value(aP1);
	int aIndSom =  mIndSoms[aK];
	mDevTri->KthSom(aIndSom).AddPt2(Proj(aQ1),1.0);
        // if  we are in mode bench, then make some litle check
	if (The_MMVII_DebugLevel>=The_MMVII_DebugLevel_InternalError_tiny )
	{
           tPt3D aP2 = aTriC.PtCirc(aK+1);
           tPt3D aQ2 = aIsometry.Value(aP2);
           MMVII_INTERNAL_ASSERT_bench(std::abs(aQ1.z())<1e-10,"z-init in dev surf"); // check on plane Z=0
           MMVII_INTERNAL_ASSERT_bench(std::abs(Norm2(aP1-aP2) - Norm2(aQ1-aQ2))<1e-10,"dist-init in dev surf"); // chek isometry
	   if (aK==aInd)
	   {
               MMVII_INTERNAL_ASSERT_bench(Norm2(aQ1)<1e-10,"z-init in dev surf");  // Check firt point is 0
               MMVII_INTERNAL_ASSERT_bench(std::abs(aQ2.y())<1e-10,"z-init in dev surf"); // check  first seg on OX
	   }
	}
    }
}

tCoordDevTri cFaceDevT3D::DistortionDist() const
{
    tCoordDevTri aSomDif = 0;
    tCoordDevTri aSomDist = 0;
    for (int aK=0 ; aK<3 ; aK++)
    {
       cSomDevT3D&  aS1 =   mDevTri->KthSom(mIndSoms[aK]);
       cSomDevT3D&  aS2 =   mDevTri->KthSom(mIndSoms[(aK+1)%3]);
       
       tCoordDevTri aD2 = Norm2(aS1.Pt2()-aS2.Pt2());
       tCoordDevTri aD3 = Norm2(aS1.Pt3()-aS2.Pt3());
       aSomDif += std::abs(aD2-aD3);
StdOut() <<  "KK " << aK  <<  " DD=" << std::abs(aD2-aD3) << "\n";
       aSomDist += aD2+aD3;
    }

    return aSomDif / aSomDist;
}


/* ******************************************************* */
/*                                                         */
/*                    cDevTriangu3d                        */
/*                                                         */
/* ******************************************************* */


cDevTriangu3d::cDevTriangu3d(tTriangulation3D & aTri) :
     mNumCurStep  (0),
     mNbFaceReached (0),
     mTri         (aTri),
     mDualGr      (mTri.DualGr()),
     mStepReach_S (mTri.NbPts() ,NO_STEP),
     mVPtsDev     (mTri.NbPts()),
     mStepReach_F (mTri.NbFace(),NO_STEP),
     mOldIndexFC     (NO_STEP),
     mFaceC       (NO_STEP,NO_STEP,NO_STEP),
     mNumGen      (0)
{
   //  generate topology
   aTri.MakeTopo();

   // create the vector of points
   for (size_t aKPt=0 ; aKPt<mTri.NbPts() ; aKPt++)
   {
      mVSoms.push_back(cSomDevT3D(this,aKPt,mTri.KthPts(aKPt)));
   }

   // create the vector of faces
   for (size_t aKF=0 ; aKF<mTri.NbFace() ; aKF++)
   {
       mVFaces.push_back(cFaceDevT3D(this,aKF,mTri.KthFace(aKF)));
   }

   AddOneFace(mTri.IndexCenterFace(),true);
   size_t aIndNewF0 = 0;

   // iterate as long as we found new soms
   while (aIndNewF0!=mVReachedFaces.size())
   {
        mNumGen++;
        StdOut() << "IIIi " << aIndNewF0 << " " << mVReachedFaces.size() << "\n"; 
        size_t aIndNewF1 = mVReachedFaces.size();  // memorize size to avoid doing everything in one step

	// parse all face that where reached in previous step
        for (size_t aIndFace=aIndNewF0 ; aIndFace<aIndNewF1 ; aIndFace++)
        {
            std::vector<int> aVN;
	    int aOldF = mVReachedFaces.at(aIndFace);
            mDualGr.GetFacesNeighOfFace(aVN,aOldF); // get all face touching old one
            for (const auto & aNewF : aVN) // parse "new" face
            {
                 cFaceDevT3D & aFace = mVFaces.at(aNewF);
                 if (! aFace.IsReached())  // are they really new ?
                 {
                     AddOneFace(aNewF,false);
                 }
            }
        }

        aIndNewF0 = aIndNewF1;
   }

   MMVII_INTERNAL_ASSERT_tiny(mVReachedFaces.size()==mTri.NbFace(),"in Dev : Pb in reached face");  // Check firt point is 0
   MMVII_INTERNAL_ASSERT_tiny(mVReachedSoms.size() ==mTri.NbPts (),"in Dev : Pb in reached face");  // Check firt point is 0
   StdOut() << mVReachedFaces.size() << " " << mTri.NbFace() << "\n";
   StdOut() << mVReachedSoms.size() << " " << mTri.NbPts() << "\n";
   StdOut() << "Wwwww\n"; getchar();
}

const tTriangulation3D & cDevTriangu3d::Tri() const {return  mTri;}
cSomDevT3D&   cDevTriangu3d::KthSom(size_t aKSom) {return mVSoms.at(aKSom);}

void  cDevTriangu3d::AddOneFace(int aKFace,bool IsFaceC)
{
   cFaceDevT3D & aFace = mVFaces.at(aKFace);
   aFace.SetReached(mNumGen);
   mVReachedFaces.push_back(aKFace);

   size_t aNbS0 = mVReachedSoms.size();
   int aIndK0=-1;
   for (int aK3=0 ; aK3<3; aK3++)
   {
       cSomDevT3D & aSom = mVSoms.at(aFace.IndKthSom(aK3));
       if (! aSom.IsReached())
       {
           aSom.SetReached(mNumGen);
	   mVReachedSoms.push_back(aSom.NumObj());
	   aIndK0=aK3;
       }
   }
   // StdOut() <<  "HHHH  " << mVReachedSoms.size() - aNbS0 << "\n";

   int aNbNewS = mVReachedSoms.size()-aNbS0;
   if (IsFaceC)
   {
      MMVII_INTERNAL_ASSERT_tiny((aNbNewS==3),"Bad size for AddOneFace");
      aFace.InitGeomAsFaceCenter(); //  init the geometry using rotation
   }
   else
   {
      MMVII_INTERNAL_ASSERT_tiny(aNbNewS<=1,"Bad size for AddOneFace");
      if (aNbNewS)
      {
          int aIndK1 = (aIndK0+1)%3;
          int aIndK2 = (aIndK0+2)%3;

          tPt2D   aP1 = mVSoms.at(aFace.IndKthSom(aIndK1)).Pt2();
          tPt2D   aP2 = mVSoms.at(aFace.IndKthSom(aIndK2)).Pt2();
          tTri3D  aTri3D = mTri.KthTri(aKFace);  // 3D triangle
          tSim3D  aSim = tSim3D::FromTriInAndSeg(aP1,aP2,aIndK1,aTri3D);
          tPt2D   aPDev =  Proj(aSim.Value(mTri.KthPts(aIndK0)));


          mVSoms.at(aFace.IndKthSom(aIndK0)).AddPt2(aPDev,1.0);

          StdOut() << "DIST " << aFace.DistortionDist() <<  aSim.Value(mTri.KthPts(aIndK0)) << "\n";
          StdOut() <<  aSim.Value(mTri.KthPts(aIndK1)) << " " <<  aSim.Value(mTri.KthPts(aIndK2)) << "\n";
          getchar();
      }
   }

}

/*

void cDevTriangu3d::OldAddOneFace(int aKFace)
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
     const tFace & aFace = mTri.KthFace(aKFace);  // get the face

     std::vector<int>  aRes;
     for (int aNumV=0 ; aNumV<3 ; aNumV++) // parse 3 submiy
     {
         int aKS= aFace[aNumV];  // get the submit
         if (mStepReach_S.at(aKS) == NO_STEP) // if not reached
	 {
            aRes.push_back(aNumV);  // add it
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
         OldAddOneFace(aKF);

     // Mark face that containt 3 reaches vertices (may have been created
     // by previous step)
     for (int aKF=0 ; aKF<mTri.NbFace() ; aKF++)
     {
         if (VNumsUnreached(aKF).size()==0)
            OldAddOneFace(aKF);
     }

     return aVFNeigh.size();
}


void cDevTriangu3d::SetFaceC(int aNumF)
{
    mOldIndexFC = aNumF;
    mFaceC   = mTri.KthFace(mOldIndexFC);
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
    if (mOldIndexFC==-1)
        SetFaceC(mTri.IndexCenterFace());

    // 1.2  Make face and soms marked
    OldAddOneFace(mOldIndexFC); 

    // 1.3 computes its geometry
    {
       tTri3D  aTriC = mTri.KthTri(mOldIndexFC);  // 3D triangle
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
*/

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
   InitOutFromIn(mNameCloudOut,"Dev_"+mNameCloudIn);

   tTriangulation3D  aTri(mNameCloudIn);

   cDevTriangu3d aDev(aTri);



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


};
