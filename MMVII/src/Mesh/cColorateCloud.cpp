#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PointCloud.h"


namespace MMVII
{

typedef cDataMapping<tREAL8,3,3> tProjPC;

class cOrthoProj  : public  tProjPC
{
    public :
       typedef std::vector<cPt3dr> tVecP3;

       cOrthoProj (const cPt3dr & aDir);
        
    private  :
       const  tVecP3 &  Values(tVecP3 &,const tVecP3 & ) const override;
       cPt3dr mDir;
};

cOrthoProj::cOrthoProj (const cPt3dr & aDir) :
   mDir (aDir/aDir.z())
{
}

const  std::vector<cPt3dr> &  cOrthoProj::Values(tVecP3 & aVOut,const tVecP3 & aVIn ) const 
{
   aVOut.clear();
   for (size_t aK=0 ; aK<aVIn.size() ; aK++)
   {
       const cPt3dr & aPIn = aVIn.at(aK);

       aVOut.push_back(cPt3dr(aPIn.x()-aPIn.z()*mDir.x(),aPIn.y()-aPIn.z()*mDir.y(),aPIn.z()));
   }
   return aVOut;
}


class cProjPointCloud
{
     public :
         cProjPointCloud(cPointCloud&,tREAL8 aSurResol );

         void ProcessOneProj(const tProjPC &,tREAL8 aW);
         
         void ColorizePC();
     private :
         cPointCloud&           mPC;
         tREAL8                 mSumW;
         std::vector<cPt3dr>    mVPtsInit;
         std::vector<cPt3dr>    mVPtsProj;
         std::vector<tREAL4>    mSumRad;
         tREAL8                 mSurResol;
};

cProjPointCloud::cProjPointCloud(cPointCloud& aPC,tREAL8 aSurResol) :
   mPC       (aPC),
   mSumW     (0.0),
   mSumRad   (aPC.NbPts(),0.0),
   mSurResol (aSurResol)
{
   mVPtsInit.reserve(aPC.NbPts());

   for (size_t aKPt=0 ; aKPt<aPC.NbPts() ; aKPt++)
       mVPtsInit.push_back(aPC.KthPt(aKPt));
   mVPtsProj = mVPtsInit;
}

void cProjPointCloud::ColorizePC()
{
   mPC.SetMulDegVis(1e4);
   for (size_t aK=0 ; aK<mVPtsProj.size() ; aK++)
   {
       mPC.SetDegVis(aK,mSumRad.at(aK)  / mSumW);
   }
}
void cProjPointCloud::ProcessOneProj(const tProjPC & aProj,tREAL8 aW)
{
     mSumW += aW;
     tREAL8 aMinInfty;

     aProj.Values(mVPtsProj,mVPtsInit);
     cTplBoxOfPts<tREAL8,2> aBOP;

     for (const auto & aPt : mVPtsProj)
     {
         aBOP.Add(Proj(aPt));
     }
     cBox2dr aBox = aBOP.CurBox();
     // tREAL8 aSurf = aBox.NbElem() /mVPtsProj.size();
     tREAL8 aAvgD = std::sqrt(1.0/mPC.Density());

     tREAL8 aStep = aAvgD / mSurResol;

     cTplBoxOfPts<int,2>  aBoxInd;
     std::vector<cPt2di>  mVIndexes;
     for (const auto & aPt : mVPtsProj)
     {
         cPt2di anInd = ToI((Proj(aPt)-aBox.P0()) / aStep);
         aBoxInd.Add(anInd);
         mVIndexes.push_back(anInd);
     }


     cPt2di aSzIm = aBoxInd.CurBox().Sz() + cPt2di(1,1);
     cIm2D<tREAL8> aImProf(aSzIm);
     cDataIm2D<tREAL8> & aDImProf = aImProf.DIm();
     aDImProf.InitCste(aMinInfty);

     std::vector<std::vector<cPt2di>> aVVdisk(256);
     for (int aK=0 ; aK<=255 ; aK++)
     {
         // tREAL8 aSzL = (2.0 * aAvgD) / aStep;
         // std::vector<cPt2di>  aVDisk =  VectOfRadius(-1,aSzL);
         tREAL8 aSzL = mPC.ConvertInt2SzLeave(aK);
         aVVdisk.at(aK) = VectOfRadius(-1,aSzL/aStep);
     }

     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++)
     {
         const cPt2di  & aCenter = mVIndexes.at(aKPt);
         tREAL8   aProf      = mVPtsProj.at(aKPt).z() - aAvgD/100.0;
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh : aVDisk)
         {
             cPt2di aPt = aCenter + aNeigh;
             if (aDImProf.Inside(aPt))
             {
                 aDImProf.SetMax(aPt,aProf);
             } 
         }
     }

     for (size_t aKPt=0 ; aKPt<mVPtsProj.size() ; aKPt++)
     {
         const cPt2di  & aCenter = mVIndexes.at(aKPt);
         tREAL8   aProf      = mVPtsProj.at(aKPt).z();
         int aNbVis = 0;
         const auto & aVDisk = aVVdisk.at(mPC.GetIntSzLeave(aKPt));
         for (const auto & aNeigh :aVDisk)
         {
             cPt2di aPt = aCenter + aNeigh;
             if (aDImProf.DefGetV(aPt,aMinInfty) <= aProf)
             {
                aNbVis++;
             } 
         }
         mSumRad.at(aKPt) +=  (aW * aNbVis) / aVDisk.size();
     }
}

/* =============================================== */
/*                                                 */
/*             cAppli_MMVII_CloudImProj            */
/*                                                 */
/* =============================================== */

class cAppli_MMVII_CloudImProj : public cMMVII_Appli
{
     public :

        cAppli_MMVII_CloudImProj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // --- Mandatory ----
	std::string   mNameCloudIn;
        // --- Optionnal ----
        std::string   mNameImageOut;
};

/*
cAppli_MMVII_CloudImProj::cAppli_MMVII_CloudImProj
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppli_MMVII_CloudImProj::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
   ;
}

cCollecSpecArg2007 & cAppli_MMVII_CloudImProj::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mNameImageOut,CurOP_Out,"Name of image  file, def= Ima+Input")
   ;
}
*/


/* =============================================== */
/*                                                 */
/*                 cAppli_MMVII_CloudColorate      */
/*                                                 */
/* =============================================== */


class cAppli_MMVII_CloudColorate : public cMMVII_Appli
{
     public :

        cAppli_MMVII_CloudColorate(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        // --- Mandatory ----
	std::string   mNameCloudIn;
        // --- Optionnal ----
        std::string mNameCloudOut;

        tREAL8   mPropRayLeaf;
        tREAL8   mSurResol;
        int      mNbSampS;
};

cAppli_MMVII_CloudColorate::cAppli_MMVII_CloudColorate
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli      (aVArgs,aSpec),
     mPropRayLeaf          (2.0),
     mSurResol         (2.0),
     mNbSampS          (5)
{
}

cCollecSpecArg2007 & cAppli_MMVII_CloudColorate::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
   ;
}


cCollecSpecArg2007 & cAppli_MMVII_CloudColorate::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
          << AOpt2007(mNameCloudOut,CurOP_Out,"Name of output file, def=Colorate_+InPut")
          << AOpt2007(mPropRayLeaf,"RayLeaves","Ray of leaves (/ avg dist)",{eTA2007::HDV})
          << AOpt2007(mSurResol,"SurResol","Sur resol in computation (/ avg dist)",{eTA2007::HDV})
          << AOpt2007(mNbSampS,"NbSampS","Number of sample/face for sphere discretization",{eTA2007::HDV})
   ;
}

int  cAppli_MMVII_CloudColorate::Exe()
{
   if (! IsInit(&mNameCloudOut))
      mNameCloudOut = "Colorate_"+ mNameCloudIn;

  
   cAutoTimerSegm aTSRead(TimeSegm(),"Read");
   cPointCloud   aPC_In ;
   ReadFromFile(aPC_In,mNameCloudIn);

   // generate the sz of leaves
   if (! aPC_In.SzLeavesIsInit())
   {
       aPC_In.SetLeavesUnit(0.05,SVP::Yes);  // fix unit step,
       tREAL8  aRayLeaf  = mPropRayLeaf  / std::sqrt(aPC_In.Density());
       for (size_t aKPt=0 ; aKPt<aPC_In.NbPts() ; aKPt++)
       {
           aPC_In.SetSzLeaves(aKPt,aRayLeaf);
       }
   }

   cAutoTimerSegm aTSInit(TimeSegm(),"Init");
   cProjPointCloud  aPPC(aPC_In,mSurResol);

    
   cAutoTimerSegm aTSProj(TimeSegm(),"1Proj");

   cSampleSphere3D aSampS(mNbSampS);
   int aNbStd=0;
   for (int aK=0 ; aK< aSampS.NbSamples() ; aK++)
   {
       cPt3dr aDir = aSampS.KthPt(aK);
       if (aDir.z() >= 0.2)
       {
           aPPC.ProcessOneProj( cOrthoProj(aDir),1.0);
           aNbStd++;
           StdOut() << "Still " << aSampS.NbSamples() - aK << "\n";
       }
   }

   aPPC.ProcessOneProj( cOrthoProj  (cPt3dr(0.0,1.0,1.0)),aNbStd);

   aPPC.ColorizePC();
   SaveInFile(aPC_In,mNameCloudOut);


   return EXIT_SUCCESS;
}

     /* =============================================== */
     /*                       ::                        */
     /* =============================================== */

tMMVII_UnikPApli Alloc_MMVII_CloudColorate(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_CloudColorate(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_MMVII_CloudColorate
(
     "CloudMMVIIColorate",
      Alloc_MMVII_CloudColorate,
      "Generate a colorate version of  MMVII-Cloud",
      {eApF::Cloud},
      {eApDT::Ply},
      {eApDT::Ply},
      __FILE__
);
#if (0)
#endif


};
