#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PointCloud.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Interpolators.h"
#include "MMVII_PCSens.h"

#include "cColorateCloud.h"

namespace MMVII
{

/* =============================================== */
/*                                                 */
/*                 cAppli_MMVII_CloudColorate      */
/*                                                 */
/* =============================================== */

enum eModeCloudCol
{
    eColShade,
    eColZ,
    eColXY
};

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

        cPt2dr   mPropRayLeaf;
        tREAL8   mSurResol;
        int      mNbSampS;
        cPt3dr   mSun;
        bool     mShowMsg;
        bool     mExportIm;
        bool     mProfIsZ0;
        int           mIMCol;
        eModeCloudCol mModeCol;
};

cAppli_MMVII_CloudColorate::cAppli_MMVII_CloudColorate
(
     const std::vector<std::string> & aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli    (aVArgs,aSpec),
     mPropRayLeaf    (1.0,1.0),
     mSurResol       (2.0),
     mNbSampS        (5),
     mShowMsg        (false),
     mExportIm       (false),
     mProfIsZ0       (false),
     mIMCol          (0),
     mModeCol        (eModeCloudCol::eColShade)
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
          << AOpt2007(mSun,"Sun","Sun : Dir3D=(x,y,1)  ,  Z=WEIGHT !! ")
          << AOpt2007(mShowMsg,"ShowMsg","Print detailled message at each computation",{{eTA2007::HDV},{eTA2007::Tuning}})
          << AOpt2007(mExportIm,"ExportIm","Export all individual images",{{eTA2007::HDV},{eTA2007::Tuning}})
          << AOpt2007(mProfIsZ0,"ProfIsZ0","Prof is ZInit/\"Z in Dir proj\"",{{eTA2007::HDV},{eTA2007::Tuning}})
          << AOpt2007(mIMCol,"ICol","Col mode 0-Shde 1-Z 2-XY")
  ;
}


int  cAppli_MMVII_CloudColorate::Exe()
{
   if (! IsInit(&mNameCloudOut))
      mNameCloudOut = "Colorate_"+ mNameCloudIn;

  
   mModeCol = eModeCloudCol(mIMCol);
   cAutoTimerSegm aTSRead(TimeSegm(),"Read");
   cPointCloud   aPC_In ;
   ReadFromFile(aPC_In,mNameCloudIn);

   StdOut() << "ColNB COLIN=" << aPC_In.GetNbColours() << "\n";

   // generate the sz of leaves
   if (! aPC_In.LeavesIsInit())
   {
       aPC_In.SetLeavesUnit(0.05,SVP::Yes);  // fix unit step,
       for (size_t aKPt=0 ; aKPt<aPC_In.NbPts() ; aKPt++)
       {
           tREAL8  aRayLeaf  = RandInInterval(mPropRayLeaf)  * aPC_In.GroundSampling();
           aPC_In.SetSzLeaves(aKPt,aRayLeaf);
       }
   }

   cAutoTimerSegm aTSInit(TimeSegm(),"Init");
   tREAL8 aWeightInit = (mNbSampS==0);
   cProjPointCloud  aPPC(aPC_In,aWeightInit);  // Weight Init 0 if NbS ,  
// cProjPointCloud::cProjPointCloud(cPointCloud& aPC,tREAL8 aWeightInit) :

    
   cAutoTimerSegm aTSProj(TimeSegm(),"1Proj");

   int aNbStd=0;
   aPC_In.SetMulDegVis(1e4);
   if (mModeCol ==eModeCloudCol::eColShade)
   {
       if (mNbSampS>0)
       {
           cSampleSphere3D aSampS(mNbSampS);
           for (int aK=0 ; aK< aSampS.NbSamples() ; aK++)
           {
               cPt3dr aDir = VUnit(aSampS.KthPt(aK));
               if (aDir.z() >= 0.2)
               {
                   std::unique_ptr<cCamOrthoC> aCam (aPPC.PPC_CamOrtho(aK,mProfIsZ0,aDir));
                   cPt3di aDirI = ToI(aDir*100.0);
                   std::string aMsg = ToStr(aDirI.x()) + "_" +  ToStr(aDirI.y()) + "_" +  ToStr(aDirI.z());
                   aPPC.ProcessOneProj(mSurResol,*aCam,1.0,false,aMsg,mShowMsg,mExportIm);
                   aNbStd++;
                   StdOut() << "Still " << aSampS.NbSamples() - aK << "\n";
               }
           }
        }
        if (IsInit(&mSun))
        {
            tREAL8 aW0  = mNbSampS ? aNbStd : 1.0;
            std::unique_ptr<cCamOrthoC> aCam (aPPC.PPC_CamOrtho(0,mProfIsZ0,cPt3dr(mSun.x(),mSun.y(),1.0)));
            aPPC.ProcessOneProj(mSurResol,*aCam,aW0 * mSun.z(),false,"",false,false);
        }
        aPPC.ColorizePC();
   }
   else
   {
       cBoundVals<tREAL8> aBounds;
       for (size_t aKPt=0 ; aKPt<aPC_In.NbPts(); aKPt++)
       {
           cPt3dr aPt = aPC_In.KthPt(aKPt);
           tREAL8 aDegVis=0.0;
           if (mModeCol==eModeCloudCol::eColXY)
           {
               int Ix=  round_ni(aPt.x());
               int Iy=  round_ni(aPt.y());

              // aDegVis =   ((1+std::sin(aPt.x() /3.0 )) * (1+std::sin(aPt.y() /9.0 )))  / 4.0 ;
               aDegVis =   (Ix/ 100 + Iy/100)%2;

              // aDegVis =  ((Ix%60)<5) || ((Iy%60)<5);
              // aDegVis = aDegVis>0.5;
           }
           aPC_In.SetDegVis(aKPt,aDegVis);
           aBounds.Add(aPC_In.GetDegVis(aKPt) );
       }

       int aNbT=-1;
       for (int aKT=0 ; aKT<aNbT ; aKT++)
       {
           int aKPt = (aPC_In.NbPts()*aKT) / aNbT;
           StdOut() << aPC_In.GetDegVis(aKPt) << "\n";
       }
       StdOut() << "ColNB COLOUT=" << aPC_In.GetNbColours() << " DVInt="<< aBounds.VMin() << " " << aBounds.VMax() << "\n";

   }


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


};
