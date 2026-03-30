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
/*             cAppli_MMVII_CloudImProj            */
/*                                                 */
/* =============================================== */

class cAppli_MMVII_CloudImProj : public cMMVII_Appli
{
     public :

        cAppli_MMVII_CloudImProj(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,int aMode);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        void ProcessOrthoMode(cPointCloud  &,cProjPointCloud&);
        void ProcessConikMode(cPointCloud  &,cProjPointCloud&);

        int                     mMode;
        cPhotogrammetricProject mPhProj;
        // --- Mandatory ----
        std::string   mNameCloudIn;
        // --- Optionnal ----
        tREAL8  mSurResolSun;
        std::string   mPrefixOut;
        bool mMakeImRectified;
        bool mSaveImDepth;
        std::string mPrefixImGen ;


        tREAL8        mResolOrthoC;
        cPt2di        mSzIm;
        cPt2dr        mOverLap;
        tREAL8        mRInsideMin;
        tREAL8        mFOV;
        //cPt2di        mNbBande;
        //cPt2dr        mBSurH;
        
        cPerspCamIntrCalib * mCalib;

        cPt3dr        mSun;
        std::string   mNameSavePCSun;
        std::vector<tREAL8>  mParamP;
        std::vector<int>     mVDeltaPax;  /// Delta for computing depth like paralax
        bool                 mShow;


        tREAL8               mSensDownSample; /// Sub
        tREAL8               mSurResCloud;
};

cAppli_MMVII_CloudImProj::cAppli_MMVII_CloudImProj
(
     const std::vector<std::string> &    aVArgs,
     const cSpecMMVII_Appli &            aSpec,
     int                                 aMode
) :
     cMMVII_Appli      (aVArgs,aSpec),
     mMode             (aMode),
     mPhProj           (*this),
     mSurResolSun      (2.0),
     mMakeImRectified  (false),
     mSaveImDepth      (false),
     mPrefixImGen      ("C_"),
     mResolOrthoC      (0.2),
     mOverLap          (0.8,0.6),
     mRInsideMin      (0.5),
     mFOV              (0.7),
  //   mNbBande          (5,1),
  //   mBSurH            (0.1,0.2),
   //  mFocal            (-1),
     mCalib            (nullptr),
     mVDeltaPax        {-1,1},
     mShow             (false),
     mSensDownSample   (2.0),
     mSurResCloud      (2.0)

{
    FakeUseIt(mResolOrthoC);
}

cCollecSpecArg2007 & cAppli_MMVII_CloudImProj::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
  anArgObl
	  <<   Arg2007(mNameCloudIn,"Name of input cloud/mesh", {eTA2007::FileDirProj,eTA2007::FileDmp})
   ;

  if (mMode==0)
  {
     anArgObl
        <<  Arg2007(mParamP,"Param of projection [Phi,Nb,Teta1,Teta2?=-T1,NbE?=1],",{{eTA2007::ISizeV,"[3,5]"}})
      ;
  }
  else if (mMode==1)
  {
      anArgObl
          << Arg2007(mSzIm,"Size of resulting image")
          << mPhProj.DPOrient().ArgDirOutMand()
      ;
  }

   return anArgObl;
}

cCollecSpecArg2007 & cAppli_MMVII_CloudImProj::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   anArgOpt
          << AOpt2007(mPrefixOut,CurOP_Out,"Preifix for out images, def= Ima+Input")
          << AOpt2007(mSun,"Sun","Sun : Dir3D=(x,y,1)  ,  Z=WEIGHT !! ")
          << AOpt2007(mNameSavePCSun,"CloudSun","Name of cloud with sun, if sun was added")
          << AOpt2007(mShow,"Show","Show images &  messages for tuning",{eTA2007::HDV})
   ;

   if (mMode==0)
   {
       anArgOpt
           << AOpt2007(mVDeltaPax,"DeltaPax","Delta between image were compte paralax",{eTA2007::HDV})
           << AOpt2007(mMakeImRectified,"MkImSupRect","Generate Superposed Rectified Images",{eTA2007::HDV})
           << AOpt2007(mSaveImDepth,"SaveImD","Save depth per image",{eTA2007::HDV})
       ;
   }
   else if (mMode==1)
   {
       anArgOpt
           << AOpt2007(mOverLap,"Overlap","Ratio of overlap between images",{eTA2007::HDV})
           << AOpt2007(mRInsideMin,"MinInside","Minimal insideness ratio ",{eTA2007::HDV})
           << AOpt2007(mFOV,"FOV","Field of view, in radian",{eTA2007::HDV})
           << AOpt2007(mPrefixImGen,"PrefixIm","Prefix for generating names",{eTA2007::HDV})
       ;
   }

   return anArgOpt;

}

int  cAppli_MMVII_CloudImProj::Exe()
{
   mPhProj.FinishInit();
   if (!IsInit(&mPrefixOut))
      mPrefixOut =  "ImProj_" + LastPrefix(mNameCloudIn) ;


   cPointCloud   aPC_In ;
   ReadFromFile(aPC_In,mNameCloudIn);

   cProjPointCloud  aPPC(aPC_In,1.0);


   if  (IsInit(&mSun))
   {
      // cPt3dr aDirSun = VUnit(cPt3dr(mSun.x(),mSun.y(),1.0));

       cPt3dr aDirSun = VertSphericalDir(mSun);
       if (mShow)
          StdOut() << " SUN, cart:" << aDirSun  << "\n";
       std::unique_ptr<cCamOrthoC> aCam (aPPC.PPC_CamOrtho(0,false,aDirSun));

       aPPC.ProcessOneProj(mSurResolSun,*aCam,mSun.z(),false,"",false,false);

       aPPC.ColorizePC();

       if (IsInit(&mNameSavePCSun))
           SaveInFile(aPC_In,mNameSavePCSun);
   }

   if (mMode==0)
   {
       ProcessOrthoMode(aPC_In,aPPC);
   }
   else
   {
       ProcessConikMode(aPC_In,aPPC);
   }

   return EXIT_SUCCESS;
}


void cAppli_MMVII_CloudImProj::ProcessConikMode(cPointCloud  & aPC_In,cProjPointCloud& aPPC)
{
    aPPC.SetComputeProfMax(false);
    //cBox3dr   aBox3 = aPC_In.Box3d();
    cBox2dr   aBox2Glob = aPC_In.Box2d();
    cPt3dr aCenter = aPC_In.Centroid();

    tREAL8 aFocal = Norm2(mSzIm) / mFOV ;
    mCalib = cPerspCamIntrCalib::SimpleCalib
            (
                "Cam-" + mPrefixImGen,
                eProjPC::eStenope,
                mSzIm,  // sz
                cPt3dr(mSzIm.x()/2.0,mSzIm.y()/2.0,aFocal),  // PP + F
                cPt3di(0,0,0)  // degree
            );

    tREAL8  aResol = aPC_In.GroundSampling() *  mSensDownSample;

    tREAL8  aHeight = aResol * aFocal;
    tREAL8  aZ     = aCenter.z() + aHeight;
    cPt2dr aGFootP = ToR(mSzIm) * aResol;   // Ground Foot Print
    cPt2dr aDeltaIm = MulCByC(aGFootP,cPt2dr(1.0,1.0) - mOverLap);

    cPt2di aNb = Pt_round_up(DivCByC(aBox2Glob.Sz(),aDeltaIm*2.0));


    cRect2  aRect(-aNb,aNb+cPt2di(1,1));

    for (const auto & aPix : aRect)
    {
        cPt2dr aC2 = aBox2Glob.Middle() + MulCByC(ToR(aPix),aDeltaIm);
        cBox2dr aBoxLoc(aC2-aGFootP/2.0,aC2+aGFootP/2.0);
        cBox2dr aBoxI = aBoxLoc.Inter(aBox2Glob);
        tREAL8 aRatio = aBoxI.NbElem() / aBoxLoc.NbElem();

        if (aRatio > mRInsideMin)
        {
            cPt3dr aC3(aC2.x(),aC2.y(),aZ);
            tPoseR aPose(aC3,tRotR::RotFromCanonicalAxes("i-j-k"));
            std::string aPrefix =  mPrefixImGen + ToStr(aPix.x()+aNb.x()) + "-" +  ToStr(aPix.y()+aNb.y());

            std::string aNameImage = aPrefix +"-Radiom-"+".tif";
            cSensorCamPC aCam(aNameImage,aPose,mCalib);


             std::string aDirIm =  mPhProj.DPOrient().FullDirOut();


            aPPC.ProcessOneProj(mSurResCloud*mSensDownSample,aCam,0.0,true,"",false,false); // HERE
            cResImagesPPC aResIm = aPPC.ProcessImage(mSurResCloud*mSensDownSample,aCam);

            aResIm.mImRadiom.DIm().ToFile(aDirIm+aNameImage);
            aResIm.mImWeight.DIm().ToFile(aDirIm+aPrefix+"Weight-"+".tif");
            aResIm.mImDepth.DIm().ToFile (aDirIm+aPrefix+"Depth-" +".tif");

            mPhProj.SaveCamPC(aCam);

            StdOut() << "Doneee=" << aNameImage << "\n";
        }

    }
    //mOverLap
    StdOut() << " RESOL= " << aResol << " H=" << aHeight << " aNb=" << aNb  << " DELTAIM=" << aDeltaIm << "\n";
            //aPC_In.Centroid()

    delete mCalib;

}


void cAppli_MMVII_CloudImProj::ProcessOrthoMode(cPointCloud  & aPC_In,cProjPointCloud& aPPC)
{
    // Prof is Z0 is required for depth to have the same meaning in both image for paralax compute
    bool mProfIsZ0 = true;


//   int aNbPos = 1;
  // tREAL8 aSensDownSample = 2.0;
  // tREAL8 aSurResCloud = 2.0;
   // tREAL8 aSousResIm = 0.5;

   // [Phi,Nb,Teta1,Teta2?=-T1,NbE?=1], [Phi,Nb,Teta1,Teta2?=-T1,NbE?=1],
   tREAL8 aPhi = mParamP.at(0) * M_PI/2;
   int    aNb = round_ni(mParamP.at(1));
   tREAL8 aTeta1 = mParamP.at(2);
   tREAL8 aTeta2 = GetDef(mParamP,3,-aTeta1);

   // Compute the orthographic camera in aVCamO, we do it in advance for epip
   std::vector<cCamOrthoC *> aVCamO;
   for (int aKT=0 ; aKT<= aNb ; aKT++)
   {
       tREAL8 aTeta =  (aTeta1*(aNb-aKT) + aTeta2*aKT) / aNb;
       aTeta = M_PI/2 + aTeta;
       cPt3dr aAxeK = spher2cart(cPt3dr(aPhi,aTeta,1.0));
       cPt3dr aAxeI = spher2cart(cPt3dr(aPhi,aTeta+0.5,1.0));
       cPt3dr aAxeJ = VUnit(aAxeK ^ aAxeI);
       aAxeI = VUnit(aAxeJ^aAxeK);
       tRotR aRot(aAxeI,aAxeJ,aAxeK,false);


       cCamOrthoC* aCamO = aPPC.PPC_CamOrtho(aKT,mProfIsZ0,aRot,mSensDownSample);
       aVCamO.push_back(aCamO);
   }

   typedef std::pair<cPt2di,cIm2D<tREAL4>> tPPax;
   std::vector<cIm2D<tU_INT1>> aVImRad;
   std::vector<tPPax>          aVPPax;
   std::string aDir = mPhProj.DirVisuAppli();
   // use them to generate images
   for (size_t aKC1=0 ; aKC1<aVCamO.size() ; aKC1++)
   {
       cCamOrthoC* aCam1 =  aVCamO.at(aKC1);

       if (mShow)
           StdOut() << "Doing image : " << aCam1->NameImage() << " SzPixInit=" << aCam1->Sz() << "\n";
       aPPC.ProcessOneProj(mSurResCloud*mSensDownSample,*aCam1,0.0,true,"",false,false); // HERE
       cResImagesPPC aResIm = aPPC.ProcessImage(mSurResCloud*mSensDownSample,*aCam1);

      std::string aPost =aCam1->NameImage()+".tif";

      aResIm.mImRadiom.DIm().ToFile(aDir+mPrefixOut+ "Radiom"+aPost);
      if (mMakeImRectified)
          aVImRad.push_back(aResIm.mImRadiom);

      aResIm.mImWeight.DIm().ToFile(aDir+mPrefixOut+ "Weight"+aPost);

      if (mSaveImDepth)
      {
         aResIm.mImDepth.DIm().ToFile(aDir+mPrefixOut+ "Depth"+aPost);
      }


      // compute the vector images of pax
      for (const auto aDeltaPx : mVDeltaPax)
      {
          // cCamOrthoC * aCam2 = aVCamO.at(aKPax);
          int aKC2 = (int)aKC1 + aDeltaPx;
          if ((aKC2>=0)&& (aKC2<int(aVCamO.size())))
          {
              if (mShow)
                  StdOut() << "  * doing pair : " << aCam1->NameImage() << " " << aCam1->NameImage() << "\n";
              tREAL8 aMaxPxTrsv;
              cCamOrthoC * aCam2 = aVCamO.at(aKC2);
              cIm2D<tREAL4> aImPax =  aCam1->ImageDepth2ImagePax(aResIm.mImDepth,*aCam2,&aMaxPxTrsv);

             /* StdOut()  << "KKPax " << aCam1->NameImage()
                        << aCam2->NameImage()
                        << " " << aMaxPxTrsv << "\n";*/

              aImPax.DIm().ToFile(aDir+mPrefixOut+"Px"+aCam1->NameImage()+"-"+aCam2->NameImage()+".tif");

              if (mMakeImRectified)
              {
                  aVPPax.push_back(tPPax(cPt2di(aKC1,aKC2),aImPax));
              }
          }
      }
   }

   if (mMakeImRectified)
   {
       auto anInterp = new   cTabulatedDiffInterpolator(cSinCApodInterpolator(5.0,5.0));

       for (const auto & [aPt,aImPax] : aVPPax)
       {
           cCamOrthoC* aCam1 =  aVCamO.at(aPt.x());
           cCamOrthoC* aCam2 =  aVCamO.at(aPt.y());

           const cDataIm2D<tU_INT1> & aDI1 = aVImRad.at(aPt.x()).DIm();
           const cDataIm2D<tU_INT1> & aDI2 = aVImRad.at(aPt.y()).DIm();
           const cDataIm2D<tREAL4> & aDPax = aImPax.DIm();
           cRGBImage aImSuperp(aDI1.Sz());

           for (const auto & aPix1 : aDI1)
           {
               tREAL8 aR1 = aDI1.GetV(aPix1);
               cPt2dr aPix2 = ToR(aPix1) + cPt2dr(aDPax.GetV(aPix1),0.0);
               tREAL8 aR2 = 0.0;
               if (aDI2.InsideInterpolator(*anInterp,aPix2))
                   aR2 = aDI2.GetValueInterpol(*anInterp,aPix2);
               aImSuperp.SetRGBPix(aPix1,aR1,aR2,aR2);
           }

           aImSuperp.ToFile(aDir+mPrefixOut+"Superp" +aCam1->NameImage()+"-"+aCam2->NameImage() + ".tif");
           StdOut() << " SZ=" << aDI1.Sz()  << aDI2.Sz() << "\n";
       }
       delete anInterp;
   }

   // We test the "perfect" epipolarity of the geometry ...
   for (size_t aKC1=0 ; aKC1<aVCamO.size() ; aKC1++)
   {
       for (size_t aKC2=aKC1+1 ; aKC2<aVCamO.size() ; aKC2++)
       {
           cBox3dr   aBox3 = aPC_In.Box3d();
           for (int aKPt =0 ; aKPt < 5 ; aKPt++)
           {
               cPt3dr aPt = aBox3.GeneratePointInside();
               cPt2dr aPIm1 = aVCamO.at(aKC1)->Ground2Image(aPt);
               cPt2dr aPIm2 = aVCamO.at(aKC2)->Ground2Image(aPt);

               tREAL8 aDY = std::abs(aPIm1.y()-aPIm2.y());
               MMVII_INTERNAL_ASSERT_always(aDY<1e-5,"Epip in CloudMMVIIImProj");
           }
       }
   }
   DeleteAllAndClear(aVCamO);

   StdOut() << "NbLeaves "<< aPC_In.LeavesIsInit () << "\n";

}

     /* =============================================== */
     /*                       MMVII                     */
     /* =============================================== */

tMMVII_UnikPApli Alloc_MMVII_GT_EpipOrthoC(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_CloudImProj(aVArgs,aSpec,0));
}

cSpecMMVII_Appli  TheSpec_MMVII_GT_EpipOrthoC
(
     "CloudMMVII_GT_EpipOrthoC",
      Alloc_MMVII_GT_EpipOrthoC,
      "Generate Epipolar Ground Truth of Ortho-Centric camera",
      {eApF::Cloud,eApF::Simul},
      {eApDT::MMVIICloud},
      {eApDT::Image},
      __FILE__
);

tMMVII_UnikPApli Alloc_MMVII_GT_MultiView(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_MMVII_CloudImProj(aVArgs,aSpec,1));
}

cSpecMMVII_Appli  TheSpec_MMVII_GT_MultiView
(
     "CloudMMVII_GT_MultiView",
      Alloc_MMVII_GT_MultiView,
      "Generate Multi-view Ground Truth with conik camera",
      {eApF::Cloud,eApF::Simul},
      {eApDT::MMVIICloud},
      {eApDT::Image,eApDT::Orient},
      __FILE__
);


};
