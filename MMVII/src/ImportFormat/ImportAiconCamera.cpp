#include "MMVII_PCSens.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Mappings.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_ReadFileStruct.h"

/**
   \file ImportAiconCamera.cpp  

   \brief import a set of internal calibration from Aicon files
*/


namespace MMVII
{

/**  A class for representing the Aicon camera.

     The mathematicall model of projection is equivalent to "fraser's model",
or to  MMVII's 311/stenope perpective model. There is only a difference of coding :

    * AICON represent in mm
    * the "R0" factor, used to reduce correlation between distorsion and focal
    R0 doesn't had degree of freedom because it's effect can be absorbed by focal

      So, up to the numericall accuracy, the conversion Aicon <-> MMVII  can be "perfect".
    For the sake of simplicity (for developper) we simply export  3D/2D correspondances,
    tha can be used later to compute a MMVII camera (using space resection and bundle adjustment)

*/

static const tREAL8 theWBorder = 1e-6;

typedef std::pair<cPerspCamIntrCalib*,cStdStatRes> tResAiconConv;

class cAiconCamera
{
     public :
         cAiconCamera(const std::vector<tREAL8> & aParams);

         tResAiconConv GenMMVIICam(int aNbXY,int aNbZ,const std::string & aName="FromAicon");

         cPt2dr  PLoc2PIm(const cPt3dr & aPLoc) const;
         cPt3dr  PIm2DirB(const cPt2dr & aPIm) const;
         cPt2dr  Distord(const cPt2dr & aPLoc) const;
         cPt2dr  InvDist(const cPt2dr & aPLoc) const;

          // Sz Capt pix
          tREAL8  SzX_pix() const   {return mParams.at(13);}
          tREAL8  SzY_pix() const   {return mParams.at(14);}
          cPt2dr  Sz_pix() const    {return cPt2dr(SzX_pix(),SzY_pix());}

          // Sz Capt mm
          tREAL8  SzX_mm() const   {return mParams.at(11);}
          tREAL8  SzY_mm() const   {return mParams.at(12);}
          cPt2dr  Sz_mm() const    {return cPt2dr(SzX_mm(),SzY_mm());}
     private :
          tREAL8 WeigthOk(tREAL8 aW) {return std::max(theWBorder,std::min(aW,1-theWBorder));}
          // tREAL8 aX = (aKX*SzX_pix()) / aNbXY ;


          std::vector<tREAL8> mParams;
          tREAL8 mPixInMM;

          tREAL8  Focal() const {return mParams.at(0);}
          tREAL8  FocalPix() const {return -Focal() / mPixInMM;}

          tREAL8  PPx() const   {return mParams.at(1);}
          tREAL8  PPy() const   {return -mParams.at(2);}
          cPt2dr PP() const     {return cPt2dr(PPx(),PPy());}

          tREAL8  A1() const    {return mParams.at(3);}
          tREAL8  A2() const    {return mParams.at(4);}
          tREAL8  R0() const    {return mParams.at(5);}
          tREAL8  A3() const    {return mParams.at(6);}
           // Decentrik
          tREAL8  B1() const    {return mParams.at(7);}
          tREAL8  B2() const    {return mParams.at(8);}

         // Affine
          tREAL8  C1() const    {return mParams.at(9);}
          tREAL8  C2() const    {return mParams.at(10);}


/*
FOCAL  -24.90098
PPX   -0.10482 
PPY  -0.00659
A1 -1.57981e-004
A2  2.60388e-007    
R0  9.00
A3  -9.75344e-011
B1  -1.48754e-005   // Decentrik
B2 7.47493e-006
C1 9.77902e-005    // Affine
C2  -8.03252e-005
SzCapteur 24.00000    16.00000
SzCapteur 6000  4000
*/

};

cAiconCamera::cAiconCamera(const std::vector<tREAL8> & aVParams) :
   mParams (aVParams)
{
   MMVII_INTERNAL_ASSERT_tiny(aVParams.size()==15,"Bad size for cAiconCamera");
   mPixInMM =  SzX_mm() / SzX_pix();
   tREAL8 aCheck = RelativeDifference(mPixInMM,SzY_mm() / SzY_pix());
   MMVII_INTERNAL_ASSERT_tiny(aCheck<1e-8,"Sz of pixel incoherent in cAiconCamera ");
}


cPt2dr cAiconCamera::Distord(const cPt2dr & aP) const
{
    tREAL8 aR2 = SqN2(aP);
    tREAL8 aR0_2 = Square(R0());

    cPt2dr aRadCorr =  aP * ( A1()*(aR2-aR0_2)  +  A2()*(Square(aR2)-Square(aR0_2)) + A3()*(Cube(aR2)-Cube(aR0_2)));

    cPt2dr aDecCorr
           (
                B1()*(aR2+2*Square(aP.x()))  + B2() * 2*aP.x()*aP.y(),
                B2()*(aR2+2*Square(aP.y()))  + B1() * 2*aP.x()*aP.y()
           );

    cPt2dr aAffCorr (C1()*aP.x()+C2()*aP.y(),0.0);

    return aP + aRadCorr + aDecCorr + aAffCorr;
}

cPt2dr  cAiconCamera::PLoc2PIm(const cPt3dr & aPLoc) const
{
    cPt2dr aPPhgr = cPt2dr(aPLoc.x(),aPLoc.y()) / (-aPLoc.z());

    cPt2dr  aPCmm = aPPhgr * Focal() ;

    aPCmm = Distord(aPCmm) + PP() ;

    return (aPCmm + Sz_mm()/2.0) / mPixInMM;
}

cPt3dr  cAiconCamera::PIm2DirB(const cPt2dr & aPImPix) const
{
 // StdOut() << "aPImPix " << aPImPix << "\n";
    cPt2dr aPC_mm =  aPImPix * mPixInMM - Sz_mm()/2.0;
 // StdOut() << "aPC_mm " << aPC_mm << "\n";
    aPC_mm = InvDist(aPC_mm - PP());
 // StdOut() << "aPC_mm " << aPC_mm << "\n";
 // getchar();
    cPt2dr aPPhgr = aPC_mm /Focal();

    return VUnit(cPt3dr(aPPhgr.x(),aPPhgr.y(),-1));
}

class cAiconCamera_DistInv : public cDataNxNMapping<tREAL8,2>
{
    public :
       cAiconCamera_DistInv(const cAiconCamera & anAC) : mAC (anAC) {}
       cPt2dr Value(const cPt2dr & aPt) const override {return mAC.Distord(aPt);}
    private :
        const cAiconCamera & mAC;
};


cPt2dr  cAiconCamera::InvDist(const cPt2dr & aP) const
{
    cAiconCamera_DistInv aACI(*this);

    return aACI.InvertQuasiTrans(aP,aP,1e-8,10);
}

tResAiconConv cAiconCamera::GenMMVIICam(int aNbXY,int aNbZ,const std::string & aName)
{
    cPt2dr aMil = Sz_pix() /2.0;
    cPt3dr  aPPF(aMil.x(),aMil.y(),FocalPix());


    cPerspCamIntrCalib * aRes = cPerspCamIntrCalib::SimpleCalib
                                (
                                     aName,
                                     eProjPC::eStenope,
                                     ToI(Sz_pix()),
                                     aPPF,
                                     cPt3di(3,1,1)
                                );
    cSensorCamPC aCam(aName,tPoseR::Identity(),aRes);

    cSet2D3D aSet23;
    for (int aKX=0 ; aKX<=aNbXY ; aKX++)
    {
        tREAL8 aX = WeigthOk (aKX/tREAL8(aNbXY)) *SzX_pix() ;
        for (int aKY=0 ; aKY<=aNbXY ; aKY++)
        {
            tREAL8 aY = WeigthOk (aKY/tREAL8(aNbXY)) *SzY_pix() ;
            cPt2dr aPIm(aX,aY);
            cPt3dr aDirB = PIm2DirB(aPIm);


            for (int  aKZ=1 ; aKZ<=aNbZ ; aKZ++)
                 aSet23.AddPair(aPIm,aDirB*tREAL8(-aKZ));
        }
    }
    cCorresp32_BA aBA(&aCam,aSet23);
    for (int aKIt=0 ; aKIt<4 ; aKIt++)
    {
        aBA.OneIteration();
    }

    cStdStatRes aStatRes;
    for (const auto aPair : aSet23.Pairs())
        aStatRes.Add(Norm2(aPair.mP2 - aCam.Ground2Image(aPair.mP3)));

    return tResAiconConv(aRes,aStatRes);
}

void BenchAiconCamera()
{
    cAiconCamera aAiconCam(
       {
         -24.90098, -0.10482,  -0.00659,  // FF PPX PPY
         -1.57981e-004, 2.60388e-007,9.00, -9.75344e-011, // K1 K2 R0 K3
         -1.48754e-005  , 7.47493e-006,  // B1 B2
          9.77902e-005 ,  -8.03252e-005,  // C1 C2
          24.00000, 16.00000,             // Sz Capt mm
          6000,  4000                     // Sz Capt pix
       }
    );

    // Check that the MMVII camera gives the same result than the value  computed by Florian Barcet
    // ---  Make the test on a single point
    {
        std::vector<cPt2dr>  aVPImTest{{473.79500000, 1.64750000}};
        std::vector<cPt3dr>  aVDirBTest{{-0.35937477, -0.28762563, 0.88776194}} ;

        for (size_t aKP=0 ; aKP<aVPImTest.size() ; aKP++)
        {
            tREAL8 aDif = Norm2( aVPImTest.at(aKP)-aAiconCam.PLoc2PIm(aVDirBTest.at(aKP)));
            MMVII_INTERNAL_ASSERT_bench(aDif<1e-3,"Aicon Dist/DistInv");
        }
    }
    // ---  Make the test on point  stored in the file "043-V1.txt";
    {
        // stuf to process txt file 
        std::string aName = cMMVII_Appli::InputDirTestMMVII() + "043-V1.txt";    
        cNewReadFilesStruct aRFS;
        std::string aFormat = "XiYiXgYgZg";
        aRFS.SetFormat( "??"+aFormat,aFormat,aFormat);
        aRFS.ReadFile(aName,cNRFS_ParamRead(0,-1,'#'));

        // check  3d proj = ref im 
        for (size_t aKL=0 ; aKL<aRFS.NbLineRead() ; aKL++)  
        {
            cPt2dr aPIm = aRFS. GetPt2dr(aKL,"Xi","Yi");
            cPt3dr aPGLoc = aRFS. GetPt3dr(aKL,"Xg","Yg","Zg");

            tREAL8 aDif = Norm2(aPIm- aAiconCam.PLoc2PIm(aPGLoc));
            MMVII_INTERNAL_ASSERT_bench(aDif<1e-3,"Aicon Dist/DistInv");
        }
   }


    // check the correction of inverse function 
    // --    InvDist * Distord = Id
    //                                    
    //                PIm2DirB                 PLoc2PIm
    // --    Im1   --------------->  Bundle -----------------> Im2  : check Im1 = Im2

    int aNB = 20;
    for (int aKX= 0 ; aKX<= aNB; aKX++)
    {
        tREAL8 aWX = aKX/tREAL8(aNB);
        for (int aKY= 0 ; aKY<= aNB; aKY++)
        {
            tREAL8 aWY = aKY/tREAL8(aNB);
            cPt2dr aPW(aWX,aWY);
            cPt2dr aP0  =  MulCByC(aAiconCam.Sz_mm(),aPW - cPt2dr(0.5,0.5));
            
            cPt2dr aP1 = aAiconCam.InvDist(aP0);
            cPt2dr aP2 = aAiconCam.Distord(aP1);
 
            tREAL8 aDist = Norm2(aP0-aP2);
            MMVII_INTERNAL_ASSERT_bench(aDist<1e-8,"Aicon Dist/DistInv");

            cPt2dr aPIm = MulCByC(aAiconCam.Sz_pix(),aPW);
            cPt3dr aDirB = aAiconCam.PIm2DirB(aPIm);
            cPt2dr aPIm2 = aAiconCam.PLoc2PIm(aDirB);
 
            aDist = Norm2(aPIm-aPIm2);
            MMVII_INTERNAL_ASSERT_bench(aDist<1e-4,"Aicon Dist/DistInv");
        }
    } 

    auto [aMMVIICalib,aStat]  = aAiconCam.GenMMVIICam(50,3);

    // test that Aicon & MMVII are ~ equal, on learning data
    MMVII_INTERNAL_ASSERT_bench(aStat.Max()<1e-4,"Aicon/MMVII conv");

    // test that Aicon & MMVII are ~ equal, on new data
    cSensorCamPC aSensPC("Aicon",tPoseR::Identity(),aMMVIICalib);
    for (int aK=0 ; aK< 1000 ; aK++)
    {
        cPt3dr aPG = aSensPC.RandomVisiblePGround(0.1,2.0);
        cPt2dr aPIm1 = aSensPC.Ground2Image(aPG);
        cPt2dr aPIm2 = aAiconCam.PLoc2PIm(aPG);

        tREAL8 aDist = Norm2(aPIm1-aPIm2);
        MMVII_INTERNAL_ASSERT_bench(aDist<1e-4,"Aicon/MMVII conv");
    }
    
    delete aMMVIICalib;
}



   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportAiconCamera                   */
   /*                                                            */
   /* ********************************************************** */


class cAppli_ImportAiconCamera : public cMMVII_Appli
{
     public :
        cAppli_ImportAiconCamera(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
     private :

	cPhotogrammetricProject  mPhProj;
        std::vector<std::string> mNameImages;

	// Mandatory Arg
	std::string              mNameAiconFile;
	std::string              mExtCalib;

	std::vector<std::string>  Samples() const override;
};

cAppli_ImportAiconCamera::cAppli_ImportAiconCamera(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli  (aVArgs,aSpec),
   mPhProj       (*this)
{
}

cCollecSpecArg2007 & cAppli_ImportAiconCamera::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameAiconFile ,"Name of Aicon file containing ",{eTA2007::FileAny})
	      <<  Arg2007(mNameImages ,"Name of images, can be tab of a single (multiple) pattern")
	      // <<  Arg2007(mExtCalib ,"Calib to import")
              <<  mPhProj.DPOrient().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportAiconCamera::ArgOpt(cCollecSpecArg2007 & anArgObl) 
{
    
    return anArgObl
       //  << AOpt2007(mNameGCP,"NameGCP","Name of GCP set")
       //  << AOpt2007(mNbDigName,"NbDigName","Number of digit for name, if fixed size required (only if int)")
       //  << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       //  << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       //  << AOpt2007(mPatternTransfo,"PatName","Pattern for transforming name (first sub-expr)")
    ;
}


int cAppli_ImportAiconCamera::Exe()
{
    mPhProj.FinishInit();

    //cMMVII_Ifs aInFile(mNameAiconFile, eFileModeIn::Text);
    cAr2007 *  anAr = AllocArFromFile(mNameAiconFile,true,false,eTypeSerial::etxt);

    
    // Parse the differenr pattern
    for (const auto & aPat : mNameImages)
    {
        // if empty string, used to skeep a file
        std::vector<std::string>  aSetFile = {""};
	if (aPat != "")
	{
           aSetFile = ToVect(SetNameFromString(aPat,true));
           if (aSetFile.empty())
              aSetFile = std::vector<std::string>({aPat});
	}
	// StdOut() << "Pat=" << aPat << " S={" << aSetFile << "}\n";
	// parse the different file
        for (const auto&  aNameFile : aSetFile)
        {
            std::vector<tREAL8> aVParam;
            for (int aKP=0 ; aKP<17 ; aKP++)
            {
                tREAL8 aV;
                anAr->RawAddDataTerm(aV);
		// the two firt values are not used
                if (aKP>=2)
                   aVParam.push_back(aV);
            }
	    if (aPat != "")
	    {
                cAiconCamera anAC(aVParam);
	        std::string aNameCalib = mPhProj.StdNameCalibOfImage( aNameFile);
                StdOut() << "====== " << anAC.Sz_pix() << " " << aNameCalib << "\n";
                auto [aCam,aStat] = anAC.GenMMVIICam(50,3,aNameCalib);
		mPhProj.SaveCalibPC(*aCam);
                delete aCam;
	    }
         }
    }

   

    delete anAr;
    return EXIT_SUCCESS;
}


std::vector<std::string>  cAppli_ImportAiconCamera::Samples() const
{
     return 
     {
          "MMVII ImportAiconCalib Data-Input/20250627_CCAM01.ior [.*_1065.tif]",
          "MMVII ImportAiconCalib Data-Input/20250627_CCAM01.ior [043_1065.tif,671_1065.tif,948_1065.tif,949_1065.tif]",
          "MMVII ImportAiconCalib Data-Input/20250627_CCAM01.ior [,671_1065.tif,,949_1065.tif]",
          "MMVII ImportAiconCalib Data-Input/20250627_CCAM01.ior [043_1065.tif,,948_1065.tif,,]"
     };
}


tMMVII_UnikPApli Alloc_ImportAiconCamera(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportAiconCamera(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportAiconCamera
(
     "ImportAiconCalib",
      Alloc_ImportAiconCamera,
      "Import a camera calibration from an Aicon file",
      {eApF::Ori},
      {eApDT::Ori},
      {eApDT::Ori},
      __FILE__
);
/*
*/


}; // MMVII

