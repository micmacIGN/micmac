#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"


/**
   \file CheckGCPDist.cpp

   \brief file for checking correctness of 3D point distribution
          (enough ?  Not Alined ? Not Coplonar ? ...)

 */

namespace MMVII
{

/* ==================================================== */
/*                                                      */
/*          cAppli_CheckGCPDist                         */
/*                                                      */
/* ==================================================== */

class cAppli_CheckGCPDist : public cMMVII_Appli
{
     public :
        typedef std::vector<cPerspCamIntrCalib *> tVCal;

        cAppli_CheckGCPDist(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :

        ///  Process 1 image
        void MakeOneIm(const std::string & aName);

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;

        int                      mNbMin11;
        tREAL8                   mMinPlan11;
        int                      mNbMinResec;
        tREAL8                   mMinLineResec;
        std::string              mPrefSave;
        std::vector<double>      mVSpecCernPanel;

        tNameSet                 mSetOK11;
        tNameSet                 mSetNotOK11;
        tNameSet                 mSetOKResec;
        tNameSet                 mSetNotOKResec;
};

cAppli_CheckGCPDist::cAppli_CheckGCPDist
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mNbMin11      (12),
     mMinPlan11    (1e-2),
     mNbMinResec   (6),
     mMinLineResec (1e-1),
     mPrefSave     ("SetFiltered_GCP")
{
}



cCollecSpecArg2007 & cAppli_CheckGCPDist::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPPointsMeasures().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CheckGCPDist::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	      << AOpt2007(mNbMin11,"NbMin11P","Number minimal of point for 11 Param",{eTA2007::HDV})
	      << AOpt2007(mMinPlan11,"MinPlane11P","Minim planarity index of 11 Param",{eTA2007::HDV})
	      << AOpt2007(mNbMinResec,"NbMinResec","Number minimal of point for space resection",{eTA2007::HDV})
	      << AOpt2007(mMinLineResec,"MinPlane11P","Mimimal linearity index of space resection",{eTA2007::HDV})
	      << AOpt2007(mPrefSave,"PrefSave","Prefix for saved files",{eTA2007::HDV})
	      << AOpt2007(mVSpecCernPanel,"SpecCernPan","[H0 H1 Delta Nb Tile Dx Dy] : spec CERN for 11P",
                          {{eTA2007::ISizeV,"[8,8]"}})
    ;
}


void cAppli_CheckGCPDist::MakeOneIm(const std::string & aNameIm)
{
    cSetMesImGCP  aSetMes;
    cSet2D3D      aSet23 ;

    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,aNameIm);
    aSetMes.ExtractMes1Im(aSet23,aNameIm);

    std::vector<cPt3dr> aVP3 = aSet23.VP3();

    // Specific test for CERN PANEL
    bool NbPlaneRefut = false;
    bool CoordRefut = false;

    if (IsInit(&mVSpecCernPanel))
    {
       // First we test if enouh point on each of the 2 plane
       CoordRefut = true;
       tREAL8 aH0 = mVSpecCernPanel.at(0);
       tREAL8 aH1 = mVSpecCernPanel.at(1);
       tREAL8 aDH = mVSpecCernPanel.at(2);
       int    aNbMinByPlane = round_ni(mVSpecCernPanel.at(3));

       tREAL8 aSzTile = mVSpecCernPanel.at(4);

       int aNb0 = 0;
       int aNb1 = 0;

       std::vector<cPt2di> aVIndex;
       cPt2di aSzMax(0,0);
       for (const auto & aPt : aVP3)
       {
            tREAL8 aH = aPt.z();
            aNb0 += (aH>aH0-aDH) && (aH<aH0+aDH);
            aNb1 += (aH>aH1-aDH) && (aH<aH1+aDH);

            cPt2di aIndex = ToI(cPt2dr(aPt.x(),aPt.y()) / aSzTile);
            aVIndex.push_back(aIndex);
            SetSupEq(aSzMax,aIndex);
       }
       NbPlaneRefut = (aNb0<aNbMinByPlane) || (aNb1<aNbMinByPlane) ;

       // Now we test if there exist a grid Nx*Ny or Ny*Nx of target (we dont want a single line)
       if (! aVIndex.empty())
       {
           int    aNbX    = round_ni(mVSpecCernPanel.at(5));
           int    aNbY    = round_ni(mVSpecCernPanel.at(6));
           int    aNbXYMin    = round_ni(mVSpecCernPanel.at(7));

           int aRab = std::max(aNbX,aNbY);
           cIm2D<tU_INT1> aIm(aSzMax+cPt2di(aRab,aRab),nullptr,eModeInitImage::eMIA_Null);
           cDataIm2D<tU_INT1> & aDIm = aIm.DIm();
           for (const auto & aIndex : aVIndex)
               aDIm.SetV(aIndex,1);

           for (const auto & aPix : cRect2(cPt2di(0,0),aSzMax) )
           {
                if (    
                         (SumIm(aDIm,cRect2(aPix,aPix+cPt2di(aNbX,aNbY)))>= aNbXYMin)
                     ||  (SumIm(aDIm,cRect2(aPix,aPix+cPt2di(aNbY,aNbX)))>= aNbXYMin)
                   )
                   CoordRefut = false;
           }
       }
    }

    if (((int)aVP3.size() <mNbMin11) || (L2_PlanarityIndex(aVP3)<mMinPlan11)   || NbPlaneRefut || CoordRefut)
    {
       mSetNotOK11.Add(aNameIm);
    }
    else
    {
        mSetOK11.Add(aNameIm);
    }


    if (((int)aVP3.size() <mNbMinResec) || (L2_LinearityIndex(aVP3)<mMinLineResec) || CoordRefut)
    {
       mSetNotOKResec.Add(aNameIm);
    }
    else
    {
        mSetOKResec.Add(aNameIm);
    }
    
}


int cAppli_CheckGCPDist::Exe()
{
    mPhProj.FinishInit();

    // By default print detail if we are not in //
    for (const auto & aNameIm : VectMainSet(0))
    {
        MakeOneIm(aNameIm);
    }

     SaveInFile(mSetNotOK11,mPrefSave+"_NotOK_11Param.xml");
     SaveInFile(mSetOK11,mPrefSave+"_OK_11Param.xml");
     SaveInFile(mSetNotOKResec,mPrefSave+"_NotOK_Resec.xml");
     SaveInFile(mSetOKResec,mPrefSave+"_OK_Resec.xml");

/*
    mPhProj.LoadGCP(mSetMes);
    mPhProj.LoadIm(mSetMes,aNameIm);
    mSetMes.ExtractMes1Im(mSet23,aNameIm);
*/

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_CheckGCPDist(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_CheckGCPDist(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_OriCheckGCPDist
(
     "OriPoseEstimCheckGCPDist",
      Alloc_CheckGCPDist,
      "Check GCP distribution for pose estimation",
      {eApF::Ori},
      {eApDT::GCP},
      {eApDT::Orient},
      __FILE__
);





}; // MMVII

