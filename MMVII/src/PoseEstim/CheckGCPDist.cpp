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

class cGeomCERNPannel
{
     public :
	     const std::string TheNameClass;

         // Geometry of the pannel (could be extracted autom)
	     tREAL8  mH0;       ///  hight of main plane
	     tREAL8  mH1;       ///  hight of the 3 or 6 target outside the plane
	     tREAL8  mDH;       ///  hight of main plane
	     tREAL8  mDistPlani;   ///  size of tiling in main plane => to compute an index

         // 
             int     mNbMinByP; ///  Number minimal of point in each plane
	     cPt2di  mSzRect;   /// Size of the rect in which we search consecutive target
             int     mNbMinRect;  /// number min of Pts InRect

	     void AddData(const cAuxAr2007 & anAux0);
     private :
};

template <>  const std::string cStrIO<cGeomCERNPannel>::msNameType = "GeomCERNPannel";


void cGeomCERNPannel::AddData(const cAuxAr2007 & anAux0)
{
    cAuxAr2007 anAux(cStrIO<cGeomCERNPannel>::msNameType,anAux0);

    MMVII::AddData(cAuxAr2007("H0",anAux),mH0);
    MMVII::AddData(cAuxAr2007("H1",anAux),mH1);
    MMVII::AddData(cAuxAr2007("Delta",anAux),mDH);
    MMVII::AddData(cAuxAr2007("DistPlani",anAux),mDistPlani);
    MMVII::AddData(cAuxAr2007("MinNbByPlane",anAux),mNbMinByP);
    MMVII::AddData(cAuxAr2007("SzRect",anAux),mSzRect);
    MMVII::AddData(cAuxAr2007("NbMinRect",anAux),mNbMinRect);
}

void AddData(const cAuxAr2007 & anAux,cGeomCERNPannel & aGCP)
{
	aGCP.AddData(anAux);
}
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

	bool                     mGenSpecCERN;  ///< do we generate the specification of cGeomCERNPannel
	bool                     mUseGCP;       ///< Do we use a CERN Pannel Filter
	cGeomCERNPannel          mGCP;          ///< parametrs specific to CERN Panel
	std::string              mNameGCP;      ///< Name of mGCP

        int                      mNbMin11;      ///< number minimal of point for 11 parameters estimation 
        tREAL8                   mMinPlan11;    ///<  minimal value of non planarity
        int                      mNbMinResec;   ///<  minimal number of point for space resection
        tREAL8                   mMinLineResec;  ///< minimal value for non linearity 
        std::string              mPrefSave;
        std::vector<double>      mVSpecCernPanel;

	// ---  Set ok for different processing, will be used in next steps
        tNameSet                 mSetOK11;        ///< Set of point OK for 11 parameters estimation
        tNameSet                 mSetOKResec;      ///< Set of  point ok for space resection

	// ---- Set not OK, more for documentation & inspection
        tNameSet                 mSetNotOK11;     ///< Set of point NOT Ok for 11 P
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
	      << AOpt2007(mNameGCP,"CERNFilter","File for Geometric of CERN-like pannel, for special filtering",{{eTA2007::XmlOfTopTag,cStrIO<cGeomCERNPannel>::msNameType}})

	      << AOpt2007(mNbMin11,"NbMin11P","Number minimal of point for 11 Param",{eTA2007::HDV})
	      << AOpt2007(mMinPlan11,"MinPlane11P","Minim planarity index of 11 Param",{eTA2007::HDV})
	      << AOpt2007(mNbMinResec,"NbMinResec","Number minimal of point for space resection",{eTA2007::HDV})
	      << AOpt2007(mMinLineResec,"MinPlane11P","Mimimal linearity index of space resection",{eTA2007::HDV})
	      << AOpt2007(mPrefSave,"PrefSave","Prefix for saved files",{eTA2007::HDV})
	      << AOpt2007(mVSpecCernPanel,"SpecCernPan","[H0 H1 Delta Nb Tile Dx Dy] : spec CERN for 11P",
                          {{eTA2007::ISizeV,"[8,8]"}})
	      << AOpt2007(mGenSpecCERN,"GenarateSpecfifCERNPannel","for editing to a usable value",{eTA2007::HDV})
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
    bool RefuteByNbInPlanes = false;
    bool CoordRefut = false;

    if (mUseGCP)
    {
       // First we test if enouh point on each of the 2 plane
       CoordRefut = true;
       tREAL8 aDH = mGCP.mDH;

       int aNb0 = 0;
       int aNb1 = 0;

       std::vector<cPt2di> aVIndex;
       cPt2di aSzMax(0,0);
       for (const auto & aPt : aVP3)
       {
            tREAL8 aH = aPt.z();
	    bool Is0 = (aH>mGCP.mH0-aDH) && (aH<mGCP.mH0+aDH);
            aNb0 += Is0;
            aNb1 += (aH>mGCP.mH1-aDH) && (aH<mGCP.mH1+aDH);

	    if (Is0)
	    {
                cPt2di aIndex = ToI(cPt2dr(aPt.x(),aPt.y()) / mGCP.mDistPlani);
	        StdOut()  << "JJJJ " << cPt2dr(aPt.x(),aPt.y()) / mGCP.mDistPlani << "\n";
                aVIndex.push_back(aIndex);
                SetSupEq(aSzMax,aIndex);
	    }
       }
       RefuteByNbInPlanes = (aNb0<mGCP.mNbMinByP) || (aNb1<mGCP.mNbMinByP) ;

       StdOut() << "aNb0 " << aNb0 << " aNb1 " << aNb1 << "\n";

return;

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

    if (((int)aVP3.size() <mNbMin11) || (L2_PlanarityIndex(aVP3)<mMinPlan11)   || RefuteByNbInPlanes || CoordRefut)
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

    if (mGenSpecCERN)
    {
       SpecificationSaveInFile<cGeomCERNPannel>();
       return EXIT_SUCCESS;
    }

    mUseGCP =IsInit(&mNameGCP);
    if (mUseGCP)
    {
       ReadFromFile(mGCP,mNameGCP);
    }

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

