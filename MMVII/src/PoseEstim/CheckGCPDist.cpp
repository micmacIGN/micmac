#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PCSens.h"


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

             int     mNbMinZ0; ///  Number minimal of point in each plane
             int     mNbMinZ1; ///  Number minimal of point in each plane

	     tREAL8  mThreshHomogNoCal;  ///  Threshold on homgraphy if initial no calib is given
	     tREAL8  mThreshHomogCalib;  ///   Threshold on homgraphy when we have initial caliv to correct distorsion
	     cPt2di  mSzRect;   /// Size of the rect in which we search consecutive target
             int     mNbMinRect;  /// number min of Pts InRect

             /// Pts with  grid number established, used to give grid id to all
             std::map<std::string,cPt2di>  mPtGrid;

	     void AddData(const cAuxAr2007 & anAux0);

             bool IsZ0(tREAL8 aZ) {return IsAlmostZ(aZ,mH0);}
             bool IsZ1(tREAL8 aZ) {return IsAlmostZ(aZ,mH1);}

     private :
             bool IsAlmostZ(tREAL8 aZ,tREAL8 aRef) {return std::abs(aZ-aRef) < mDH;}
};

template <>  const std::string cStrIO<cGeomCERNPannel>::msNameType = "GeomCERNPannel";


void cGeomCERNPannel::AddData(const cAuxAr2007 & anAux0)
{
    cAuxAr2007 anAux(cStrIO<cGeomCERNPannel>::msNameType,anAux0);

    MMVII::AddData(cAuxAr2007("H0",anAux),mH0);
    MMVII::AddData(cAuxAr2007("H1",anAux),mH1);
    MMVII::AddData(cAuxAr2007("Delta",anAux),mDH);
    MMVII::AddData(cAuxAr2007("DistPlani",anAux),mDistPlani);
    MMVII::AddData(cAuxAr2007("NbMinZ0",anAux),mNbMinZ0);
    MMVII::AddData(cAuxAr2007("NbMinZ1",anAux),mNbMinZ1);

    MMVII::AddData(cAuxAr2007("ThresholdHomographyNOCal",anAux),mThreshHomogNoCal);
    MMVII::AddData(cAuxAr2007("ThresholdHomographyCalib",anAux),mThreshHomogCalib);

    MMVII::AddData(cAuxAr2007("SzRect",anAux),mSzRect);
    MMVII::AddData(cAuxAr2007("NbMinRect",anAux),mNbMinRect);
    MMVII::AddData(cAuxAr2007("PtGrid",anAux),mPtGrid);
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

class  cPtCheck
{
     public :
       const cMultipleImPt *  mIm;
       cPt2dr                 mP2ImInit;
       cPt2dr                 mP2ImCorr;
       const cMes1Gnd3D *       mGr;
       cPt2di                 mId;
       bool                   mIsOk;
       bool                   mIsZ0;
};

class cAppli_CheckGCPDist : public cMMVII_Appli
{
     public :
        typedef std::vector<cPerspCamIntrCalib *> tVCal;

        cAppli_CheckGCPDist(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :

	/// give the point their grid Id, init IsZ0,  init Nb0, Nb1
        void   ComputeGridId(const cSetMesGndPt & aSetMes);
	///  compute homography Plane -> Im for point in the main plane (Z~0 in CERN-pannel)
        bool   ComputeMainHomogr(cHomogr2D<tREAL8> &);
	///  for the point up (Z~200 in CERN pannel), compute the homography as a perturbation to previous one
        bool    ComputeHomogrZ1(const cHomogr2D<tREAL8> &);
	///  compute the max number of point in  a given rectangle
	int    ComputeSquareAdj();


        ///  Process 1 image
        bool MakeOneIm(const std::string & aName);

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;

	bool                     mGenSpecCERN;  ///< do we generate the specification of cGeomCERNPannel
	bool                     mUseGCP;       ///< Do we use a CERN Pannel Filter
	cGeomCERNPannel          mPannel;          ///< parametrs specific to CERN Panel
	std::string              mNameGCP;      ///< Name of mGCP

        int                      mNbMin11P;      ///< number minimal of point for 11 parameters estimation 
        tREAL8                   mMinPlan11;    ///<  minimal value of non planarity
        int                      mNbMinResec;   ///<  minimal number of point for space resection
        tREAL8                   mMinLineResec;  ///< minimal value for non linearity 
        std::string              mPrefSave;
        std::vector<double>      mVSpecCernPanel;
        cPerspCamIntrCalib *     mCalib;
        tREAL8                   mThresholdHomog; ///< Thresholds for homograhic fitting on Z0 && Z1
        std::string              mCurNameIm;

	bool                     mSaveMeasures;
	// ---  Set ok for different processing, will be used in next steps
        tNameSet                 mSetOK11;        ///< Set of point OK for 11 parameters estimation
        tNameSet                 mSetOKResec;      ///< Set of  point ok for space resection

        std::vector<cPtCheck>    mPtCheck;
        int                      mNbZ0 ;  ///< number of validated point in plane 0 of CERN PANNEL
        int                      mNbZ1 ;  ///< number of validated point in other plane  of CERN PANNEL
	cPt2di                   mSupId;
        cIm2D<tU_INT1>           mImId;
	std::string              mNameReportErrTarget;
	std::string              mNameReportImages;
};

cAppli_CheckGCPDist::cAppli_CheckGCPDist
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mNbMin11P     (6),
     mMinPlan11    (1e-2),
     mNbMinResec   (4),
     mMinLineResec (1e-1),
     mPrefSave     ("SetFiltered_GCP"),
     mCalib        (nullptr),
     mImId         (cPt2di(1,1))
{
}



cCollecSpecArg2007 & cAppli_CheckGCPDist::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPGndPt3D().ArgDirInMand()
              <<  mPhProj.DPGndPt2D().ArgDirInMand()
           ;
}

cCollecSpecArg2007 & cAppli_CheckGCPDist::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	      << AOpt2007(mNameGCP,"CERNFilter","File for Geometric of CERN-like pannel, for special filtering",
			      {{eTA2007::XmlOfTopTag,cStrIO<cGeomCERNPannel>::msNameType}})
	      << AOpt2007(mNbMin11P,"NbMin11P","Number minimal of point for 11 Param",{eTA2007::HDV})
	      << AOpt2007(mMinPlan11,"MinPlane11P","Minim planarity index of 11 Param",{eTA2007::HDV})
	      << AOpt2007(mNbMinResec,"NbMinResec","Number minimal of point for space resection",{eTA2007::HDV})
	      << AOpt2007(mMinLineResec,"MinLineResec","Mimimal linearity index of space resection",{eTA2007::HDV})
	      << AOpt2007(mPrefSave,"PrefSave","Prefix for saved files",{eTA2007::HDV})
	      << AOpt2007(mVSpecCernPanel,"SpecCernPan","[H0 H1 Delta Nb Tile Dx Dy] : spec CERN for 11P",
                          {{eTA2007::ISizeV,"[8,8]"}})
	      << AOpt2007(mGenSpecCERN,"GenarateSpecfifCERNPannel","for editing to a usable value",{eTA2007::HDV})
              << mPhProj.DPGndPt2D().ArgDirOutOpt("DirFiltered","Directory for filtered point")
              << mPhProj.DPOrient().ArgDirInOpt("Calib","Internal calibration folder is any")
    ;
}



void   cAppli_CheckGCPDist::ComputeGridId(const cSetMesGndPt & aSetMes)  
{
    mSupId = cPt2di(-1000,-1000);
    cHomogr2D<tREAL8> aHNum ;
    std::vector<cPt2dr>  aVPlani;
    std::vector<cPt2dr>  aVNum;
    for (const auto & aPair : mPannel.mPtGrid)
    {
        aVNum.push_back(ToR(aPair.second));
	SetSupEq(mSupId,aPair.second);
        const cMes1Gnd3D & aGCP = aSetMes.MesGCPOfName(aPair.first);
        aVPlani.push_back(Proj(aGCP.mPt));
    }
    mSupId += cPt2di(1,1);
    aHNum =  cHomogr2D<tREAL8>::StdGlobEstimate(aVPlani,aVNum);

    mNbZ0 = 0;
    mNbZ1 = 0;
    for (auto & aPC : mPtCheck)
    {
       cPt3dr aPGr = aPC.mGr->mPt;
       aPC.mIsZ0 = (mPannel.IsZ0(aPGr.z()));
       if (aPC.mIsZ0)
       {
            cPt2dr aP2 = Proj(aPGr);
            aPC.mId  =  ToI(aHNum.Value(aP2));
	    //  should not happen,  by the way never too cautious
	    if ( !( SupEq(aPC.mId,cPt2di(0,0)) && InfStr(aPC.mId,mSupId)))
	    {
               aPC.mIsOk = false;
	    }
	    else
	    {
                mNbZ0++;
	    }
        }
        else
        {
            MMVII_INTERNAL_ASSERT_tiny(mPannel.IsZ1(aPGr.z()),"CERN pannel nor z0 nor z1 ");
            mNbZ1++;
        }
     }

     int aRab = NormInf(mPannel.mSzRect);
     mImId = cIm2D<tU_INT1>(mSupId+cPt2di(aRab,aRab),nullptr,eModeInitImage::eMIA_Null);
}

bool   cAppli_CheckGCPDist::ComputeMainHomogr(cHomogr2D<tREAL8> & aHG2I)
{
  if (ComputeSquareAdj() < mPannel.mNbMinRect) 
     return false;

   mNbZ0 = 0;
   std::vector<cPt2dr>  aVPlani;
   std::vector<cPt2dr>  aVIm;
   for (auto & aPC : mPtCheck)
   {
           if (aPC.mIsZ0 && aPC.mIsOk)
           {
              aVPlani.push_back( Proj(aPC.mGr->mPt));
              aVIm.push_back(aPC.mP2ImCorr);
	      mNbZ0++;
           }
   }

   if (mNbZ0<mPannel.mNbMinZ0)
   {
      return false;
   }

   mNbZ0 = 0;

   cAvgAndBoundVals<tREAL8>  aAvgD;
   aHG2I =  cHomogr2D<tREAL8>::RansacL1Estimate(aVPlani,aVIm,1000);
   // cHomogr2D<tREAL8> aHG2I =  cHomogr2D<tREAL8>::RansacL1Estimate(aVPlani,aVIm,1000);
   for (auto & aPC : mPtCheck)
   {
          if (aPC.mIsZ0 && aPC.mIsOk)
          {
              tREAL8 aDif = Norm2(aHG2I.Value(Proj(aPC.mGr->mPt)) - aPC.mP2ImCorr);
              aAvgD.Add(aDif);

              if (aDif >  mThresholdHomog)
              {
                 aPC.mIsOk = false;
		 AddOneReportCSV(mNameReportErrTarget,{mCurNameIm,aPC.mGr->mNamePt,"MainHom",ToStr(aDif)});

                 // StdOut() << "##############  DIFFF " << aDif << " Pt=" <<  aPC.mGr->mNamePt << " Im=" << mCurNameIm << "#### " << std::endl;
              }
	      else
                 mNbZ0++;
          }
   }

  if (ComputeSquareAdj() < mPannel.mNbMinRect) 
     return false;

   return  (mNbZ0 >= mPannel.mNbMinZ0) ;
}


bool  cAppli_CheckGCPDist::ComputeHomogrZ1(const cHomogr2D<tREAL8> & aH0)
{
   std::vector<cPt2dr>  aVPlani;
   std::vector<cPt2dr>  aVIm;
   for (auto & aPC : mPtCheck)
   {
          if ((!aPC.mIsZ0) && aPC.mIsOk)
          {
              aVPlani.push_back( Proj(aPC.mGr->mPt));
              aVIm.push_back(aPC.mP2ImCorr);
	  }
   }

   if (int(aVPlani.size()) < mPannel.mNbMinZ1) 
      return false;

   mNbZ1=0;
   cHomogr2D<tREAL8> aH1 = aH0.RansacParalPlaneShift(aVPlani,aVIm,2,4);

   for (auto & aPC : mPtCheck)
   {
          if ((!aPC.mIsZ0) && aPC.mIsOk)
          {
              tREAL8 aDif = Norm2(aH1.Value(Proj(aPC.mGr->mPt)) - aPC.mP2ImCorr);
              if (aDif >  mThresholdHomog)
              {
                 aPC.mIsOk = false;
                 StdOut() << "##############  DIFFF " << aDif << " Pt=" <<  aPC.mGr->mNamePt << " Im=" << mCurNameIm << "#### " << std::endl;
              }
	      else
	      {
                  mNbZ1++;
	      }
	  }
   }

   return (mNbZ1 >= mPannel.mNbMinZ1);
}

int   cAppli_CheckGCPDist::ComputeSquareAdj()
{
   cDataIm2D<tU_INT1> & aDIm = mImId.DIm();
   for (auto & aPC : mPtCheck)
   {
       if (aPC.mIsZ0 && aPC.mIsOk)
       {
            aDIm.SetV(aPC.mId,1);
       }
   }
   int aRes=0;

    for (const auto & aPix : cRect2(cPt2di(0,0),mSupId) )
    {
        int aS1 = SumIm(aDIm,cRect2(aPix,aPix +        mPannel.mSzRect));
        int aS2 = SumIm(aDIm,cRect2(aPix,aPix + PSymXY(mPannel.mSzRect)));

        UpdateMax(aRes,std::max(aS1,aS2));
    }
   return aRes;
}

bool cAppli_CheckGCPDist::MakeOneIm(const std::string & aNameIm)
{
    mCurNameIm = aNameIm;

    cSetMesGndPt  aSetMes;
    mPhProj.LoadGCP3D(aSetMes);
    mPhProj.LoadIm(aSetMes,aNameIm,nullptr,nullptr,SVP::Yes);

    if (mPhProj.DPOrient().DirInIsInit())
    {
        mCalib =  mPhProj.InternalCalibFromStdName(aNameIm);
    }

    // Create a structure with GCP + image-points, so that we can add information
    mPtCheck.clear() ;
    for (const auto & aMes : aSetMes.MesImOfPt())
    {
        if (aMes.VImages().size()==1)  // there can be empty images because its a structure for multiple points
        {
             cPtCheck aPC;

	     // compute measue image
             aPC.mP2ImInit = aMes.VMeasures().at(0);
             aPC.mP2ImCorr=  mCalib ? mCalib->Undist(aPC.mP2ImInit ) : aPC.mP2ImInit;

             aPC.mIm = &aMes;
             aPC.mGr = & aSetMes.MesGCPOfMulIm(aMes);
             aPC.mId = cPt2di::Dummy();  // just to be detect use w/o init
             aPC.mIsOk = true;  //  up to now, everything is ok

             mPtCheck.push_back(aPC);
        }
    }

    bool  OkResec = true;
    bool  Ok11P   = true;
    if (mUseGCP)
    {

       mThresholdHomog  =  mCalib ?  mPannel.mThreshHomogCalib : mPannel.mThreshHomogNoCal;
       ComputeGridId(aSetMes);
       cHomogr2D<tREAL8> aHG2I;

       if (OkResec) 
          OkResec = ComputeMainHomogr(aHG2I);

       if (OkResec)
       {
	  Ok11P = ComputeHomogrZ1(aHG2I);
          //  if not validated, the point at Z1 are not valide
          if (! Ok11P)
          {
             for (auto & aPC : mPtCheck)
                 if (! aPC.mIsZ0)
                    aPC.mIsOk = false;
          }
       }
       else // if cant comput homogr all point false
       {
          Ok11P = false;
          for (auto & aPC : mPtCheck)
              aPC.mIsOk = false;
       }
    }

    mCalib = nullptr;


    std::vector<cPt3dr>  aVP3;
    for (auto & aPC : mPtCheck)
    {
        if (aPC.mIsOk)
        {
           aVP3.push_back(aPC.mGr->mPt);
	}
    }

    double aPlanarityIndex = -1;
    if ((int)aVP3.size() >=mNbMin11P)
       aPlanarityIndex = L2_PlanarityIndex(aVP3);

    double aLinearityIndex = -1;
    if ((int)aVP3.size() >=mNbMinResec) 
        aLinearityIndex = L2_LinearityIndex(aVP3);

    if (0)   StdOut() << " Im=" << mCurNameIm << " PlaneInd=" << aPlanarityIndex  << " LineInd=" << aLinearityIndex << std::endl;


    Ok11P = Ok11P && (aPlanarityIndex > mMinPlan11);
    if (Ok11P)
    {
        mSetOK11.Add(aNameIm);
    }

    OkResec = OkResec && (aLinearityIndex > mMinLineResec);
    if (OkResec)
    {
       mSetOKResec.Add(aNameIm);
    }

    AddOneReportCSV(mNameReportImages,{ToStr(Ok11P),ToStr(OkResec),aNameIm,ToStr(aVP3.size()),ToStr(aLinearityIndex),ToStr(aPlanarityIndex)});

    if (mSaveMeasures)
    {
       if (OkResec)
       {
          cFilterMesIm aFMIM(mPhProj,mCurNameIm);
          for (const auto & aPC : mPtCheck)
          {          
               aFMIM.AddInOrOut(aPC.mP2ImInit,aPC.mGr->mNamePt,aPC.mIsOk);
          }
          aFMIM.SetFinished();
          aFMIM.Save();
       }
    }

    return OkResec || Ok11P;
}



int cAppli_CheckGCPDist::Exe()
{
    mPhProj.FinishInit();

    mNameReportErrTarget = "RejectedTarget";
    mNameReportImages = "ReportImage";
    InitReportCSV(mNameReportErrTarget,"csv",false);
    InitReportCSV(mNameReportImages,"csv",false);

    AddOneReportCSV(mNameReportErrTarget,{"Image","GCP","Cause","Err"});

    AddOneReportCSV(mNameReportImages,{"Ok11P","OKResec","Image","NbGCP","LineIndex","PlaneIndex"});


    //  if we just need a pattern for xml file
    if (mGenSpecCERN)
    {
       SpecificationSaveInFile<cGeomCERNPannel>();
       return EXIT_SUCCESS;
    }

    mUseGCP =IsInit(&mNameGCP);
    if (mUseGCP)
    {
       ReadFromFile(mPannel,mNameGCP);
    }

    mSaveMeasures = mPhProj.DPGndPt2D().DirOutIsInit();
    // For now this case woul be probably not coherent , but I dont see how avoid it before, so test it now
    if (mUseGCP!=mSaveMeasures)
    {
        MMVII_UnclasseUsEr("Must save measure iff have CERN Pannel");
    }




    // make computation for each image (not in //, very fast)
    auto aNbIm = VectMainSet(0).size();
    int aNbUsed = 0;
    for (const auto & aNameIm : VectMainSet(0))
    {
        if (MakeOneIm(aNameIm))
            ++aNbUsed;
    }

    StdOut() << "Image files: " << aNbIm << ", used:" << aNbUsed << "\n";

    auto aFileNameOK11 = mPrefSave+"_OK_11Param.xml";
    SaveInFile(mSetOK11,aFileNameOK11);
    auto aFileNameOKResec = mPrefSave+"_OK_Resec.xml";
    SaveInFile(mSetOKResec,aFileNameOKResec);

    StdOut() << "Output: " << aFileNameOK11 << " " << aFileNameOKResec << std::endl;

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
      {eApDT::ObjMesInstr, eApDT::ObjCoordWorld},
      {eApDT::Orient},
      __FILE__
);





}; // MMVII

