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
             int     mNbMin0; ///  Number minimal of point in each plane
             int     mNbMin1; ///  Number minimal of point in each plane
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
    MMVII::AddData(cAuxAr2007("NbMin0",anAux),mNbMin0);
    MMVII::AddData(cAuxAr2007("NbMin1",anAux),mNbMin1);
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
       cPt2dr                 mP2Im;
       const cMes1GCP *       mGr;
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

        ///  Process 1 image
        void MakeOneIm(const std::string & aName);

        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;

	bool                     mGenSpecCERN;  ///< do we generate the specification of cGeomCERNPannel
	bool                     mUseGCP;       ///< Do we use a CERN Pannel Filter
	cGeomCERNPannel          mPannel;          ///< parametrs specific to CERN Panel
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

        std::vector<cPtCheck>    mPtCheck;
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
    //  load GCP & Im in aSetMes

    cSetMesImGCP  aSetMes;
    mPhProj.LoadGCP(aSetMes);
    mPhProj.LoadIm(aSetMes,aNameIm);

    // Create a structure with GCP of image, so that we can add information
    mPtCheck.clear() ;
    for (const auto & aMes : aSetMes.MesImOfPt())
    {
        if (aMes.VImages().size()==1)
        {
             cPtCheck aPC;
 
             aPC.mP2Im = aMes.VMeasures().at(0);
             aPC.mIm = &aMes;
             aPC.mGr = & aSetMes.MesGCPOfMulIm(aMes);
             aPC.mId = cPt2di::Dummy();  // just to be detect use w/o init
             aPC.mIsOk = true;
             mPtCheck.push_back(aPC);
        }
    }

    bool RefuteByNbInPlanes = false;
    if (mUseGCP)
    {
       // Compute the mapping (homography 4 now) that associate a grid-num to planar coordinates
       cHomogr2D<tREAL8> aHNum ;
       {
           std::vector<cPt2dr>  aVPlani;
           std::vector<cPt2dr>  aVNum;
           for (const auto & aPair : mPannel.mPtGrid)
           {
                aVNum.push_back(ToR(aPair.second));
                const cMes1GCP & aGCP = aSetMes.MesGCPOfName(aPair.first);
                aVPlani.push_back(Proj(aGCP.mPt));
           }
           aHNum =  cHomogr2D<tREAL8>::StdGlobEstimate(aVPlani,aVNum);
       }

       /// Set the identifier + store the point for homography computing
       int aNb0 = 0;
       int aNb1 = 0;
       std::vector<cPt2dr>  aVPlani;
       std::vector<cPt2dr>  aVIm;
       for (auto & aPC : mPtCheck)
       {
           cPt3dr aPGr = aPC.mGr->mPt;
           aPC.mIsZ0 = (mPannel.IsZ0(aPGr.z()));
           if (aPC.mIsZ0)
           {
              cPt2dr aP2 = Proj(aPGr);
              aPC.mId  =  ToI(aHNum.Value(aP2));
              aNb0++;
              aVPlani.push_back(aP2);
              aVIm.push_back(aPC.mP2Im);
           }
           else
           {
                MMVII_INTERNAL_ASSERT_tiny(mPannel.IsZ1(aPGr.z()),"CERN pannel nor z0 nor z1 ");
                aNb1++;
           }
       }

       RefuteByNbInPlanes = (aNb0<mPannel.mNbMin0) || (aNb1<mPannel.mNbMin1) ;

       if (! RefuteByNbInPlanes)
       {
            cHomogr2D<tREAL8> aHG2I =  cHomogr2D<tREAL8>::RansacL1Estimate(aVPlani,aVIm,1000);
            for (auto & aPC : mPtCheck)
            {
                if (aPC.mIsZ0)
                {
                   cPt2dr aDif = aHG2I.Value(Proj(aPC.mGr->mPt)) - aPC.mP2Im;

                   if (Norm2(aDif) > 10.0)
                   {
                       StdOut() << "DIFFF " << aDif << " Pt=" <<  aPC.mGr->mNamePt << " Im=" << aNameIm << "\n";
                   }
              
                }
            }
        }

       // StdOut() << "aNb0 " << aNb0 << " aNb1 " << aNb1 << "\n";
   }

FakeUseIt(RefuteByNbInPlanes);
}


#if (0)
       // First we test if enouh point on each of the 2 plane
/*
       CoordRefut = true;
       tREAL8 aDH = mPannel.mDH;


       std::vector<cPt2di> aVIndex;
       cPt2di aSzMax(0,0);
       for (const auto & aPt : aVP3)
       {
            tREAL8 aH = aPt.z();
	    bool Is0 = (aH>mPannel.mH0-aDH) && (aH<mPannel.mH0+aDH);
            aNb0 += Is0;
            aNb1 += (aH>mPannel.mH1-aDH) && (aH<mPannel.mH1+aDH);

	    if (Is0)
	    {
                cPt2di aIndex = ToI(cPt2dr(aPt.x(),aPt.y()) / mPannel.mDistPlani);
	        // StdOut()  << "JJJJ " << cPt2dr(aPt.x(),aPt.y()) / mPannel.mDistPlani << "\n";
                aVIndex.push_back(aIndex);
                SetSupEq(aSzMax,aIndex);
	    }
       }
*/


       // Now we test if there exist a grid Nx*Ny or Ny*Nx of target (we dont want a single line)
/*
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
*/
#endif
    


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
       ReadFromFile(mPannel,mNameGCP);
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

