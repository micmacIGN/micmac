#include "include/MMVII_all.h"
#include "include/V1VII.h"

namespace MMVII
{
namespace NS_FormatTDEDM
{


class cAppliFormatTDEDM ;  // format 4 training data on epipolar dense matching
class cWT_AppliFormatTDEDM ;  // format 4 training data on epipolar dense matching


class cNameFormatTDEDM 
{
    protected :
        static std::string  PrefixAll()   {return "DMTrain_";}

        static std::string  Im1()   {return "Im1";}
        static std::string  Im2()   {return "Im2";}
        static std::string  Px1()   {return "Pax1";}
        static std::string  Px2()   {return "Pax2";}
        static std::string  Masq1() {return "Masq1";}
        static std::string  Masq2() {return "Masq2";}

        static std::string MakeName(const std::string & aName,const std::string & aPref) 
        {
             return PrefixAll() + aName + "_" + aPref + ".tif";
        }

        static void GenConvertIm(const std::string & aInput, const std::string & aOutput)
        {
            std::string aCom =   "convert -colorspace Gray -compress none " + aInput + " " + aOutput;
            GlobSysCall(aCom);
        }

        static std::string NameIm1(const std::string & aName) {return MakeName(aName,Im1());}
        static std::string NameIm2(const std::string & aName) {return MakeName(aName,Im2());}
        static std::string NamePx1(const std::string & aName) {return MakeName(aName,Px1());}
        static std::string NamePx2(const std::string & aName) {return MakeName(aName,Px2());}
        static std::string NameMasq1(const std::string & aName) {return MakeName(aName,Masq1());}
        static std::string NameMasq2(const std::string & aName) {return MakeName(aName,Masq2());}

        static std::string NameRedrIm1(const std::string & aName) {return MakeName(aName,"REDRIn_"+Im1());}
        static std::string NameRedrIm2(const std::string & aName) {return MakeName(aName,"REDRIn_"+Im2());}

        static void ConvertIm1(const std::string & aInput,const std::string & aName) {GenConvertIm(aInput,NameIm1(aName));}
        static void ConvertIm2(const std::string & aInput,const std::string & aName) {GenConvertIm(aInput,NameIm2(aName));}
};



/*  ============================================== */
/*                                                 */
/*       cMDLB_AppliFormatTDEDM                    */
/*                                                 */
/*  ============================================== */



class cMDLB_AppliFormatTDEDM : public cMMVII_Appli,
                              public cNameFormatTDEDM 
{
     public :

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cMDLB_AppliFormatTDEDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

        void MakePxSym(const std::string & aDir,int aK);
        cIm2D<tU_INT1> ComputeAR(double aAmpl,int aK);
        void  MakeMaskAR(double aPixThresh,int aK);
        void  DoRectifiedImage(int aK);
     private :
       // =========== Data ========
            // Mandatory args
            std::string mPatDir;
            int  mYear;
            bool mDoRectif;  ///< Do rectification of images
            std::string mCurPref;

            std::string mNameIm[2];
            std::string mNamePx[2];


            // std::string mTmpPxSym[2];
            // std::string mTmpAR[2];

            std::string NameMasq(int aK) {return (aK==0) ? NameMasq1(mCurPref) : NameMasq2(mCurPref) ;}
            std::string NamePx  (int aK) {return (aK==0) ? NamePx1(mCurPref)   : NamePx2(mCurPref)   ;}
            std::string NameIm  (int aK) {return (aK==0) ? NameIm1(mCurPref)   : NameIm2(mCurPref)   ;}
            std::string NameRedr(int aK) {return (aK==0) ? NameRedrIm1(mCurPref)   : NameRedrIm2(mCurPref)   ;}

};

cMDLB_AppliFormatTDEDM::cMDLB_AppliFormatTDEDM
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli (aVArgs,aSpec),
   mDoRectif    (false)
{
}

cCollecSpecArg2007 & cMDLB_AppliFormatTDEDM::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return 
      anArgObl  
         << Arg2007(mPatDir,"Directory of file")
         << Arg2007(mYear,"Year of midlebury dataset")
   ;
}

cCollecSpecArg2007 & cMDLB_AppliFormatTDEDM::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
         << AOpt2007(mDoRectif,"DoRectif","Compute rectified images, for check",{eTA2007::HDV})
   ;
}

void cMDLB_AppliFormatTDEDM::MakePxSym(const std::string & aDir,int aK)
{
  std::string aTmp = "SymTmp.tif";
  std::string aComSym = "mm3d Turn90Im " + aDir + mNamePx[aK] + "  NumGeom=6 Out=" +  aTmp;
  GlobSysCall(aComSym);
       //==== Create a Px without inf/nan just in case ====
  int aSign = (aK==0) ? -1 : 1;
  int aDef = aSign * -10000;
  std::string aComPx = "mm3d Nikrup  \"* " + ToStr(aSign) + " F1F2BN " + aTmp   +" "+ ToStr(aDef) +  " \" "  + NamePx(aK);
  GlobSysCall(aComPx);
  RemoveFile(aTmp,false);
}

cIm2D<tU_INT1> cMDLB_AppliFormatTDEDM::ComputeAR(double aAmpl,int aK)
{
     cIm2D<tREAL4> aIm1 =  cIm2D<tREAL4>::FromFile(NamePx(aK));
     cIm2D<tREAL4> aIm2 =  cIm2D<tREAL4>::FromFile(NamePx(1-aK));
     cDataIm2D<tREAL4> & aDI1 = aIm1.DIm();
     cDataIm2D<tREAL4> & aDI2 = aIm2.DIm();

     int aSzX = aDI1.Sz().x();
     int aSzY = std::min(aDI1.Sz().y(),aDI2.Sz().y());

     cIm2D<tU_INT1> aImRes(aIm1.DIm().Sz());
     aImRes.DIm().InitCste(255);

     for (const auto & aPix1 : cPixBox<2>(cBox2di(cPt2di(aSzX,aSzY))))
     {
        cPt2dr aPR2 = ToR(aPix1)  + cPt2dr(aDI1.GetV(aPix1),0.0);
        cPt2dr aPI1_Back = aPR2 + cPt2dr(aDI2.DefGetVBL(aPR2,-1000.0),0.0);
        double aDX = std::abs(aPix1.x()-aPI1_Back.x());
        aImRes.DIm().SetVTrunc(aPix1,aAmpl*aDX);
     }
     return aImRes;
}

void  cMDLB_AppliFormatTDEDM::DoRectifiedImage(int aK)
{
    cIm2D<tREAL4> aImPx =  cIm2D<tREAL4>::FromFile(NamePx(aK));
    cIm2D<tU_INT1> aImRect(aImPx.DIm().Sz());
    aImRect.DIm().InitCste(0);
    cIm2D<tU_INT1> aImOther = cIm2D<tU_INT1>::FromFile(NameIm(1-aK));
    cIm2D<tU_INT1> aImMasq =  cIm2D<tU_INT1>::FromFile(NameMasq(aK));

    for (const auto & aPix : aImPx.DIm())
    {
        if (aImMasq.DIm().GetV(aPix))
        {
            cPt2dr aP2 = ToR(aPix) + cPt2dr(aImPx.DIm().GetV(aPix),0);
            aImRect.DIm().SetV(aPix,aImOther.DIm().DefGetVBL(aP2,0));
        }
    }

    aImRect.DIm().ToFile(NameRedr(aK));
}

void  cMDLB_AppliFormatTDEDM::MakeMaskAR(double aPixThresh,int aK)
{
    std::string aTmpAR("TmpAR.tif");
    int aMul = 100;
    double aAmpl = double(aMul)  / aPixThresh;

    cIm2D<tU_INT1>  aImAR = ComputeAR(aAmpl,aK);
    aImAR.DIm().ToFile(aTmpAR);

    std::string aComMasq = 
               std::string("mm3d Nikrup " )
             + "\"* 255  < " + aTmpAR + " " + ToStr(aMul) + "\" "
             +  NameMasq(aK)  + " Type=u_int1";
    GlobSysCall(aComMasq);
    RemoveFile(aTmpAR,false);
}

int cMDLB_AppliFormatTDEDM::Exe()
{
   switch(mYear)
   {
        case 2006 :
           mNameIm[0]  = "view1.png";
           mNameIm[1]  = "view5.png";
           mNamePx[0]  = "disp1.png";
           mNamePx[1]  = "disp5.png";
        break;

        case 2014 :
           mNameIm[0]  = "im0.png";
           mNameIm[1]  = "im1.png";
           mNamePx[0]  = "disp0.pfm";
           mNamePx[1]  = "disp1.pfm";
        break;

        default :
              MMVII_UsersErrror(eTyUEr::eUnClassedError,"Year specified not avalaible");
        break;
   }

   std::vector<std::string>  aVS = RecGetFilesFromDir("./",AllocRegex(mPatDir+DirSeparator()+mNameIm[0]),1,2);
   for (auto aFullName : aVS)
   {
       std::vector<std::string> aVSep = SplitString(aFullName,"/");
       std::string aNameIm  = aVSep.at(aVSep.size()-2);
       mCurPref  = "MDLB" + ToStr(mYear) + "-" + aNameIm;
       std::string aDir =  DirOfPath(aFullName);

       ConvertIm1(aDir+mNameIm[0],mCurPref);
       ConvertIm2(aDir+mNameIm[1],mCurPref);

       if (mYear==2014)
       {
           MakePxSym(aDir,0);
           MakePxSym(aDir,1);

           MakeMaskAR(2.0,0);
           MakeMaskAR(2.0,1);

       }
       if (mDoRectif)
       {
          DoRectifiedImage(0);
          DoRectifiedImage(1);
       }
   }

   return EXIT_SUCCESS;
}

tMMVII_UnikPApli Alloc_Format_MDLB_TDEDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cMDLB_AppliFormatTDEDM(aVArgs,aSpec));
}

/*  ============================================== */
/*                                                 */
/*       cETHZ_AppliFormatTDEDM                    */
/*                                                 */
/*  ============================================== */

/*  ============================================== */
/*                                                 */
/*       cWT_AppliFormatTDEDM                      */
/*                                                 */
/*  ============================================== */


class cWT_AppliFormatTDEDM : public cMMVII_Appli,
                             public cNameFormatTDEDM 
{
     public :

       // =========== Declaration ========
                 // --- Method to be a MMVII application
        cWT_AppliFormatTDEDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
       // =========== Data ========
            // Mandatory args
            std::string mPatIm1;
            std::string mDirIm;
            std::string mNameBatch;
};


cWT_AppliFormatTDEDM::cWT_AppliFormatTDEDM
(
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli &          aSpec
)  :
   cMMVII_Appli(aVArgs,aSpec)
{
}


cCollecSpecArg2007 & cWT_AppliFormatTDEDM::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   
   return 
      anArgObl  
         << Arg2007(mPatIm1,"Pattern or Xml for image1")
         << Arg2007(mNameBatch,"Name of batch files belongs to")
   ;
}

cCollecSpecArg2007 & cWT_AppliFormatTDEDM::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
   ;
}


int cWT_AppliFormatTDEDM::Exe()
{
   std::vector<std::string>  aVS = RecGetFilesFromDir("./",AllocRegex(".*/colored_0/"+mPatIm1),0,10);
   for (auto aFullName : aVS)
   {
       StdOut () << "FFF " << aFullName << "\n";

       std::string aDir =  DirOfPath(aFullName);
       std::string aFile =  FileOfPath(aFullName);

       std::string aPref =  mNameBatch + "-" +  Prefix(aFile);
       ConvertIm1(aFullName,aPref);
       ConvertIm2(aDir+"../colored_1/"+aFile,aPref);

       std::string aTmpPx =   PrefixAll() + "_tmppx.tif";
       GenConvertIm(aDir+"../disp_occ/"+aFile,aTmpPx);
       std::string aComDyn = "mm3d Nikrup \"/ " + aTmpPx + " -256.0\" " + NamePx1(aPref);
       GlobSysCall(aComDyn);

       std::string aComMasq = "mm3d Nikrup \"* 255 !=  " + aTmpPx + " 0\" " + NameMasq1(aPref)  + " Type=u_int1";
       GlobSysCall(aComMasq);

       RemoveFile(aTmpPx,false);
    }

   return EXIT_SUCCESS;
}


/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_Format_WT_TDEDM(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cWT_AppliFormatTDEDM(aVArgs,aSpec));
}

}; //  NS_FormatTDEDM
}


namespace MMVII
{

cSpecMMVII_Appli  TheSpecFormatTDEDM_WT
(
     "DMFormatTD_WT",
      NS_FormatTDEDM::Alloc_Format_WT_TDEDM,
      "Dense Match: Format Training data from Wu-Teng to MMVII",
      {eApF::Match},
      {eApDT::Image,eApDT::FileSys},
      {eApDT::Image,eApDT::FileSys},
      __FILE__
);

cSpecMMVII_Appli  TheSpecFormatTDEDM_MDLB
(
     "DMFormatTD_MDLB",
      NS_FormatTDEDM::Alloc_Format_MDLB_TDEDM,
      "Dense Match: Format Training data from Middelburry to MMVII, run in Mdlb-xxx/",
      {eApF::Match},
      {eApDT::Image,eApDT::FileSys},
      {eApDT::Image,eApDT::FileSys},
      __FILE__
);

};

