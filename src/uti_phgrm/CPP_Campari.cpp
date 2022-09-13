/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"
#include <algorithm>
#include "TapasCampari.h"


/*
Parametre de Tapas :

   - calibration In : en base de donnees ou deja existantes.


*/

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

#define  NbModele 10


/*
   Campari avec Block Adj :

      <BlockCamera>
              <NameFile>  Stereo-Apero-EstimBlock.xml </NameFile>
              <Id> TheBlock </Id>
              <UseForBundle>
                        <GlobalBundle >  false     </GlobalBundle>
                        <RelTimeBundle >  true    </RelTimeBundle>
              </UseForBundle>
      </BlockCamera>


      <ObsBlockCamRig>
             <Id>  TheBlock  </Id>
             <Show> true </Show>
             <!-- <GlobalPond> </GlobalPond> -->
             <RelTimePond>
                            <PondOnTr >  1e+2 </PondOnTr>
                            <PondOnRot>  2e+3 </PondOnRot>
             </RelTimePond>
       </ObsBlockCamRig>


        <ExportBlockCamera>
                         <Id> TheBlock</Id>
                         <NameFile> Bloc_Cmp_LR.xml </NameFile>
        </ExportBlockCamera>



         
*/

/****************************************************/
/*                                                  */
/*             cAppli_Tapas_Campari                 */
/*                                                  */
/****************************************************/

std::string BlQUOTE (const std::string & aStr)
{
   if (aStr.empty()) return aStr;

    return " " + QUOTE(aStr) + " ";
}

cAppli_Tapas_Campari::cAppli_Tapas_Campari() :
   mWithBlock       (false),
   LocDegGen(100),
   LocLibDec(true),
   LocLibCD(true),
   LocDRadMaxUSer(100),
   LocLibPP(true),
   LocLibFoc(true),
   LocDegAdd(0),
   IsAutoCal(false),
   IsFigee(false),
   PropDiag(-1.0),
   GlobLibAff(true),
   GlobLibDec(true),
   GlobLibPP(true),
   GlobLibCD(true),
   GlobLibFoc(true),
   GlobDRadMaxUSer(100),
   GlobDegGen(100),
   GlobDegAdd(0),
   ModeleAdditional(false),
   ModeleAddFour(false),
   ModeleAddPoly(false),
   TheModelAdd(""),
   mSauvAutom       (""),
   mRatioMaxDistCS  (30.0),
   mNamesBlockInit  (false),
   mDSElimB         (1),
   mExportMatrixMarket        (false),
   mArg             (new LArgMain)
{
    (*mArg) << EAM(mVBlockGlob,"BlocGlob",true,"Param for Glob bloc compute [File,SigmaCenter,SigmaRot,?MulFinal,?Export]")
            << EAM(mVBlockDistGlob,"DistBlocGlob",true,"Param for Dist Glob bloc compute [File,SigmaDist,?MulFinal,?Export]")
            << EAM(mVBlockRel,"BlocTimeRel",true,"Param for Time Reliative bloc compute [File,SigmaCenter,SigmaRot,?MulFinal,?Export]")
            << EAM(mVOptGlob,"OptBlocG",true,"[SigmaTr,SigmaRot]")
            << EAM(GlobLibFoc,"FocFree",true,"Foc Free (Def=true)", eSAM_IsBool)
            << EAM(GlobLibPP,"PPFree",true,"Principal Point Free (Def=true)", eSAM_IsBool)
            << EAM(GlobLibAff,"AffineFree",true,"Affine Parameter (Def=true)", eSAM_IsBool)
            << EAM(GlobDegAdd,"DegAdd",true, "When specified, degree of additionnal parameter")
            << EAM(GlobDegGen,"DegFree",true, "When specified degree of freedom of parameters generiqs")
            << EAM(GlobDRadMaxUSer,"DRMax",true, "When specified degree of freedom of radial parameters")
            << EAM(GlobLibCD,"LibCP",true,"Free distorsion center, Def context dependant", eSAM_IsBool)
            // alias
            << EAM(GlobLibCD,"LibCD",true,"Free distorsion center, Def context dependant. Principal Point should be also free if CD is free", eSAM_IsBool)
            << EAM(GlobLibDec,"LibDec",true,"Free decentric parameter, Def context dependant", eSAM_IsBool)
            << EAM(mRapOnZ,"RapOnZ",true,"Force Rappel on Z [Z,Sigma,KeyGrp]")
            << EAM(mDSElimB,"SElimB",true,"Print stat on reason for bundle elimination (0,1,2)")
            << EAM(mExportMatrixMarket,"ExpMatMark",true,"Export Cov Matrix to Matrix Market Format+Eigen/cmp")
            << EAM(mSauvAutom,"SauvAutom",true, "Save intermediary results to, Set NONE if dont want any", eSAM_IsOutputFile)
            << EAM(mRatioMaxDistCS,"RatioMaxDistCS",true, "Ratio max of distance P-Center ", eSAM_IsOutputFile)
               ;
}


LArgMain &   cAppli_Tapas_Campari::ArgATP()
{
   return (*mArg);
}


void cAppli_Tapas_Campari::AddParamBloc(std::string & mCom)
{
    // if (EAMIsInit(&mDSElimB))
    {
       mCom = mCom + " +DSElimB=" + ToString(mDSElimB) + " ";
    }
    if (ExportMatrixMarket())
    {
       mCom = mCom + " +ExportMatrixMarket=true ";
    }

    if (mSauvAutom!="")
    {
        if (mSauvAutom=="NONE")
           mCom =   mCom + " +DoSauvAutom=false";
        else
           mCom =   mCom + " +SauvAutom="+mSauvAutom;
    }
    if (EAMIsInit(&mRatioMaxDistCS))
    {
        mCom = mCom + " +RatioMaxDistCS=" + ToString(mRatioMaxDistCS) + " ";
    }


    if (EAMIsInit(&mRapOnZ))
    {
	 ELISE_ASSERT(mRapOnZ.size()==3,"Bad size for RapOnZ");
	 mCom = mCom + " +WithRapOnZ=true" 
		     + " +ZRapOnZ=" + mRapOnZ[0] 
		     + " +SigmaRapOnZ=" + mRapOnZ[1] 
		     + " +KeyGrpRapOnZ=" + mRapOnZ[2] + " "
		 ;
    }

    AddParamBloc(mCom,mVBlockRel,"TimeRel",true);
    AddParamBloc(mCom,mVBlockGlob,"Glob",true);
    AddParamBloc(mCom,mVBlockDistGlob,"DistGlob",false);


    if (EAMIsInit(&mVOptGlob))
    {
       ELISE_ASSERT(EAMIsInit(&mVBlockGlob)|| EAMIsInit(&mVBlockDistGlob),"OptBlocG without BlocGlob");
       ELISE_ASSERT(mVOptGlob.size()>=2,"Not enough arg in OptBlocG");

       double aSigTr,aSigRot;
       FromString(aSigTr,mVOptGlob[0]);
       FromString(aSigRot,mVOptGlob[1]);
       if ((aSigTr<=0) || (aSigRot<=0))
       {
          ELISE_ASSERT((aSigTr==aSigRot) &&((aSigTr==-1)||(aSigTr==-2)),"Bad neg value in OptBlocG");
       }

       if (aSigTr>0)
       {
          mCom +=   std::string(" +WBG_Sigma=true ")
                  + " +WBG_Center=" + ToString(1/ElSquare(aSigTr))
                  + " +WBG_Ang=" + ToString(1/ElSquare(aSigRot))
                  + " " ;
       }
       if (aSigTr==-1)
       {
          mCom += std::string(" +WBG_Stricte=true ");
       }
    }


    if (1)
    {
       mStrParamBloc = mStrParamBloc + BlQUOTE(StrInitOfEAM(&mVOptGlob));
        
/*
       std::cout << "00:" << mCom << "\n";
       std::cout << "11:" << mStrParamBloc << "\n";
       getchar();
*/
    }

}

void cAppli_Tapas_Campari::AddParamBloc(std::string & mCom,std::vector<std::string> & aVBL,const std::string & aPref,bool ModeRot)
{
    mStrParamBloc = mStrParamBloc +  BlQUOTE(StrInitOfEAM(&aVBL)) ;
    int IndRot = ModeRot ? 1 : 0;
    if (!EAMIsInit(&aVBL)) return;
    ELISE_ASSERT(int(aVBL.size()) >= 2+IndRot ,"Not enough param in AddParamBloc");
    ELISE_ASSERT(int(aVBL.size()) <= 4+IndRot,"Too many param in AddParamBloc");


    // Gere le fait que ou Blox initialise une seule fois, ou alors toujours avec le meme nom
    if (!mWithBlock)
    {
        mWithBlock = true;
        mCom = mCom + " +WithBloc=true ";
        mNameInputBloc = aVBL[0];
        mCom = mCom + " +NameInputBloc=" + mNameInputBloc +" ";
        mNameOutputBloc = "Out-" + mNameInputBloc;
        mSBC =   StdGetFromPCP(mNameInputBloc,StructBlockCam);
    }
    else
    {
        ELISE_ASSERT(mNameInputBloc==aVBL[0],"Variable name in NameInputBloc");
    }

    double aSigmaTr0,aSigmaRot0=1;
    FromString(aSigmaTr0,aVBL[1]);
    if (ModeRot)
       FromString(aSigmaRot0,aVBL[2]);

    double aMulFin = 1.0;
    if (int(aVBL.size()) >= 3+IndRot)
       FromString(aMulFin,aVBL[2+IndRot]);

    if (int(aVBL.size())>=4+IndRot)
       mNameOutputBloc = aVBL[3+IndRot];


    double aSigmaTrFin = aSigmaTr0 * aMulFin;
    double aSigmaRotFin = aSigmaRot0 * aMulFin;

    mCom = mCom + " +WithBloc_" + aPref + "=true ";
    mCom = mCom + " +PdsBlocTr0_"  + aPref + "=" + ToString(1.0/ElSquare(aSigmaTr0))  + " ";
    mCom = mCom + " +PdsBlocTrFin_"  + aPref + "=" + ToString(1.0/ElSquare(aSigmaTrFin))  + " ";
    if (ModeRot)
    {
       mCom = mCom + " +PdsBlocRot0_" + aPref + "=" + ToString(1.0/ElSquare(aSigmaRot0)) + " ";
       mCom = mCom + " +PdsBlocRotFin_" + aPref + "=" + ToString(1.0/ElSquare(aSigmaRotFin)) + " ";
    }

    mCom = mCom + " +NameOutputBloc=" + mNameOutputBloc +" ";
}

std::string  cAppli_Tapas_Campari::TimeStamp(const std::string & aName,cInterfChantierNameManipulateur * anICNM)
{
   return anICNM->Assoc2To1(mSBC.KeyIm2TimeCam(),aName,true).first;
}


std::string   cAppli_Tapas_Campari::ExtendPattern
                           (
                                      const std::string & aPatGlob,
                                      const std::string & aPatCenter,
                                      cInterfChantierNameManipulateur * anICNM
                           )
{
   const cInterfChantierNameManipulateur::tSet *  aSetGlob = anICNM->Get(aPatGlob);
   std::string aKey = mSBC.KeyIm2TimeCam();

   const cInterfChantierNameManipulateur::tSet *  aSetCenter = anICNM->Get(aPatCenter);
   cPatOfName aPat;

   for (const auto & aImCenter : *aSetCenter)
   {
      std::string aTimeC = anICNM->Assoc2To1(mSBC.KeyIm2TimeCam(),aImCenter,true).first;
      for (const auto & aName : *aSetGlob)
      {
         std::pair<std::string,std::string> aPair = anICNM->Assoc2To1(mSBC.KeyIm2TimeCam(),aName,true);
         if (aPair.first == aTimeC) 
            aPat.AddName(aName);
            // std::cout << "PAIR " << aPair.first << " *** " <<  aPair.second << "\n";

      }
   }


   return aPat.Pattern();
}

const cStructBlockCam &  cAppli_Tapas_Campari::SBC() const {return mSBC;}
const std::string & cAppli_Tapas_Campari::StrParamBloc() const {return mStrParamBloc;}



void cAppli_Tapas_Campari::InitAllImages(const std::string & aPat,cInterfChantierNameManipulateur * anICNM)
{
    ELISE_ASSERT(mWithBlock,"cAppli_Tapas_Campari::InitAllImages");

    cInterfChantierNameManipulateur::tSet   aSetGlob = *(anICNM->Get(aPat));
    std::vector<std::pair<std::string,std::string> > aVP;
    for (const auto & aS : aSetGlob)
    {
        aVP.push_back(std::pair<std::string,std::string>(TimeStamp(aS,anICNM),aS));
        mBlocCptTime[aVP.back().first]++;
    }
    std::sort(aVP.begin(),aVP.end());
    for (const auto & aPair : aVP)
    {
       mBlocTimeStamps.push_back(aPair.first);
       mBlocImagesByTime.push_back(aPair.second);
    }
}

const std::vector<std::string> & cAppli_Tapas_Campari::BlocImagesByTime() const {return mBlocImagesByTime;}
const std::vector<std::string> & cAppli_Tapas_Campari::BlocTimeStamps() const   {return mBlocTimeStamps;}
std::map<std::string,int> & cAppli_Tapas_Campari::BlocCptTime() {return mBlocCptTime;}

int  cAppli_Tapas_Campari::NbInBloc() const
{
    return mSBC.LiaisonsSHC().Val().ParamOrientSHC().size();
}


int  cAppli_Tapas_Campari::LongestBloc(int aK0,int aK1)
{
     int aLong=1;
     int aLongMax=1;
     for (int aK=aK0+1 ; aK<aK1 ; aK++)
     {
         if (mBlocTimeStamps[aK]  == mBlocTimeStamps[aK-1])
         {
            aLong++;
            aLongMax = ElMax(aLongMax,aLong);
         }
         else
            aLong=1;
     }
     return aLongMax;
}



/****************************************************/
/*                                                  */
/*                    ::                            */
/*                                                  */
/****************************************************/

void Campari_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************************\n";
    std::cout <<  " *     C-ompensation of                      *\n";
    std::cout <<  " *     A-lter                                *\n";
    std::cout <<  " *     M-easurements for                     *\n";
    std::cout <<  " *     P-hotomatric                          *\n";
    std::cout <<  " *     A-djustment after                     *\n";
    std::cout <<  " *     R-otation (and position and etc...)   *\n";
    std::cout <<  " *     I-nitialisation                       *\n";
    std::cout <<  " *********************************************\n\n";
}

/****************************************************/
/*                                                  */
/*             cAppli_Campari                       */
/*                                                  */
/****************************************************/

class cAppli_Campari : public cAppli_Tapas_Campari
{
     public :
       cAppli_Campari(int argc,char ** argv);

       // void AddParamBloc(std::vector<std::string> & aVBL,const std::string & aPref,bool ModeRot);


       int RTA();

       std::string mCom;

       int mResult;
       bool mExe;
       double mGcpGrU;
       double mGcpImU;
       std::vector<double>      mMulRTA;
       std::vector<std::string> GCP;
       std::vector<std::string> GCPRTA;
       std::string mDir,mPat;
       std::string mStr0;
       std::string AeroOut;
       std::string mNameRTA;

       cInterfChantierNameManipulateur * mICNM;

/*
       cAppli_Tapas_Campari      mTPC;
       bool                      mWithBlock;
       std::string               mNameInputBloc;
       std::string               mNameOutputBloc;
       std::vector<std::string>  mVBlockRel;
       std::vector<std::string>  mVBlockGlob;
       std::vector<std::string>  mVBlockDistGlob;
       std::vector<std::string>  mVOptGlob;
*/

       std::vector<double>   mPdsErrorGps;
       std::string  mStrDebugVTP;  // Debug sur les tie points

       int  mNumPtsAttrNewF;
       std::vector<std::string>  mROP;

       std::string  mFileObsPlane;
       double       mWeigthObsPlane;
       double       mExtenZ;
};



cAppli_Campari::cAppli_Campari (int argc,char ** argv) :
    AeroOut          (""),
    mNameRTA         ("SauvRTA.xml"),
    mNumPtsAttrNewF  (-1),
    mWeigthObsPlane  (1.0),
    mExtenZ          (0)
{
    mStr0 = MakeStrFromArgcARgv(argc,argv,true);
    MMD_InitArgcArgv(argc,argv);

    std::string aFullDir= "";
    std::string AeroIn= "";

    bool  CPI1 = false;
    bool  CPI2 = false;
    GlobLibFoc = false;
    GlobLibPP =  false;
    GlobLibAff=  false;
    GlobLibDec = false;
    GlobLibCD=   false;
    // local variable: in campari, used only with GradualRefineCal argument
    LocLibFoc = true;
    LocLibPP =  true;
    LocLibDec = true;
    LocLibCD=   true;
    bool  AllFree = false; 
    std::string AllFreePattern;	
    std::string CalibMod2Refine;
    bool AddViscInterne=false;
    double ViscosInterne=0.1;

    bool  AllPoseFigee = false;
    std::string  PatPoseFigee;
    std::string  PatCentreFigee;
    std::string  PatAngleFigee;

    double aSigmaTieP = 1;
    double aFactResElimTieP = 5;

    std::vector<std::string> EmGPS;
    bool DetailAppuis = false;
    double Viscos = 1.0;
    bool ExpTxt = false;
    std::vector<std::string> aImMinMax;

    Pt3dr aGpsLA;

    GlobDegAdd = 0;
    GlobDegGen=0;
    GlobDRadMaxUSer=0;

    bool AcceptGB=true;
    std::string aSetHom="";
    int aNbIterFin = 4;

    int aNbLiais=100;
    double aPdsGBRot=0.002;
    double aPdsGBId=0.0;
    double aPdsGBIter=1e-6;
    bool   aExportSensib = false;

    bool   aUseGaussJ = false;
    int    NormaliseEq = 3;
    
    std::string RapTxt="";
    std::vector<std::string> aParamCCCC;

    std::vector<double> aVRegulDist;
    std::vector<double> aVExpImRes;

    std::string              aPatGPS;
    std::vector<std::string> aVMultiLA;
    Pt3dr                    aIncLA;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Input Orientation", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Output Orientation", eSAM_IsOutputDirOri),
        LArgMain()  << EAM(GCP,"GCP",true,"[GrMes.xml,GrUncertainty,ImMes.xml,ImUnc]", eSAM_NoInit)
                    << EAM(EmGPS,"EmGPS",true,"Embedded GPS [Gps-Dir,GpsUnc, ?GpsAlti?], GpsAlti if != Plani", eSAM_NoInit)
                    << EAM(aGpsLA,"GpsLa",true,"Gps Lever Arm, in combination with EmGPS", eSAM_NoInit)
                    << EAM(mPdsErrorGps,"PdsResiduGps",true,"Gps weigthing according to error [Mode,MaxPlani,SigmaPlani,MaxAlti,SigmaAlti] Mode=2 (Gauss), 1 (L1 sec)", eSAM_NoInit)
                    << EAM(aVMultiLA,"MultiLA",true,"If multiple LA indicates the patterns of different subsets (first pattern being implicitely first mandatory parameter) ", eSAM_NoInit)
                    << EAM(aIncLA,"IncLA",true,"Inc on initial value of LA (Def not used)")
                    << EAM(aPatGPS,"PatGPS",true,"When EmGPS, filter images where GPS is used")
                    << EAM(aSigmaTieP,"SigmaTieP", true, "Sigma use for TieP weighting (Def=1)")
                    << EAM(aFactResElimTieP,"FactElimTieP", true, "Fact elimination of tie point (prop to SigmaTieP, Def=5)")
                    << EAM(CPI1,"CPI1",true,"Calib Per Im, Firt time", eSAM_IsBool)
                    << EAM(CPI2,"CPI2",true,"Calib Per Im, After first time, reUsing Calib Per Im As input", eSAM_IsBool)
                    << EAM(AllFree,"AllFree",true,"Refine all calibration parameters (Def=false)", eSAM_IsBool)
                    << EAM(AllFreePattern,"AllFreePat",true,"Pattern of images that will be subject to AllFree (Def=.*)", eSAM_IsBool)
                    << EAM(CalibMod2Refine,"GradualRefineCal",true,"Calibration model to refine gradually",eSAM_None)
                    << EAM(DetailAppuis,"DetGCP",true,"Detail on GCP (Def=false)", eSAM_IsBool)
                    << EAM(Viscos,"Visc",true,"Viscosity on external orientation in Levenberg-Marquardt like resolution (Def=1.0)")
                    << EAM(AddViscInterne,"AddViscInterne",true,"Add Viscosity on calibration parameter (Def=false, exept for GradualRefineCal)")
                    << EAM(ViscosInterne,"ViscInterne",true,"Viscosity on calibration parameter (Def=0.1), use it with AddViscInterne=true")
                    << EAM(ExpTxt,"ExpTxt",true, "Export in text format (Def=false)",eSAM_IsBool)
                    << EAM(aImMinMax,"ImMinMax",true, "Im max and min to avoid tricky pat")
                    << EAM(AllPoseFigee,"PoseFigee",true,"Does the external orientation of the cameras are frozen or free (Def=false, i.e. camera poses are free)", eSAM_IsBool)
                    << EAM(PatPoseFigee,"FrozenPoses",true,"List of frozen poses (pattern)")
                    << EAM(PatCentreFigee,"FrozenCenters",true,"List of frozen poses (pattern)")
                    << EAM(PatAngleFigee,"FrozenOrients",true,"List of frozen poses (pattern)")
                    << EAM(AcceptGB,"AcceptGB",true,"Accepte new Generik Bundle image, Def=true, set false for perfect backward compatibility")
                    << EAM(mMulRTA,"MulRTA",true,"Rolling Test Appuis , multiplier ")
                    << EAM(mNameRTA,"NameRTA",true,"Name for save results of Rolling Test Appuis , Def=SauvRTA.xml")
                    << EAM(GCPRTA,"GCPRTA",true,"Internal Use, GCP for RTA ")
                    << EAM(aSetHom,"SH",true,"Set of Hom, Def=\"\", give MasqFiltered for result of HomolFilterMasq, set NONE if unused")
                    << EAM(aNbIterFin,"NbIterEnd",true,"Number of iteration at end, Def = 4")
                    <<  ArgATP()
                    << EAM(aNbLiais,"NbLiais",true,"Param for relative weighting for tie points")
                    << EAM(aPdsGBRot,"PdsGBRot",true,"Weighting of the global rotation constraint (Generic bundle Def=0.002)")
                    << EAM(aPdsGBId,"PdsGBId",true,"Weighting of the global deformation constraint (Generic bundle Def=0.0)")
                    << EAM(aPdsGBIter,"PdsGBIter",true,"Weighting of the change of the global rotation constraint between iterations (Generic bundle Def=1e-6)")
                    << EAM(aExportSensib,"ExportSensib",true,"Export sensiblity (accuracy) estimator : correlation , variance, inverse matrix variance ... ")
                    << EAM(aUseGaussJ,"UseGaussJ",true,"Use GaussJ instead of Cholesky (Def depend of others) ")
                    << EAM(NormaliseEq,"NormEq",true,"Flag for Norm Eq, 1->Sc, 2-Tr, Def=3 (All), tuning purpose ")
                    << EAM(aParamCCCC,"ContrCalCamCons",true,"Constraint on calibration for conseq camera [Key,Simga] ")
                    << EAM(aVRegulDist,"RegulDist",true,"Parameter fo RegulDist [Val,Grad,Hessian,NbCase,SeuilNb]")
                    << EAM(RapTxt,"RapTxt",true,"Output report of residual for each point")
                    << EAM(aVExpImRes,"ExpImRes",true,"Sz of Im Res=[Cam,Pose,Pair]")
                    << EAM(mStrDebugVTP,"StrDebugVTP",true,"String of debug for tie points")

                    << EAM(mNumPtsAttrNewF,"NAWNF",true,"Num Attribute for Weigthing in New Format")
                    << EAM(mROP,"ROP",true,"Rappel On Pose [IdOr,SigmaC,SigmaOr,Pattern]")
                    << EAM(mFileObsPlane,"FOP",true,"File for plane observations on centers")
                    << EAM(mWeigthObsPlane,"WOP",true,"Weight of plane observation on centers")
                    << EAM(mExtenZ,"ExtIntZ",true,"Extension of Z Interval for elimination")
    );


    if (!MMVisualMode)
    {
    #if (ELISE_windows)
         replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
        SplitDirAndFile(mDir,mPat,aFullDir);
        StdCorrecNameOrient(AeroIn,mDir);
        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

        std::string aSetIm = "NKS-Set-OfPattern@[[" + mPat + "]]";



        if (EAMIsInit(&aImMinMax))
        {
            // Pt2dr Focales(0,100000);
            // std::string aParamPatFocSetIm = "@" + aPat + "@" + ToString(Focales.x) + "@" + ToString(Focales.y) ;
            ELISE_ASSERT(aImMinMax.size()==2,"Bad size in vect");
            aSetIm =  "NKS-Set-OfPatternAndInterv@" + mPat + "@" + aImMinMax[0] + "@" + aImMinMax[1];
        }

        if (! EAMIsInit(&aUseGaussJ)) 
           aUseGaussJ = aExportSensib ;

        bool LastIterSupl = false;
       if (aExportSensib || EAMIsInit(&aVExpImRes) )
       {
            if (EAMIsInit(&aNbIterFin))
            {
                if (aNbIterFin>0) aNbIterFin--;
               LastIterSupl = true;
            }
            else
            {
               aNbIterFin = 0;
            }
       }
       if ( (!EAMIsInit(&aNbIterFin)) && (DSElimB()>=3))
       {
          aNbIterFin = 0;
       }

       if (EAMIsInit(&CalibMod2Refine))
       {

           InitVerifModele(CalibMod2Refine,mICNM);
           if (!EAMIsInit(&AddViscInterne)) AddViscInterne=1;
       }

       if (!GlobLibPP && GlobLibCD) std::cout << "Warning, distorsion center is set to free but Principal point is set to frozen.\n I will not adjust Distorsion center.\n";

       mCom =     MM3dBinFile_quotes( "Apero" )
                           +  ToStrBlkCorr( Basic_XML_MM_File("Apero-Compense.xml") )
                           +  std::string(" DirectoryChantier=") + mDir + " "
                           +  std::string(" +SetIm=") + QUOTE(aSetIm) + " "
                           +  std::string(" +PatterIm0=") + QUOTE(mPat) + " "
                           +  std::string(" +AeroIn=-") + AeroIn + " "
                           +  std::string(" +AeroOut=-") + AeroOut + " "
                           +  std::string(" +NbMinIterFin=") + ToString(aNbIterFin) + " "
                           +  std::string(" +NbMaxIterFin=") + ToString(aNbIterFin) + " "
                           +  std::string(" +NbLiais=") + ToString(aNbLiais) + " "
                           +  std::string(" +PdsGBRot=") + ToString(aPdsGBRot) + " "
                           +  std::string(" +PdsGBId=") + ToString(aPdsGBId) + " "
                           +  std::string(" +PdsGBIter=") + ToString(aPdsGBIter) + " "
                          ;

        if (LastIterSupl)  mCom +=          " +LastIterSupl=true";
        if (CPI1 || CPI2) mCom       +=     " +CPI=true ";
        if (CPI2) mCom       +=             " +CPIInput=true ";
        if (GlobLibFoc) mCom    +=          " +FocFree=true ";
        if (GlobLibPP) mCom    +=           " +PPFree=true ";
        if (GlobLibAff) mCom +=             " +AffineFree=true ";
        if (GlobLibCD) mCom +=              " +LibCD=true ";
        if (GlobLibDec) mCom +=              "+LibDec=true ";
        if (AllFree) mCom    +=             " +AllFree=true ";
        if (ExpTxt) mCom += std::string(" +Ext=") + (ExpTxt?"txt ":"dat ")  ;
        if (EAMIsInit(&RapTxt)) mCom += std::string(" +RapTxt=") + RapTxt + " ";
    if (AllPoseFigee) mCom    +=            " +PoseFigee=true ";

        if (EAMIsInit(&AllFreePattern))
        {
            mCom    += "  +AllFreePattern=" + AllFreePattern + " ";
        }
        if (EAMIsInit(&PatPoseFigee))
        {
            mCom    += " +WithPatPoseFigee=true +PatPoseFigee=" + PatPoseFigee + " ";
        }
        if (EAMIsInit(&PatCentreFigee))
        {
            mCom    += " +WithPatCentreFigee=true +PatCentreFigee=" + PatCentreFigee + " ";
        }
        if (EAMIsInit(&PatAngleFigee))
        {
            mCom    += " +WithPatOrientFigee=true +PatOrientFigee=" + PatAngleFigee + " ";
        }

        if (EAMIsInit(&aFactResElimTieP))
           mCom =  mCom+ " +FactMaxRes=" + ToString(aFactResElimTieP) + " ";


       if (Viscos<=0) 
       {
          mCom  +=  " +UseSLM=false ";
          Viscos = 1; // Pour eviter une / par 0 en xml
       }
       if (EAMIsInit(&Viscos)) mCom  +=  " +Viscos=" + ToString(Viscos) + " ";

       if (EAMIsInit(&AddViscInterne)) mCom  +=  " +AddViscInt=true +IntrLVM=" + ToString(ViscosInterne) + " ";

       if (EAMIsInit(&DetailAppuis)) mCom += " +DetailAppuis=" + ToString(DetailAppuis) + " ";


       if (EAMIsInit(&aVExpImRes))
       {
            ELISE_ASSERT
            (
                   (int(aVExpImRes.size()>=3))
                && (int(aVExpImRes.size()<=4)),
                "Bad size for ExpImRes"
            );
            double aNbByC = 10;
            if (int(aVExpImRes.size()) >=4) aNbByC = aVExpImRes[3];
         
            mCom +=    std::string(" +DoUseExportImageResidu=true ")
                    +  std::string(" +UEIR_ByCam=")  + ToString(aVExpImRes[0])
                    +  std::string(" +UEIR_ByPose=") + ToString(aVExpImRes[1])
                    +  std::string(" +UEIR_ByPair=") + ToString(aVExpImRes[2]) 
                    +  std::string(" +UEIR_NbMesByCase=") + ToString(aNbByC) 
                    +  std::string(" ")
           ;
                    
       }
        if (EAMIsInit(&GCP))
        {
            if (EAMIsInit(&GCPRTA))
            {
               GCP = GCPRTA;
            }

            ELISE_ASSERT(GCP.size()==4,"Mandatory part of GCP requires 4 arguments");
            mGcpGrU = RequireFromString<double>(GCP[1],"GCP-Ground uncertainty");
            mGcpImU = RequireFromString<double>(GCP[3],"GCP-Image  uncertainty");

            std::cout << "GPC UNCERTAINCY, Ground : " << mGcpGrU << " === Image : " << mGcpImU << "\n";

            mCom =   mCom
                   + std::string("+WithGCP=true ")
                   + std::string("+FileGCP-Gr=") + GCP[0] + " "
                   + std::string("+FileGCP-Im=") + GCP[2] + " "
                   + std::string("+GrIncGr=") + ToString(mGcpGrU) + " "
                   + std::string("+GrIncIm=") + ToString(mGcpImU) + " ";
        }
        if (GlobDegAdd>0)  mCom = mCom + " +HasModeleAdd=true  +ModeleAdditionnel=eModelePolyDeg" +  ToString(GlobDegAdd);
        if (GlobDegGen>0)  mCom = mCom + " +DegGen=" +  ToString(GlobDegGen);
        if (GlobDRadMaxUSer>0)   mCom = mCom + " +DRMax=" +  ToString(GlobDRadMaxUSer);

        if (EAMIsInit(&EmGPS))
        {
            ELISE_ASSERT((EmGPS.size()>=2) && (EmGPS.size()<=3) ,"Mandatory part of EmGPS requires 2 arguments");
            StdCorrecNameOrient(EmGPS[0],mDir);
            double aGpsU = RequireFromString<double>(EmGPS[1],"GCP-Ground uncertainty");
            double aGpsAlti = aGpsU;
            if (EmGPS.size()>=3)
               aGpsAlti = RequireFromString<double>(EmGPS[2],"GCP-Ground Alti uncertainty");
            mCom = mCom +  " +BDDC=" + EmGPS[0]
                        +  " +SigmGPS=" + ToString(aGpsU)
                        +  " +SigmGPSAlti=" + ToString(aGpsAlti)
                        +  " +WithCenter=true";

            if (EAMIsInit(&aGpsLA))
            {
                mCom = mCom + " +WithLA=true +LaX="  + ToString(aGpsLA.x)
                                         + " +LaY=" + ToString(aGpsLA.y)
                                         + " +LaZ=" + ToString(aGpsLA.z)
                                         + " ";
            }
            if (EAMIsInit(&aVMultiLA))
            {
                ELISE_ASSERT(aVMultiLA.size()<=4,"RTA without GCP");
                for (size_t aK=0 ; aK<aVMultiLA.size() ; aK++)
                {
                     mCom = mCom + " +WithLA" + ToString(int(aK+1)) + std::string("=true ")
                                 + " +PatImLA"+ToString(int(aK+1)) + "=" + QUOTE(aVMultiLA[aK]);
                }
            }
            if (EAMIsInit(&aPatGPS))
            {
                 mCom = mCom + " +PatternGPS=" + QUOTE(aPatGPS);
            }

            if (EAMIsInit(&aIncLA))
            {
                mCom = mCom  + std::string(" +WithIncLA=true")
                             + " +IncLaX=" + ToString(aIncLA.x)
                             + " +IncLaY=" + ToString(aIncLA.y)
                             + " +IncLaZ=" + ToString(aIncLA.z) ;
            }
        }

        if (EAMIsInit(&mPdsErrorGps))
        {
            ELISE_ASSERT(mPdsErrorGps.size()==5,"Bad size for PdsResiduGps");
            std::string aModePond;
            if (mPdsErrorGps[0]==1)  
               aModePond="eL1Secured";
            else if (mPdsErrorGps[0]==2)  
               aModePond= "ePondGauss";
            else
            {
               ELISE_ASSERT(false,"Bad size for PdsResiduGps");
            }
            mCom = mCom + " +ModePondCentre=" + aModePond
                        + " +EcartMaxPlaniPondCentre=" + ToString(mPdsErrorGps[1])
                        + " +SigmaPlaniPondCentre=" + ToString(mPdsErrorGps[2])
                        + " +EcartMaxAltiPondCentre=" + ToString(mPdsErrorGps[3])
                        + " +SigmaPlaniPondCentre=" + ToString(mPdsErrorGps[4])  + " ";
        }


        if (EAMIsInit(&mNumPtsAttrNewF))
        {
           mCom = mCom + " +NumAttrPdsNewF=" + ToString(mNumPtsAttrNewF) + " ";
        }

        if (EAMIsInit(&mROP))
        {
           ELISE_ASSERT(mROP.size()==4,"Bad size for Rappel On Pose (ROP)");
           StdCorrecNameOrient(mROP.at(0),mDir);
           mCom = mCom +  " +WithROP=true"
                       +  " +ROPOrient="+ mROP.at(0)
                       +  " +ROPSigmaC="+ mROP.at(1)
                       +  " +ROPSigmaR="+ mROP.at(2)
                       +  " +ROPPattern="+ QUOTE(mROP.at(3))
                       +  " ";
        }
    

        if (aSetHom=="NONE")
        {
            mCom = mCom + " +UseHom=false ";
        }
        else
        {
           StdCorrecNameHomol(aSetHom,mDir);
           if (EAMIsInit(&aSetHom))
           {
               mCom = mCom + std::string(" +SetHom=") + aSetHom;
           }
        }


        if (EAMIsInit(&aSigmaTieP)) mCom = mCom + " +SigmaTieP=" + ToString(aSigmaTieP);


        if (EAMIsInit(&mMulRTA))
        {
            ELISE_ASSERT(EAMIsInit(&GCP),"RTA without GCP");
        }

        AddParamBloc(mCom);
        if (aUseGaussJ)
        {
           mCom +=   std::string(" +ModeResolSysLin=eSysPlein");
        }
        if (EAMIsInit(&mStrDebugVTP))
        {
           mCom += " +StrDebugVecElimTieP=" + mStrDebugVTP + " ";
        }

        if (aExportSensib) 
        {
           mCom +=   std::string(" +NormaliseEqSc=false")
                   + std::string(" +ExportSensib=true") 
                   + std::string(" +DirExportSensib=/Ori-")+ AeroOut + std::string("/");
        }
        if (EAMIsInit(&NormaliseEq))
        {
           mCom +=   std::string(" +NormaliseEqSc=") + ToString((NormaliseEq&1) != 0)
                   + std::string(" +NormaliseEqTr=") + ToString((NormaliseEq&2) != 0)
                   + std::string(" ");
        }

        if (EAMIsInit(&aParamCCCC))
        {
            ELISE_ASSERT(aParamCCCC.size() >=2,"ContrCalCamCons requires at least 2 vals");
            std::string aKey = aParamCCCC[0];
            double      aSigma ;
            FromString(aSigma,aParamCCCC[1]);
 
            mCom +=    std::string(" +With-ContrCamConseq=true")
                     + std::string(" +Key-CCC=") + aKey
                     + std::string(" +Sigma-CCC=") + ToString(aSigma) + " ";
        }

        if (EAMIsInit(&aVRegulDist))
        {
           ELISE_ASSERT(aVRegulDist.size()>=3,"Not enough parameter in RegulDist")
           double aNbCase = (aVRegulDist.size() >= 4) ? round_ni(aVRegulDist[3])  : 7;
           double aSeuilNbPts = (aVRegulDist.size() >= 5) ? aVRegulDist[4]  : 5.0;
           mCom = mCom  + std::string(" +UseRegulDist=true")
                        + std::string(" +RegDist0=") + ToString(aVRegulDist[0])
                        + std::string(" +RegDist1=") + ToString(aVRegulDist[1])
                        + std::string(" +RegDist2=") + ToString(aVRegulDist[2])
                        + std::string(" +RegDistNbCase=") + ToString(aNbCase)
                        + std::string(" +RegDistSeuil=") + ToString(aSeuilNbPts);
        }

        if (EAMIsInit(&mFileObsPlane))
        {
          mCom =    mCom 
                 +   std::string(" +WithObsPlane=true")
                 +   std::string(" +FileObsPlane=") + mFileObsPlane
                 +   std::string(" +WeightObsPlane=") + ToString(mWeigthObsPlane);
        }

        if (EAMIsInit(&mExtenZ))
	{
          mCom =    mCom 
                 +   std::string(" +WithExtenZ=true")
                 +   std::string(" +ExtenZ=") + ToString(mExtenZ);
	}

        mExe = (! EAMIsInit(&mMulRTA)) || (EAMIsInit(&GCPRTA));

        if (mExe)
        {

            std::cout << mCom << "\n";
            int aRes = System(mCom.c_str());

            Campari_Banniere();
            BanniereMM3D();

            mResult = aRes;
        }
    }
    else
        mResult = EXIT_SUCCESS;
}

int cAppli_Campari::RTA()
{
    // std::cout << "CCCCCCC=[" <<  mStr0 << "]\n";
    cDicoAppuisFlottant aDAF = StdGetFromPCP(mDir +GCP[0] ,DicoAppuisFlottant);
    cSetOfMesureAppuisFlottants aMAF = StdGetFromPCP(mDir +GCP[2] ,SetOfMesureAppuisFlottants);

    cXmlResultRTA aResGlobRTA;
    aResGlobRTA.BestMult() = 0.0;
    aResGlobRTA.BestMoyErr() = 1e20;


    std::string aTmpDAF = "Tmp-RTA-DAF"+ mNameRTA;
    for (int aKMul=0 ; aKMul<int(mMulRTA.size()) ; aKMul++)
    {
         aResGlobRTA.RTA().push_back(cXmlOneResultRTA());
         cXmlOneResultRTA & aResRTA = aResGlobRTA.RTA().back();
         char aBuf[1000];
         double aMul= mMulRTA[aKMul];
         aResRTA.Mult() = aMul;

         sprintf(aBuf,"[%s,%lf,%s,%lf]",aTmpDAF.c_str(),mGcpGrU*aMul,GCP[2].c_str(),mGcpImU*aMul);

         std::string aCom = mStr0 + " GCPRTA=" + std::string(aBuf);
         double aSomDist = 0;
         int    aNbist = 0;

         for
         (
              std::list<cOneAppuisDAF>::iterator itDAF=aDAF.OneAppuisDAF().begin();
              itDAF!=aDAF.OneAppuisDAF().end();
              itDAF ++
         )
         {
               int aNbMesIm =0 ;
               for (std::list<cMesureAppuiFlottant1Im>::const_iterator itMIm=aMAF.MesureAppuiFlottant1Im().begin() ; itMIm!=aMAF.MesureAppuiFlottant1Im().end() ; itMIm++)
               {
                     for (std::list<cOneMesureAF1I>::const_iterator itMp=itMIm->OneMesureAF1I().begin();itMp!=itMIm->OneMesureAF1I().end();itMp++)
                     {
                         aNbMesIm += (itMp->NamePt()==itDAF->NamePt());
                     }
               }



               Pt3dr anI = itDAF->Incertitude();
               if (  ((anI.x>0) || (anI.y>0) || (anI.z>0)) && (aNbMesIm>=2) && (itDAF->UseForRTA().Val()))
               {
                  itDAF->Incertitude() = Pt3dr(-1,-1,-1);
                  MakeFileXML(aDAF,mDir+aTmpDAF);
                  
                  int aResult = System(aCom.c_str());
                  if (aResult != EXIT_SUCCESS) 
                  {
                     return aResult;
                  }

                  std::string aName = mDir + "Ori-" + AeroOut + "/Residus.dmp";
                  cXmlSauvExportAperoGlob aEG = StdGetFromAp(aName,XmlSauvExportAperoGlob);
                  const cXmlSauvExportAperoOneIter &  anIt = aEG.Iters().back();
                  const cXmlSauvExportAperoOneAppuis * TheApp=0;

                  for (std::list<cXmlSauvExportAperoOneAppuis>::const_iterator itAp=anIt.OneAppui().begin() ; itAp!=anIt.OneAppui().end() ; itAp++)
                  {
                       if (itAp->Name() == itDAF->NamePt())
                       {
                           ELISE_ASSERT(TheApp==0,"Multiple name in XmlSauvExportAperoOneAppuis");
                           TheApp = & (*itAp);
                       }
                  }
                  ELISE_ASSERT(TheApp!=0,"No name in XmlSauvExportAperoOneAppuis");

                  // cXmlSauvExportAperoGlob =  
                  // std::cout << "PT=" << TheApp->Name() << "\n"; std::cout << " " << TheApp->EcartFaiscTerrain().Val() << "\n"; getchar();

                  aResRTA.OneAppui().push_back(*TheApp);

                  if (TheApp->DistFaiscTerrain().IsInit())
                  {
                       aSomDist += TheApp->DistFaiscTerrain().Val();
                       aNbist++;
                       aResRTA.MoyErr() = aSomDist / aNbist;
                  }

                  itDAF->Incertitude() = anI;
               }
               MakeFileXML(aResGlobRTA,mNameRTA);
         }
         if (aResGlobRTA.BestMoyErr() > aResRTA.MoyErr())
         {
              aResGlobRTA.BestMoyErr() = aResRTA.MoyErr();
              aResGlobRTA.BestMult() =  aMul;
         }
         MakeFileXML(aResGlobRTA,mNameRTA);
    }
    MakeFileXML(aResGlobRTA,mNameRTA);

    return EXIT_SUCCESS;
}

int Campari_main(int argc,char ** argv)
{
    cAppli_Campari anAppli(argc,argv);

     if (anAppli.mExe || MMVisualMode)
        return anAppli.mResult;

    return anAppli.RTA();
}

int AperoProg_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string aFullDir= "";
    std::string AeroIn= "";
    std::string AeroOut="";


    /*double aSigmaTieP = 1;
    double aFactResElimTieP = 5;
    double Viscos = 1.0;
    bool ExpTxt = false;*/

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile)
                     << EAMC(AeroIn,"Input Orientation", eSAM_IsExistDirOri)
                     << EAMC(AeroOut,"Output Orientation", eSAM_IsOutputDirOri),
         LArgMain()
    );
    if (!MMVisualMode)
    {
        std::string aDir,aPat;
    #if (ELISE_windows)
         replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
        SplitDirAndFile(aDir,aPat,aFullDir);
        StdCorrecNameOrient(AeroIn,aDir);
    }

    return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
