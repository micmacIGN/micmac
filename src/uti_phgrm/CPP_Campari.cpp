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


class cAppli_Campari
{
     public :
       cAppli_Campari(int argc,char ** argv);

       void AddParamBloc(std::vector<std::string> & aVBL,const std::string & aPref);


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
       bool                      mWithBlock;
       std::string               mNameInputBloc;
       std::string               mNameOutputBloc;
       std::vector<std::string>  mVBlockRel;
       std::vector<std::string>  mVBlockGlob;
       std::vector<std::string>  mVOptGlob;
};


void cAppli_Campari::AddParamBloc(std::vector<std::string> & aVBL,const std::string & aPref)
{
    if (!EAMIsInit(&aVBL)) return;
    ELISE_ASSERT(aVBL.size() >= 3,"Not enough param in AddParamBloc");
    ELISE_ASSERT(aVBL.size() <= 5,"Too many param in AddParamBloc");


    if (!mWithBlock)
    {
        mWithBlock = true;
        mCom = mCom + " +WithBloc=true ";
        mNameInputBloc = aVBL[0];
        mCom = mCom + " +NameInputBloc=" + mNameInputBloc +" ";
        mNameOutputBloc = "Out-" + mNameInputBloc;
    }
    else
    {
        ELISE_ASSERT(mNameInputBloc==aVBL[0],"Variable name in NameInputBloc");
    }

    double aSigmaTr0,aSigmaRot0;
    FromString(aSigmaTr0,aVBL[1]);
    FromString(aSigmaRot0,aVBL[2]);

    double aMulFin = 1.0;
    if (aVBL.size() >= 4)
       FromString(aMulFin,aVBL[3]);

    if (aVBL.size()>=5) 
       mNameOutputBloc = aVBL[4];


    double aSigmaTrFin = aSigmaTr0 * aMulFin;
    double aSigmaRotFin = aSigmaRot0 * aMulFin;

    mCom = mCom + " +WithBloc_" + aPref + "=true ";
    mCom = mCom + " +PdsBlocTr0_"  + aPref + "=" + ToString(1.0/ElSquare(aSigmaTr0))  + " ";
    mCom = mCom + " +PdsBlocRot0_" + aPref + "=" + ToString(1.0/ElSquare(aSigmaRot0)) + " ";
    mCom = mCom + " +PdsBlocTrFin_"  + aPref + "=" + ToString(1.0/ElSquare(aSigmaTrFin))  + " ";
    mCom = mCom + " +PdsBlocRotFin_" + aPref + "=" + ToString(1.0/ElSquare(aSigmaRotFin)) + " ";

    mCom = mCom + " +NameOutputBloc=" + mNameOutputBloc +" ";
}



cAppli_Campari::cAppli_Campari (int argc,char ** argv) :
    AeroOut    (""),
    mNameRTA   ("SauvRTA.xml"),
    mWithBlock (false)
{
    mStr0 = MakeStrFromArgcARgv(argc,argv);
    MMD_InitArgcArgv(argc,argv);

    std::string aFullDir= "";
    std::string AeroIn= "";

    bool  CPI1 = false;
    bool  CPI2 = false;
    bool  FocFree = false;
    bool  PPFree = false;
    bool  AffineFree = false;
    bool  AllFree = false;

    bool  PoseFigee = false;

    double aSigmaTieP = 1;
    double aFactResElimTieP = 5;

    std::vector<std::string> EmGPS;
    bool DetailAppuis = false;
    double Viscos = 1.0;
    bool ExpTxt = false;
    std::vector<std::string> aImMinMax;

    Pt3dr aGpsLA;

    int aDegAdd = 0;
    int aDegFree = 0;
    int aDrMax = 0;
    bool AcceptGB=true;
    std::string aSetHom="";
    int aNbIterFin = 4;

    int aNbLiais=100;



    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Input Orientation", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Output Orientation", eSAM_IsOutputDirOri),
        LArgMain()  << EAM(GCP,"GCP",true,"[GrMes.xml,GrUncertainty,ImMes.xml,ImUnc]", eSAM_NoInit)
                    << EAM(EmGPS,"EmGPS",true,"Embedded GPS [Gps-Dir,GpsUnc, ?GpsAlti?], GpsAlti if != Plani", eSAM_NoInit)
                    << EAM(aGpsLA,"GpsLa",true,"Gps Lever Arm, in combination with EmGPS", eSAM_NoInit)
                    << EAM(aSigmaTieP,"SigmaTieP", true, "Sigma use for TieP weighting (Def=1)")
                    << EAM(aFactResElimTieP,"FactElimTieP", true, "Fact elimination of tie point (prop to SigmaTieP, Def=5)")
                    << EAM(CPI1,"CPI1",true,"Calib Per Im, Firt time", eSAM_IsBool)
                    << EAM(CPI2,"CPI2",true,"Calib Per Im, After first time, reUsing Calib Per Im As input", eSAM_IsBool)
                    << EAM(FocFree,"FocFree",true,"Foc Free (Def=false)", eSAM_IsBool)
                    << EAM(PPFree,"PPFree",true,"Principal Point Free (Def=false)", eSAM_IsBool)
                    << EAM(AffineFree,"AffineFree",true,"Affine Parameter (Def=false)", eSAM_IsBool)
                    << EAM(AllFree,"AllFree",true,"Affine Parameter (Def=false)", eSAM_IsBool)
                    << EAM(DetailAppuis,"DetGCP",true,"Detail on GCP (Def=false)", eSAM_IsBool)
                    << EAM(Viscos,"Visc",true,"Viscosity in Levenberg-Marquardt like resolution (Def=1.0)")
                    << EAM(ExpTxt,"ExpTxt",true, "Export in text format (Def=false)",eSAM_IsBool)
                    << EAM(aImMinMax,"ImMinMax",true, "Im max and min to avoid tricky pat")
                    << EAM(aDegAdd,"DegAdd",true, "When specified, degree of additionnal parameter")
                    << EAM(aDegFree,"DegFree",true, "When specified degree of freedom of parameters generiqs")
                    << EAM(aDrMax,"DRMax",true, "When specified degree of freedom of radial parameters")
 		    << EAM(PoseFigee,"PoseFigee",true,"Does the external orientation of the cameras are frozen or free (Def=false, i.e. camera poses are free)", eSAM_IsBool)
                    << EAM(AcceptGB,"AcceptGB",true,"Accepte new Generik Bundle image, Def=true, set false for perfect backward compatibility")
                    << EAM(mMulRTA,"MulRTA",true,"Rolling Test Appuis , multiplier ")
                    << EAM(mNameRTA,"NameRTA",true,"Name for save results of Rolling Test Appuis , Def=SauvRTA.xml")
                    << EAM(GCPRTA,"GCPRTA",true,"Internal Use, GCP for RTA ")
                    << EAM(aSetHom,"SH",true,"Set of Hom, Def=\"\", give MasqFiltered for result of HomolFilterMasq, set NONE if unused")
                    << EAM(aNbIterFin,"NbIterEnd",true,"Number of iteration at end, Def = 4")
                    // << EAM(GCP,"MulRTA",true,"Rolling Test Appuis , multiplier ")
                    << EAM(mVBlockGlob,"BlocGlob",true,"Param for Glob bloc compute [File,SigmaCenter,SigmaRot,?MulFinal,?Export]")
                    << EAM(mVOptGlob,"OptBlocG",true,"[SigmaTr,SigmaRot]")
                    << EAM(mVBlockRel,"BlocTimeRel",true,"Param for Time Reliative bloc compute [File,SigmaCenter,SigmaRot,?MulFinal,?Export]")
                    << EAM(aNbLiais,"NbLiais",true,"Param for relative weighting for tie points")

    );


    if (!MMVisualMode)
    {
    #if (ELISE_windows)
         replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
        SplitDirAndFile(mDir,mPat,aFullDir);
        StdCorrecNameOrient(AeroIn,mDir);

        std::string aSetIm = "NKS-Set-OfPattern@" + mPat;



        if (EAMIsInit(&aImMinMax))
        {
            // Pt2dr Focales(0,100000);
            // std::string aParamPatFocSetIm = "@" + aPat + "@" + ToString(Focales.x) + "@" + ToString(Focales.y) ;
            ELISE_ASSERT(aImMinMax.size()==2,"Bad size in vect");
            aSetIm =  "NKS-Set-OfPatternAndInterv@" + mPat + "@" + aImMinMax[0] + "@" + aImMinMax[1];
        }




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
                          ;

        if (CPI1 || CPI2) mCom       += " +CPI=true ";
        if (CPI2) mCom       += " +CPIInput=true ";
        if (FocFree) mCom    += " +FocFree=true ";
        if (PPFree) mCom    += " +PPFree=true ";
        if (AffineFree) mCom += " +AffineFree=true ";
        if (AllFree) mCom    += " +AllFree=true ";
        if (ExpTxt) mCom += std::string(" +Ext=") + (ExpTxt?"txt ":"dat ")  ;

 	if (PoseFigee) mCom    += " +PoseFigee=true ";

        if (EAMIsInit(&aFactResElimTieP))
           mCom =  mCom+ " +FactMaxRes=" + ToString(aFactResElimTieP);


       if (EAMIsInit(&Viscos)) mCom  +=  " +Viscos=" + ToString(Viscos) + " ";

       if (EAMIsInit(&DetailAppuis)) mCom += " +DetailAppuis=" + ToString(DetailAppuis) + " ";

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
        if (aDegAdd>0)  mCom = mCom + " +HasModeleAdd=true  +ModeleAdditionnel=eModelePolyDeg" +  ToString(aDegAdd);
        if (aDegFree>0)  mCom = mCom + " +DegGen=" +  ToString(aDegFree);
        if (aDrMax>0)   mCom = mCom + " +DRMax=" +  ToString(aDrMax);

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

        AddParamBloc(mVBlockRel,"TimeRel");
        AddParamBloc(mVBlockGlob,"Glob");
        if (EAMIsInit(&mVOptGlob))
        {
           ELISE_ASSERT(EAMIsInit(&mVBlockGlob),"OptBlocG without BlocGlob");
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

         // std::cout << "GPPPPP " << aBuf << "\n";
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
