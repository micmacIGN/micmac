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
#include "Apero/Apero.h"
#define DEF_OFSET -12349876

//                   1  X  Y  X2 XY Y2
int NLDDegFlagX[6] ={0 ,1 ,0 ,2 ,1 ,0 };
int NLDDegFlagY[6] ={0 ,0 ,1 ,0 ,1 ,2 };

extern const char* NamePolQuadXY[6];
// static const char *  PolXY[6] = {"1","X","Y","X2","XY","Y2"};

int FlagOfDeg(const std::vector<std::string> & aDXY)
{
   int aRes = 0;
   for (int aK=0 ; aK<int(aDXY.size()) ; aK++)
   {
       int aGot = -1;
       for (int aP=0 ; aP<6 ; aP++)
       {
           if (aDXY[aK] == NamePolQuadXY[aP])
           {
              aGot= aP;
           }
       }
       if (aGot >= 0)
       {
          aRes |= (1<<aGot);
       }
       else
       {
           std::cout << "For monom =" << aDXY[aK] << "\n";
           ELISE_ASSERT(false,"Is no a valid monom");
       }
   }

   return aRes;
}

int FlagOfDeg(const Pt3di & aDXY)
{
   int aRes = 0;
   for (int aK=0 ; aK<6 ; aK++)
   {
        if (      (aDXY.x>=NLDDegFlagX[aK]) 
               && (aDXY.y>=NLDDegFlagY[aK]) 
               && (aDXY.z >= (NLDDegFlagX[aK]+NLDDegFlagY[aK]))
           )
           aRes |= 1<< aK;
   }
   return aRes;
}

int GCPBascule_main(int argc,char ** argv)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;


    std::string AeroOut;
    std::string AeroIn;
    std::string DicoPts;
    std::string MesureIm;
    bool        ModeL1 = false;
    bool        CPI = false;
    bool ShowUnused = true;
    bool ShowDetail = false;
    bool NLDShow = false;
    bool NLDFTR = true;
    std::string ForceSol;

    std::string aPatNLD;
    std::vector<std::string> NLDDegX;  NLDDegX.push_back("1");  NLDDegX.push_back("X"); NLDDegX.push_back("Y");
    std::vector<std::string> NLDDegY;  NLDDegY.push_back("1");  NLDDegY.push_back("X"); NLDDegY.push_back("Y");
    std::vector<std::string> NLDDegZ;  NLDDegZ.push_back("1");  NLDDegZ.push_back("X"); NLDDegZ.push_back("X2");
/*
    Pt3di NLDDegX(1,1,1);
    Pt3di NLDDegY(1,1,1);
    Pt3di NLDDegZ(2,0,2);
*/


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(AeroOut,"Orientation out", eSAM_IsOutputDirOri)
                    << EAMC(DicoPts,"Ground Control Points File", eSAM_IsExistFile)
                    << EAMC(MesureIm,"Image Measurements File", eSAM_IsExistFile),
        LArgMain()
                    <<  EAM(ModeL1,"L1",true,"L1 minimisation vs L2 (Def=false)", eSAM_IsBool)
                    <<  EAM(CPI,"CPI",true,"when Calib Per Image has to be used", eSAM_IsBool)
                    <<  EAM(ShowUnused,"ShowU",true,"Show unused point (def=true)", eSAM_IsBool)
                    <<  EAM(ShowDetail,"ShowD",true,"Show details (def=false)", eSAM_IsBool)
                    <<  EAM(aPatNLD,"PatNLD",true,"Pattern for Non linear deformation, with aerial like geometry (def,unused)")
                    <<  EAM(NLDDegX,"NLDegX",true,"Non Linear monoms for X, when PatNLD, (Def =[1,X,Y])")
                    <<  EAM(NLDDegY,"NLDegY",true,"Non Linear monoms for Y, when PatNLD, (Def =[1,X,Y])")
                    <<  EAM(NLDDegZ,"NLDegZ",true,"Non Linear monoms for Z, when PatNLD, (Def =[1,X,X2])")
                    <<  EAM(NLDFTR,"NLFR",true,"Non Linear : Force True Rot (Def=true)",eSAM_IsBool)
                    <<  EAM(NLDShow,"NLShow",true,"Non Linear : Show Details (Def=false)",eSAM_IsBool)
                    <<  EAM(ForceSol,"ForceSol",true,"To Force Sol from existing solution (xml file)",eSAM_IsExistFile)
    );

    if (!MMVisualMode)
    {
    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);


    MMD_InitArgcArgv(argc,argv);

    std::string aCom =   MM3dBinFile_quotes( "Apero" )
                       + ToStrBlkCorr( MMDir()+"include/XML_MicMac/Apero-GCP-Bascule.xml" )+" "
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +AeroIn=") + AeroIn
                       + std::string(" +AeroOut=") +  AeroOut
                       + std::string(" +DicoApp=") +  QUOTE(DicoPts)
                       + std::string(" +SaisieIm=") +  QUOTE(MesureIm)
                    ;

    if (EAMIsInit(&ShowUnused)) aCom = aCom + " +ShowUnused=" + ToString(ShowUnused);
    if (EAMIsInit(&ShowDetail)) aCom = aCom + " +ShowDetail=" + ToString(ShowDetail);

    if (EAMIsInit(&ForceSol))
    {
       aCom = aCom + " +DoForceSol=true  +NameForceSol=" + ForceSol + " ";
    }

    if (ModeL1)
    {
        aCom = aCom+ std::string(" +L2Basc=") + ToString(!ModeL1);
    }

    if (CPI) aCom += " +CPI=true ";


    if (EAMIsInit(&aPatNLD))
    {
       aCom = aCom + " +UseNLD=true +PatNLD=" + QUOTE(aPatNLD)
                   + " +NLFlagX=" + ToString(FlagOfDeg(NLDDegX))
                   + " +NLFlagY=" + ToString(FlagOfDeg(NLDDegY))
                   + " +NLFlagZ=" + ToString(FlagOfDeg(NLDDegZ))
                   + " +NLDForceTR=" + ToString(NLDFTR)
                   + " +NLDShow=" + ToString(NLDShow)
              ;
    }


    std::cout << "Com = " << aCom << "\n";
    int aRes = System(aCom.c_str(),false,true,true);


    return aRes;

    }
    else return EXIT_SUCCESS;
}


int GCPCtrl_main(int argc,char ** argv)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;


    std::string AeroIn;
    std::string DicoPts;
    std::string MesureIm;
    bool        CPI = false;
    bool ShowUnused = true;

    std::string OutTxt {"ResRoll.txt"};
    bool BoolOutTxt =false;

    std::string OutJSON {"Res.geojson"};
    bool BoolOutJSON =false;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(DicoPts,"Ground Control Points File", eSAM_IsExistFile)
                    << EAMC(MesureIm,"Image Measurements File", eSAM_IsExistFile),
        LArgMain()
                    <<  EAM(CPI,"CPI",true,"when Calib Per Image has to be used", eSAM_IsBool)
                    <<  EAM(ShowUnused,"ShowU",true,"Show unused point (def=true)", eSAM_IsBool)
                    <<  EAM(OutTxt,"OutTxt",true,"Name TXT file for Ctrl result (def=false)")
                    <<  EAM(OutJSON,"OutJSON",true,"Name .geojson file for Ctrl result (def=false)")
                );

    if (!MMVisualMode)
    {
    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);



    std::string aCom =   MM3dBinFile_quotes( "Apero" )
                       + ToStrBlkCorr( MMDir()+"include/XML_MicMac/Apero-GCP-Control.xml" )+" "
                       + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                       + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                       + std::string(" +AeroIn=") + AeroIn
                       + std::string(" +DicoApp=") +  DicoPts
                       + std::string(" +SaisieIm=") +  MesureIm
                    ;

    if (EAMIsInit(&ShowUnused)) aCom = aCom + " +ShowUnused=" + ToString(ShowUnused);
    if (CPI) aCom += " +CPI=true ";
    if (EAMIsInit(&OutTxt)) BoolOutTxt=true;
    aCom += " +BoolOutTxt=" + ToString(BoolOutTxt) + " +OutTxt=" +OutTxt;

    if (EAMIsInit(&OutJSON)) BoolOutJSON=true;
    aCom += " +BoolOutJSON=" + ToString(BoolOutJSON) + " +OutJSON=" +OutJSON;

    std::cout << "Com = " << aCom << "\n";
    int aRes = System(aCom.c_str(),false,true,true);


    return aRes;

    }
    else return EXIT_SUCCESS;
}

//extern bool L2SYM;
extern const char * theNameVar_ParamApero[];
int GCPCtrlPly_main(int argc,char ** argv)
{
    
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;


    std::string AeroIn;
    std::string DicoPts;
    std::string MesureIm;
    bool        CPI = false;
    bool ShowUnused = true;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(DicoPts,"Ground Control Points File", eSAM_IsExistFile)
                    << EAMC(MesureIm,"Image Measurements File", eSAM_IsExistFile),
        LArgMain()
                    <<  EAM(CPI,"CPI",true,"when Calib Per Image has to be used", eSAM_IsBool)
                    <<  EAM(ShowUnused,"ShowU",true,"Show unused point (def=true)", eSAM_IsBool)
    );

    if (!MMVisualMode)
    {
    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    }
    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);
  

    /* Prepare the input arguments */    
    argc = 7;
    char** n_argv = new char *[argc];

    std::string aArg1 = "Apero";
    std::string::iterator aArgIt=aArg1.begin();
    char* aChr = &(*aArgIt);
    n_argv[0] = aChr;
    
    std::string aArg3 = ( MMDir()+"include/XML_MicMac/Apero-GCP-Control.xml" );
    std::string::iterator aArgIt3=aArg3.begin();
    char* aChr3 = &(*aArgIt3);
    n_argv[1] = aChr3;

    std::string aArg4 = std::string("DirectoryChantier=") +aDir;
    std::string::iterator aArgIt4=aArg4.begin();
    char* aChr4 = &(*aArgIt4);
    n_argv[2] = aChr4;

    std::string aArg5 = std::string("+PatternAllIm=") + QUOTE(aPat);
    std::string::iterator aArgIt5=aArg5.begin();
    char* aChr5 = &(*aArgIt5);
    n_argv[3] = aChr5;

    std::string aArg6 = std::string("+AeroIn=") + AeroIn;
    std::string::iterator aArgIt6=aArg6.begin();
    char* aChr6 = &(*aArgIt6);
    n_argv[4] = aChr6;

    std::string aArg7 = std::string("+DicoApp=") +  DicoPts;
    std::string::iterator aArgIt7=aArg7.begin();
    char* aChr7 = &(*aArgIt7);
    n_argv[5] = aChr7;

    std::string aArg8 = std::string("+SaisieIm=") +  MesureIm;
    std::string::iterator aArgIt8=aArg8.begin();
    char* aChr8 = &(*aArgIt8);
    n_argv[6] = aChr8;

    MMD_InitArgcArgv(argc,n_argv);



    AddEntryStringifie
    (
#if ELISE_windows
        "include\\XML_GEN\\ParamApero.xml",
#else
        "include/XML_GEN/ParamApero.xml",
#endif
        theNameVar_ParamApero,
        true
     );

    std::string aNameSauv = "SauvApero.xml";
    if ( isUsingSeparateDirectories() ) aNameSauv=MMLogDirectory()+aNameSauv;

    cResultSubstAndStdGetFile<cParamApero> aP2
    (
         argc-2,n_argv+2,
         n_argv[1],
         StdGetFileXMLSpec("ParamApero.xml"),
         "ParamApero",
         "ParamApero",
         "DirectoryChantier",
         "FileChantierNameDescripteur",
         aNameSauv.c_str()
    );
    ::DebugPbCondFaisceau = aP2.mObj->DebugPbCondFaisceau().Val();

    //L2SYM = aP2.mObj->AllMatSym().Val();
    cAppliApero   anAppli (aP2);


    if (anAppli.ModeMaping())
    {
        anAppli.DoMaping(argc,n_argv);
    }
    else
    {
        anAppli.DoCompensation();
    }


    cElWarning::ShowWarns( ( isUsingSeparateDirectories()?MMTemporaryDirectory():anAppli.DC() ) + "WarnApero.txt");
    ShowFClose();
    


    /*delete aChr;
    delete aChr3;
    delete aChr4;
    delete aChr5;
    delete aChr6;
    delete aChr7;
    delete aChr8;
*/
    return 0;
}


int GCPVisib_main(int argc,char ** argv)
{
    
    MMD_InitArgcArgv(argc,argv);
    
    cInterfChantierNameManipulateur * aICNM;
    std::string aFullDir, aDir, aPat, aAeroIn, DicoPts, aFilePtsOut="GCPVisibility.txt";
    std::list<std::string> aListFile;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(aAeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(DicoPts,"Ground Control Points File", eSAM_IsExistFile),
        LArgMain()
                    <<  EAM(aFilePtsOut,"Out",true,"Output filename")
    );


    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
    #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
   

    /* Get the list of orientations/cameras */
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    aListFile = aICNM->StdGetListOfFile(aPat);
    StdCorrecNameOrient(aAeroIn,aDir);
    

    /* Read the groud control points */
    int aNb=0;
    std::vector<Pt3dr> aVPts;
    std::vector<std::string> aVName;

    cDicoAppuisFlottant aD = StdGetObjFromFile<cDicoAppuisFlottant>
        (
            DicoPts,
            StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
            "DicoAppuisFlottant",
            "DicoAppuisFlottant"
         );
    
    std::list<cOneAppuisDAF>::iterator itA=aD.OneAppuisDAF().begin();
    for( ; itA!=aD.OneAppuisDAF().end(); itA++ )
    {
        aVPts.push_back(itA->Pt());
        aVName.push_back(itA->NamePt());

        aNb++;
    }


    /* Check visibility of the GCP in the list of cameras */
    FILE *  aFOut = FopenNN((aDir+aFilePtsOut).c_str(),"w","GCPVisib");

    int aK=0;
    cBasicGeomCap3D * aBGC =0;
    std::list<std::string>::iterator itL = aListFile.begin();
    for( ;
         itL != aListFile.end();
         itL++
       )
    {
        
        aBGC = aICNM->StdCamGenerikOfNames(aAeroIn, (*itL));
        
        for( aK=0; aK<aNb; aK++ )
        {

            if( aBGC->PIsVisibleInImage(aVPts.at(aK)) )
            {
                Pt2dr aPIm = aBGC->Ter2Capteur(aVPts.at(aK));
                fprintf(aFOut,"%s %s %lf %lf\n", 
                        aVName.at(aK).c_str(), (*itL).c_str(), aPIm.x, aPIm.y);
            }
        }
    }
    ElFclose(aFOut);
            


    return EXIT_SUCCESS;
}

int GCP_Fusion(int argc,char ** argv)
{
    std::vector<std::string> aVIntPut;
    std::string              aNameRes;
    std::map<std::string,int>    aCPt;

    std::vector<std::string> aVChInc;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aVIntPut,"[Inputs]")
                    << EAMC(aNameRes,"Name Result"),
        LArgMain()
                    <<  EAM(aVChInc,"SetInc",true,"[Pat1,Ix1,Iy1,Iz1,Pat2,Ix2...]")
    );

    cDicoAppuisFlottant aRes;

    ELISE_ASSERT((aVChInc.size())%4==0,"Bad size for SetInc");
    std::vector<cElRegex *> aVAutom;
    for (int aKInc = 0 ; aKInc<int(aVChInc.size()) ; aKInc+=4)
    {
         aVAutom.push_back(new cElRegex(aVChInc[aKInc],10));
    }

    for (int aKInput=0 ; aKInput<int(aVIntPut.size()) ; aKInput++)
    {
       //cSetOfMesureAppuisFlottants
       cDicoAppuisFlottant aDAF = StdGetFromPCP(aVIntPut[aKInput],DicoAppuisFlottant);
       for (std::list<cOneAppuisDAF>::const_iterator itP=aDAF.OneAppuisDAF().begin() ; itP!=aDAF.OneAppuisDAF().end() ; itP++)
       {
           cOneAppuisDAF aPt = *itP;
           aCPt[aPt.NamePt()] ++;
           if (aCPt[aPt.NamePt()]==1)
           {
              for (int aKInc = 0 ; aKInc<int(aVAutom.size()) ; aKInc+=1)
              {
                    if (aVAutom[aKInc]->Match(aPt.NamePt()))
                    {
                       FromString(aPt.Incertitude().x,aVChInc[4*aKInc+1]);
                       //FromString(aVChInc[4*aKInc+1],aPt.Incertitude().x);
                       // aPt.Incertitude() = Pt3dr(
                    }
              }
              aRes.OneAppuisDAF().push_back(aPt);
           }
       }
    }

    MakeFileXML(aRes,aNameRes);
    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
