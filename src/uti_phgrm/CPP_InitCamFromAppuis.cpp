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

const cOneAppuisDAF * GetDAFFromName(const cDicoAppuisFlottant & aDic,const std::string & aName)
{
   for
   (
       std::list<cOneAppuisDAF>::const_iterator itOAD=aDic.OneAppuisDAF().begin();
       itOAD!=aDic.OneAppuisDAF().end();
       itOAD++
   )
   {
        if (itOAD->NamePt() == aName)
           return &(*itOAD);
   }
   return 0;
}

class cInitCamAppuis
{
    public :

       cInitCamAppuis(int argc,char ** argv,LArgMain &,const std::string & aOri);
       std::string NameOri(const std::string & aNameIm)
       {
           return aICNM->Assoc1To1(mKeyOri,aNameIm,true);
       }

       bool InitPts(const cMesureAppuiFlottant1Im &);

       std::string aNameFile3D;
       std::string aNameFile2D;
       std::string         mFilter;

       cSetOfMesureAppuisFlottants mSMAF;
       cDicoAppuisFlottant         mDicApp;

       std::vector<Pt3dr> mVCPCur;
       std::vector<Pt2dr> mVImCur;
       std::list<Appar23> mL32;
       cElRegex *         mAutoF;
       cInterfChantierNameManipulateur * aICNM;
       std::string                       mKeyOri;
};

cInitCamAppuis::cInitCamAppuis(int argc,char ** argv,LArgMain & ArgOpt,const std::string & anOri) :
    mFilter (".*")
{
   ArgOpt = ArgOpt <<   EAM(mFilter,"Filter",true,"Filter for Image (Def=.*)");

   ElInitArgMain
   (
       argc,argv,
       LArgMain()  << EAMC(aNameFile3D,"Name File for GCP",eSAM_IsExistFile)
                   << EAMC(aNameFile2D,"Name File for Image Measures",eSAM_IsExistFile),
       ArgOpt
   );

   if (!MMVisualMode)
   {
       mDicApp = StdGetFromPCP(aNameFile3D,DicoAppuisFlottant);
       mSMAF =  StdGetFromPCP(aNameFile2D,SetOfMesureAppuisFlottants);

       mAutoF = new cElRegex(mFilter,10);
       aICNM = cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aNameFile3D));
       mKeyOri = "NKS-Assoc-Im2Orient@"+anOri;
   }
}

bool cInitCamAppuis::InitPts(const cMesureAppuiFlottant1Im & aMAF)
{
   mVCPCur.clear();
   mVImCur.clear();
   mL32.clear();

   if (! mAutoF->Match(aMAF.NameIm()))
      return false;

   for
   (
         std::list<cOneMesureAF1I>::const_iterator itM=aMAF.OneMesureAF1I().begin();
         itM!=aMAF.OneMesureAF1I().end();
         itM++
   )
   {
         const cOneAppuisDAF * aOAF = GetDAFFromName(mDicApp,itM->NamePt());
         if (aOAF)
         {
             mVCPCur.push_back(aOAF->Pt());
             mVImCur.push_back(itM->PtIm());
             mL32.push_back(Appar23(itM->PtIm(),aOAF->Pt()));
         }
   }
   return true;
}

//==============================================

class cAppli_Aspro //  :  public cAppliWithSetImage
{
    public :

       cAppli_Aspro(int argc,char ** argv);
  
       cElemAppliSetFile mEASF;
       std::string mNameIm;
       std::string mNameCalib;
       std::string mNameFile3D;
       std::string mNameFile2D;
       std::string mOriOut;
};

cAppli_Aspro::cAppli_Aspro(int argc,char ** argv) ://  : cAppliWithSetImage (argc-1,argv+1,0)
    mOriOut ("Aspro")
{

   ElInitArgMain
   (
       argc,argv,
       LArgMain()  << EAMC(mNameIm,"Name File for images",eSAM_IsExistFile)
                   << EAMC(mNameCalib,"Name File for input calibratio,",eSAM_IsExistFile)
                   << EAMC(mNameFile3D,"Name File for GCP",eSAM_IsExistFile)
                   << EAMC(mNameFile2D,"Name File for Image Measures",eSAM_IsExistFile),
       LArgMain() <<  EAM(mOriOut,"Out",true,"Out orientation, def=Aspro")
   );

   
   mEASF.Init(mNameIm);
   StdCorrecNameOrient(mNameCalib,mEASF.mDir);

   std::string aCom =          MM3dBinFile_quotes("Apero")
                       + " " + XML_MM_File("Apero-GCP-Init.xml")
                       + " DirectoryChantier=" + mEASF.mDir
                       + " +PatternAllIm=" + mEASF.mPat
                       + " +CalibIn="      + mNameCalib
                       + " +AeroOut=" + mOriOut
                       + " +DicoApp="  + mNameFile3D
                       + " +SaisieIm=" + mNameFile2D;

   // std::cout <<  aCom << "\n";
   System(aCom);

   std::cout << "**********************************************\n";
   std::cout << "*   (only on medical prescription in         *\n";
   std::cout << "        case of Apero's abuse)               *\n";
   std::cout << "*                                            *\n";
   std::cout << "*        A-utomatic                          *\n";
   std::cout << "*        SP-ace                              *\n";
   std::cout << "*        R-esection for                      *\n";
   std::cout << "*        O-rientation                        *\n";
   std::cout << "*                                            *\n";
   std::cout << "**********************************************\n";
/*
Apero  /home/mpd/MMM/culture3d/include/XML_MicMac/Apero-GCP-Init.xml DirectoryChantier=/media/data2/Jeux-Test/Dino/ +PatternAllIm=_MG_0080.CR2 +CalibIn=AllFix +AeroOut=Aspro +DicoApp=TestOPA-S3D.xml +SaisieIm=TestOPA-S2D.xml
*/

/*
    std::string anOriCalib;
    LArgMain ArgOpt;
    ArgOpt <<  EAM(anOriCalib,"OriCalib",true,"Folder for calibration (if none use no distorsion)");

    cInitCamAppuis aICA(argc,argv,ArgOpt,"-Aspro");

    if (anOriCalib!="")
          aICA.aICNM->CorrecNameOrient(anOriCalib);

    int aNbRansac = 10000;


    for
    (
        std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF = aICA.mSMAF.MesureAppuiFlottant1Im().begin();
        itMAF != aICA.mSMAF.MesureAppuiFlottant1Im().end();
        itMAF++
    )
    {
        if (aICA.InitPts(*itMAF) && (int(aICA.mVCPCur.size())>=3))
        {
             std::string aNameIm = itMAF->NameIm();
             std::cout << "Init 6 Param :" << aNameIm << "\n";
             CamStenope * aCam = aICA.aICNM->GlobCalibOfName(aNameIm,anOriCalib);
             // ElRotation3D aR = aCam->RansacOFPA(true,aNbRansac
        }
    }
*/

}
int Init3App_Main(int argc,char ** argv)
{
    cAppli_Aspro anAppli(argc,argv);
    return EXIT_SUCCESS;
}


std::string Dir11Param = "11Param";

int Init11Param_Main(int argc,char ** argv)
{
    bool        isFraserModel = false;
    std::vector<std::string> aRansacParam;
    LArgMain ArgOpt;
    ArgOpt <<  EAM(isFraserModel,"FM",true,"Fraser Mode, use all affine parmeters (def=false)")
            << EAM(aRansacParam,"Rans",true,"Parameters for Ransac, [NbTirage,PropInlier]");

    cInitCamAppuis aICA(argc,argv,ArgOpt,"-" + Dir11Param);


    bool RansacMode =EAMIsInit(&aRansacParam);
    int     aNbTirage,aNbMaxP=0;
    double  aPropInlier;
    if (RansacMode)
    {
         ELISE_ASSERT(aRansacParam.size()==2,"Bad size for ransac param");
         FromString(aNbTirage  ,aRansacParam[0]);
         FromString(aPropInlier,aRansacParam[1]);

         aNbMaxP = 6 + aNbTirage/20;
    }
    
    std::vector<cQual12Param>  aVQual;
    std::vector<std::string>   aVName;
    for
    (
        std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF = aICA.mSMAF.MesureAppuiFlottant1Im().begin();
        itMAF != aICA.mSMAF.MesureAppuiFlottant1Im().end();
        itMAF++
    )
    {
        if (aICA.InitPts(*itMAF) && (int(aICA.mVCPCur.size())>=6))
        {
             cQual12Param aQual;
             std::string aNameIm = itMAF->NameIm();
             std::cout << "Init11Param :" << aNameIm << "\n";
             double Alti,Prof;
             cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aNameIm);
             Pt2di aSzCam = aMDP.SzImTifOrXif();
             CamStenope * aCS =
                        RansacMode                                                                              ?
                        cEq12Parametre::RansacCamera11Param(aQual,aSzCam,isFraserModel,aICA.mVCPCur,aICA.mVImCur,Alti,Prof,aNbTirage,aPropInlier,aNbMaxP) :
                        cEq12Parametre::Camera11Param(aQual,aSzCam,isFraserModel,aICA.mVCPCur,aICA.mVImCur,Alti,Prof) ;

             if (aCS)
             {
                 cOrientationConique anEC = aCS->ExportCalibGlob(aSzCam,Alti,Prof ,false,true,(char*)0);
                 MakeFileXML(anEC,aICA.NameOri(aNameIm));
             }

             std::string aBlk = " ";
             std::string aVirg = ",";
             std::string aCom = MM3dBinFile_quotes("Campari")
                                + aBlk  + aNameIm 
                                + aBlk  + Dir11Param 
                                + aBlk  + Dir11Param + "Comp "
                                + " SH=NONE "
                                + std::string(" AffineFree=1 FocFree=1 PPFree=1 CPI2=1 GCP=[")
                                +  aICA.aNameFile3D + aVirg + ToString(1e-3) + aVirg
                                +  aICA.aNameFile2D + aVirg + ToString(1e3) + "]";

	     // std::cout << aCom << "\n"; getchar();
              System(aCom);
	      aVQual.push_back(aQual);
	      aVName.push_back(aNameIm);
                              
        }
        else
        {
             std::cout << "for " << itMAF->NameIm() << " only " << aICA.mVCPCur.size() << " measurements\n";
        }
    }
    
    if (! aVQual.empty())
    {
       std::cout  <<  " =============================================\n";
       std::cout  <<  " Qualities before compensation :\n\n";
       for (size_t aK=0; aK< aVQual.size() ; aK++)
       {
           std::cout  <<  aVName[aK] << " : ";
           aVQual[aK].Show();
           std::cout  <<  "\n";
       }
    }

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
