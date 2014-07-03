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
         }
   }
   return true;
}

//==============================================


int Init11Param_Main(int argc,char ** argv)
{
    bool        isFraserModel = false;
    LArgMain ArgOpt;
    ArgOpt <<  EAM(isFraserModel,"FM",true,"Fraser Mode, use all affine parmeters (def=false)");

    cInitCamAppuis aICA(argc,argv,ArgOpt,"-11Param");



    for
    (
        std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF = aICA.mSMAF.MesureAppuiFlottant1Im().begin();
        itMAF != aICA.mSMAF.MesureAppuiFlottant1Im().end();
        itMAF++
    )
    {
        if (aICA.InitPts(*itMAF) && (int(aICA.mVCPCur.size())>=6))
        {
             std::string aNameIm = itMAF->NameIm();
             std::cout << "Init11Param :" << aNameIm << "\n";
             cEq12Parametre anEq12;
             Pt3dr aPMoy(0,0,0);
             for (int aK=0 ; aK<int(aICA.mVCPCur.size()) ; aK++)
             {
                 anEq12.AddObs(aICA.mVCPCur[aK],aICA.mVImCur[aK],1.0);
                 aPMoy = aPMoy+aICA.mVCPCur[aK];
             }
             aPMoy = aPMoy/double(aICA.mVCPCur.size());
             std::pair<ElMatrix<double>,ElRotation3D > aPair = anEq12.ComputeOrtho();
             ElMatrix<double> aMat = aPair.first;
             ElRotation3D aR = aPair.second;

             double aFX =  aMat(0,0);
             double aFY =  aMat(1,1);
             Pt2dr aPP(aMat(2,0),aMat(2,1));
             double aSkew =  aMat(1,0);

             Pt3dr aCenter =  aR.ImAff(Pt3dr(0,0,0));
             double Alti = aPMoy.z;
             double Prof = euclid(aPMoy-aCenter);
/*
             std::cout << "FX=" <<  aFX  << " FY=" << aFY << " PP=" << aPP << " Skew=" << aSkew << "\n";
             std::cout << "C=" <<  aR.ImAff(Pt3dr(0,0,0))  << "  "   << aR.ImRecAff(Pt3dr(0,0,0)) << "\n";
             std::cout << "VVVV " <<  anEq12.ComputeNonOrtho().second << "\n";
*/

             cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aNameIm);

             Pt2di aSz = aMDP.SzImTifOrXif();
             Pt2dr aRSz = Pt2dr(aSz);

             ElDistRadiale_PolynImpair aDR((1.1*euclid(aRSz))/2.0,aPP);

             CamStenope * aCS=0;
             std::vector<double> aPAF;
             if (isFraserModel)
             {
                 cDistModStdPhpgr aDPhg(aDR);
                 aDPhg.b1() = (aFX-aFY)/ aFY;
                 aDPhg.b2() = aSkew / aFY;
                 aCS = new cCamStenopeModStdPhpgr(true,aFY,aPP,aDPhg,aPAF);
             }
             else
             {
                  aCS = new cCamStenopeDistRadPol(true,(aFX+aFY)/2.0,aPP,aDR,aPAF);
             }

             if (aCS)
             {
                 aCS->SetOrientation(aR.inv());
                 cOrientationConique anEC = aCS->ExportCalibGlob(aSz,Alti,Prof ,false,true,(char*)0);
                 MakeFileXML(anEC,aICA.NameOri(aNameIm));
             }
        }
    }

/*
    std::string aNameFile3D;
    std::string aNameFile2D;
    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(aNameFile3D,"Name File for GCP",eSAM_IsExistFile)
                   << EAMC(aNameFile3D,"Name File for GCP",eSAM_IsExistFile),
       LArgMain()
                    << EAM(isFraserModel,"FM",true,"Fraser Mode, use all affine parmeters (def=false)")
    );
*/


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
