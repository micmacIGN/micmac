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

#include "NewOri.h"


cNewO_NameManager::cNewO_NameManager
(
     const std::string  & aDir,
     const std::string  & anOriCal,
     const std::string  & aPostHom
) :
    mICNM      (cInterfChantierNameManipulateur::BasicAlloc(aDir)),
    mDir       (aDir),
    mOriCal    (anOriCal),
    mPostHom   (aPostHom)
{
   StdCorrecNameOrient(mOriCal,mDir);
}

const std::string & cNewO_NameManager::Dir() const
{
   return mDir;
}


ElPackHomologue cNewO_NameManager::PackOfName(const std::string & aN1,const std::string & aN2) const
{
    std::string aNameH = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@@"+mPostHom,aN1,aN2,true);
    if (!  ELISE_fp::exist_file(aNameH))
       return ElPackHomologue();

    return ElPackHomologue::FromFile(aNameH);
}


CamStenope * cNewO_NameManager::CamOfName(const std::string  & aName) 
{

   cMetaDataPhoto aMTD = cMetaDataPhoto::CreateExiv2(mDir+aName);

   if (mOriCal =="")
   {
        std::vector<double> aPAF;
        double aFPix  = aMTD.FocPix();
        Pt2di  aSzIm  = aMTD.TifSzIm();
        Pt2dr  aPP = Pt2dr(aSzIm) / 2.0;

        bool IsFE;
        FromString(IsFE,mICNM->Assoc1To1("NKS-IsFishEye",aName,true));
        CamStenope * aRes = 0;

        if (IsFE)
        {
            std::vector<double> aVP;
            std::vector<double> aVE;
            aVE.push_back(aFPix);
            aVP.push_back(aPP.x);
            aVP.push_back(aPP.y);
            aRes = new cCamLin_FishEye_10_5_5
                       (
                            false,
                            aFPix,aPP,Pt2dr(aSzIm),
                            aPAF,
                            &aVP,
                            &aVE
                       );

        }
        else
        {
             aRes = new CamStenopeIdeale(false,aFPix,aPP,aPAF);
        }
        aRes->SetSz(aSzIm);
        return aRes;
   }


   std::string  aNC = mICNM->StdNameCalib(mOriCal,aName);


   return CamOrientGenFromFile(aNC,mICNM);
}
/*
*/
CamStenope *  cInterfChantierNameManipulateur::StdCamOfNames(const std::string & aNameIm,const std::string & anOri)
{

     std::string aKey = "NKS-Assoc-Im2Orient@-"+ anOri ;
     std::string aNameCam =  Assoc1To1(aKey,aNameIm,true);
     return CamOrientGenFromFile(aNameCam,this);
}

CamStenope * cNewO_NameManager::CamOriOfName(const std::string & aNameIm,const std::string & anOri)
{
    return mICNM->StdCamOfNames(aNameIm,anOri);
/*
     std::string aKey = "NKS-Assoc-Im2Orient@-"+ anOri ;
     std::string aNameCam =  mICNM->Assoc1To1(aKey,aNameIm,true);
     return CamOrientGenFromFile(aNameCam,mICNM);
*/
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
