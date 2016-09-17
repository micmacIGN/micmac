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

#include "OriTiepRed.h"

NS_OriTiePRed_BEGIN

/**********************************************************************/
/*                                                                    */
/*                         cLnk2ImTiepRed                             */
/*                                                                    */
/**********************************************************************/

cLnk2ImTiepRed::cLnk2ImTiepRed(cCameraTiepRed * aC1 ,cCameraTiepRed * aC2) :
    mCam1   (aC1),
    mCam2   (aC2),
    mCsRel1 (0),
    mCsRel2 (0)
{
    cAppliTiepRed & anAppli = aC1->Appli();
    if (anAppli.OrLevel()==eLevO_ByCple)
    {
         cVirtInterf_NewO_NameManager & aNM = anAppli.NM();
         cResVINM  aRV = aNM.ResVINM(mCam1->NameIm() , mCam2->NameIm());
         mCsRel1  = aRV.mCam1;
         mCsRel2  = aRV.mCam2;
         mHom     = new cElHomographie(aRV.mHom);
    }
}

bool cLnk2ImTiepRed::HasOriRel() const
{
   return (mCsRel1!=0) && (mCsRel2!=0);
}

CamStenope & cLnk2ImTiepRed::CsRel1()
{
   if (mCsRel1==0)
   {
       std::cout << "NAMME " << mCam1->NameIm() << " " << mCam2->NameIm() << "\n";
       ELISE_ASSERT(false,"cLnk2ImTiepRed::CsRel1");
   }
   return *mCsRel1;
}
CamStenope & cLnk2ImTiepRed::CsRel2()
{
   ELISE_ASSERT(mCsRel2!=0,"cLnk2ImTiepRed::CsRel1");
   return *mCsRel2;
}

cElHomographie & cLnk2ImTiepRed::Hom()
{
   ELISE_ASSERT(mHom!=0,"cLnk2ImTiepRed::CsRel1");
   return *mHom;
}




cCameraTiepRed &     cLnk2ImTiepRed::Cam1() {return *mCam1;}
cCameraTiepRed &     cLnk2ImTiepRed::Cam2() {return *mCam2;}
std::vector<Pt2df>&  cLnk2ImTiepRed::VP1()  {return mVP1;}
std::vector<Pt2df>&  cLnk2ImTiepRed::VP2()  {return mVP2;}
std::vector<Pt2df>&  cLnk2ImTiepRed::VPPrec1()  {return mVPPrec1;}
std::vector<Pt2df>&  cLnk2ImTiepRed::VPPrec2()  {return mVPPrec2;}


std::vector<Pt2df> & cLnk2ImTiepRed::VSelP1()
{
    return mVSelP1;
}
std::vector<Pt2df> & cLnk2ImTiepRed::VSelP2()
{
    return mVSelP2;
}
std::vector<U_INT1> & cLnk2ImTiepRed::VSelNb()
{
    return mVSelNb;
}


// Add all the tie points to the merging structur
void cLnk2ImTiepRed::Add2Merge(tMergeStr * aMergeStr)
{
    int aKCam1 =  mCam1->Num();
    int aKCam2 =  mCam2->Num();


    // Parse the point 
    for (int aKP=0 ; aKP<int(mVP1.size()) ; aKP++)
    {
         aMergeStr->AddArc(mVP1[aKP],aKCam1,mVP2[aKP],aKCam2,cCMT_U_INT1(ORR_MergeNew));
    }

    for (int aKP=0 ; aKP<int(mVPPrec1.size()) ; aKP++)
    {
         aMergeStr->AddArc(mVPPrec1[aKP],aKCam1,mVPPrec2[aKP],aKCam2,cCMT_U_INT1(ORR_MergePrec));
    }
}

NS_OriTiePRed_END



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
