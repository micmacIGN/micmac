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




/*************************************************/
/*                                               */
/*                XXXXXXX                        */
/*                                               */
/*************************************************/

cSP_PointeImage::cSP_PointeImage
(
   cOneSaisie *      aSIm,
   cImage *          anIm,
   cSP_PointGlob  *  aPGl
) :
  mSIm      (aSIm),
  mIm       (anIm),
  mGl       (aPGl),
  mVisible  (true)
{
}

cOneSaisie *    cSP_PointeImage::Saisie() {return mSIm;}
cImage *        cSP_PointeImage::Image()  {return mIm;}
cSP_PointGlob * cSP_PointeImage::Gl()     {return mGl;}


bool & cSP_PointeImage::Visible() {return mVisible;}

bool cSP_PointeImage::BuildEpipolarLine(Pt2dr &pt1, Pt2dr &pt2)
{
    if (mGl)
    {
        eEtatPointeImage aState = mSIm->Etat();

        cBasicGeomCap3D * aCap3D = mIm->Capt3d();

        if (aCap3D && mGl->PG()->PS1().IsInit() && ((aState==eEPI_NonSaisi) || (aState==eEPI_Refute)))
        {
            pt1 = aCap3D->Ter2Capteur(mGl->PG()->PS1().Val());
            pt2 = aCap3D->Ter2Capteur(mGl->PG()->PS2().Val());

            return true;
        }
    }
    return false;
}

bool cSP_PointeImage::BuildEpipolarLine(std::vector<Pt2dr> & aResVPt,std::vector<bool> & aV3DInCap)
{
    if (mGl)
    {
        eEtatPointeImage aState = mSIm->Etat();

        cBasicGeomCap3D * aCap3D = mIm->Capt3d();

        if (aCap3D && mGl->PG()->PS1().IsInit() && ((aState==eEPI_NonSaisi) || (aState==eEPI_Refute)))
        {
            aResVPt.clear();
            aV3DInCap.clear();
            std::vector<Pt3dr> aVPtIn = mGl->PG()->VPS();
            if (aVPtIn.empty())
            {
                aVPtIn.push_back(mGl->PG()->PS1().Val());
                aVPtIn.push_back(mGl->PG()->PS2().Val());
            }
            for (int aK=0 ; aK<int(aVPtIn.size()) ; aK++)
            {
                aV3DInCap.push_back(aCap3D->PIsVisibleInImage(aVPtIn[aK]));
                aResVPt.push_back(aCap3D->Ter2Capteur(aVPtIn[aK]));
            }

            return true;
        }
    }
    return false;
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
