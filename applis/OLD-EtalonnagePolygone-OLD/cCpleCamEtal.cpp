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

#include "all_etal.h"

/************************************************************/
/*                                                          */
/*                   cCpleCamEtal                           */
/*                                                          */
/************************************************************/



cCpleCamEtal::cCpleCamEtal
(
      bool isC2M,
      cSetEqFormelles & aSet,
      cCamIncEtalonage * aCam1,
      cCamIncEtalonage * aCam2
)  :
     pCam1 (aCam1),
     pCam2 (aCam2),
     pCpleRes1 (isC2M ? aSet.NewCpleCam(pCam1->CF(),pCam2->CF(),cNameSpaceEqF::eResiduIm1) : 0),
     pCpleRes2 (isC2M ? aSet.NewCpleCam(pCam1->CF(),pCam2->CF(),cNameSpaceEqF::eResiduIm2) : 0)
{
    cSetPointes1Im::tCont & l1 = aCam1->SetPointes().Pointes();
    cSetPointes1Im        &S2  = aCam2->SetPointes();

    for 
    (
       cSetPointes1Im::tCont::iterator it1=l1.begin();
       it1 != l1.end();
       it1++
    )
    {
        cPointeEtalonage * p2 = S2.PointeOfIdSvp(it1->Cible().Ind());
        if (p2)
        {
            mPack.Cple_Add(ElCplePtsHomologues(it1->PosIm(),p2->PosIm(),1.0));
        }
    }
}

REAL cCpleCamEtal::DCopt() const
{
   return euclid(pCam1->CF().CurRot().tr()-pCam2->CF().CurRot().tr());
}

void cCpleCamEtal::AddLiaisons(cEtalonnage & anEt)
{
     REAL aDCOpt = DCopt();
     if (aDCOpt > 0.01)
     {
        for
        (
             ElPackHomologue::const_iterator it=mPack.begin();
             it!=mPack.end();
             it++
       )
       {
            ELISE_ASSERT(pCpleRes1!=0,"Liaison en mode M2C");
            REAL aP = it->Pds();
            REAL Ecart = pCpleRes1->AddLiaisonP1P2(it->P1(),it->P2(),aP,false);
            anEt.AddErreur(ElAbs(Ecart)*anEt.FocAPriori(),aP);
            Ecart = pCpleRes2->AddLiaisonP1P2(it->P1(),it->P2(),aP,false);
            anEt.AddErreur(ElAbs(Ecart)*anEt.FocAPriori(),aP);
       }
    }
}


void cCpleCamEtal::Show()
{
   cout << "D COPT " 
	<< euclid(pCam1->CF().CurRot().tr()-pCam2->CF().CurRot().tr())
	<< "\n";
}

cCamIncEtalonage * cCpleCamEtal::Cam1() 
{
    return pCam1;
}
cCamIncEtalonage * cCpleCamEtal::Cam2() 
{
    return pCam2;
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
