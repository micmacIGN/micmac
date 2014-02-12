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


using namespace NS_SaisiePts;


/*************************************************/
/*                                               */
/*                XXXXXXX                        */
/*                                               */
/*************************************************/

const Pt2dr cWinIm::PtsEchec (-100000,-10000);

Pt2dr cWinIm::FindPoint(const Pt2dr & aPIm,eTypePts aType,double aSz,cPointGlob * aPG)
{
     Tiff_Im aTF = mCurIm->Tif();
     Pt2di aSzT = aTF.sz();

     int aRab = 5 + round_up(aSz);
     if ((aPIm.x <aRab) || (aPIm.y <aRab) || (aPIm.x >aSzT.x-aRab)|| (aPIm.y >aSzT.y-aRab))
         return PtsEchec;


     Pt2di aMil  = mAppli.SzRech() / 2;
     Im2D_INT4 aImA = mAppli.ImRechAlgo();
     mAppli.DecRech() = round_ni(aPIm) - aMil;
     Pt2di aDec = mAppli.DecRech();
     ELISE_COPY
     (
         aImA.all_pts(),
         mCurIm->FilterImage(trans(aTF.in_proj(),aDec),aType,aPG),
         aImA.out()
     );
     ELISE_COPY
     (
         aImA.all_pts(),
         trans(aTF.in_proj(),aDec),
         //  mCurIm->FilterImage(trans(aTF.in_proj(),aDec),aType),
         mAppli.ImRechVisu().out()
     );


     if (aType==eNSM_Pts)
     {
        return aPIm;
     }



     Pt2dr aPosImInit = aPIm-Pt2dr(aDec);



     bool aModeExtre = (aType == eNSM_MaxLoc) ||  (aType == eNSM_MinLoc) || (aType==eNSM_GeoCube);
     bool aModeMax = (aType == eNSM_MaxLoc) ||  (aType==eNSM_GeoCube);


     if (aModeExtre)
     {
          aPosImInit = Pt2dr(MaxLocEntier(aImA,round_ni(aPosImInit),aModeMax,2.1));
          aPosImInit = MaxLocBicub(aImA,aPosImInit,aModeMax);


          return aPosImInit+ Pt2dr(aDec);
     }


     return aPIm;
}


void  cWinIm::CreatePoint(const Pt2dr & aPW,eTypePts aType,double aSz)
{
    Pt2dr aPGlob = FindPoint(mScr->to_user(aPW),aType,aSz,0);

    if (aPGlob==PtsEchec) return;

    mAppli.Interface()->DrawZoom(aPGlob);

   //TODO:
    cCaseNamePoint * aCNP = mAppli.Interface()->GetIndexNamePoint();

    if (aCNP && aCNP->mFree && (aCNP->mTCP != eCaseCancel))

        mCurIm->CreatePGFromPointeMono(aPGlob,aType,aSz,aCNP);

    else

        mAppli.Interface()->MenuNamePoint()->W().lower();
}

void cX11_Interface::DrawZoom(const Pt2dr & aPGlob)
{
     double aZoom = 10.0;

     Pt2dr aPIm = aPGlob- Pt2dr(mAppli->DecRech());
     Pt2dr aPMil = Pt2dr(mWZ->sz())/(2.0*aZoom);

     Video_Win aWC = mWZ->chc(aPIm-aPMil,Pt2dr(aZoom,aZoom));
     ELISE_COPY
     (
                aWC.all_pts(),
                mAppli->ImRechVisu().in(0),
                aWC.ogray()
     );

     aWC.draw_circle_abs(aPIm,4.0,Line_St(aWC.pdisc()(P8COL::blue),3.0));
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
