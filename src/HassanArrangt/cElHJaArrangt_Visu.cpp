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




cElHJaArrangt_Visu::cElHJaArrangt_Visu
(
    Pt2di aSzG,
    Pt2di aNbVisu,
    Pt2di aSzPl
)  :
   mSzG    (aSzG),
   mWG     (Video_Win::WStd(mSzG,1.0)),
   mWG2    (mWG,Video_Win::eBasG,mSzG,1),
   mWG3    (mWG2,Video_Win::eBasG,mSzG,1),
   mNbVisu (aNbVisu),
   mSzPl   (aSzPl)
{
   ELISE_COPY
   (
       mWG.all_pts(),
       255,
       mWG.ogray() | mWG2.ogray()
   );

   Video_Win * aWPrec = &mWG;
   Video_Win * aWColPrec = &mWG;
   mVallW.push_back(mWG);
   for (INT anX = 0 ; anX<mNbVisu.x ; anX++)
   {
       for (INT anY = 0 ; anY<mNbVisu.y ; anY++)
       {
            if(anY==0) 
              aWPrec = aWColPrec;
            mVWPl.push_back
            (
	        Video_Win
		(
                      *aWPrec,
                      (anY==0) ? Video_Win::eDroiteH : Video_Win::eBasG,
		      mSzPl,
		      2
		)
            );
	    mVallW.push_back(mVWPl.back());
	    aWPrec = &mVWPl.back();
	    ELISE_COPY
            (
	         aWPrec->all_pts(),
                 225,//100+anX*100+anY*20)%256,
		 aWPrec->ogray()
            );
            if(anY==0) 
              aWColPrec = aWPrec;
       }
   }
}

Video_Win *cElHJaArrangt_Visu::WinOfPl(INT aK)
{
    return ((aK>=0) && (aK<INT(mVWPl.size()))) ? & mVWPl[aK] : 0;
}

Video_Win cElHJaArrangt_Visu::WG () {return mWG;}
Video_Win cElHJaArrangt_Visu::WG2() {return mWG2;}
Video_Win cElHJaArrangt_Visu::WG3() {return mWG3;}

void cElHJaArrangt_Visu::GetPolyg
     (
          std::vector<Pt2dr> & aPolyg,
	  std::vector<int> &   aVSelect
     )
{
     aPolyg.clear();
     aVSelect.clear();

     bool first = true;
     Pt2dr aLastP;
     Video_Win aW = mVWPl[0];
     while(1)
     {
         Clik aClik = aW.disp().clik();
	 Pt2dr aP = aClik._pt;
	 bool  isSel = ! aClik.shifted();
	 bool  isEnd = (aClik._b==3) && (!first);


	 if (isEnd)
	 {
            aP = aLastP;
            aLastP = aPolyg[0];
	 }
	 if (!  first)
         {
	    aVSelect.push_back(isSel);
	    for (INT aK=0 ; aK<INT(mVallW.size()) ; aK++)
	    {
               Video_Win aW = mVallW[aK];
               aW.draw_seg(aP,aLastP,aW.pdisc()(isSel ? P8COL::black : P8COL::green));
	    }
	 }
	 if (isEnd)
            return;

	 for (INT aK=0 ; aK<INT(mVallW.size()) ; aK++)
	 {
             Video_Win aW = mVallW[aK];
             aW.draw_circle_loc(aP,2.0,aW.pdisc()(P8COL::green));
	 }
	 aPolyg.push_back(aP);


         aLastP = aP;
	 first = false;
     }
}


cElHJaFacette * cElHJaArrangt_Visu::GetFacette
                (const std::vector<cElHJaFacette *> & aVF)
{
    Video_Win aW = mVWPl[0];
    while (1)
    {
         Clik aClik = aW.disp().clik();
         Pt2dr aP = aClik._pt;
         aW = aClik._w;
         for (INT aK=0; aK<INT(aVF.size()) ; aK++)
         {
             cElHJaFacette * aF = aVF[aK];
             if (*(aF->Plan()->W()) == aW)
             {
                if(aF->PointInFacette(aP))
                  return aF;
             }
         }
    }
    return 0;
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
