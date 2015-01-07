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
// #include "anag_all.h"

#include "CmpMNT.h"

namespace NS_CmpMNT
{

void cOneMnt::EcartZ(Im2D_REAL4 aRes,cOneMnt & aM2) 
{
    ELISE_COPY
    (
        mIm.all_pts(),
	mIm.in()-aM2.mIm.in(),
	aRes.out()
    );
}


void cOneMnt::CorrelPente
     (
        Im2D_REAL4               aRes,
	cOneMnt &                aM2,
	const cCorrelPente &     aCP
     )
{
   Symb_FNum sGrA(Grad());
   Symb_FNum sGrB(aM2.Grad());

   Symb_FNum sGrAX (sGrA.v0());
   Symb_FNum sGrAY (sGrA.v1());
   Symb_FNum sGrBX (sGrB.v0());
   Symb_FNum sGrBY (sGrB.v1());

   Fonc_Num aF = Virgule
                 (
		     Virgule
		     (
		         1,
		         sGrAX,
		         sGrAY,
		         sGrBX,
		         sGrBY
		     ) ,
		     Square(sGrAX)+Square(sGrAY),
		     Square(sGrBX)+Square(sGrBY),
		     sGrAX*sGrBX+sGrAY*sGrBY
		 );

   Symb_FNum sAllSom = rect_som(aF,aCP.SzWCP());

   Symb_FNum s1 = sAllSom.kth_proj(0);
   Symb_FNum sAx =  sAllSom.kth_proj(1) / s1;
   Symb_FNum sAy =  sAllSom.kth_proj(2) / s1;
   Symb_FNum sBx =  sAllSom.kth_proj(3) / s1;
   Symb_FNum sBy =  sAllSom.kth_proj(4) / s1;

   Symb_FNum sAA =  sAllSom.kth_proj(5) / s1 -(Square(sAx)+Square(sAy));
   Symb_FNum sBB =  sAllSom.kth_proj(6) / s1 -(Square(sBx)+Square(sBy));

   Symb_FNum sAB = sAllSom.kth_proj(7)/s1 -(sAx*sBx+sAy*sBy);

   Symb_FNum sCor = sAB / sqrt(sAA*sBB+ElSquare(aCP.GrMinCP()));

   ELISE_COPY(mIm.all_pts(),sCor,aRes.out());
}



};



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
