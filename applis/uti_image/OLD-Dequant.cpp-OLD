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
#include "general/all.h"
#include "private/all.h"
#include <algorithm>

Im2DGen AllocImGen(Pt2di aSz,const std::string & aName)
{
    return D2alloc_im2d(type_im(aName),aSz.x,aSz.y);
}



int main(int argc,char ** argv)
{
     std::string aNameIn;
     std::string aNameOut;

     Pt2di aP0Glob(0,0),aSzGlob(0,0);
     INT aSzMaxDalles = 3000;
     INT aSzRecDalles = 300;

     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAM(aNameIn) ,
           LArgMain() << EAM(aNameOut,"Out",true)
                      << EAM(aP0Glob,"P0",true)
                      << EAM(aSzGlob,"Sz",true)
		      << EAM(aSzMaxDalles,"SzMaxDalles",true)
		      << EAM(aSzRecDalles,"SzRecDalles",true)
    );	


    if (aNameOut=="")
       aNameOut = StdPrefix(aNameIn) +std::string("Deq.tif");

     Tiff_Im aFileIn(aNameIn.c_str());
     if (aSzGlob== Pt2di(0,0))
        aSzGlob = aFileIn.sz();

    Tiff_Im  aTifOut
             (
                    aNameOut.c_str(),
                    aSzGlob,
                    GenIm::real4,
	            Tiff_Im::No_Compr,
	            Tiff_Im::BlackIsZero
             );

     Pt2di aPRD(aSzRecDalles,aSzRecDalles);
     cDecoupageInterv2D aDecoup
	                (
                            Box2di(aP0Glob,aP0Glob+aSzGlob),
			    Pt2di(aSzMaxDalles,aSzMaxDalles),
			    Box2di(-aPRD,aPRD)
			);

     ElImplemDequantifier aDeq(aDecoup.SzMaxIn());
     for (int aKDec=0; aKDec<aDecoup.NbInterv() ; aKDec++)
     {

         Box2di aBoxIn = aDecoup.KthIntervIn(aKDec);
	 Pt2di aSzIn = aBoxIn.sz();
	 Pt2di aP0In = aBoxIn.P0();


         aDeq.SetTraitSpecialCuv(true);
         aDeq.DoDequantif(aSzIn, trans(aFileIn.in(),aP0In),1);


         Fonc_Num aFoncRes = aDeq.ImDeqReelle();

         Box2di aBoxOut = aDecoup.KthIntervOut(aKDec);
         ELISE_COPY
         (
              rectangle(aBoxOut.P0()-aP0Glob,aBoxOut.P1()-aP0Glob),
	      trans(aFoncRes,aP0Glob-aP0In),
	      aTifOut.out()
         );
     }
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
