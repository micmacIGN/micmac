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


Im2D_REAL4 ImCercle()
{
   Im2D_REAL4 aI(256,256);

   ELISE_COPY
   (
       disc(Pt2di(128,128),80),
       255,
       aI.out()
   );

   return aI;
}

Im2D_REAL4 ImLena()
{
   Im2D_REAL4 aI(256,256);

   ELISE_COPY
   (
       aI.all_pts(),
       Tiff_Im("../TMP/lena.tif").in(),
       aI.out()
   );

   return aI;
}



Im2D_REAL4 Convol(Im2D_REAL4 aI)
{
    Im2D_REAL4 aRes(aI.tx(),aI.ty());

    Fonc_Num aF = aI.in(0);
    for (int aK = 0 ; aK< 5 ; aK++)
        aF = rect_som(aF,2) / 25.0;

    ELISE_COPY(aRes.all_pts(),aF,aRes.out());

    return aRes;
}

void Show(Im2D_REAL4 aIm,Fonc_Num aF, std::string aName)
{

    aName =  std::string("../TMP/") + aName + ".tif";

    L_Arg_Opt_Tiff aL = Tiff_Im::Empty_ARG;
    aL = aL + Arg_Tiff(Tiff_Im::ANoStrip());
    Tiff_Im aRes
            (
               aName.c_str(),
               aIm.sz(),
               GenIm::u_int1,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero,
               aL
            );
    ELISE_COPY
    (
       aIm.all_pts(),
       Max(0,Min(255,aF)),
       aRes.out()
         
    );
}


int main(int argc,char ** argv)
{
   Im2D_REAL4 aI = ImCercle();
   Im2D_REAL4 aGauss = Convol(ImCercle());
   Im2D_REAL4 aLena = ImLena();

   Show(aI,aI.in(),"Cercle");
   Show(aI,aGauss.in(),"Gauss-Cercle");

   Show(aI,128+(aI.in()-trans(aI.in(0),Pt2di(1,0)))/2,"GX-Cercle");
   Show(aI,128+(aGauss.in()-trans(aGauss.in(0),Pt2di(0,1)))*3,"GY-Gauss");


   Im2D_REAL8 aLapl(3,3,
                       "0 -1 0 "
                       "-1 4 -1 "
                       " 0 -1 0"
                  );

  Show(aI,128+(som_masq(aGauss.in(0),aLapl))*5,"GY-Lapl");



  Show(aI,(aLena.in(0)*4)%256,"Lena-Deb");
  Show(aI,128+(som_masq(aLena.in(0),aLapl))*5,"Lena-Lapl");
  Show(aI,mod(128+(som_masq(aLena.in(0),aLapl))*5,256),"Lena-Lapl-Deb");

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
