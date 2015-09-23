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


/*
    Petit utilitaire pour creer un masque d'eau dans le cadre de la corelation d'images
    QuickBird.  Suppose evidemment que l'image soit cooperative, l'eau est caracterise
    par une plage de radiometrie + homogeneite.
*/


#include "general/all.h"
#include "private/all.h"
#include <algorithm>



#define DEF_OFSET -12349876


int main(int argc,char ** argv)
{
    std::string aNameIn;
    std::string aNameOut;

    int   aSzEct = 3;
    int   aValOut = 0;

    int aVMax;
    int aFiltrFin; 
    int   anEcMax;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameIn)
                    << EAM(aVMax)
		    << EAM(anEcMax)
		    <<  EAM(aFiltrFin),

	LArgMain()  << EAM(aNameOut,"Out",true)
	            << EAM(aValOut,"ValOut",true)
	            << EAM(aSzEct,"SzEct",true)
    );	


    Tiff_Im aTifIn = Tiff_Im::StdConv(aNameIn.c_str());
    Pt2di aSz = aTifIn.sz();
    if (aNameOut == "")
    {
       if (IsPostfixed(aNameIn)) 
          aNameOut = StdPrefix(aNameIn)+std::string("_Masq.tif");
       else
          aNameOut = aNameIn+std::string("_Masq.tif");
    }


     L_Arg_Opt_Tiff aLArgTiff = Tiff_Im::Empty_ARG;
     aLArgTiff =  aLArgTiff+ Arg_Tiff(Tiff_Im::AFileTiling(Pt2di(-1,-1)));



    Tiff_Im aTifOut (
                              aNameOut.c_str(),
                              aSz,
                              // GenIm::u_ist1,
                              GenIm::bits1_msbf,
                              Tiff_Im::No_Compr,
                              // Tiff_Im::Group_4FAX_Compr,
                              Tiff_Im::BlackIsZero,
			      aLArgTiff
                    );

     Fonc_Num aFoncMasque = aTifIn.in_proj() > aVMax;

    double aNbV = ElSquare(1+2*aSzEct);

    Fonc_Num  Ect =  rect_som(Square(aTifIn.in_proj()),aSzEct)/aNbV
                    -Square(rect_som(aTifIn.in_proj(),aSzEct)/aNbV);

     aFoncMasque = aFoncMasque || (Ect > Square(anEcMax));
     
     if (aValOut >= 0) 
         aFoncMasque = aFoncMasque && (aTifIn.in_proj() != aValOut);

     aFoncMasque  =  rect_som(aFoncMasque,aFiltrFin) > (ElSquare(1+2*aFiltrFin)/2);


     ELISE_COPY
     (
          aTifOut.all_pts(),
	  aFoncMasque,
          aTifOut.out() | Video_Win::WiewAv(aSz)
     );

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
