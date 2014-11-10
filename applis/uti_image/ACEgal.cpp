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



int main(int argc,char ** argv)
{

	for (REAL r = -2; r<=2 ; r+=0.2)
		cout << "Erf(" << r << ")=" << erfcc(r) << "\n";

    std::string aNameIn;
    std::string aNameOut;
    REAL  PdsGlob= 0.8;
    INT   NbIter = 4;
    INT   NbVois = 20;


    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameIn) ,
	LArgMain()  << EAM(aNameOut,"Out",true)
    );	


    if (aNameOut == "")
       aNameOut = StdPrefix(aNameIn)+std::string("_8Bits.tif");

    Tiff_Im tiff = Tiff_Im::StdConv(aNameIn.c_str());
    Pt2di aSz = tiff.sz();
    INT NbH = 256*256;
    Im1D_REAL8 Histo(NbH,0.0);

    ELISE_COPY
    (
        tiff.all_pts(),
	1,
	Histo.histo().chc(tiff.in())
    );
    REAL8 * DH = Histo.data();
    for(INT aK = 1 ;aK<NbH ; aK++)
        DH[aK] += DH[aK-1];
 
    ELISE_COPY(Histo.all_pts(),Histo.in() * 255.0 /(aSz.x*aSz.y),Histo.out());

    cout << "END HISTO \n";

    Tiff_Im TiffOut  = Tiff_Im 
                       (
                              aNameOut.c_str(),
                              aSz,
                              GenIm::u_int1,
                              Tiff_Im::No_Compr,
                              Tiff_Im::BlackIsZero
                       );


     Fonc_Num  fRes = 0;  
          
     Symb_FNum  Fonc (tiff.in(0));
     Symb_FNum  Pond (tiff.inside());
          
     Fonc_Num fSom = Virgule(Rconv(Pond),Fonc,ElSquare(Fonc));
     for (INT k=0; k< NbIter ; k++)
          fSom = rect_som(fSom,NbVois)/ElSquare(1.0+2.0*NbVois);  // Pour Eviter les divergences
     Symb_FNum  S012 (fSom);
          
     Symb_FNum s0 (Rconv(S012.v0()));
     Symb_FNum s1 (S012.v1()/s0);
     Symb_FNum s2 (S012.v2()/s0-Square(s1));
     Symb_FNum ect  (sqrt(Max(5,s2)));
     fRes = 255*erfcc( (2/PI)*(tiff.in()-s1)/ect);


      ELISE_COPY
      (
         tiff.all_pts(),
         Histo.in()[tiff.in()] * PdsGlob + (1-PdsGlob) * fRes,
         TiffOut.out() | Video_Win::WiewAv(tiff.sz())
      );


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
