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



#define DEF_OFSET -12349876


int main(int argc,char ** argv)
{
    std::string aName ("/home/pierrot/Data/AutoCalAmiens/Im16_217/amiensquadri.16_217.THM");
    Elise_Tiled_File_Im_2D  aTF = Elise_Tiled_File_Im_2D::Thom(aName.c_str());

}
/*
{

    std::string aNameIn;
    std::string aNameOut;
    REAL EcMin = 0.0;


    INT Brd = -1;
    INT NbIter =-1;
    REAL Dyn = 1.0;

    INT Offset = DEF_OFSET;
    INT Circ = 0;

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameIn) ,
	LArgMain()  << EAM(EcMin,"EcMin",true)
                    << EAM(aNameOut,"Out",true)
                    << EAM(Brd,"Brd",true)
                    << EAM(NbIter,"NbIter",true)
                    << EAM(Dyn,"Dyn",true)
                    << EAM(Offset,"Offset",true)
                    << EAM(Circ,"Circ",true)
    );	

    Tiff_Im tiff = Tiff_Im::StdConv(aNameIn.c_str());
 

    GenIm::type_el aType = tiff.type_el();


    if (Offset == DEF_OFSET)
    {
       Offset = (signed_type_num(aType) ?  128 : 0);
    }


    if (aNameOut == "")
        aNameOut = StdPrefix(aNameIn)+std::string("_8Bits.tif");


    Tiff_Im TiffOut  = 
                         Circ                ?
                           Tiff_Im 
                           (
                              aNameOut.c_str(),
                              tiff.sz(),
                              GenIm::u_int1,
                              Tiff_Im::No_Compr,
                              Disc_Pal::PCirc(256)
                          )                    :
                           Tiff_Im 
                           (
                              aNameOut.c_str(),
                              tiff.sz(),
                              GenIm::u_int1,
                              Tiff_Im::No_Compr,
                              Tiff_Im::BlackIsZero
                          );

   Symb_FNum  FoncInit(tiff.in(0));
   Fonc_Num  fRes = 0;


    if ((Brd > 0) && (NbIter >0))
    {
        Fonc_Num Masq =  (FoncInit!= -(1<<15)) ;

        Symb_FNum  Fonc (FoncInit * Masq);
        Symb_FNum  Pond (tiff.inside()*Masq);

        Fonc_Num fSom = Virgule(Rconv(Pond),Fonc,ElSquare(Fonc));
        for (INT k=0; k< NbIter ; k++)
            fSom = rect_som(fSom,Brd)/ElSquare(1.0+2.0*Brd);  // Pour Eviter les divergences
        Symb_FNum  S012 (fSom);

        Symb_FNum s0 (Rconv(S012.v0()));
        Symb_FNum s1 (S012.v1()/s0);
        Symb_FNum s2 (S012.v2()/s0-Square(s1) + EcMin);
        Symb_FNum ect  (sqrt(Max(0.01,s2)));
        fRes = 255*erfcc((tiff.in()-s1)/ect);
    }
    else
    {
       fRes = FoncInit * Dyn;
       if (Offset != 0)
           fRes = fRes + Offset;
    }
    

    if (Circ)
       fRes = mod(fRes,256);
    else
       fRes = Min(255,Max(0,fRes));

    ELISE_COPY
    (
         tiff.all_pts(),
         fRes,
         TiffOut.out() | Video_Win::WiewAv(tiff.sz())
    );


    return 0;
}



*/


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
