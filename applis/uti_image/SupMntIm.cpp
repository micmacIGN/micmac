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

    Tiff_Im::SetDefTileFile(1000000);

    std::string aNameIm;
    std::string aNameMnt;
    std::string aNameOut;

    REAL DynCoul = 1.0;
    REAL DynGray = 1.0;
    REAL OffsetGray = 0.0;
    REAL GamaGray = 1.0;
    INT NoVal = -(1<<15);
    double Sat = 0.5;
    int ShowSat = 0;
    int Grad = 0;
    double  CDN=0;

    Pt3dr aCoulCDN(0,0,0);

    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aNameIm)
                    << EAM(aNameMnt),
	LArgMain()  << EAM(aNameOut,"Out",true)
                    << EAM(DynGray,"DynGray",true)
                    << EAM(OffsetGray,"OffsetGray",true)
                    << EAM(GamaGray,"GamaGray",true)
	            << EAM(DynCoul,"DynCoul",true)
                    << EAM(NoVal,"NoVal",true)
                    << EAM(Sat,"Sat",true)
                    << EAM(ShowSat,"ShowSat",true)
                    << EAM(Grad,"Grad",true)
                    << EAM(CDN,"CDN",true)
                    << EAM(aCoulCDN,"CoulCDN",true)
    );	

    if (aNameOut == "")
       aNameOut = StdPrefix(aNameIm) + "Superp.tif";

    Tiff_Im TifIm  = Tiff_Im::StdConv(aNameIm.c_str());
    Tiff_Im Mnt = Tiff_Im::StdConv(aNameMnt.c_str());

    Tiff_Im TiffOut  = Tiff_Im 
                       (
                              aNameOut.c_str(),
                              TifIm.sz(),
                              GenIm::u_int1,
                              Tiff_Im::No_Compr,
                              Tiff_Im::RGB
		       );

   Fonc_Num aFin = 0;

   for (int aK=0 ; aK<TifIm.nb_chan() ; aK++)
       aFin = aFin + TifIm.in_proj().kth_proj(aK);
    aFin = aFin / TifIm.nb_chan();
    // Rconv(TifIm.in_proj().kth_proj(0));

   // Fonc_Num aFin =0;
    if (Grad)
    {
        int aNbV= 1;
        for (int aK=0 ; aK<3 ; aK++)
            aFin = rect_som(aFin,aNbV) /ElSquare(1.0+2*aNbV);
        aFin =  128 + Laplacien(aFin);
    }
    Symb_FNum fGr (aFin);
    if ((DynGray != 1.0) || (OffsetGray!=0.0))
       fGr =  OffsetGray + fGr*DynGray;
    if (GamaGray != 1.0)
       fGr = 255.0 * pow(fGr/255.0,1/GamaGray);

    Fonc_Num aRes =  its_to_rgb( Virgule
                      (
                            fGr,
                            Mnt.in() * DynCoul,
                            Sat * 255 * ( aFin != NoVal)
                       ));
    
    if (CDN > 0)
    {
       Fonc_Num aCDN = cdn(Mnt.in_proj() / CDN);
       aRes = aRes * (!aCDN)  +  Fonc_Num(aCoulCDN.x,aCoulCDN.y,aCoulCDN.z) * aCDN; // Virgule(!cdn(Mnt.in_proj()),1,1);
    }
    if (ShowSat)
       aRes =  its_to_rgb(Virgule
                (
                     fGr,
                     0,
                     Mnt.in() * DynCoul
                ));

    ELISE_COPY
    (
         TifIm.all_pts(),
         aRes,
         TiffOut.out() | Video_Win::WiewAv(TifIm.sz())
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
