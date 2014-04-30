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
#include <algorithm>



#define DEF_OFSET -12349876


int Impainting_main(int argc,char ** argv)
{
    std::string aNameIn;
    std::string aNameMasqOK;
    std::string aNameMasq2FIll;
    std::string aNameOut;


    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aNameIn,"Name of Input image")
                    << EAMC(aNameMasqOK,"Name Of Ok Masq,0 values means OK"),
	LArgMain()  << EAM(aNameOut,"Out",true,"Name of Result")
                    << EAM(aNameMasq2FIll,"2Fill","Masq of point 2 fill, def = all")
    );	

    Tiff_Im aFileIm = Tiff_Im::UnivConvStd(aNameIn.c_str());
    Pt2di aSzIm = aFileIm.sz();
    int aNBC = aFileIm.nb_chan();

    std::vector<Im2D_REAL4> aVIm;
    Output anOut = Output::onul(0);
    for (int aK=0 ; aK<aNBC ; aK++)
    {
         Im2D_REAL4  anIm(aSzIm.x,aSzIm.y);
         aVIm.push_back(anIm);
         anOut = (aK==0) ? anIm.out() : Virgule(anOut,anIm.out());
    }
    ELISE_COPY(aFileIm.all_pts(),aFileIm.in(),anOut);


    Tiff_Im aFileMasq(aNameMasqOK.c_str());
    Im2D_Bits<1> aMasq(aSzIm.x,aSzIm.y,1);
    ELISE_COPY(aFileMasq.all_pts(),!aFileMasq.in_bool(),aMasq.out());


    Im2D_Bits<1> aMasq2Fill(aSzIm.x,aSzIm.y,1);
    if (EAMIsInit(&aNameMasq2FIll))
    {
        Tiff_Im aFileMasq(aNameMasq2FIll.c_str());
        ELISE_COPY(aFileMasq.all_pts(),!aFileMasq.in_bool(),aMasq2Fill.out());
    }


    Fonc_Num aFRes=0;
    for (int aK=0 ; aK<aNBC ; aK++)
    {
         aVIm[aK] = ImpaintL2(aMasq,aMasq2Fill,aVIm[aK]);
         aFRes = (aK==0) ? aVIm[aK].in() : Virgule(aFRes,aVIm[aK].in());
    }

    if (!EAMIsInit(&aNameOut))
    {
       aNameOut = StdPrefix(aNameIn) + "_Impaint.tif";
    }
    Tiff_Im aTifOut
            (
                aNameOut.c_str(),
                aSzIm,
                aFileIm.type_el(),
                Tiff_Im::No_Compr,
                aFileIm.phot_interp()
            );

    ELISE_COPY(aTifOut.all_pts(),aFRes,aTifOut.out());

    return EXIT_SUCCESS;
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
