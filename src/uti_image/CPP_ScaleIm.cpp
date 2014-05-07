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


int ScaleIm_main(int argc,char ** argv)
{
    std::string aNameIn;
    std::string aNameOut;
    std::string aNameType;

    double aScX,aScY=0;
    Pt2dr  aP0(0,0);
    Pt2dr  aSz(-1,-1);
    double aFactMult=1.0;
    double anOffset=0;

    double aDilate = 1.0;
    Pt2dr  aDilXY (-1,-1);

    int aDebug=0;
    int Tile = -1;

    bool aForceGray  = false;
    bool aForce8B  = false;


    ElInitArgMain
    (
    argc,argv,
                LArgMain()  << EAMC(aNameIn, "Image", eSAM_IsExistFile)
                << EAMC(aScX, "Scale"),
    LArgMain()  << EAM(aNameOut,"Out",true)
                    << EAM(aScY,"YScale",true)
                    << EAM(aSz,"Sz",true)
                    << EAM(aP0,"P0",true)
                    << EAM(aNameType,"Type",true, "Type", eSAM_None, ListOfVal(GenIm::bits1_msbf, ""))
                    << EAM(aFactMult,"Mult",true)
                    << EAM(aDilate,"Dilate",true)
                    << EAM(aDilXY,"DilXY",true)
                    << EAM(aDebug,"Debug",true)
                    << EAM(anOffset,"Offset",true)
                    << EAM(Tile,"Tile",true)
                    << EAM(aForceGray,"FG",true,"Force gray (Def=false)")
                    << EAM(aForce8B,"F8B",true,"Force 8 bits (Def=false)")
    );
    if (Tile<0)
       Tile = 1<<30;
     Tiff_Im::SetDefTileFile(Tile);

    if (aDilXY.x<0)
       aDilXY = Pt2dr(aDilate,aDilate);

    if (aScY==0)
        aScY= aScX;


    std::string aNameTif = NameFileStd(aNameIn,aForceGray ? 1 :-1,!aForce8B ,true,true);

    Tiff_Im tiff = Tiff_Im::StdConvGen(aNameIn.c_str(),aForceGray ? 1 :-1,!aForce8B ,true);

    aP0.SetSup(Pt2dr(0,0));
    Pt2di aSzG = tiff.sz();
    if (aDebug==1)
    {
        // ELISE_COPY(tiff.all_pts(),(FX/30)%2,tiff.out());
        // ELISE_COPY(tiff.all_pts(),1,tiff.out());
    }
    Pt2dr  aSzMax = Pt2dr(aSzG)-aP0;
    if (aSz== Pt2dr(-1,-1))
    {
        aSz = aSzMax;
    }
    aSz.SetInf(aSzMax);


    aSz.x = round_ni(aSz.x/aScX);
    aSz.y = round_ni(aSz.y/aScY);
    ELISE_ASSERT((aSz.x>0)&&(aSz.y>0),"Taille Insuffisante");
    // aP0.x = aScX;
    // aP0.y = aScY;


    if (aNameOut == "")
    {
       if (IsPostfixed(aNameIn))
          aNameOut = StdPrefix(aNameIn)+std::string("_Scaled.tif");
       else
          aNameOut = aNameIn+std::string("_Scaled.tif");
    }

    GenIm::type_el aType = tiff.type_el();
    if (aNameType!="")
       aType = type_im(aNameType);


    Tiff_Im TiffOut  =     (tiff.phot_interp() == Tiff_Im::RGBPalette)  ?
                           Tiff_Im
                           (
                              aNameOut.c_str(),
                              Pt2di(aSz),
                              aType,
                              Tiff_Im::No_Compr,
                              tiff.pal(),
                              ArgOpTiffMDP(aNameTif)
                          )                    :
                           Tiff_Im
                           (
                              aNameOut.c_str(),
                              Pt2di(aSz),
                              aType,
                              Tiff_Im::No_Compr,
                  tiff.phot_interp(),
                              ArgOpTiffMDP(aNameTif)
                          );

    std::cout << "P0 " << aP0 << " Sc " << aScX << " " << aScY << "\n";

    Fonc_Num aFIn = StdFoncChScale
                 (
                       //aDebug ? ((FX/30)%2) && tiff.in_proj() : tiff.in_proj(),
                       aDebug ? tiff.in(0) : tiff.in_proj(),
                       Pt2dr(aP0.x,aP0.y),
                       Pt2dr(aScX,aScY),
                       aDilXY
                 );
    aFIn = aFactMult * aFIn;
    aFIn = anOffset + aFIn;
    aFIn = Tronque(aType,aFIn);
    ELISE_COPY
    (
         TiffOut.all_pts(),
         aFIn,
     TiffOut.out()
    );

    return EXIT_SUCCESS;
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
