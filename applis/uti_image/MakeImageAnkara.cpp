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

std::string aName1 = "/bdisque/Data/HRS_Epir/AllSeg/TmpCoxRoy/Id00";
std::string aName2 = "ZPx_51C_H1_EPI_REF.HDR51C_H1_EPI_SEC.HDRReduc_";
std::string aName3 = ".tif";

int aReduc0 = 32;
double aDyn1 = 1.0;
double aOffset = 128;

int ReducOfId(int anId) {return ElMax(1,aReduc0/(1<<anId));}

std::string NameFile(int anId)
{
   return   aName1
          + ToString(anId)
          + aName2
          + ToString(ReducOfId(anId))
          + aName3;
}


void MakeImage(int anId,Pt2di aP0_Z1,Pt2di aSz_Z1)
{
    int aZ = ReducOfId(anId);
    Pt2di aP0 = aP0_Z1 / aZ;
    Pt2di aSz = aSz_Z1 / aZ;


    Tiff_Im aFileIn = Tiff_Im::StdConv(NameFile(anId));

    Im2D_U_INT1 aI0(aSz.x,aSz.y);
    ELISE_COPY
    (
        aI0.all_pts(),
        Max(0,Min(255,trans(aFileIn.in(),aP0)*aZ*aDyn1+aOffset)),
        aI0.out()
    );
    std::string aNameOut = 
                            "../TMP/MNT" 
                         +  ToString(anId) 
                         +  std::string("_R") 
                         +  ToString(aZ) 
                         +  ".tif";

    Tiff_Im::Create8BFromFonc
    (
        aNameOut,
        aSz_Z1,
        aI0.in_proj()[Virgule(FX/aZ,FY/aZ)]
    );
}


Pt2di aP0 (1650*8,900*8);  // [13200,7200]

Pt2di aSz(800,800);

int main(int argc,char ** argv)
{
    cout << aP0 << "\n";

    for (int anId = 1 ; anId<9 ; anId++)
        MakeImage(anId,aP0,aSz);
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
