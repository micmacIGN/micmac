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

#include "general/all.h"
#include "private/all.h"

#include "XML_GEN/all.h"
#include "im_tpl/image.h"
#include "im_tpl/reduc_im.h"


using namespace NS_ParamChantierPhotogram;



void TestNew()
{
     int aNb = 1000000000;
     std::vector<char *> aVC;
     for (int aK=0 ; aK< 1000000 ; aK++)
     {
        char * aTb = new char [aNb];
        aVC.push_back(aTb);
        if (1)
        {
            for (int i=0 ; i<aNb ; i++)
                aTb[i] = 0;
        }
        // STD_NEW_TAB_USER(1000000000,char);
        std::cout << "K=" << aK << "\n";
     }
}


void TestTime()
{
   int aNB=100000000;

   ElTimer aChrono;
   int aResI=0;

   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResI = aK /1023;
   }
   std::cout << "Time  div I   " << aChrono.uval() << "\n";
   // std::cout << "Time  div I   " << aChrono.uval() << " " << aResI << "\n";
   aChrono.reinit();

   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResI = aK << 10 ; 
   }
   std::cout << "Time  shif I I   " <<  aChrono.uval() << " " << aResI << "\n";
   aChrono.reinit();

   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResI = aK * 1025 ; 
   }
   std::cout << "Time  Mul I I   " <<  aChrono.uval() << " " << aResI << "\n";
   aChrono.reinit();


   double aResR;
   for  (int aK=1 ; aK<aNB ; aK++)
   {
      aResR = aK /1023.0;
   }
   std::cout << "Time  div R   " << aChrono.uval() << " " << aResR <<  "\n";
   aChrono.reinit();


}


int main (int argc,char** argv)
{

   TestTime();
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
