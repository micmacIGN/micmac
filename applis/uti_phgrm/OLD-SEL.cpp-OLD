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


void Banniere_SEL()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     S-aisie d'                *\n";
   std::cout <<  " *     E-lements de              *\n";
   std::cout <<  " *     Liaison                   *\n";
   std::cout <<  " *********************************\n";

}

void Sys(const std::string & aStr)
{
   VoidSystem(aStr.c_str());
}

int main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);


    Pt2di aSzW(1000,900);
    if (! ELISE_fp::exist_file(MMDir() + "bin/MICMACSaisieLiaisons"))
       VoidSystem("make -f MakeMICMAC  bin/MICMACSaisieLiaisons");

    std::string aDir;
    std::string aN1;
    std::string aN2;
    std::string aKeyH;

    int aRedr=0;
    std::string aFilter="";

    std::string aKeyCompl="Cple2HomAp";
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(aDir) 
                       << EAM(aN1) 
                      << EAM(aN2) ,
           LArgMain() << EAM(aRedr,"R",true)
                      << EAM(aFilter,"F",true)
                      << EAM(aKeyH,"KH",true)
                      << EAM(aKeyCompl,"KCpl",true)
                      << EAM(aSzW,"SzW",true)
    );

    std::string aCom =    MMDir()
                       +  std::string("bin/MICMACSaisieLiaisons ")
                       + MMDir()+std::string("applis/XML-Pattron/Pattron-MicMacLiaison.xml ")
                       + " WorkDir=" + aDir
                       + " \%Im1=" + aN1
                       + " \%Im2=" + aN2
                       + " \%SL_XSzW=" + ToString(aSzW.x)
                       + " \%SL_YSzW=" + ToString(aSzW.y)
                     ;

    if (aRedr)
       aCom = aCom + " SL_NewRedrCur=true";

   if (aFilter!="")
       aCom = aCom
              /* + " SL_TJS_FILTER=true" */
	      +  " SL_FILTER=" +aFilter;

   if (aKeyH!="")
   {
       if (aKeyH=="P")
       {
          aKeyCompl = "PastisHom";
       }
       else if (aKeyH=="PB")
       {
          aKeyCompl = "Key-Assoc-CpleIm2HomolPastisBin";
       }
       else if (aKeyH=="PBR")
       {
          aKeyCompl = "Key-Assoc-SsRes-CpleIm2HomolPastisBin";
       }
       else if (aKeyH=="M")
       {
          aKeyCompl = "MarcHom";
       }
       else if (aKeyH=="S")
       {
          // aKeyCompl = "Key-Assoc-StdHom";
          aKeyCompl = "NKS-Assoc-CplIm2Hom@-Man@xml";
       }
       else if (aKeyH=="NB")
       {
          aKeyCompl = "NKS-Assoc-CplIm2Hom@@dat";
       }
       else if (aKeyH=="NT")
       {
          aKeyCompl = "NKS-Assoc-CplIm2Hom@@txt";
       }
       else
       {
           std::cout << "For Key=[" << aKeyH << "]\n";
           ELISE_ASSERT(false,"Do Not know key");
       }
   }

   aCom = aCom + " FCND_CalcHomFromI1I2=" + aKeyCompl;


    std::cout << aCom << "\n";
    Sys(aCom);
    Banniere_SEL();

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
