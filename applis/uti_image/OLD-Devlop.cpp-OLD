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

    MicMa cis an open source software specialized in image matching
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
    std::string aFullDir;
    std::string aPostAdd="None";
    std::string aNameOut="";
    int  En8B = 1;
    int  EnGray = 1;
    int  ConsCol = 0;
    std::string aSplit = "";


    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(aFullDir),
	LArgMain()  << EAM(En8B,"8B",true)	
                    << EAM(EnGray,"Gray",true)	
                    << EAM(aPostAdd,"Post",true)	
                    << EAM(ConsCol,"ConsCol",true)	
                    << EAM(aNameOut,"NameOut",true)	
                    << EAM(aSplit,"Split",true)	
    );

    std::string aDir,aPatFile;
    SplitDirAndFile(aDir,aPatFile,aFullDir);

    std::string aPost = StdPostfix(aPatFile);
    std::string aPref = StdPrefix(aPatFile);


    std::string aCom = MMBin() + "MapCmd ";
    std::string aPat= QUOTE("P=" + aDir+ "("+aPref+")." + aPost);

    // if (aNameOut =="") aNameOut = "\\$1" + (aPostAdd=="None"?"":aPostAdd)  + ".tif" ;

    std::string aSubst=  "\\$1" + (aPostAdd=="None"?"":aPostAdd)  + ".tif";

    {
       aCom = aCom + MMBin() +"MpDcraw " + aPat + " Add16B8B=0 ";
       if (aSplit!="")
       {
             aCom = aCom + " " + QUOTE("Split="+aSplit);
       }
       else
       {
          aCom = aCom +  (EnGray ? " GB=1 " : " CB=1 ");
       }
       aCom = aCom +  " 16B=" + (En8B ? "0 " : "1 ") ;
       aCom = aCom +  " ExtensionAbs="  + aPostAdd ;
       aCom = aCom +  " ConsCol=" + ToString(ConsCol) ;
       if (aNameOut != "")
       {
          aCom = aCom +  QUOTE(" NameOut=" + aNameOut);
          aCom = aCom +  " " + QUOTE("T=" + aNameOut);
       }
       else
       {
          aCom = aCom +  " " + QUOTE("T="  +  aSubst)  ;
       }
    }

    aCom = aCom+ " M=MkDevlop";
    // std::cout << aCom << "\n"; getchar();
     System(aCom);

     aCom = "make all -f MkDevlop  -j" + ToString(MMNbProc());
     System(aCom);


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
