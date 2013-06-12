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
#include <cstring>


// P,p   Pattern  (P est conservee pas p)
// N Niv
// M  Makefile
// K  
// A ->  Ne Rajoute pas la directory dans le C=
// C  Change
// T Target
// c  Change+Target:

int main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aPatDir;
   int aNiv=1;

   std::string aMKF;
   cEl_GPAO * aGPAO=0;

   bool NameCompl=false;
   bool AjouteDir = true;  // Compatibilite

// std::cout << "ARGC " << argc << "\n";

   for (int aK=1 ; aK< argc ; aK++)
   {
      const char * aN = argv[aK];
      // std::cout << aN << "\n";
      if ((strlen(aN) > 2) && (aN[1]=='='))
      {
        if ((aN[0]=='P') || (aN[0]=='p'))
        {
           ELISE_ASSERT(aPatDir=="","Multiple pattern");
           aPatDir = std::string(aN+2);
        }
        if (aN[0]=='N')
        {
	   sscanf(aN+2,"%d",&aNiv);
        }
        if (aN[0]=='M')
        {
            ELISE_ASSERT(aGPAO==0,"Multiple Makefile");
            aGPAO = new cEl_GPAO;
            aMKF=std::string(aN+2);
        }

        if (aN[0]=='K')
        {
            NameCompl = (aN[2]!='0');
        }
        if (aN[0]=='A')
        {
            AjouteDir = (aN[2]!='0');
        }
      }
   }

   ELISE_ASSERT(aPatDir!="","No pattern");

   std::string aPatIn,aDirIn;

   SplitDirAndFile(aDirIn,aPatIn,aPatDir);

   cElRegex anAuto(aPatIn,10);

   std::list<std::string>  aLNameI = RegexListFileMatch(aDirIn,aPatIn,aNiv,NameCompl);


   for 
   ( 
       std::list<std::string>::iterator itS=aLNameI.begin();
       itS!=aLNameI.end();
       itS++
   )
   {

// std::cout << "ITSSSS " << *itS << "\n";


       std::string aName = *itS;
       if ((! NameCompl) && (AjouteDir))
         aName = aDirIn+aName;
       std::string aCom;
       std::string aTarget;
       for (int aK=1 ; aK< argc ; aK++)
       {
          if (aK > 1)
             aCom +=  std::string(" ");
          const char * aN = argv[aK];
          if ((strlen(aN) > 2) && (aN[1]=='='))
          {
            if (aN[0]=='P')
            {
                aCom += aName;
            }
            if ((aN[0]=='C' ) || (aN[0]=='T') || (aN[0]=='c' ))
            {
	        std::string aDirCur,aNameCur;
		SplitDirAndFile(aDirCur,aNameCur,aName);

                if (! AjouteDir) aDirCur="";


                std::string aNp2 (aN+2);
                std::string aPrefEq="";

                if (aNp2.find('=')!=std::string::npos)
                {
                      std::string aNewNp2;
                      SplitIn2ArroundEq(aNp2,aPrefEq,aNewNp2);
                      aNp2 = aNewNp2;
                      aPrefEq=aPrefEq+"=";
                }
// std::cout << "xxxxxxxxxx"<< aNameCur << " " << aNp2 << "\n";

		std::string aRepl = MatchAndReplace(anAuto,aNameCur,aNp2); // std::string(aN+2));

                if ((aN[0]=='T') || (aN[0]=='c'))
                {
                    ELISE_ASSERT(aTarget=="","Multiple cible");
                   aTarget = aDirCur+aRepl;
                }
                if ((aN[0]=='C') || (aN[0]=='c'))
                {
		   aCom += aPrefEq+aDirCur+aRepl;
                }
            }
          }
	  else
	  {
	     aCom += std::string(aN);
	  }
       }
       std::cout << "COM=[" << aCom << "]\n";
       if (aGPAO)
       {
          ELISE_ASSERT(aTarget!="","No  target in Makefile mode");
          aGPAO->GetOrCreate(aTarget,aCom);
          aGPAO->TaskOfName("all").AddDep(aTarget);
          //aGPAO->TaskOfName("all")
       }
       else
       {
          ElTimer aChrono;
          VoidSystem(aCom.c_str());
          std::cout << "Time = " << aChrono.uval() << "\n";
       }
   }

   if (aGPAO)
      aGPAO->GenerateMakeFile(aMKF);

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
