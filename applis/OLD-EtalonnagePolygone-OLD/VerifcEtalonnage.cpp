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


/*
     Erreurs possibles :
 
  Erreur de syntaxe sur fichier Param,
  Fichier Tiff inexistant
  Ni Fichiers PointInit existant( ni dans les ref pour les fichier non Primaire)
  Memes erreurs sur les fichier references

*/


#include "all_etal.h"
#include <algorithm>


/************************************************************/
/*                                                          */
/*                   cParamEtal                             */
/*                                                          */
/************************************************************/

void cEtalonnage::VerifTifFile(const std::string & aName,bool UseRat)
{
     std::string aTifName = NameTiffIm(aName);
     if (! ELISE_fp::exist_file(aTifName.c_str()))
     {
         cout << aName << "=> " << aTifName << " : Tiff File Does not exist\n";
	 exit(-1);
     }
     Tiff_Im::BasicConvStd(aTifName);

      std::string aNamePointe = NamePointeInit(aName);

     if (! ELISE_fp::exist_file(aNamePointe.c_str()))
     {
        if (UseRat)
        {
            if (mParam.HasFileRat())
	    {
                if (pEtalRattachement == 0)
		{
                   cout << "\n";
                   cout << "TEST RATACHE A " << mParam.NameFileRatt() << "\n";
                   cout << "\n";
		}	
		cEtalonnage & ERat = EtalRatt();
		ERat.VerifTifFile(aName,false);
		ERat.VerifTifFile(mParam.NamePosRatt(),false);
		VerifTifFile(mParam.NamePosRatt(),false);
	    }
	    else
	    {
                cout << aNamePointe << " File Point Does not exist \n";
		cout << "Pas de manip de rattachement \n";
	        exit(-1);
	    }
        }
	else
	{
            cout << aNamePointe << " File Point Does not exist \n";
	    exit(-1);
	}
     }
     else
     {
         cSetPointes1Im aSet(mPol,aNamePointe,false);
         if (INT(aSet.Pointes().size()) < 4)
         {
            cout << aNamePointe << " File Point Does , insuff \n";
         }
     }
}

void cEtalonnage::VerifTifFile(const std::vector<std::string> & VNames,bool UseRat)
{
   for 
   (
       std::vector<std::string>::const_iterator itV = VNames.begin();
       itV != VNames.end();
       itV++
   )
   {
       VerifTifFile(*itV,UseRat);
   }
}


void cEtalonnage::Verif (std::list<std::string> & Verified)
{
     VerifTifFile(mParam.ImagesInit(),false);
     VerifTifFile(mParam.AllImagesCibles(),true);

     if (mParam.HasFileRat())
     {
          Verif(mIsLastEtape,mParam.NameFileRatt(),Verified);
     }
}

void cEtalonnage::Verif
     (
            bool isLastEtape,
	    const std::string & aNameFile,
	    std::list<std::string> & Verified
     )
{
     if (
             std::find(Verified.begin(),Verified.end(),aNameFile)
          != Verified.end()
	)
        return;
     Verified.push_back(aNameFile);

     char * argv[2] ;
     argv[0] = 0 ;
     argv[1] = const_cast<char *>(aNameFile.c_str());
     cParamEtal   aParam(2,argv);
     cEtalonnage  anEt(isLastEtape,aParam);
     anEt.Verif(Verified);
}

void cEtalonnage::Verif(bool IsLatEtape,int argc,char ** argv)
{
    ELISE_ASSERT(argc>=2,"Not Enough Arg in cEtalonnage::Verif");
    std::list<std::string>  Verified;
    Verif(IsLatEtape,argv[1],Verified);
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
