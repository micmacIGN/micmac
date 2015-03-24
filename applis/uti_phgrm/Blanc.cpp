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
#include "XML_GEN/all.h"

using namespace NS_ParamChantierPhotogram;

/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/


void Banniere_Blanc()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     B-andes                   *\n";
   std::cout <<  " *     L-iaisons                 *\n";
   std::cout <<  " *     A-utomatiques             *\n";
   std::cout <<  " *     N-o                       *\n";
   std::cout <<  " *     C-omment                  *\n";
   std::cout <<  " *********************************\n";

}


/*********************************************/
/*                                           */
/*            cAppliBlanc                    */
/*                                           */
/*********************************************/

typedef enum
{
   eModePorto,
   eModePastis
} eModeBlanc;

class cAppliBlanc
{
    public :
       cAppliBlanc(int argc,char ** argv);
       void DoAll();

    private :

       void Sys(const std::string & aStr);
       void DoOne(const std::string &,const std::string &);

       std::string mDir;
       std::string mPat;
       int         mSz;
       int         mNbEcart;

       int mNbMaxMatch;
       int         mForce;
       std::vector<std::string> mNames;
       int                      mNb;
       int                      mSensCr;
       int                      mSensDecr;
       int                      mPurge;
       int                      mExe;
       eModeBlanc               mMode;
    
};


void cAppliBlanc::Sys(const std::string & aStr)
{
   std::cout << "DO :\n" << aStr << "\n";
   int aRes = system(aStr.c_str());
   if (aRes != 0)
   {
      std::cout  << "FAIL IN : \n";
      std::cout << aStr << "\n";
      exit(-1);
   }
}


cAppliBlanc::cAppliBlanc(int argc,char ** argv) :
   mNbMaxMatch       (16),
   mForce            (0),
   mSensCr           (1),
   mSensDecr         (0),
   mPurge            (1),
   mExe              (1)
{
    std::string aNameDir,aNameExe;
    SplitDirAndFile(aNameDir,aNameExe,argv[0]);

    if (aNameExe == "Blanc")
    {
       mMode =  eModePastis;
       ElInitArgMain
       (
           argc,argv,
           LArgMain() << EAM(mDir) 
                      << EAM(mPat) 
                      << EAM(mSz) 
                      << EAM(mNbEcart) ,
           LArgMain() << EAM(mNbMaxMatch,"NbMM",true)
                      << EAM(mForce,"Force",true)
                      << EAM(mExe,"Exe",true)
                      << EAM(mSensCr,"<",true)
                      << EAM(mSensDecr,">",true)
       );
    }
    else if (aNameExe == "Rouge")
    {
       mMode =  eModePorto;
       ElInitArgMain
       (
           argc,argv,
           LArgMain() << EAM(mDir) 
                      << EAM(mPat) 
                      << EAM(mNbEcart) ,
           LArgMain() << EAM(mForce,"Force",true)
                      << EAM(mSensCr,"<",true)
                      << EAM(mSensDecr,">",true)
                      << EAM(mExe,"Exe",true)
       );
    }
    else
    {
       ELISE_ASSERT(false,"Unknown Exe");
    }



    std::list<std::string> aLNames = RegexListFileMatch(mDir,mPat,1,false);


    mNames = std::vector<std::string>(aLNames.begin(),aLNames.end());
    mNb = mNames.size();

    std::sort(mNames.begin(),mNames.end());
}

void cAppliBlanc::DoAll()
{
    for (int aK1=0 ; aK1<mNb ; aK1++)
    {
        int aK2Deb = ElMax(0,aK1+ (mSensDecr ? -mNbEcart : 0)) ;
        int aK2Fin = ElMin(mNb-1,aK1+ (mSensCr ? mNbEcart : 0)) ;
        
        std::cout << mNames[aK1] << "\n";
	for (int aK2 = aK2Deb ; aK2 <= aK2Fin ; aK2++)
	{
	    if (aK1 != aK2)
	    {
	        // std::cout << "  - " <<  mNames[aK2] << "\n";
		DoOne(mNames[aK1],mNames[aK2]);
	    }
	}
    }
}


void cAppliBlanc::DoOne(const std::string & aN1,const std::string & aN2)
{
     std::string aCom;
     if (mMode==eModePastis)
     {
           aCom =     std::string("bin/Pastis ")
                          + mDir + std::string(" ")
			  + aN1  + std::string(" ")
			  + aN2  + std::string(" ")
			  + ToString(mSz) + std::string(" ");
     }

     if (mMode==eModePorto)
     {
           aCom =     std::string("bin/Porto ")
                          + mDir + std::string(" ")
			  + aN1  + std::string(" ")
			  + aN2  + std::string(" ")
			  + std::string("Purge=") +ToString(mPurge)+ std::string(" ")
			  + std::string("Force=") +ToString(0)+ std::string(" ")
	           ;
            // std::cout << aCom << "\n";
     }

     if (mExe)
        Sys(aCom);
     else
        std::cout << aCom << "\n";
}



   //===========================================

int main(int argc,char ** argv)
{
    system("make bin/Pastis");

    cAppliBlanc aAP(argc,argv);

    aAP.DoAll();

    Banniere_Blanc();

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
