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
#include "algorithm"

using namespace NS_ParamChantierPhotogram;


// Pour recopier une arborescence d'ensemble de fichier en les remplacant par des fichiers vides
/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/

struct  cMov
{
    std::string mNameIn;
    std::string mNameOut;

    cMov(const std::string & aNameIn,const std::string & aNameOut) :
        mNameIn   (aNameIn),
        mNameOut  (aNameOut)
    {
    }

    bool operator < (const cMov & aM2) const
    {
       return mNameOut < aM2.mNameOut;
    }
};


class cAppliCp
{
    public :
       cAppliCp(int argc,char ** argv);

       void OneTest();

    private :
       std::string mDir;
       std::string mDirOut;
       std::string mPat;
       std::string mRepl;
       std::string mFile2M;
       int         mExe;
       int         mNiv;
       int         mForce;
       int         mForceDup;
};

cAppliCp::cAppliCp(int argc,char ** argv)  :
    mExe     (0) ,
    mNiv      (1),
    mForce    (0),
    mForceDup (0)
{
    std::string aDP;
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(aDP) 
	              << EAM(mDirOut),
           LArgMain() << EAM(mExe,"Exe",true)
                      << EAM(mNiv,"Niv",true)
                      << EAM(mForce,"F",true)
                      << EAM(mForceDup,"FD",true)
                      << EAM(mFile2M,"File2M",true)
    );
    SplitDirAndFile(mDir,mPat,aDP);

    std::list<std::string> aLIn = RegexListFileMatch(mDir,mPat,mNiv,false);




    for (int aTime=0; aTime<2 ;aTime++)
    {
       bool OverW=false;
       for
       (
           std::list<std::string>::const_iterator itS=aLIn.begin();
           itS!=aLIn.end();
	   itS++
       )
       {
           std::string aSubD;
           std::string aName;
           SplitDirAndFile(aSubD,aName,*itS);
           std::string  aNameOut = mDirOut+ aName;


           if (aTime==0)
           {
              if (ELISE_fp::exist_file(aNameOut))
              {
                 OverW=true;
                 std::cout << "Over write " << aNameOut<< "\n";
              }
           }
           
           if (aTime==1)
           {
              std::cout  << "CREATE " << aNameOut << "\n";
              if (mExe && (! ELISE_fp::exist_file(aNameOut)))
              {
                 FILE * aFP = ElFopen(aNameOut.c_str(),"w");
                 ElFclose(aFP);
              }
           }
        }

         if (OverW)
         {
             ELISE_ASSERT(mForce,"Use force to overwrite existing file");
         }
    }

    if (! mExe)
       std::cout << "Use Exe=1 to really create files \n";
}


   //===========================================

int main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    cAppliCp  aRename(argc,argv);


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
