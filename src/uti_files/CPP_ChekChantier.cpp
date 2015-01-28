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

/*
*/

#define DEF_OFSET -12349876

#define  NbModele 10


/********************************************************/
/*                                                      */
/*                    GESTION DES ERREURS               */
/*                                                      */
/********************************************************/

void MakeFileJunk(const std::string & aName)
{
     std::cout << "ERREUR SUR " << aName << "\n";
     std::string JUNK_PREF = "MMJunk";
     std::string aDest = DirOfFile(aName) + JUNK_PREF + "-" + NameWithoutDir(aName) + "." + JUNK_PREF;
     ELISE_fp::MvFile ( aName, aDest);
}

class cTetstFileErrorHandler : public cElErrorHandlor
{
     public :
         cTetstFileErrorHandler(const std::string & aName) :
             mName (aName)
         {
             if (! ELISE_fp::exist_file(aName))
             {
                  std::cout << "Warn  " << aName << " does not exist \n";
                  exit(EXIT_SUCCESS);
             }
         }

         void OnError()
         {
              MakeFileJunk(mName);
              exit(EXIT_SUCCESS);
         }

         std::string mName;
};

void InitJunkErrorHandler(const std::string & aName)
{
    TheCurElErrorHandlor = new cTetstFileErrorHandler(aName);
}

std::string InitJunkErrorHandler(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);
    if (argc <  2) 
    {
        std::cout << "Warn not enough arg \n";
        exit(EXIT_SUCCESS);
    }
    std::string aName = argv[1];
    TheCurElErrorHandlor = new cTetstFileErrorHandler(aName);

    return aName;
}

void CheckSetFile(const std::string & aDir,const std::string & aKey,const std::string & aKeyCom)
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aVName = aICNM->Get(aKey);

    std::list<std::string> aLCom;
    for (int aK=0; aK< int (aVName->size()) ; aK++)
    {
        std::string aCom =  MM3dBinFile(" TestLib " + aKeyCom ) + aDir+(*aVName)[aK];
        aLCom.push_back(aCom);
    }

     cEl_GPAO::DoComInParal(aLCom);
}


/********************************************************/
/*                                                      */
/*     Check Tiff                                       */
/*                                                      */
/********************************************************/

int CheckOneTiff_main(int argc,char ** argv)
{
   std::string  aName = InitJunkErrorHandler(argc,argv);
   Tiff_Im aTF(aName.c_str());
   ELISE_COPY(aTF.all_pts(),aTF.in(),Output::onul());
   return EXIT_SUCCESS;
}

int CheckAllTiff_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);
   
    std::string aDir;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory "),
        LArgMain()  
    );

    CheckSetFile(aDir,"NKS-Set-TmpTifFile","Check1Tiff");

    return EXIT_SUCCESS;
}

/********************************************************/
/*                                                      */
/*     Check Hom                                        */
/*                                                      */
/********************************************************/

int CheckOneHom_main(int argc,char ** argv)
{
   std::string  aName = InitJunkErrorHandler(argc,argv);
   ElPackHomologue::FromFile(aName);
   return EXIT_SUCCESS;
}

int CheckAllHom_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);
   
    std::string aDir;
    std::string aExt ="";
    std::string aPost = "dat";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory "),
        LArgMain()  << EAM(aExt,"Extension ",true,"Like _SRes, Def=\"\"")
                    << EAM(aPost,"Extension ",true,"Post , Def = dat")
    );

    CheckSetFile(aDir,"NKS-Set-Homol@"+aExt+ "@"+aPost,"Check1Hom");

    return EXIT_SUCCESS;
}




/********************************************************/
/*                                                      */
/*                    GESTION DES ERREURS               */
/*                                                      */
/********************************************************/




/*
int CheckOneTiffFile_main(int argc,char ** argv)
{
}
*/


int Check_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv,2);

    std::string  aDir,aPat,aFullDir;
    int  aNbMax=10;

    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile),
    LArgMain()  << EAM(aNbMax,"Nb",true,"Nb Max printed (def=10)")
    );


#if (ELISE_windows)
    replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
#endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const cInterfChantierNameManipulateur::tSet * mSetIm = aICNM->Get(aPat);


    int aNb = ElMin(aNbMax,int(mSetIm->size()));

    for (int aK=0 ; aK< aNb ; aK++)
    {
         std::string aName = (*mSetIm)[aK];
         printf("%3d ",aK);
         std::cout << aName ;
         std::cout  << "\n";
    }

    if (1)
    {
         // std::list<std::string>  aL = RegexListFileMatch(aDir,aPat,1,false);
          std::cout << "NB  BY RFLM " << mSetIm->size() << "\n";
    }



    return 1;
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
