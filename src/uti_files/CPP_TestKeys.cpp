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

int Recover_Main(int argc, char ** argv)
{
    std::string aNameIn,aNameOut;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIn, "Name of Input File")
                    << EAMC(aNameOut, "Name of Input File"),
        LArgMain()  
    );

    FILE * aFPin = FopenNN(aNameIn,"r","In recover");
    FILE * aFPOut = FopenNN(aNameOut,"w","In recover");

    int aC;
    while (  (aC= fgetc(aFPin)) != EOF)
    {
        if (  
                  ( (aC>='a') && (aC<='z'))
               || ( (aC>='A') && (aC<='Z'))
               || ( (aC>='0') && (aC<='9'))
               || (aC=='\n')
           )
           fputc(aC,aFPOut);
    }
    fclose(aFPin);
    fclose(aFPOut);

    return EXIT_SUCCESS;
}


/*
Parametre de Tapas :

   - calibration In : en base de donnees ou deja existantes.


*/

// bin/Tapioca MulScale "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" 300 -1 ExpTxt=1
// bin/Tapioca All  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1  ExpTxt=1
// bin/Tapioca Line  "../micmac_data/ExempleDoc/Boudha/IMG_[0-9]{4}.tif" -1   3 ExpTxt=1
// bin/Tapioca File  "../micmac_data/ExempleDoc/Boudha/MesCouples.xml" -1  ExpTxt=1

#define DEF_OFSET -12349876

#define  NbModele 10

int TestNameCalib_main(int argc,char ** argv)
{
    std::string aFullNameIm,aNameCalib="TestNameCalib";
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullNameIm, "Name of images", eSAM_IsPatFile),
        LArgMain()  << EAM(aNameCalib,"Nb",true,"Name of caib (def=TestNameCalib)")
    );

    std::string aDir,aNameIm;

    SplitDirAndFile(aDir,aNameIm,aFullNameIm);
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    std::cout << aICNM->StdNameCalib(aNameCalib,aNameIm) << "\n";

    return EXIT_SUCCESS;
}

int TestSet_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv,2);

    std::string  aDir,aPat,aFullDir,aKeyAssoc,aFileCpl;
    int  aNbMax=10;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full Directory (Dir+Pattern)", eSAM_IsPatFile),
        LArgMain()  << EAM(aNbMax,"Nb",true,"Nb Max printed (def=10)")
                    << EAM(aKeyAssoc,"KeyAssoc",true,"Key for association")
                    << EAM(aFileCpl,"NameCple",true,"Name of XML file to save couples determined with keyAssoc")    
    );

    if (MMVisualMode) return EXIT_SUCCESS;

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
         printf("Num=%3d ",aK);
         std::cout << " Name=" << aName ;
         if (EAMIsInit(&aKeyAssoc))
         {
             //std::cout <<  " Key=" << aICNM->Assoc1To1(aKeyAssoc,aName,true) ;

             std::vector<std::string> aInput;
             aInput.push_back(aName);
             cInterfChantierNameManipulateur::tNuplet aRes= aICNM->Direct(aKeyAssoc,aInput);
             std::cout <<  " Key=";
             for (auto& a:aRes)
                 std::cout<<a<<" ";
             //std::cout<<"\n";
         }
         std::cout  << "\n";
    }

    if (1)
    {
         // std::list<std::string>  aL = RegexListFileMatch(aDir,aPat,1,false);
          std::cout << "NB  BY RFLM " << mSetIm->size() << "\n";
    }
    
    if (EAMIsInit(&aFileCpl) && EAMIsInit(&aKeyAssoc))
    {
		 std::cout << "Export couples to file  " << aFileCpl << "\n";
		 cSauvegardeNamedRel aVCpl;
		 for (auto & im1 : *mSetIm){
			 
			    cCpleString aCpl(im1,aICNM->Assoc1To1(aKeyAssoc,im1,true));
                aVCpl.Cple().push_back(aCpl);
                cCpleString aCpl2(aICNM->Assoc1To1(aKeyAssoc,im1,true),im1);
                aVCpl.Cple().push_back(aCpl2);
		 }
		 MakeFileXML(aVCpl,aFileCpl);
	}


    return EXIT_SUCCESS;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
