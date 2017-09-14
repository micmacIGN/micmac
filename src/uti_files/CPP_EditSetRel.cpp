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

void MYFromString(std::vector<std::string> & aRes,const std::string & aStr)
{
     stringstream aStream(aStr);
     ElStdRead(aStream,aRes,ElGramArgMain::StdGram);
}


std::vector<std::string>  Str2VecStr(const std::string & aStr)
{
   std::vector<std::string> aRes;
   if (aStr[0]=='[')
   {
       MYFromString(aRes,aStr);
   }
   else
   {
      aRes.push_back(aStr);
   }
   return aRes;
}

std::vector<std::string>  PatOrFile2VecStr(cInterfChantierNameManipulateur * anICNM,const std::string & aStr)
{
   if (IsPostfixed(aStr) && (StdPostfix(aStr)=="xml") && ELISE_fp::exist_file(aStr))
   {
       cListOfName aLON = StdGetFromPCP(aStr,ListOfName);
       return std::vector<std::string>(aLON.Name().begin(),aLON.Name().end());
   }

   return *(anICNM->Get(aStr));
}


std::vector<std::string>  VecPatOrFile2VecStr(cInterfChantierNameManipulateur * anICNM,const std::string & aStr)
{
    std::vector<std::string> aVStr = Str2VecStr(aStr);
    std::vector<std::string> aRes;

    for (int aK=0 ; aK<int(aVStr.size()) ; aK++)
    {
        std::vector<std::string> OneSet = PatOrFile2VecStr(anICNM,aVStr[aK]);
        std::copy(OneSet.begin(),OneSet.end(),back_inserter(aRes));
    }

    return aRes;
}





class cAppliEditSet
{
    public :
       cAppliEditSet(int argc,char**argv);

    private :
        std::string                       mDir;
        cInterfChantierNameManipulateur * mICNM;
        std::string                       mSetIn;
        std::string                       mSetOut;
        std::string                       mAdd;
        std::string                       mSuppr;
        std::string                       mInter;
        std::set<std::string>             mSet;
};


cAppliEditSet::cAppliEditSet(int argc,char**argv)
{
    ElInitArgMain
    (
        argc,argv,
        LArgMain() <<  EAMC(mSetIn,"Input set", eSAM_IsPatFile)
                   <<  EAMC(mSetOut,"Ouput set"),
        LArgMain() <<   EAM(mAdd,"Add",true,"Files to add")
                   <<   EAM(mSuppr,"Supr",true,"Files to supress")
                   <<   EAM(mInter,"Inter",true,"Files to intersect")
    );

    mDir = DirOfFile(mSetIn);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
     
    if (ELISE_fp::exist_file(mSetIn))
    {
       cListOfName aLON = StdGetFromPCP(mSetIn,ListOfName);
       for (std::list<std::string>::const_iterator itL=aLON.Name().begin(); itL!=aLON.Name().end()  ; itL++)
       {
           mSet.insert(*itL);
       }
    }


    if (EAMIsInit(&mAdd))
    {
         std::vector<std::string>  aV2Add =  VecPatOrFile2VecStr(mICNM,mAdd);
         mSet.insert(aV2Add.begin(),aV2Add.end());
    }
    if (EAMIsInit(&mSuppr))
    {
       std::vector<std::string>  aV2Supr =  VecPatOrFile2VecStr(mICNM,mSuppr);
       for (int aK=0 ; aK<int(aV2Supr.size()) ; aK++)
       {
           mSet.erase(aV2Supr[aK]);
       }
    }

    if (EAMIsInit(&mInter))
    {
       std::set<std::string>  aNewSet;
       std::vector<std::string>  aV2Inter =  VecPatOrFile2VecStr(mICNM,mInter);
       for (int aK=0 ; aK<int(aV2Inter.size()) ; aK++)
       {
           if (DicBoolFind(mSet,aV2Inter[aK]))
              aNewSet.insert(aV2Inter[aK]);
       }
       mSet = aNewSet;
    }

    cListOfName aRes;
    std::copy(mSet.begin(),mSet.end(),back_inserter(aRes.Name()));


    MakeFileXML(aRes,mSetOut);
    
}

int CPP_EditSet(int argc,char**argv)
{
   cAppliEditSet anAppli(argc,argv);

   return EXIT_SUCCESS;
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
