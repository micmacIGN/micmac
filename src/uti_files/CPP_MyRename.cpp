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


// bin/MyRename "/media/MYPASSPORT/Archi/ArcTriomphe/(F[0-9]{3}_IMG_[0-9]{4})_MpDcraw16B_GR.tif"   "\$1\$2_MpDcraw16B_GR.tif"  File2M="\$1(_.*)_MpDcraw8B_GR.tif"

/*

mm3d MyRename "(IMG_010.*)" "F\$2_\$1" AddFoc=1

mm3d MyRename "(IMG_010.*)" "F\$2_\$1" AddFoc=1 PatSub="(.*)"



*/
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


class cAppliMyRename
{
    public :
       cAppliMyRename(int argc,char ** argv);

       void OneTest();

    private :
       std::string mDir;
       std::string mPat;
       std::string mRepl;
       std::string mFile2M;
       std::string mPatSubst;
       int         mExe;
       int         mNiv;
       int         mForce;
       int         mForceDup;
       int         mAddF;
       bool        mFull;
};

cAppliMyRename::cAppliMyRename(int argc,char ** argv)  :
    mExe     (0) ,
    mNiv      (1),
    mForce    (0),
    mForceDup (0),
    mAddF     (0),
    mFull     (false)
{

    std::string aDP;
    ElInitArgMain
    (
           argc,argv,
                LArgMain() << EAMC(aDP,"Full name: Dir + images", eSAM_IsPatFile)
                << EAMC(mRepl, "Directory", eSAM_IsDir),
           LArgMain() << EAM(mExe,"Exe",true)
                      << EAM(mNiv,"Niv",true)
                      << EAM(mForce,"F",true)
                      << EAM(mForceDup,"FD",true)
                      << EAM(mAddF,"AddFoc",true)
                      << EAM(mFile2M,"File2M",true)
                      << EAM(mFull,"Full",true)
                      << EAM(mPatSubst,"PatSub","Can be diff from Pattern when use key")
    );
    SplitDirAndFile(mDir,mPat,aDP);

/*
    std::list<std::string> aLIn = RegexListFileMatch(mDir,mPat,mNiv,mFull);
*/
   cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   const cInterfChantierNameManipulateur::tSet * aVecIm = aICNM->Get(mPat);
   std::list<std::string> aLIn (aVecIm->begin(),aVecIm->end());


   if (EAMIsInit(&mPatSubst))
      mPat = mPatSubst;

    std::vector<cMov> aVM;
    if (mAddF)
       mPat = mPat + "@(.*)";


    cElRegex *  anF2Autom=0;
    if (mFile2M!="")
    {
        ELISE_ASSERT(!mAddF,"AddF and File2M incomptabible");
        anF2Autom= new cElRegex(mPat,10);
        // mPat = mPat + "@" + mFile2M;
    }

    cElRegex * anAutom = new cElRegex(mPat,10);

    bool anOverW=false;
    for
    (
        std::list<std::string>::const_iterator itS=aLIn.begin();
        itS!=aLIn.end();
    itS++
    )
    {
        std::string aName=*itS;
        if (mAddF)
        {
             cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(mDir+aName);
             std::string aF = ToString(round_ni(aMDP.FocMm()));
             while (aF.size() < 3) aF="0"+aF;
             aName = aName + "@" + aF;
        }
        if (mFile2M!="")
        {
            std::string aPat2 = MatchAndReplace(*anF2Autom,*itS,mFile2M);
            std::list<std::string>  aL2 = RegexListFileMatch(mDir,aPat2,mNiv,false);
            if (aL2.size()!=1)
            {
                std::cout << "NB Got = " << aL2.size() << "\n";
                ELISE_ASSERT(false,"mFile2M Must Got 1 File");

            }
            aName = aName + "@" + *(aL2.begin());
            anAutom = new cElRegex(mPat+"@"+aPat2,10);
        // mPat = mPat + "@" + mFile2M;

        }

    std::string aNOut = MatchAndReplace(*anAutom,aName,mRepl);
    if (!mForce)
    {
        if (ELISE_fp::exist_file(mDir+aNOut))
        {
                std::cout << *itS << " -> " << aNOut << "\n";
            std::cout << "FILE [" <<mDir+aNOut<< "]Already exist\n";
        std::cout << "Use F=1 to overwrite\n\n";
        anOverW = true;
        }
    }
    aVM.push_back(cMov(*itS,aNOut));
    }
    ELISE_ASSERT(!anOverW,"Cannot overwrite !! ");

    std::sort(aVM.begin(),aVM.end());

    if (!mForceDup)
    {
        bool aGotDup = false;

        for (int aK=1 ; aK <int(aVM.size()) ; aK++)
        {
            if (aVM[aK-1].mNameOut == aVM[aK].mNameOut)
            {
                std::cout << aVM[aK-1].mNameIn << "->"  << aVM[aK-1].mNameOut << "\n";
                std::cout << aVM[aK].mNameIn << "->"  << aVM[aK].mNameOut << "\n";
            std::cout << "Found pontential duplicata in renaming\n";
        std::cout << "Use FD=1 to pass over\n\n";

        aGotDup = true;
            }
        }
        ELISE_ASSERT(!aGotDup,"Cannot force duplicata !! ");
    }

    for (int aK=0 ; aK <int(aVM.size()) ; aK++)
    {
         std::string aSys = string(SYS_MV) + ' ' + ToStrBlkCorr(mDir+aVM[aK].mNameIn) + " " + ToStrBlkCorr(mDir+aVM[aK].mNameOut);
     std::cout << aSys << "\n";
     if (mExe)
     {
             VoidSystem(aSys.c_str());
     }
    }
    if (!mExe)
       std::cout << "\n     Use Exe=1 to execute moves !!\n";

}


   //===========================================

int MyRename_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    cAppliMyRename  aRename(argc,argv);


    return 0;
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
