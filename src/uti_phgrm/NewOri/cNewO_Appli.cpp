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

#include "NewOri.h"

class cAppli_Martini
{
      public :
          cAppli_Martini(int argc,char ** argv,bool Quick);
          void DoAll();
          void Banniere(bool Quick);
      private :

          void StdCom(const std::string & aCom,const std::string & aPost="");
          std::string mNameOriCalib;
          std::string mPat;
          bool        mExe;
          bool        mQuick;
          ElTimer     aChrono;
};

void cAppli_Martini::StdCom(const std::string & aCom,const std::string & aPost)
{
    std::string  aFullCom = MM3dBinFile_quotes( "TestLib ") + aCom + " "   + QUOTE(mPat);
    if (EAMIsInit(&mNameOriCalib))  aFullCom = aFullCom + " OriCalib=" + mNameOriCalib;
    aFullCom += " Quick=" + ToString(mQuick);

    aFullCom = aFullCom + aPost;


    if (mExe)
       System(aFullCom);
    else
       std::cout << "COM= " << aFullCom << "\n";

    std::cout << " DONE " << aCom << " in time " << aChrono.uval() << "\n";
}

void cAppli_Martini::Banniere(bool Quick)
{
    if (Quick)
    {
        std::cout <<  "\n";
        std::cout <<  " *********************************************\n";
        std::cout <<  " *     MART-ingale d'                        *\n";
        std::cout <<  " *     INI-tialisation                       *\n";
        std::cout <<  " *********************************************\n\n";
    }
    else
    {
        std::cout <<  "\n";
        std::cout <<  " *********************************************\n";
        std::cout <<  " *     MARTIN                                *\n";
        std::cout <<  " *     Gale d'                               *\n";
        std::cout <<  " *     IN-itialisation (stronger version)    *\n"; 
        std::cout <<  " *********************************************\n\n";
    }

}

void cAppli_Martini::DoAll()
{
     //  Calcul de toute les orientations relatives entre paires d'images
     StdCom("NO_AllOri2Im");
     // Homologues flottants
     StdCom("NO_AllHomFloat");
     // Generation des triplet de points hom flottants
     StdCom("NO_AllImTriplet");
     // Generation  des triplet
     StdCom("NO_GenTripl"," Show=false");
     // Optimisation des triplets
     StdCom("NO_AllImOptTrip");
}





cAppli_Martini::cAppli_Martini(int argc,char ** argv,bool Quick) :
    mExe   (true),
    mQuick (Quick)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mPat,"Image Pat", eSAM_IsPatFile),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(mExe,"Exe",true,"Execute commands, def=true (if false, only print)")
                   // << EAM(mQuick,"Quick",true,"Quick version")
   );
}


int CPP_Gene_Martini_main(int argc,char ** argv,bool Quick)
{
   MMD_InitArgcArgv(argc,argv);
   cAppli_Martini anAppli(argc,argv,Quick);
   if (MMVisualMode) return EXIT_SUCCESS;
   anAppli.DoAll();
   anAppli.Banniere(Quick);
   return EXIT_SUCCESS;
}

int CPP_Martini_main(int argc,char ** argv)
{
    return CPP_Gene_Martini_main(argc,argv,true);
}


int CPP_MartiniGin_main(int argc,char ** argv)
{
    return CPP_Gene_Martini_main(argc,argv,false);
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
