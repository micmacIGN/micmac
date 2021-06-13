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

#if ELISE_unix
    #define SYS_RMR "\\rm -r"   // definition un rm du linux -- ELISE_fp marche pas ???
#endif


int GetSpace_main(int argc,char ** argv)
{
        string aDir = ELISE_Current_DIR;
        bool exe= false;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()
                    << EAMC(aDir, "Directory Chantier",  eSAM_IsDir)
                    ,
                    //optional arguments
                    LArgMain()
                    << EAM(exe, "exe", true, "Do delete temp")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;
        cout<<"Dir : "<<aDir<<endl;
        vector<string> VTmp;
        vector<string> VTmpExist;
        string aTmpMMDir = aDir + "Tmp-MM-Dir" + ELISE_CAR_DIR;
          VTmp.push_back(aTmpMMDir);
        string aTmpZBuf= aDir + "Tmp-ZBuffer" + ELISE_CAR_DIR;
          VTmp.push_back(aTmpZBuf);
        string aTmpLiai= aDir + "Tmp-LIAISON";
          VTmp.push_back(aTmpLiai);
        string aTmpSaisie= aDir + "Tmp-SaisieAppuis" + ELISE_CAR_DIR;
          VTmp.push_back(aTmpSaisie);
        string aBigMac = aDir + "PIMs-BigMac" + ELISE_CAR_DIR;
          VTmp.push_back(aBigMac);
        string aMicMac = aDir + "PIMs-MicMac" + ELISE_CAR_DIR;
           VTmp.push_back(aMicMac);
        string aQuickMac = aDir + "PIMs-QuickMac" + ELISE_CAR_DIR;
           VTmp.push_back(aQuickMac);
        string aStatue = aDir + "PIMs-Statue" + ELISE_CAR_DIR;
           VTmp.push_back(aStatue);
        string aForest = aDir + "PIMs-Forest" + ELISE_CAR_DIR;
           VTmp.push_back(aForest);
        string aPyram = aDir + "Pyram" + ELISE_CAR_DIR;
           VTmp.push_back(aPyram);
        string aMMPyram = aDir + "MM-Pyram" + ELISE_CAR_DIR;
           VTmp.push_back(aMMPyram);
       string aPastis = aDir + "Pastis" + ELISE_CAR_DIR;
           VTmp.push_back(aPastis);
       string aOrthoMecMalt = aDir + "Ortho-MEC-Malt" + ELISE_CAR_DIR;
           VTmp.push_back(aOrthoMecMalt);
       string aOKOrtho = aDir + "Qk-ORTHO" + ELISE_CAR_DIR;
           VTmp.push_back(aOKOrtho);
       string aMECMalt = aDir + "MEC-Malt" + ELISE_CAR_DIR;
           VTmp.push_back(aMECMalt);
       string aTA = aDir + "TA" + ELISE_CAR_DIR;
           VTmp.push_back(aTA);
       string aHomol_SRes = aDir + "Homol_SRes" + ELISE_CAR_DIR;
           VTmp.push_back(aHomol_SRes);

        for (uint i=0; i<VTmp.size(); i++)
        {
            if (ELISE_fp::IsDirectory(VTmp[i]))
            {
                cout<<"Found : "<<VTmp[i]<<endl;
                VTmpExist.push_back(VTmp[i]);
            }
        }

        if (!exe)
        {
            cout<<"Set exe = 1 if you want to delete all temp"<<endl;
        }
        if (exe && VTmpExist.size() > 0)
        {
            cout<<"Sure to delete ? [y/n]"<<endl;
            char ch1 = static_cast<char>(getc(stdin));
            if (ch1 == 'y')
            {
                for (uint i=0; i<VTmpExist.size(); i++)
                {
                    cout<<"Del : "<<VTmpExist[i]<<" ... ";

#if ELISE_unix
                    std::string aNameCom = std::string(SYS_RMR)+ " " +VTmpExist[i];
                    ::System(aNameCom.c_str());
#else

                    ELISE_fp::PurgeDirRecursif(VTmpExist[i]);
#endif
                    cout<<" done ! "<<endl;
                }
            }
            else
            {
                return EXIT_SUCCESS;
            }
        }
        else
        {
            cout<<"Nothing to delete ! "<<endl;
        }
        return EXIT_SUCCESS;
    }




