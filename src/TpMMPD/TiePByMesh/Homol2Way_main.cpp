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

#include "InitOutil.h"
#include <stdio.h>


    /******************************************************************************
    The main function.
    ******************************************************************************/

void cplExistInHomol ( vector<CplString> & VCpl,
                       vector<string> & VImgs,
                       string & Homol,
                       cInterfChantierNameManipulateur * aICNM,
                       bool  isTxtHomol )
{
    cout<<" ++ Creat couple image depends on exist homol structure: "<<endl;
    string extHomol = isTxtHomol ? std::string("txt"):std::string("dat");
    cout<<"Name Homol: "<<Homol<<" -Ext : "<<extHomol<<endl;
    string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(Homol)
                       +  std::string("@")
                       +  extHomol;

    for (uint i=0; i<VImgs.size()-1; i++)
    {
        for (uint j=i+1; j<VImgs.size(); j++)
        {
            string pic1 = VImgs[i];
            string pic2 = VImgs[j];

            string aHomoIn = aICNM->Assoc1To2(aKHIn, pic1, pic2, true);

            StdCorrecNameHomol_G(aHomoIn, aICNM->Dir());

            bool Exist= ELISE_fp::exist_file(aHomoIn);

            if (Exist)
            {
                CplString aCplHomo;
                aCplHomo.img1 = pic1;
                aCplHomo.img2 = pic2;
                VCpl.push_back(aCplHomo);
                //cout<<"..exist!";
            }
            //cout<<endl;
        }
    }
}

int Homol2Way_main(int argc,char ** argv)
{
    string aFullPattern;
    string aSHIn = "Homol";
    string aSHOut = "_2Way";
    bool skipVide = false;
    cout<<"*************************************************************************"<<endl;
    cout<<"*    Creat same pack homol in 2 way by combination 2 pack of each way   *"<<endl;
    cout<<"*************************************************************************"<<endl;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile),
                    //optional arguments
                    LArgMain()
                    << EAM(aSHIn, "SH", true, "Input homol folder (default = Homol)")
                    << EAM(aSHOut, "SHOut", true, "Output homol folder")
                    << EAM(skipVide, "skipVide", true, "don't write out pack Homol vide")
                );

        if (MMVisualMode) return EXIT_SUCCESS;

        std::string aDir,aNameImg;
        SplitDirAndFile(aDir,aNameImg,aFullPattern);
        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        vector<string> VImgs = *(aICNM->Get(aNameImg));
        cout<<"Nb Imgs : "<<VImgs.size()<<endl;
        vector<CplString> VCpl;

        //StdCorrecNameHomol_G(aSHIn, aDir);
        if (!ELISE_fp::IsDirectory(aSHIn))
            aSHIn = "Homol" + aSHIn;

        cplExistInHomol      ( VCpl,
                               VImgs,
                               aSHIn,
                               aICNM,
                               false );


        string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                           +  std::string(aSHIn)
                           +  std::string("@")
                           +  std::string("dat");
        string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                            +  std::string(aSHOut)
                            +  std::string("@")
                            +  std::string("dat");

        cout<<"ToTal: "<<VCpl.size()<<" cpl founed"<<endl;


        for (uint aKCpl=0; aKCpl<VCpl.size(); aKCpl++)
        {
            CplString aCpl = VCpl[aKCpl];
            string pic1 = aCpl.img1;
            string pic2 = aCpl.img2;
            cout<<"Cpl : "<<pic1 <<" - "<< pic2<<endl;

            ElPackHomologue aPckCmbn;
            string aHmIn= aICNM->Assoc1To2(aKHIn, pic1, pic2, true);
            string aHmInIv= aICNM->Assoc1To2(aKHIn, pic2, pic1, true);

            bool Exist= ELISE_fp::exist_file(aHmIn);
            if (Exist)
            {
                aPckCmbn =  ElPackHomologue::FromFile(aHmIn);
                cout<<" ++"<<pic1<<" : "<<aPckCmbn.size()<<" Pts"<<endl;
                if (ELISE_fp::exist_file(aHmInIv))
                {
                    ElPackHomologue aPck2 = ElPackHomologue::FromFile(aHmInIv);
                    aPck2.SelfSwap();
                    aPckCmbn.Add(aPck2);
                    cout<<" ++"<<pic2<<" : "<<aPck2.size()<<" Pts"<<endl;
                }
            }
            else
            {
                StdCorrecNameHomol_G(aSHIn, aICNM->Dir());
                aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                                   +  std::string(aSHIn)
                                   +  std::string("@")
                                   +  std::string("dat");
                aHmIn= aICNM->Assoc1To2(aKHIn, pic1, pic2, true);
                aHmInIv= aICNM->Assoc1To2(aKHIn, pic2, pic1, true);
                Exist= ELISE_fp::exist_file(aHmIn);
                if (Exist)
                {
                    aPckCmbn =  ElPackHomologue::FromFile(aHmIn);
                    cout<<" ++"<<pic1<<" : "<<aPckCmbn.size()<<" Pts"<<endl;
                    if (ELISE_fp::exist_file(aHmInIv))
                    {
                        ElPackHomologue aPck2 = ElPackHomologue::FromFile(aHmInIv);
                        aPck2.SelfSwap();
                        aPckCmbn.Add(aPck2);
                        cout<<" ++"<<pic2<<" : "<<aPck2.size()<<" Pts"<<endl;
                    }
                }
            }
            //write aPckCmbn to disk
            string aHmOut = aICNM->Assoc1To2(aKHOutDat, pic1, pic2, true);
            cout<<" ++ Pck Combine : "<<aPckCmbn.size()<<" Pts"<<endl;
            cout<<" ++ Wtite : "<<aHmOut<<endl;
            if (skipVide)
            {
                if (aPckCmbn.size() > 0)
                {
                    aPckCmbn.StdPutInFile(aHmOut);
                    aHmOut = aICNM->Assoc1To2(aKHOutDat, pic2, pic1, true);
                    aPckCmbn.SelfSwap();
                    aPckCmbn.StdPutInFile(aHmOut);
                }
            }
            else
            {
                aPckCmbn.StdPutInFile(aHmOut);
                aHmOut = aICNM->Assoc1To2(aKHOutDat, pic2, pic1, true);
                aPckCmbn.SelfSwap();
                aPckCmbn.StdPutInFile(aHmOut);
            }
        }
        cout<<"If Total 0 Cpl founded, add Homol_ in parameter SH"<<endl;
        return EXIT_SUCCESS;
    }
