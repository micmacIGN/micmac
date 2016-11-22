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
int Homol2Way_main(int argc,char ** argv)
{
    string aFullPattern;
    string aSHIn = "Homol";
    string aSHOut = "_2Way";
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
                );

        if (MMVisualMode) return EXIT_SUCCESS;
        InitOutil * aChain = new InitOutil(aFullPattern, "NONE", aSHIn);
        cInterfChantierNameManipulateur * mICNM = aChain->getPrivmICNM();
        aChain->load_Im();
        vector<CplPic> VCpl = aChain->loadCplPicExistHomol();
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
            CplPic aCpl = VCpl[aKCpl];
            pic * pic1 = aCpl.pic1;
            pic * pic2 = aCpl.pic2;
            cout<<"Cpl : "<<aCpl.pic1->getNameImgInStr() <<" - "<< aCpl.pic2->getNameImgInStr()<<endl;
            ElPackHomologue aPckCmbn;
            string aHmIn= mICNM->Assoc1To2(aKHIn, pic1->getNameImgInStr(), pic2->getNameImgInStr(), true);
            string aHmInIv= mICNM->Assoc1To2(aKHIn, pic2->getNameImgInStr(), pic1->getNameImgInStr(), true);
            bool Exist= ELISE_fp::exist_file(aHmIn);
            if (Exist)
            {
                aPckCmbn =  ElPackHomologue::FromFile(aHmIn);
                cout<<" ++"<<pic1->getNameImgInStr()<<" : "<<aPckCmbn.size()<<" Pts"<<endl;
                if (ELISE_fp::exist_file(aHmInIv))
                {
                    ElPackHomologue aPck2 = ElPackHomologue::FromFile(aHmInIv);
                    aPck2.SelfSwap();
                    aPckCmbn.Add(aPck2);
                    cout<<" ++"<<pic2->getNameImgInStr()<<" : "<<aPck2.size()<<" Pts"<<endl;
                }
            }
            else
            {
                StdCorrecNameHomol_G(aSHIn, aChain->getPrivmICNM()->Dir());
                aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                                   +  std::string(aSHIn)
                                   +  std::string("@")
                                   +  std::string("dat");
                aHmIn= mICNM->Assoc1To2(aKHIn, pic1->getNameImgInStr(), pic2->getNameImgInStr(), true);
                aHmInIv= mICNM->Assoc1To2(aKHIn, pic2->getNameImgInStr(), pic1->getNameImgInStr(), true);
                Exist= ELISE_fp::exist_file(aHmIn);
                if (Exist)
                {
                    aPckCmbn =  ElPackHomologue::FromFile(aHmIn);
                    cout<<" ++"<<pic1->getNameImgInStr()<<" : "<<aPckCmbn.size()<<" Pts"<<endl;
                    if (ELISE_fp::exist_file(aHmInIv))
                    {
                        ElPackHomologue aPck2 = ElPackHomologue::FromFile(aHmInIv);
                        aPck2.SelfSwap();
                        aPckCmbn.Add(aPck2);
                        cout<<" ++"<<pic2->getNameImgInStr()<<" : "<<aPck2.size()<<" Pts"<<endl;
                    }
                }
            }
            //write aPckCmbn to disk
            string aHmOut = mICNM->Assoc1To2(aKHOutDat, pic1->getNameImgInStr(), pic2->getNameImgInStr(), true);
            cout<<" ++ Pck Combine : "<<aPckCmbn.size()<<" Pts"<<endl;
            cout<<" ++ Wtite : "<<aHmOut<<endl;
            aPckCmbn.StdPutInFile(aHmOut);
            aHmOut = mICNM->Assoc1To2(aKHOutDat, pic2->getNameImgInStr(), pic1->getNameImgInStr(), true);
            aPckCmbn.SelfSwap();
            aPckCmbn.StdPutInFile(aHmOut);
        }



       
        return EXIT_SUCCESS;
    }
