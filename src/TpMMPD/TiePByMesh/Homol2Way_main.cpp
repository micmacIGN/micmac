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

class cAppliHomol_2Way;
class cImgTiep2Way;


class cAppliHomol_2Way
{
    public:
        cAppliHomol_2Way(string aFullPattern);
        cInterfChantierNameManipulateur * ICNM() {return mICNM;}
        vector<string> & VImg() {return mVImg;}
        vector<cImgTiep2Way*> & VImgTiep2W() {return mVImgTiep2W;}
        void DoFusion2Way(bool isTxt, string extHomol);
        void ExportHom(bool isTxt, string extHomol);
    private:
        cInterfChantierNameManipulateur * mICNM;
        vector<string> mVImg;
        string mDir;
        vector<cImgTiep2Way*> mVImgTiep2W;
};


class cImgTiep2Way
{
    public:
        cImgTiep2Way(int ind, cAppliHomol_2Way* aAppli);
        cAppliHomol_2Way* Appli() {return mAppli;}
        int & Ind() {return mInd;}
        string & Name() {return mAppli->VImg()[mInd];}
        vector<ElPackHomologue*> & VPackHom() {return mVPackHom;}
    private:
        cAppliHomol_2Way* mAppli;
        int mInd;
        vector<ElPackHomologue*> mVPackHom;
};


cAppliHomol_2Way::cAppliHomol_2Way(string aFullPattern)
{
    std::string aNameImg;
    SplitDirAndFile(mDir,aNameImg,aFullPattern);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mVImg = *(mICNM->Get(aNameImg));

    for (uint aKImg=0; aKImg<mVImg.size(); aKImg++)
    {
        cImgTiep2Way * aImg = new cImgTiep2Way(aKImg, this);
        mVImgTiep2W.push_back(aImg);
    }
}

cImgTiep2Way::cImgTiep2Way(int ind, cAppliHomol_2Way* aAppli):
    mAppli (aAppli),
    mInd (ind),
    mVPackHom (vector<ElPackHomologue*>(mAppli->VImg().size()))
{
    for (uint aKPack=0; aKPack<mVPackHom.size(); aKPack++)
    {
        ElPackHomologue * aPck = new ElPackHomologue();
        mVPackHom[aKPack] = aPck;
    }
}

void cAppliHomol_2Way::DoFusion2Way(bool isTxt, string extHomol)
{
    string ExpFormat = isTxt ? std::string("txt"):std::string("dat");
    string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(extHomol)
                       +  std::string("@")
                       +  ExpFormat;

    for (uint aKIm=0; aKIm<mVImgTiep2W.size(); aKIm++)
    {
        cImgTiep2Way * aIm1 = mVImgTiep2W[aKIm];
        for (uint aKImB=0; aKImB<mVImgTiep2W.size(); aKImB++)
        {
            if (aKImB == aKIm)
                continue;   // terminate current iteration
            else
            {
                cImgTiep2Way * aIm2 = mVImgTiep2W[aKImB];
                string aPathHom = mICNM->Assoc1To2(aKHIn, aIm1->Name(), aIm2->Name(), true);
                string aPathHomInv = mICNM->Assoc1To2(aKHIn, aIm2->Name(), aIm1->Name(), true);
                StdCorrecNameHomol_G(aPathHom, mDir);
                if (ELISE_fp::exist_file(aPathHom))
                {
                    ElPackHomologue aPck12 = ElPackHomologue::FromFile(aPathHom);
                    // ajout to aPack 12
                    aIm1->VPackHom()[aIm2->Ind()]->Add(aPck12);
                    // ajout to aPack 21
                    aPck12.SelfSwap();
                    aIm2->VPackHom()[aIm1->Ind()]->Add(aPck12);

                }
                if (ELISE_fp::exist_file(aPathHomInv))
                {
                    ElPackHomologue aPck21 = ElPackHomologue::FromFile(aPathHomInv);
                    // ajout to aPack 21
                    aIm2->VPackHom()[aIm1->Ind()]->Add(aPck21);
                    // ajout to aPack 12
                    aPck21.SelfSwap();
                    aIm1->VPackHom()[aIm2->Ind()]->Add(aPck21);
                }
            }
        }
    }
}

void cAppliHomol_2Way::ExportHom(bool isTxt, string extHomol)
{
    string ExpFormat = isTxt ? std::string("txt"):std::string("dat");
    string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(extHomol)
                        +  std::string("@")
                        +  ExpFormat;
    for (uint aKIm=0; aKIm<mVImgTiep2W.size(); aKIm++)
    {
        cImgTiep2Way * aIm1 = mVImgTiep2W[aKIm];
        for (uint aKImB=0; aKImB<mVImgTiep2W.size(); aKImB++)
        {
            if (aKImB == aKIm)
                continue;   // terminate current iteration
            else
            {
                cImgTiep2Way * aIm2 = mVImgTiep2W[aKImB];
                string aPathHom = mICNM->Assoc1To2(aKHOut, aIm1->Name(), aIm2->Name(), true);
                if (aIm1->VPackHom()[aIm2->Ind()]->size() > 0)
                    aIm1->VPackHom()[aIm2->Ind()]->StdPutInFile(aPathHom);
            }
        }
    }
}


int Homol2WayNEW_main(int argc,char ** argv)
{
    string aFullPattern;
    string aSHIn = "Homol";
    string aSHOut = "_2Way";
    bool skipVide = false;
    bool ExpTxt = false;
    cout<<"*************************************************************************"<<endl;
    cout<<"*    Creat same pack homol in 2 way by combination 2 pack of each way   *"<<endl;
    cout<<"*                   Convert homol format dat <-> txt                    *"<<endl;
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
                    << EAM(ExpTxt, "ExpTxt", true, "Output format (txt=true, dat=false - def=false")
                );

        if (MMVisualMode) return EXIT_SUCCESS;
        cAppliHomol_2Way * aAppli = new cAppliHomol_2Way(aFullPattern);
        aAppli->DoFusion2Way(ExpTxt, aSHIn);
        aAppli->ExportHom(ExpTxt, aSHOut);

   return EXIT_SUCCESS; // Warn no return
}


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

void writeHomol(string pic1, string pic2, string ExpFormat, string SHOut, cInterfChantierNameManipulateur * aICNM, ElPackHomologue aPack, bool skipVide)
{
    string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(SHOut)
                        +  std::string("@")
                        +  ExpFormat;
    string aHmOut = aICNM->Assoc1To2(aKHOutDat, pic1, pic2, true);
    cout<<" ++ Write : "<<aHmOut<<endl;
    if (skipVide)
    {
        if (aPack.size() > 0)
        {
            aPack.StdPutInFile(aHmOut);
        }
    }
    else
    {
        aPack.StdPutInFile(aHmOut);
    }
}



int Homol2Way_main(int argc,char ** argv)
{
    string aFullPattern;
    string aSHIn = "Homol";
    string aSHOut = "_2Way";
    bool skipVide = false;
    bool ExpTxt = false;
    bool IntTxt = false;
    bool OnlyConvert = false;
    cout<<"*************************************************************************"<<endl;
    cout<<"*    Creat same pack homol in 2 way by combination 2 pack of each way   *"<<endl;
    cout<<"*                   Convert homol format dat <-> txt                    *"<<endl;
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
                    << EAM(IntTxt, "IntTxt", true, "Input format (txt=true, dat=false - def=false")
                    << EAM(ExpTxt, "ExpTxt", true, "Output format (txt=true, dat=false - def=false")
                    << EAM(OnlyConvert, "OnlyConvert", true, "convert format homol only (not create 2 Way homologue) - def=false")
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
                               IntTxt );

        string ExpFormat = ExpTxt ? std::string("txt"):std::string("dat");
        string InpFormat = IntTxt ? std::string("txt"):std::string("dat");

        string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                           +  std::string(aSHIn)
                           +  std::string("@")
                           +  InpFormat;
        string aKHOutDat =   std::string("NKS-Assoc-CplIm2Hom@")
                            +  std::string(aSHOut)
                            +  std::string("@")
                            +  ExpFormat;

        cout<<"ToTal: "<<VCpl.size()<<" cpl founed"<<endl;

if (!OnlyConvert)
{
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
}
else
{
    cout<<"Convert homol format only"<<endl;
    for (uint aKCpl=0; aKCpl<VCpl.size(); aKCpl++)
    {
            CplString aCpl = VCpl[aKCpl];
            string pic1 = aCpl.img1;
            string pic2 = aCpl.img2;
            cout<<"Cpl : "<<pic1 <<" - "<< pic2<<endl;
            string aHmIn= aICNM->Assoc1To2(aKHIn, pic1, pic2, true);
            bool Exist= ELISE_fp::exist_file(aHmIn);
            if (!Exist)
            {
                StdCorrecNameHomol_G(aSHIn, aICNM->Dir());
                aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                                   +  std::string(aSHIn)
                                   +  std::string("@")
                                   +  InpFormat;
                aHmIn= aICNM->Assoc1To2(aKHIn, pic1, pic2, true);
            }
            Exist= ELISE_fp::exist_file(aHmIn);
            if (Exist)
            {
                ElPackHomologue aPack = ElPackHomologue::FromFile(aHmIn);
                writeHomol(pic1, pic2, ExpFormat, aSHOut, aICNM,  aPack, skipVide);
            }

            string aHmInIv= aICNM->Assoc1To2(aKHIn, pic2, pic1, true);
            bool ExistIv= ELISE_fp::exist_file(aHmInIv);
            if (!ExistIv)
            {
                StdCorrecNameHomol_G(aSHIn, aICNM->Dir());
                aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                                   +  std::string(aSHIn)
                                   +  std::string("@")
                                   +  InpFormat;
                aHmInIv= aICNM->Assoc1To2(aKHIn, pic2, pic1, true);
            }
            ExistIv= ELISE_fp::exist_file(aHmInIv);
            if (ExistIv)
            {
                ElPackHomologue aPack = ElPackHomologue::FromFile(aHmInIv);
                writeHomol(pic2, pic1, ExpFormat, aSHOut, aICNM,  aPack, skipVide);
            }
    }
}
        cout<<"If Total 0 Cpl founded, add Homol_ in parameter SH"<<endl;
        return EXIT_SUCCESS;
    }
