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

#define DEF_OFSET -12349876


std::string Str5OfInt(int aK)
{
    char aBuf[11];
    sprintf(aBuf,"%05d",aK);

    return aBuf;

}

class cVideoVisage
{
    public :
       cVideoVisage(int argc,char ** argv);

       std::string NameIm(const std::string & aCenter)
       {
           return  StdPrefixGen(mNameVideo) + "im" + aCenter + ".png";
       }
       bool OK_MMLD() {return ELISE_fp::exist_file(mDir+"MicMac-LocalChantierDescripteur.xml");}

    private :
       void DoImage();
       void DoHomol();
       void DoOri();

       std::string mFullNameVideo;
       std::string mDir;
       std::string mPatIm;
       const std::vector<std::string> * mSetIm;
       std::string                      mImMedian;
       std::vector<std::string>         mMasters;
       std::vector<int>                 mKMasters;
       std::string mNameVideo;
       double      mRate;
       int         mSzSift;
       int         mLineSift;
       double      mTeta;
       cInterfChantierNameManipulateur * mICNM;
};

void cVideoVisage::DoImage()
{
    std::string aStr =   std::string("ffmpeg -i ")
                       + mFullNameVideo
                       + std::string(" -r ") + ToString(mRate) + std::string("  ")
                       + NameIm("%05d");



    system_call(aStr.c_str());

}

void cVideoVisage::DoHomol()
{
    std::string aStr =    std::string("Tapioca Line ")
                       +  mPatIm
                       +  " " + ToString(mSzSift)
                       +  " " + ToString(mLineSift);
    system_call(aStr.c_str());
}


void cVideoVisage::DoOri()
{
    std::string aStr =    std::string("Tapas RadialBasic  ") + mPatIm + " Out=All" +  " ImInit=" + mImMedian;
    system_call(aStr.c_str());
    aStr =    std::string("AperiCloud  ") + mPatIm + " All " + std::string(" Out=Cam")+ StdPrefixGen(mNameVideo) + ".ply";
    system_call(aStr.c_str());
}

cVideoVisage::cVideoVisage(int argc,char ** argv) :
    mRate (4.0),
    mSzSift  (-1),
    mLineSift (8),
    mTeta     (180)
{
    MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);


    ElInitArgMain
    (
    argc,argv,
    LArgMain()   << EAMC(mFullNameVideo,"Video name", eSAM_IsExistFile),
    LArgMain()   << EAM(mRate,"Rate",true,"Number of images / sec (Def=4)")
                     << EAM(mSzSift,"SzS",true, "Size Sift, (Def=-1 max resol)")
                     << EAM(mLineSift,"LineS", true, "Line Sift (Def=8)")
                     << EAM(mTeta,"Teta", true, "Angle done (Def=180)")
    );

    if (!MMVisualMode)
    {
        int aNbB = sizeofile(mFullNameVideo.c_str());

        if (! EAMIsInit(&mRate))
        {
            // Un film de 100 Go au rate de 4 a donne 71 image sur 180, on en voulait 36
            mRate = 4.0 * ( 1.03e8 / aNbB) * (36.0 /71.0);
            std::cout << "Rate=" << mRate << "\n";
        }

        SplitDirAndFile(mDir,mNameVideo,mFullNameVideo);

        if (! OK_MMLD())
        {
            std::cout << "Add MicMac-LocalChantierDescripteur.xml !!!! \n";
            getchar();
            ELISE_ASSERT(OK_MMLD(),"No MicMac-LocalChantierDescripteur.xml, no 3D model ...");
        }
        DoImage();

        mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

        mPatIm =   NameIm("[0-9]{5}");
        mSetIm = mICNM->Get(mPatIm);
        mPatIm =  QUOTE(mPatIm) ;

        mImMedian = NameIm (Str5OfInt((int)(mSetIm->size() / 2)));


        DoHomol();
        DoOri();


        std::cout << "MASKE DONE ?????   When yes type enter \n"; getchar();


        // std::list<std::string> aListComMalt;
        std::list<std::string> aListComPly;


        ELISE_fp::MkDir("Pyram");

        for (int aK=0 ; aK<int(mSetIm->size()) ; aK++)
        {
            std::string aName = (*mSetIm)[aK];
            std::string aImMasq = mDir + StdPrefix(aName) + "_Masq.tif";
            if (ELISE_fp::exist_file(aImMasq))
            {
                mMasters.push_back(aName);
                mKMasters.push_back(aK+1);

                int aDelta = 4;
                int aStep = 1;
                std::string aPatIm =   "(";
                bool First = true;
                for (int aD = -aDelta ; aD<= aDelta ; aD++)
                {
                    int aKP = aK+1 + aD * aStep;
                    if ((aKP>=1) && (aKP<=int(mSetIm->size())))
                    {
                        if (!First) aPatIm = aPatIm+"|";
                        aPatIm = aPatIm+Str5OfInt(aKP);
                        First=false;
                    }
                }

                aPatIm = QUOTE(NameIm(aPatIm+")"));
                std::string aComMalt = "Malt GeomImage " + aPatIm
                        + " All Regul=0.1 SzW=2 ZoomF=2 AffineLast=false Master="+aName;
                system_call(aComMalt.c_str());
                //        aListComMalt.push_back(aComMalt);

                std::string aComPly = "Nuage2Ply MM-Malt-Img-"
                        +StdPrefixGen(aName)+"/NuageImProf_STD-MALT_Etape_6.xml"
                        + " Attr=" +aName
                        + std::string(" RatioAttrCarte=2 ")
                        + std::string(" Out=") +StdPrefixGen(aName)+".ply";

                aListComPly.push_back(aComPly);
                // std::cout << aComPly << "\n";
            }
        }

        // cEl_GPAO::DoComInParal(aListComMalt,"Make-Malt-Video");
        cEl_GPAO::DoComInParal(aListComPly,"Make-Nuage2Ply-Video");

        BanniereMM3D();
    }
}

int VideoVisage_main(int argc,char ** argv)
{
    cVideoVisage(argc,argv);
     return 0;
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
