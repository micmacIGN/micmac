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


class cVideoVisage
{
    public :
       cVideoVisage(int argc,char ** argv);
    private :
       void DoImage();
       void DoHomol();
       void DoOri();

       std::string mFullNameVideo;
       std::string mDir;
       std::string mPatIm;
       const std::vector<std::string> * mSetIm;
       std::vector<std::string>         mMasters;
       std::vector<int>                 mKMasters;
       std::string mNameVideo;
       double      mRate;
       int         mSzSift;
       int         mLineSift;
       cInterfChantierNameManipulateur * mICNM;
};

void cVideoVisage::DoImage()
{
    std::string aStr =   std::string("ffmpeg -i ")
                       + mFullNameVideo
                       + std::string(" -r ") + ToString(mRate) + std::string("  ")
                       + StdPrefixGen(mFullNameVideo) + "im%05d.png";



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
    std::string aStr =    std::string("Tapas RadialStd  ") + mPatIm + " Out=All";
    system_call(aStr.c_str());
    aStr =    std::string("AperiCloud  ") + mPatIm + " All ";
    system_call(aStr.c_str());
}

cVideoVisage::cVideoVisage(int argc,char ** argv) :
    mRate (4.0),
    mSzSift  (-1),
    mLineSift (8)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);


    ElInitArgMain
    (
	argc,argv,
	LArgMain()   << EAMC(mFullNameVideo,"Name Of Video"),
	LArgMain()   << EAM(mRate,"Rate",true,"Number of image / sec; def = 4")
                     << EAM(mSzSift,"SzS","Size Sift, def=-1 (Max resol)")
                     << EAM(mLineSift,"LineS","Line Sift, def=8")
    );

    SplitDirAndFile(mDir,mNameVideo,mFullNameVideo);


    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);

    mPatIm =   StdPrefixGen(mNameVideo) + "im[0-9]{5}\\.png";
    mSetIm = mICNM->Get(mPatIm);
// std::cout << mPatIm << " " << mSetIm->size() << "\n";
    mPatIm =  "\"" + mPatIm + "\"";

    // DoImage();
    // DoHomol();
    // DoOri();


     std::cout << "MASKE DONE ?????   When yes type enter \n"; getchar();
     

     
     for (int aK=0 ; aK<int(mSetIm->size()) ; aK++)
     {
          std::string aName = (*mSetIm)[aK];
          std::string aImMasq = mDir + StdPrefix(aName) + "_Masq.tif";
          if (ELISE_fp::exist_file(aImMasq))
          {
              mMasters.push_back(aName);
              mKMasters.push_back(aK+1);

              std::cout <<  aName << " " <<  aK+1 << "\n";
          }
     }


/*
	#if (ELISE_windows)
		replace( aDir.begin(), aDir.end(), '\\', '/' );
	#endif

    if (! EAMIsInit(&mDeqXY)) 
       mDeqXY = Pt2di(mDeq,mDeq);

    if (! EAMIsInit(&mDegRapXY))
       mDegRapXY = Pt2di(mDegRap,mDegRap);

    int aRes = system_call(aCom.c_str());

    BanniereMM3D();
    return aRes;
*/

}

int VideoVisage_main(int argc,char ** argv)
{
    cVideoVisage(argc,argv);
     return 0;
}




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
