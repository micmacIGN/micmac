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


int Tawny_main(int argc,char ** argv)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir;


    int mDeq = 1;
    Pt2di mDeqXY(-1,-1);
    bool mAddCste = false;
    int mDegRap = 0;
    Pt2di mDegRapXY(-1,-1);
    bool mRapGlobPhys = true;
    double mDynGlob=1.0;

    std::string mImPrio0 = ".*";
    int mSzV = 1;
    double mNbPerIm = 1e4;
    double mCorrThresh = 0.8;
    bool  DoL1Filter=true;

    double  aSatThresh = 1e9;
    string aNameOut="Ortho-Eg-Test-Redr.tif";
    ElInitArgMain
    (
    argc,argv,
    LArgMain()   << EAMC(aDir,"Directory where are the data", eSAM_IsDir),
    LArgMain()   << EAM(mDeq,"DEq",true,"Degre of equalization (Def=1)")
                 << EAM(mDeqXY,"DEqXY",true,"Degre of equalization, if diff in X and Y")
                 << EAM(mAddCste,"AddCste",true,"Add unknown constant for equalization (Def=false)", eSAM_IsBool)
                 << EAM(mDegRap,"DegRap",true,"Degre of rappel to initial values, Def = 0")
                 << EAM(mDegRapXY,"DegRapXY",true,"Degre of rappel to initial values, Def = 0")
                 << EAM(mRapGlobPhys,"RGP",true,"Rappel glob on physycally equalized, Def = true")
                 << EAM(mDynGlob,"DynG",true,"Global Dynamic (to correct saturation problems)")
                 << EAM(mImPrio0,"ImPrio",true,"Pattern of image with high prio, def=.*", eSAM_IsPatFile)
                 << EAM(mSzV,"SzV",true,"Sz of Window for equalisation (Def=1, means 3x3)")
                 << EAM(mCorrThresh,"CorThr",true,"Threshold of correlation to validate homologous (Def 0.7)")
                 << EAM(mNbPerIm,"NbPerIm",true,"Average number of point per image (Def = 1e4)")
                 << EAM(DoL1Filter,"L1F",true,"Do L1 Filter on couple, def=true (change when process is blocked)", eSAM_IsBool)
                 << EAM(aSatThresh,"SatThresh",true,"Threshold determining saturation value (pixel >SatThresh will be ignored)")
                 << EAM(aNameOut,"Out",true,"Name of output file (in the folder)", eSAM_IsOutputFile)
    );

    #if (ELISE_windows)
        replace( aDir.begin(), aDir.end(), '\\', '/' );
    #endif

    if (! EAMIsInit(&mDeqXY))
       mDeqXY = Pt2di(mDeq,mDeq);

    if (! EAMIsInit(&mDegRapXY))
       mDegRapXY = Pt2di(mDegRap,mDegRap);

    Pt2di aDegCste = mAddCste  ? Pt2di(0,0) : Pt2di(-1,-1);

    MMD_InitArgcArgv(argc,argv);

    std::string aCom =    MM3dBinFile( "Porto" )
                        + MMDir() +std::string("include/XML_MicMac/Param-Tawny.xml ")
                        + std::string(" %WD=") + aDir
                        + std::string(" +DR1X=") + ToString(mDeqXY.x)
                        + std::string(" +DR1Y=") + ToString(mDeqXY.y)
                        + std::string(" +DR0X=") + ToString(aDegCste.x)
                        + std::string(" +DR0Y=") + ToString(aDegCste.y)
                        + std::string(" +DegRapX=") + ToString(mDegRapXY.x)
                        + std::string(" +DegRapY=") + ToString(mDegRapXY.y)
                        + std::string(" +RapGlobPhys=") + ToString(mRapGlobPhys)
                        + std::string(" +DynGlob=") + ToString(mDynGlob)
                        + std::string(" +NameOrtho=") + aNameOut
                      ;

    if (mImPrio0!="") aCom = aCom+ " +ImPrio="+QUOTE(mImPrio0);
    if (EAMIsInit(&mSzV)) aCom  = aCom + " +SzV=" + ToString(mSzV);
    if (EAMIsInit(&mNbPerIm)) aCom  = aCom + " +NbPerIm=" + ToString(mNbPerIm);
    if (EAMIsInit(&mCorrThresh)) aCom  = aCom + " +CorrThresh=" + ToString(mCorrThresh);

     if (!DoL1Filter) aCom  = aCom +" +DoL1Filter=false ";

    std::cout << aCom << "\n";
    int aRes = system_call(aCom.c_str());

    BanniereMM3D();
    return aRes;
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
