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


int Devlop_main(int argc,char ** argv)
{
    std::string aFullDir;
    std::string aPostAdd="None";
    std::string aNameOut="";
    int  En8B = 1;
    int  EnGray = 1;
    int  ConsCol = 0;
    std::string aSplit = "";


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pattern)", eSAM_IsPatFile),
    LArgMain()  << EAM(En8B,"8B",true)
                    << EAM(EnGray,"Gray",true)
                    << EAM(aPostAdd,"Post",true)
                    << EAM(ConsCol,"ConsCol",true)
                    << EAM(aNameOut,"NameOut",true)
                    << EAM(aSplit,"Split",true)
    );

    if (!MMVisualMode)
    {
    std::string aDir,aPatFile;
    SplitDirAndFile(aDir,aPatFile,aFullDir);

    std::string aPost = StdPostfix(aPatFile);
    std::string aPref = StdPrefix(aPatFile);


    // std::string aCom = MMBin() + " " + MM3DStr + " MapCmd "; => le blanc fait planter
    std::string aCom = MMBin() +  MM3DStr + " MapCmd ";
    std::string aPat= QUOTE("P=" + aDir+ "("+aPref+")." + aPost);

    // if (aNameOut =="") aNameOut = "\\$1" + (aPostAdd=="None"?"":aPostAdd)  + ".tif" ;

    std::string aSubst=  "\\$1" + (aPostAdd=="None"?"":aPostAdd)  + ".tif";

    {
       aCom = aCom + MMBin() +"MpDcraw " + aPat + " Add16B8B=0 ";
       if (aSplit!="")
       {
             aCom = aCom + " " + QUOTE("Split="+aSplit);
       }
       else
       {
          aCom = aCom +  (EnGray ? " GB=1 " : " CB=1 ");
       }
       aCom = aCom +  " 16B=" + (En8B ? "0 " : "1 ") ;
       aCom = aCom +  " ExtensionAbs="  + aPostAdd ;
       aCom = aCom +  " ConsCol=" + ToString(ConsCol) ;
       if (aNameOut != "")
       {
          aCom = aCom +  QUOTE(" NameOut=" + aNameOut);
          aCom = aCom +  " " + QUOTE("T=" + aNameOut);
       }
       else
       {
          aCom = aCom +  " " + QUOTE("T="  +  aSubst)  ;
       }
    }

    aCom = aCom+ " M=MkDevlop";
    // std::cout << aCom << "\n"; getchar();
     System(aCom);

     launchMake( "MkDevlop", "all" );

    return EXIT_SUCCESS;
    }
    else
        return EXIT_SUCCESS;
}


/************************************************************/
/*                                                          */
/*                                                          */
/*                                                          */
/************************************************************/

static    int aNbH = 1<<16;
int TheNbVois = 50;
int TheNbIter = 1;

void MakeFoncRepart(Im1D_REAL8 aH,int * aVMax=0)
{
    double aNbP;
    ELISE_COPY(aH.all_pts(),aH.in(),sigma(aNbP));
    REAL8 * aDH = aH.data();
    for (int aK=1 ; aK<aNbH ; aK++)
    {
        if (aDH[aK] && aVMax) *aVMax = aK;
        aDH[aK] += aDH[aK-1];
    }
    ELISE_COPY(aH.all_pts(),aH.in() * (255.0/aNbP),aH.out());
}

int PreparSift_Main(int argc,char ** argv)
{
    double aPEg = 1.0;
    double aPSrtEg = 3.0;
    double aPM = 2.0;


    std::string  aNameIn,aNameOut="Sift.tif";

    ElInitArgMain
    (
         argc,argv,
         LArgMain()  << EAMC(aNameIn,"Full name (Dir+Pattern)", eSAM_IsPatFile),
         LArgMain()  << EAM(aNameOut,"NameOut",true)
    );

    Tiff_Im aTif = Tiff_Im::StdConvGen(aNameIn,1,true);
    
    // Init Mems
    Pt2di aSz = aTif.sz();
    Im2D_U_INT2  anIm(aSz.x,aSz.y);

    Im1D_REAL8  aH(aNbH,0.0);


    // Load image
    {
       Symb_FNum aFTif(aTif.in());
       ELISE_COPY(aTif.all_pts(),aFTif,anIm.out()| (aH.histo().chc(aFTif)<<1) );
    }
    Im1D_REAL8  aHSqrt(aNbH,0.0);
    ELISE_COPY(aH.all_pts(),sqrt(aH.in()),aHSqrt.out());



    int aVMax =0;
    // Calcul Histo
    MakeFoncRepart(aH,&aVMax);
    MakeFoncRepart(aHSqrt);

    // Fonc Loc 

    Fonc_Num aFonc(Rconv(anIm.in_proj()));
    Fonc_Num aS1S2 = Virgule(aFonc,Square(aFonc));
    for (int aK=0 ; aK< TheNbIter ; aK++)
    {
        aS1S2 = rect_som(aS1S2,TheNbVois) / ElSquare(1+2*TheNbVois);
    }
    Symb_FNum aFS1 = aS1S2.v0();
    Symb_FNum aFS2 = aS1S2.v1()-Square(aFS1);
    Fonc_Num aFLoc = (anIm.in()-aFS1) / sqrt(Max(1.0,aFS2));

    // aFLoc = aFLoc * 20;
    aFLoc = atan(aFLoc) /(PI/2);

    
    // Symb_FNum aFonc(aTif.in());

    std::cout << "MaxMin "  << aVMax << "\n";

    Fonc_Num aFEg = aH.in()[anIm.in()];
    Fonc_Num aFEgS = aHSqrt.in()[anIm.in()];
    Fonc_Num aFM  = anIm.in() * (255.0 / aVMax);


    // Fonc_Num aFRes = (aFEg*aPEg + aFM * aPM + aFEgS*aPSrtEg + aFLoc) / (aPEg + aPM + aPSrtEg);

    Symb_FNum aFMoy = (aFEg*aPEg + aFM * aPM + aFEgS*aPSrtEg ) / (aPEg + aPM + aPSrtEg);
    Symb_FNum aFMarge = Min(64,Min(aFMoy,255-aFMoy));
    Fonc_Num aFRes  = aFMoy + aFMarge * aFLoc;


/*
    Fonc_Num aFRes = Max(0,Min(255,aFMoy + 60 * aFLoc));
*/

    Tiff_Im::Create8BFromFonc(aNameOut,aSz,aFRes);
    

     
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
