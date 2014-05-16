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

/******************************/
/*	   Author: Luc Girod	  */
/******************************/

#include "StdAfx.h"
#include "hassan/reechantillonnage.h"

void Drunk_Banniere()
{
    std::cout <<  "\n";
    std::cout <<  " *********************************\n";
    std::cout <<  " *     D-istortion               *\n";
    std::cout <<  " *     R-emoving                 *\n";
    std::cout <<  " *     UN-iversal                *\n";
    std::cout <<  " *     K-it                      *\n";
    std::cout <<  " *********************************\n\n";
}

void Drunk(string aFullPattern,string aOri,string DirOut, bool Talk)
{
    string aPattern,aNameDir;
    SplitDirAndFile(aNameDir,aPattern,aFullPattern);

    //Reading input files
    list<string> ListIm=RegexListFileMatch(aNameDir,aPattern,1,false);
    int nbIm=ListIm.size();
    if (Talk){cout<<"Images to process: "<<nbIm<<endl;}

    //Paralelizing (an instance of Drunk is called for each image)
    string cmdDRUNK;
    list<string> ListDrunk;
    if(nbIm!=1)
    {
        for(int i=1;i<=nbIm;i++)
        {
            string aFullName=ListIm.front();
            ListIm.pop_front();
            cmdDRUNK=MMDir() + "bin/Drunk " + aNameDir + aFullName + " " + aOri + " Out=" + DirOut + " Talk=0";
            ListDrunk.push_back(cmdDRUNK);
        }
        cEl_GPAO::DoComInParal(ListDrunk,aNameDir + "MkDrunk");

        //Calling the banner at the end
        if (Talk){Drunk_Banniere();}
    }else{

    //Bulding the output file system
    ELISE_fp::MkDirRec(aNameDir + DirOut);

    //Processing the image
    string aNameIm=ListIm.front();
    string aNameOut=aNameDir + DirOut + aNameIm + ".tif";

    //Loading the camera
    string aNameCam="Ori-"+aOri+"/Orientation-"+aNameIm+".xml";
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aNameDir);
    CamStenope * aCam = CamOrientGenFromFile(aNameCam,anICNM);

    //Reading the image and creating the objects to be manipulated
    Tiff_Im aTF= Tiff_Im::StdConvGen(aNameDir + aNameIm,3,false);

    Pt2di aSz = aTF.sz();

    Im2D_U_INT1  aImR(aSz.x,aSz.y);
    Im2D_U_INT1  aImG(aSz.x,aSz.y);
    Im2D_U_INT1  aImB(aSz.x,aSz.y);
    Im2D_U_INT1  aImROut(aSz.x,aSz.y);
    Im2D_U_INT1  aImGOut(aSz.x,aSz.y);
    Im2D_U_INT1  aImBOut(aSz.x,aSz.y);

    ELISE_COPY
    (
       aTF.all_pts(),
       aTF.in(),
       Virgule(aImR.out(),aImG.out(),aImB.out())
    );

    U_INT1 ** aDataR = aImR.data();
    U_INT1 ** aDataG = aImG.data();
    U_INT1 ** aDataB = aImB.data();
    U_INT1 ** aDataROut = aImROut.data();
    U_INT1 ** aDataGOut = aImGOut.data();
    U_INT1 ** aDataBOut = aImBOut.data();

    //Parcours des points de l'image de sortie et remplissage des valeurs
    Pt2dr ptOut;
    for (int aY=0 ; aY<aSz.y  ; aY++)
    {
        for (int aX=0 ; aX<aSz.x  ; aX++)
        {
            ptOut=aCam->DistDirecte(Pt2dr(aX,aY));

            aDataROut[aY][aX] = Reechantillonnage::biline(aDataR, aSz.x, aSz.y, ptOut);
            aDataGOut[aY][aX] = Reechantillonnage::biline(aDataG, aSz.x, aSz.y, ptOut);
            aDataBOut[aY][aX] = Reechantillonnage::biline(aDataB, aSz.x, aSz.y, ptOut);

        }
    }

    Tiff_Im  aTOut
             (
                  aNameOut.c_str(),
                  aSz,
                  GenIm::u_int1,
                  Tiff_Im::No_Compr,
                  Tiff_Im::RGB
             );


     ELISE_COPY
     (
         aTOut.all_pts(),
         Virgule(aImROut.in(),aImGOut.in(),aImBOut.in()),
         aTOut.out()
     );
    }
}

int Drunk_main(int argc,char ** argv)
{

    //Testing the existence of argument (if not, print help file)
    if(!MMVisualMode && argc==1)
    {
        argv[1]=(char*)"";//Compulsory to call MMD_InitArgcArgv
        MMD_InitArgcArgv(argc,argv);
        string cmdhelp;
        cmdhelp=MMDir()+"bin/Drunk -help";
        system_call(cmdhelp.c_str());
    }
    else
    {
        MMD_InitArgcArgv(argc,argv);

        string aPattern,aOri;
        string DirOut="DRUNK/";
        bool Talk=true;

        //Reading the arguments
        ElInitArgMain
        (
            argc,argv,
            LArgMain()  << EAMC(aPattern,"Images Pattern", eSAM_IsPatFile)
                        << EAMC(aOri,"Orientation name", eSAM_IsExistDirOri),
            LArgMain()  << EAM(DirOut,"Out",true,"Output folder (end with /) and/or prefix (end with another char)")
                        << EAM(Talk,"Talk",true,"Turn on-off commentaries")
                    );

        //Processing the files
        Drunk(aPattern,aOri,DirOut,Talk);
    }

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
