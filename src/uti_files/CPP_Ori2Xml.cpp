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


/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/


int Ori2XML_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);

    std::string aFullOri,anOut;
    std::string toto;


    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aFullOri,"Full pattern") 
                      << EAMC(anOut,"Dir for result"),
           LArgMain() << EAM(toto,"toto",true)
    );

    std::string aDir,aFileOriIn;
    SplitDirAndFile(aDir,aFileOriIn,aFullOri);

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

   std::string aKeyOut = "NKS-Assoc-Im2Orient@-" + anOut;


    const std::vector<std::string> * aVName = anICNM->Get(aFileOriIn);

    for (int aK=0 ; aK<int(aVName->size()) ; aK++)
    {
        std::string aNameIn = (*aVName)[aK];
        std::string aNameFile = StdPrefix(aNameIn) + ".tif";

        CamStenope * aCam = CamStenope::StdCamFromFile(true,aNameIn,anICNM);
        // CamStenope * aCam = CamOrientGenFromFile(aNameIn,anICNM);


        std::string aNameOut = anICNM->Assoc1To1(aKeyOut,aNameFile,true);
        std::cout << "FFF= " << aCam->Focale() << " " << aNameOut  << "\n";
        //cCalibrationInternConique  aCIO = aCam->ExportCalibInterne2XmlStruct(aCam->Sz());
        cOrientationConique anOC = aCam->StdExportCalibGlob();

        MakeFileXML(anOC,aDir+aNameOut);
    }

/*
    CamStenope * aCam = CamStenope::StdCamFromFile(true,aFileOriIn,anICNM);

    std::cout << "FFF= " << aCam->Focale() << "\n";


    if (aFileOriOut=="")
    {
        aFileOriOut =StdPrefix(aFileOriIn) + ".xml";
    }

    Ori3D_Std anOri(aFileOriIn.c_str());
    if (aCal=="")
    {
       ElRotation3D aR = anOri.GetOrientation();
       XML_SauvFile(aR,aFileOriOut,"Ori2XML",true);
    }
    else
    {
        CamStenope * aCS=0;
        aCS = Std_Cal_From_File(aCal);


       aCS->SetOrientation(anOri.GetOrientation().inv());
       cOrientationConique anOC = aCS->ExportCalibGlob(anOri.SzIm(),1,1,3,true,(char *)0);

       MakeFileXML(anOC,aFileOriOut);
    }
*/

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
