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

#include "NewOri.h"

extern cNewO_NameManager * NM(const std::string & aDir);

/*
int TestNewOriImage_main(int argc,char ** argv)
{
   std::string aNameOri,aNameI1,aNameI2;
   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(aNameI1,"Name First Image")
                   <<  EAMC(aNameI2,"Name Second Image"),
        LArgMain() << EAM(aNameOri,"Ori",true,"Orientation ")
   );


    cNewO_NameManager aNM("./",aNameOri,"dat");

    CamStenope * aC1 = aNM.CamOfName(aNameI1);
    CamStenope * aC2 = aNM.CamOfName(aNameI2);

    ElPackHomologue aLH = aNM.PackOfName(aNameI1,aNameI2);

    std::cout << "FFF " << aC1->Focale() << " " << aC2->Focale() << " NBh : " << aLH.size() << "\n";

    return EXIT_SUCCESS;
}
*/

///Export the graph to G2O format for testing in ceres
int NewOriImage2G2O_main(int argc,char ** argv)
{
    std::string aPat,aDir;
    std::string aName="triplets_g2o.txt";

    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aPat,"Pattern of images", eSAM_IsExistFile),
        LArgMain() << EAM(aName,"Out",true,"Output file name")
    );

   #if (ELISE_windows)
        replace( aPat.begin(), aPat.end(), '\\', '/' );
   #endif

    SplitDirAndFile(aDir,aPat,aPat);

    ///ori dir manager
    cInterfChantierNameManipulateur * aICNM;
    aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        

    ///triplets dir manager
    cNewO_NameManager *  aNM = NM(aDir);
    std::string aNameLTriplets = aNM->NameTopoTriplet(true);
    cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);

    std::vector<ElRotation3D> aPVec;
    for (auto a3 : aLT.Triplets())
    {
        std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
        cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
        //aPVec.push_back(aXml3Ori);

        /*ElRotation3D aPose = Xml2El(aXmlOri.Ori2On1()) -> ElRotation3D
    mOri (aPose.ImAff(Pt3dr(0,0,0))),
    mI   (aPose.ImVect(Pt3dr(1,0,0))),
    mJ   (aPose.ImVect(Pt3dr(0,1,0))),
    mK   (aPose.ImVect(Pt3dr(0,0,1)))
*/

    }
    fstream aFp;
    aFp.open(aName.c_str(), ios::out);

    return EXIT_SUCCESS;
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
