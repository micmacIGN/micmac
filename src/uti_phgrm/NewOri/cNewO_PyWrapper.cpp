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

#include "NewO_PyWrapper.h"

RelMotion::RelMotion(cNewO_NameManager *aNM,std::string& aN1,std::string& aN2,std::string& aN3) :
	mCam1(0),
	mCam2(0),
	mCam3(0)
{
    std::string  aName3R = aNM->NameOriOptimTriplet(true,aN1,aN2,aN3);

    cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);


    //get the poses 
    ElRotation3D aP1 = ElRotation3D::Id ;
    ElRotation3D aP2 = Xml2El(aXml3Ori.Ori2On1());
    ElRotation3D aP3 = Xml2El(aXml3Ori.Ori3On1());


    mCam1 = aNM->CalibrationCamera(aN1)->Dupl();
    mCam2 = aNM->CalibrationCamera(aN2)->Dupl();
    mCam3 = aNM->CalibrationCamera(aN3)->Dupl();
	
    //update poses 
    mCam1->SetOrientation(aP1.inv());
    mCam2->SetOrientation(aP2.inv());
    mCam3->SetOrientation(aP2.inv());

}

std::vector<RelMotion> RelMotionsPyWrapper(const std::string& aImPat,const std::string& aSH,const std::string& aDir,const std::string& InCal)
{
	
	std::vector<RelMotion> aRes;

    bool aExpTxt = false;


    //file managers
    cElemAppliSetFile anEASF(aDir+aImPat);
    const std::vector<std::string> * aSetName = anEASF.SetIm();
    int aNbIm = (int)aSetName->size();
    std::cout << "Images no: " << aNbIm << "\n";




    // map to read triplets in pattern
    std::map<std::string,int> aNameMap;
    for (int aK=0; aK<aNbIm; aK++)
        aNameMap[aSetName->at(aK)] = aK;

    cNewO_NameManager * aNM = new cNewO_NameManager("",aSH,true,aDir,InCal,aExpTxt ? "txt" : "dat");

    //triplets
    std::string aNameLTriplets = aNM->NameTopoTriplet(true);

    cXml_TopoTriplet aLT;
    if (ELISE_fp::exist_file(aNameLTriplets))
    {
        aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);

        std::cout << "Triplet no: " << aLT.Triplets().size() << "\n";
    }

    for (auto a3 : aLT.Triplets())
    {
        //verify that the triplet images are in the pattern
        if ( DicBoolFind(aNameMap,a3.Name1()) &&
             DicBoolFind(aNameMap,a3.Name2()) &&
             DicBoolFind(aNameMap,a3.Name3()) )
        {
            aRes.push_back(RelMotion(aNM,a3.Name1(),a3.Name2(),a3.Name3()));
        }


    }

	return aRes;
}

int CPP_RelMotionTest_main(int argc,char ** argv)
{
	std::cout << "Snooping around? This is yet another ewelinas test;\n";


    std::string aImPat;
    std::string aDir;
    std::string InCal;
    std::string aSH="";


    ElInitArgMain
    (
        argc, argv,
        LArgMain() << EAMC(aDir,"Working directory")
                   << EAMC(aImPat,"Pattern of images"),
        LArgMain() << EAM (InCal,"Calib",true,"Input calibration")
                   << EAM (aSH,"SH",true,"Homol postfix, Def=false")
    );


	std::vector<RelMotion> aRMVec = RelMotionsPyWrapper(aImPat,aSH,aDir,InCal);

	for (auto it_motion : aRMVec)
	{
		std::cout << "+ " << it_motion.Cam1().Focale() << "\n";
	}

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
