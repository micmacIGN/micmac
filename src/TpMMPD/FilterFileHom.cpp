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
#include "DIDRO/cimgeo.h"
#include "DIDRO/cero_modelonepaire.h"


/*
	I generate File of potential homologous images with GrapheHom but I want to filter these images pairs in order to remove some pai
*/

// classes declaration    Filtre du fichier ds Homologues
class cFFH_Appli;


class cFFH_Appli
{
    public :

        cFFH_Appli(int argc,char ** argv);
        void Filter();

    private :

        std::string mFileHom;
        std::string mPat1;
        std::string mPat2;
        std::string mOut;
       
        cInterfChantierNameManipulateur * mICNM;


        std::list<std::string>  mLFile1;
        std::list<std::string>  mLFile2;
        cElemAppliSetFile mEASF;
       
};


cFFH_Appli::cFFH_Appli(int argc,char ** argv) :
      mPat1 	("img.*.tif"),
      mPat2 	("img.*.tif"), // meaning; I do not want any image pairs like img...  img...
      mOut		("FileHomFiltred.xml")
{
    ElInitArgMain
    (
    argc,argv,
        LArgMain()  << EAMC(mFileHom,"XML-File of pair", eSAM_IsExistFile)
                    << EAMC(mPat1,"Images pattern 1", eSAM_IsPatFile)
                    << EAMC(mPat2,"Images pattern 2", eSAM_IsPatFile),
        LArgMain()  << EAM(mOut,"Out", true,"xml file, default: FileHomFiltred.xml")
    );

      
    if (!MMVisualMode)
    {

	cTplValGesInit<std::string>  aTplFCND;
    mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,"./",aTplFCND);
    
    //mEASF.Init(mFileHom);
    
    mLFile1 =  mICNM->StdGetListOfFile(mPat1,1);
	mLFile2 =  mICNM->StdGetListOfFile(mPat2,1);
	
	// voir comment mm3d mmbyPair ouvre le fichier de paire pour l'exemple
	 cSauvegardeNamedRel aSNR = StdGetFromPCP(mFileHom,SauvegardeNamedRel);
	 cSauvegardeNamedRel aRel;
	 int aNb1(0);
	 int aNb2(0);
	 
	for
     (
         std::vector<cCpleString>::const_iterator itC=aSNR.Cple().begin();
         itC!=aSNR.Cple().end();
         itC++
     )	
     {
        aNb1++;
        
        // si la paire est du type "PAT1"  "PAT2" on la met de coté, on en veut pas.
        	if (std::find(std::begin(mLFile1), std::end(mLFile1), itC->N1()) != std::end(mLFile1)) {
				if (std::find(std::begin(mLFile2), std::end(mLFile2), itC->N2()) != std::end(mLFile2)) {
				aNb2++;
				continue;
			}
			}
			aRel.Cple().push_back(*itC);
     }
	
	
    MakeFileXML(aRel,mOut);
    std::cout << aNb2 << " image pairs have been removed from the " << mFileHom << ", " << aNb1-aNb2 << " pairs remain. \n";
	
	}
}


int FilterFileHom_main(int argc,char ** argv)
{
    cFFH_Appli aFFH(argc,argv);
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

A cet égard  l'attention de l'ucApplitilisateur est attirée sur les risques
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
