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

//classe d'application
class cPI_Appli;

//structure d'un couple (camera,coord 2d)
struct CamCoord {
	CamStenope * Cam;
	Pt2dr coord2d;
};

//structure d'un couple [ nom Pt, vecteur (camera,coord 2d) ]
struct PtAllInfos {
	std::string nom;
	std::vector<CamCoord> CAC;
};

class cPI_Appli
{
     public :
          cPI_Appli(int argc,char ** argv);
          void ShowArgs();
          Pt3dr IntersectionFaisceaux(
			                          const std::vector<CamStenope *> & aVCS,
			                          const std::vector<Pt2dr> & aNPts2D
		);
     private :
          std::list<std::string> mLFile;
          std::string mFullName;
          std::string mDir; 			//dossier courant
          std::string mPat; 			//pattern des images
          std::string aOriIn;			//dossier qui contient les orientations des images
		  std::string a2dPtsFile;		//fichier .xml qui contient les pointes des images (avec SaisieAppuisInit)
          cInterfChantierNameManipulateur * mICNM;
};

cPI_Appli::cPI_Appli(int argc,char ** argv)
{
     bool aShowArgs=true;
     
     ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(mFullName,"Full Name (Dir+Pat)")
					 << EAMC(aOriIn, "Directory of input orientation",  eSAM_IsExistDirOri)
					 << EAMC(a2dPtsFile, ".xml file of 2d points", eSAM_IsExistFile),
          LArgMain() << EAM(aShowArgs,"Show",false,"Gives details on arguments")
     );

     SplitDirAndFile(mDir, mPat, mFullName);
     mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
     mLFile = mICNM->StdGetListOfFile(mPat);

    if (aShowArgs) ShowArgs();
    
    //vecteur de noms de caméras
    std::vector<string> OriFiles(mLFile.size());
    //vecteur de caméras
    std::vector<CamStenope *> aCamRep(mLFile.size());
    
    //initialisation du compteur
    int cmpt1=0;
    
    for (std::list<std::string>::iterator itS=mLFile.begin(); itS != mLFile.end(); itS ++)
    {
		//remplissage avec les noms des fichiers orientations
		OriFiles.at(cmpt1) = aOriIn+"Orientation-"+*itS+".xml";
		//génération des caméras à partir des fichiers d'orientations
		aCamRep.at(cmpt1) = CamOrientGenFromFile(OriFiles.at(cmpt1),mICNM);
		//incrémentation du compteur
		cmpt1++;
	}
    
    cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(a2dPtsFile,SetOfMesureAppuisFlottants);
    
    //autant que le nombre d'images dans le directory
    //std::cout << "NbMesures = " << aDico.MesureAppuiFlottant1Im().size( ) << std::endl;
    
    std::list<cMesureAppuiFlottant1Im> & aLMAF = aDico.MesureAppuiFlottant1Im();
    
    std::vector<ElSeg3D> aVSeg;
    
    std::vector<PtAllInfos> vPtsAI;
    
    std::vector<string> vNamePts;		//contient la liste des points à transformer en 3d
    
    for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
    {
		//std::cout << "*************" ;
		//std::cout << "NameIm : " << iT1->NameIm() << "\n";
		
		std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
		
		for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
		{
			//std::cout << "NamePt : " << iT2->NamePt() << "\n";
			std::string aNamePt = iT2->NamePt();
			vNamePts.push_back(aNamePt);
			//std::cout << "CoordIm : " << iT2->PtIm() << "\n";

		}
	}
	
	std::sort(vNamePts.begin() , vNamePts.end());		//attention ce sont des strings
	
	vNamePts.erase(std::unique(vNamePts.begin(), vNamePts.end()),vNamePts.end());	//on vire les doublons
	
	//boucle sur le nombre de points à projeter
	for (unsigned int aKNP=0; aKNP<vNamePts.size(); aKNP++)
	{
		PtAllInfos aPtsAL;
		
		std::vector<CamCoord> vCameraEtCoord;
		
		std::string namePt = vNamePts.at(aKNP);
		
		aPtsAL.nom = namePt;
		
		//boucle sur les images
		for (std::list<cMesureAppuiFlottant1Im>::iterator iT1 = aLMAF.begin() ; iT1 != aLMAF.end() ; iT1++)
		{
			std::list<cOneMesureAF1I> & aMes = iT1->OneMesureAF1I();
			
			//boucle sur tous les points saisis sur l'image courante
			for (std::list<cOneMesureAF1I>::iterator iT2 = aMes.begin() ; iT2 != aMes.end() ; iT2++)
			{
				
				//si je tombe sur le point courant je dois ajouter la camera + coord dans cette caméra
				if(vNamePts.at(aKNP) == iT2->NamePt())
				{
					std::string oriNameFile = aOriIn+"Orientation-"+iT1->NameIm()+".xml";
					CamStenope * cameraCourante = CamOrientGenFromFile(oriNameFile,mICNM);
					Pt2dr coordCourant = iT2->PtIm();
					CamCoord aCameraEtCoord;
					aCameraEtCoord.Cam = cameraCourante;
					aCameraEtCoord.coord2d=coordCourant;
					vCameraEtCoord.push_back(aCameraEtCoord);
				}
			}
		}
		aPtsAL.CAC = vCameraEtCoord;
		vPtsAI.push_back(aPtsAL);
	}
	
	for(unsigned int j=0; j<vPtsAI.size(); j++)
	{
		//std::cout << "NamePt : " << vPtsAI.at(j).nom << std::endl;
		//std::cout << "NuberCam : " << vPtsAI.at(j).CAC.size() << std::endl;
	}
	
	//le vecteur des points 3d à exporter
	std::vector<Pt3dr> Pts3d;
	
	//boucle sur le nombre de points à projeter en 3d
	for(unsigned int aHG=0 ; aHG<vPtsAI.size() ; aHG++)
	{
		//vecteur de caméras
		std::vector<CamStenope *> vCSPt;
		//vecteur de coordonnées 2d
		std::vector<Pt2dr> vC2d;
		
		for(unsigned int aHF=0 ; aHF<vPtsAI.at(aHG).CAC.size() ; aHF++)
		{
			CamStenope * aCSPtC = vPtsAI.at(aHG).CAC.at(aHF).Cam;
			vCSPt.push_back(aCSPtC);
			Pt2dr aCoordPtC = vPtsAI.at(aHG).CAC.at(aHF).coord2d;
			vC2d.push_back(aCoordPtC);
		} 
			Pt3dr aPt3d = IntersectionFaisceaux(vCSPt,vC2d);
			Pts3d.push_back(aPt3d);
	}
	
	//std::cout << "Number 3d pts : " << Pts3d.size() << std::endl;
	
	std::cout << "**************" << std::endl;
	std::cout.precision(5);
	std::cout << "fixed:\n" << std::fixed;
	
	for(unsigned int aVP=0; aVP<Pts3d.size(); aVP++)
	{
		std::cout << vPtsAI.at(aVP).nom << " " << Pts3d.at(aVP).x << " " << Pts3d.at(aVP).y << " " << Pts3d.at(aVP).z << "\n" ;
	}
	
	//reprojeter le point 3D dans l'image pour calculer un résidu image
	
}

void cPI_Appli::ShowArgs()
{
		  std::cout << "DIR = " << mDir << "\n";
		  std::cout << "Pat = " << mPat << "\n";
		  
          std::cout << "Nb Files " << mLFile.size() << "\n";
          for (
               std::list<std::string>::iterator itS=mLFile.begin();
               itS != mLFile.end();
               itS ++)
                    { std::cout << "     File = " << *itS << "\n"; }

}


Pt3dr cPI_Appli::IntersectionFaisceaux
	   (
			const std::vector<CamStenope *> & aVCS,
			const std::vector<Pt2dr> & aNPts2D
		)
{
	std::vector<ElSeg3D> aVSeg;	//vecteur d'éléments segments 3d
	
	for (int aKR=0 ; aKR < int(aVCS.size()) ; aKR++)
	{
		ElSeg3D aSeg = aVCS.at(aKR)->F2toRayonR3(aNPts2D.at(aKR));
		aVSeg.push_back(aSeg);
	}
	
	Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    return aRes;
}

int PseudoIntersect_main(int argc,char **argv)
{
    
     cPI_Appli anAppli(argc,argv);
     return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
