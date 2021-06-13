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

#include "TiepTri/MultTieP.h"

typedef std::vector<Pt2dr> 				FeatureKeypoints;
typedef std::vector<std::pair<int,int>> FeatureMatches;
class cAppliOMVG
{
	public:
		cAppliOMVG(int argc, char** argv);

		void Save();

	private:
		 cInterfChantierNameManipulateur * mICNM;
		
		 std::string mPattern;
		 std::string mSH;
		 std::string mHomExp;
		 


		 template <typename T>
         void FileReadOK(FILE *fptr, const char *format, T *value);



};

void cAppliOMVG::Save()
{
	cSetTiePMul * aSPM = new cSetTiePMul(0);

	cElemAppliSetFile   anEASF(mPattern);
    aSPM->SetFilter(*(anEASF.SetIm()));


    const std::vector<std::string> * aVFileH = cSetTiePMul::StdSetName(mICNM,mSH,(mHomExp=="dat" ) ? 1 : 0 );

	//read the PMul files
	for (auto F : *aVFileH)
	{
		std::cout << F << "\n";
		aSPM->AddFile(F);
	}
	std::vector<cSetPMul1ConfigTPM *> aVSPM = aSPM->VPMul();

	//structure that will save features
    std::map<std::string,FeatureKeypoints* > aMap_Features;

	//structure that will save matches ({I,J},{FeatureId_I,FeatureId_J})
    std::map<std::pair<int,int>,FeatureMatches* > aMap_Matches;

	//ierate and save
	for (auto aConfig : aVSPM)
	{

		//vector of images in this config (track length)
		std::vector<int> aVIm = aConfig->VIdIm();


		//nb of Pts in this config
		int NbPts = aConfig->NbPts(); 

		for (int aPt=0; aPt<NbPts; aPt++)
		{
			
			std::cout << "=====Pt " << aPt << "\n";

			//add features
			for (auto ImId : aVIm)
			{
				std::string aImName = aSPM->NameFromId(ImId);

				if (aMap_Features.find(aImName) == aMap_Features.end())
					aMap_Features[aImName] = new FeatureKeypoints;

				FeatureKeypoints* aFeatures = aMap_Features[aImName];

				Pt2dr aPtCoord = aConfig->GetPtByImgId(aPt,ImId);

				aFeatures->push_back(aPtCoord);
			    std::cout << "ImId " << ImId << "\n";

			}	

			//add matches
			int NbMul = int(aVIm.size());
			for (int I=0; I<NbMul; I++)
            {
				
				std::string aIName = aSPM->NameFromId(aVIm[I]);
				int FeatureId_I = aMap_Features[aIName]->size()-1;

				for (int J=(I+1); J<NbMul; J++)
            	{

					std::string aJName = aSPM->NameFromId(aVIm[J]);
					int FeatureId_J = aMap_Features[aJName]->size()-1;

				
					if (0)
						std::cout << aIName << "-" << aJName << " I,J " << aVIm[I] << " " << aVIm[J] 
								  << ", F" << FeatureId_I << " " << FeatureId_J << "\n";
					
					if (aMap_Matches.find({aVIm[I],aVIm[J]}) == aMap_Matches.end())
						aMap_Matches[{aVIm[I],aVIm[J]}] = new FeatureMatches;

					FeatureMatches* aMatches = aMap_Matches[{aVIm[I],aVIm[J]}];
					aMatches->push_back(std::make_pair(FeatureId_I,FeatureId_J));

				}
			
			}

		}

	}

	//save to files 
	//feat file
	for (auto aImage : aMap_Features)
	{

		std::string FeatureName = StdPrefix(aImage.first.c_str()) + ".feat";
		std::ofstream file(FeatureName.c_str());
		for(auto aFeature : (*aImage.second))
		{
			file << aFeature.x << " " << aFeature.y << " 0 4\n";
		}
		file.close();

	}
	//desc file empty
	    for (auto aImage : aMap_Features)
    {

        std::string FeatureName = StdPrefix(aImage.first.c_str()) + ".desc";
        std::ofstream file(FeatureName.c_str());
        file.close();

    }


	std::ofstream match_file("match.f.txt");
	for (auto aMatch : aMap_Matches)
	{
		std::cout << aSPM->NameFromId(aMatch.first.first) << " " << aMatch.first.first << "\n";

		int NbMul = aMatch.second->size();
		match_file << aSPM->NameFromId(aMatch.first.first) << " " << aSPM->NameFromId(aMatch.first.second) << "\n"
				   << NbMul << "\n";

		for (int aP=0; aP<NbMul; aP++)
		{
			match_file << aMatch.second->at(aP).first << " " <<  aMatch.second->at(aP).second << "\n";
		}
				   
	}
	match_file.close();

}

cAppliOMVG::cAppliOMVG(int argc, char** argv) :
		mSH(""),
		mHomExp("dat")
{

	bool aExpTxt=false;


	ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(mPattern,"Pattern of images"),
        LArgMain() << EAM(mSH,"SH",true,"Homol Postfix")
				   << EAM(aExpTxt,"ExpTxt","Input PMul in txt format")
    );


	aExpTxt ? mHomExp="txt" : mHomExp="dat";
	
    #if (ELISE_windows)
        replace( mPattern.begin(), mPattern.end(), '\\', '/' );
    #endif
	
	mICNM = cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(mPattern));


}



int CPP_MM2OpenMVG_main(int argc, char** argv)
{
	cAppliOMVG anApp(argc,argv);
	anApp.Save();

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
