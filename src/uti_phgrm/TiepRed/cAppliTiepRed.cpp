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
/*
The RedTieP tool has been developed by Oscar Martinez-Rubi within the project
Improving Open-Source Photogrammetric Workflows for Processing Big Datasets
The project is funded by the Netherlands eScience Center
*/

#include "TiepRed.h"

#if (!BUG_PUSH_XML_TIEP)


bool cmpStringDesc(const pair<std::string, int>  &p1, const pair<std::string, int> &p2)
{
    return p1.first > p2.first;
}
bool cmpStringAsc(const pair<std::string, int>  &p1, const pair<std::string, int> &p2)
{
    return p1.first < p2.first;
}
bool cmpIntDesc(const pair<std::string, int>  &p1, const pair<std::string, int> &p2)
{
    return p1.second > p2.second;
}
bool cmpIntAsc(const pair<std::string, int>  &p1, const pair<std::string, int> &p2)
{
    return p1.second < p2.second;
}

/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed                              */
/*                                                                    */
/**********************************************************************/
cAppliTiepRed::cAppliTiepRed(int argc,char **argv)  :
  mNumCellsX(12),
  mNumCellsY(12),
  mAdaptive(false),
  mWeightAccGain(0.5),
  mImagesNames(0),
  mCallBack(false),
  mExpSubCom(false),
  mExpTxt(false),
  mSortByNum(false),
  mDesc(false),
  mGainMode(1),
  mMinNumHomol(20)
{
	// Read parameters
	MMD_InitArgcArgv(argc,argv);
	ElInitArgMain(argc,argv,
    LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile),
    LArgMain()  << EAM(mNumCellsX,"NumPointsX",true,"Target number of tie-points between 2 images in x axis of image space, def=12")
                << EAM(mNumCellsY,"NumPointsY",true,"Target number of tie-points between 2 images in y axis of image space, def=12")
                << EAM(mAdaptive,"Adaptive",true,"Use adaptive grids, def=false")
                << EAM(mSubcommandIndex,"SubcommandIndex",true,"Internal use")
                << EAM(mExpSubCom,"ExpSubCom",true,"Export the subcommands instead of executing them, def=false")
                << EAM(mExpTxt,"ExpTxt",true,"Export homol point in Ascii, def=false")
                << EAM(mSortByNum,"SortByNum",true,"Sort images by number of tie-points, determining the order in which the subcommands are executed, def=0 (sort by file name)")
                << EAM(mDesc,"Desc",true,"Use descending order in the sorting of images, def=0 (ascending)")
                << EAM(mWeightAccGain,"WeightAccGain",true,"Weight of median accuracy with respect to multiplicity (NumPairs) when computing Gain of multi-tie-point, i.e. K in formula Gain=NumPairs*(1/1 + (K*Acc/AccMed)^2) (if K=0 then Gain is NumPairs), def=0.5")
                << EAM(mMinNumHomol,"MinNumHomol",true,"Minimum number of tie-points for an image pair not to be excluded, def=20")
	);
	// if mSubcommandIndex was set, this is not the parent process.
  // This is child running a task/subcommand, i.e. a tie-point reduction task of a master image and its related images
	mCallBack = EAMIsInit(&mSubcommandIndex);
  //Get the parent directory of the images
	mDir = DirOfFile(mPatImage);
   //Initializes the folder name manager
	mNM = cVirtInterf_NewO_NameManager::StdAlloc(mDir, "");

  // Sets the Gain mode, if the weight is 0, then GainMode is 0. This means Gain = multiplicity (and we do not need to keep track of accuracies)
	if (mWeightAccGain == 0.){
		mGainMode = 0;
	}else{
		mGainMode = 1;
	}

	if (mCallBack){ //This is a child
		// We are running a subcommand. Read its configuration from a XML file given by the mSubcommandIndex
		mXmlParamSubcommand = StdGetFromPCP(NameParamSubcommand(mSubcommandIndex,true),Xml_ParamSubcommandTiepRed);
		// Read the images. mImagesNames[0] is the master image, the rest are related images to the master, i.e. they share tie-points
		mImagesNames = &(mXmlParamSubcommand.Images());
		// Get the initital number of tie-points that the master image had before any subcommand was executed
		mNumInit = mXmlParamSubcommand.NumInit();
    // Get the maximum number of related images for an image (this considers all the images, not only the ones related to this task)
		mMaxNumRelated = mXmlParamSubcommand.MaxNumRelated();
    // Get the number of subcommands/tasks (only for logging purposes)
		int numSubcommands = mXmlParamSubcommand.NumSubcommands();
		std::cout << "=======================   KSubcommand=" << (mSubcommandIndex+1) << "/" << numSubcommands << "  ===================\n";
	}
	else {
		// This is the parent. We get the list of images from the pattern provided by the user
		cElemAppliSetFile anEASF(mPatImage);
		mImagesNames = anEASF.SetIm();
	}
}

// Set some constants: temporal folder, output folder and JSON for commands (used for Noodles)
const std::string cAppliTiepRed::TempFolderName = "Tmp-ReducTieP-Pwork/";
const std::string cAppliTiepRed::OutputFolderName = "Homol-Red/";
const std::string cAppliTiepRed::SubComFileName = "subcommands.json";

/*
* Implement getters that depends on the previously defined constants
*/
std::string  cAppliTiepRed::NameParamSubcommand(int aK,bool Bin) const{
  return mDir+TempFolderName + "Param_" +ToString(aK) + (Bin ? ".xml" : ".dmp");
}
std::string  cAppliTiepRed::DirOneImage(const std::string &aName) const{
  return mDir+OutputFolderName + "Pastis" + aName + "/";
}
std::string  cAppliTiepRed::DirOneImageTemp(const std::string &aName) const{
  return mDir+TempFolderName + "Pastis" + aName + "/";
}
std::string  cAppliTiepRed::NameHomol(const std::string &aName1,const std::string &aName2) const{
  return DirOneImage(aName1) + aName2  + (mExpTxt ? ".txt" : ".dat");
}
std::string  cAppliTiepRed::NameHomolTemp(const std::string &aName1,const std::string &aName2) const{
  return DirOneImageTemp(aName1) + aName2  + (mExpTxt ? ".txt" : ".dat");
}

void cAppliTiepRed::ExportSubcommands(std::vector<std::string> & aVSubcommands , std::vector<std::vector< int > > & aVRelatedSubcommandsIndexes){
  // Opens the output file to write the subcommands
  ofstream scFile;
  std::string scFilePath = mDir+SubComFileName;
  scFile.open (scFilePath.c_str());

  //Write in JSON format
  scFile << "[" << endl;
  for (std::size_t i = 0 ; i < aVSubcommands.size() ; i++){
    scFile << "    {" << endl;
    scFile << "        \"id\": \"" << aVRelatedSubcommandsIndexes[i][0] << "\"," << endl;
    scFile << "        \"exclude\": [";

    for (std::size_t j = 1 ; j < aVRelatedSubcommandsIndexes[i].size() ; j++){
      scFile << aVRelatedSubcommandsIndexes[i][j];
      if (j < aVRelatedSubcommandsIndexes[i].size()-1) scFile << ",";
    }
    scFile << "]," << endl;
    scFile << "        \"command\": \"" << aVSubcommands[i] << "\"" << endl;
    scFile << "    }";
    if (i < aVSubcommands.size()-1) scFile << ",";
    scFile << endl;
  }
  scFile << "]" << endl;
  scFile.close();
}

void cAppliTiepRed::GenerateSubcommands(){
  // Create temp folder (remove it first to delete possible old data)
  ELISE_fp::PurgeDirGen(mDir+TempFolderName, true);
  ELISE_fp::MkDirSvp(mDir+TempFolderName);
  // Create the output folder
  ELISE_fp::MkDirSvp(mDir+OutputFolderName);
  // Create one subfolder for each image in the output and the temp folders
  for (std::size_t i = 0 ; i<mImagesNames->size() ; i++) {
     const std::string & aNameIm = (*mImagesNames)[i];
     ELISE_fp::MkDirSvp(DirOneImage(aNameIm));
     ELISE_fp::MkDirSvp(DirOneImageTemp(aNameIm));
  }

  // Fill a set of image names
  std::set<std::string>* mSetImagesNames = new std::set<std::string>(mImagesNames->begin(),mImagesNames->end());
  //Map of the names of the related images of for each image
  std::map<std::string,std::vector<string> > relatedImagesMap;
  //Map of number of tie-points per image (if a tie-point is in two image pairs, it counts as 2)
  std::map<std::string,int> imagesNumPointsMap;

	// Fill in the map with list of related images per image, and the map with number of tie-points per image
	for (std::size_t i = 0 ; i < mImagesNames->size() ; i++){ // for all images
    // Get the image name
		const std::string & imageName = (*mImagesNames)[i];
		// Get list of images sharing tie-points with imageName. We call these images related images
		std::list<std::string>  relatedImagesNames = mNM->ListeImOrientedWith(imageName);
		// For each related image we load the shared tie-points and update the maps
		for (std::list<std::string>::const_iterator itRelatedImageName= relatedImagesNames.begin(); itRelatedImageName!=relatedImagesNames.end() ; itRelatedImageName++){
      // Get the related image name
      const std::string & relatedImageName = *itRelatedImageName;
			// Test if the relatedImageName is in the initial set of images
			if (mSetImagesNames->find(relatedImageName) != mSetImagesNames->end()){
				if (imageName < relatedImageName){ //We add this if to guarantee we do not load the same tie-points when doing the iteration for the related image
					// Load the tie-points
					std::vector<Pt2df> aVP1,aVP2;
					mNM->LoadHomFloats(imageName, relatedImageName, &aVP1, &aVP2);
					// Update the related image names map
					relatedImagesMap[imageName].push_back(relatedImageName);
					relatedImagesMap[relatedImageName].push_back(imageName);
					// Update the number of tie-points map
					imagesNumPointsMap[imageName]+=aVP1.size();
					imagesNumPointsMap[relatedImageName]+=aVP2.size(); //should be same as aVP1
				}
			}
		}
	}
	// Get the maximum number of related images
	int maxNumRelated = 0;
	for (std::size_t i = 0 ; i < mImagesNames->size() ; i++){
		const std::string & imageName = (*mImagesNames)[i];
		int numRelated = relatedImagesMap[imageName].size();
		if 	(numRelated > maxNumRelated){
			maxNumRelated = numRelated;
		}
	}

	// Convert the map with the number of tie-points per image to a vector of pairs
	// This is required to sort them
	std::vector<pair<std::string, int> > imagesNumPointsVP;
	std::copy(imagesNumPointsMap.begin(), imagesNumPointsMap.end(), back_inserter(imagesNumPointsVP));
	// Sort the images by file name or by number or tie-points in ascending or descending order depending on user decision.
	if (mSortByNum == true){
		if (mDesc == false) std::sort(imagesNumPointsVP.begin(), imagesNumPointsVP.end(), cmpIntAsc);
		else std::sort(imagesNumPointsVP.begin(), imagesNumPointsVP.end(), cmpIntDesc);
	}else{
		if (mDesc == false) std::sort(imagesNumPointsVP.begin(), imagesNumPointsVP.end(), cmpStringAsc);
		else std::sort(imagesNumPointsVP.begin(), imagesNumPointsVP.end(), cmpStringDesc);
	}
	//Map containing for each image the order in the sorted list
	std::map<std::string,unsigned int> imagesOrderMap;
	for(std::size_t i = 0; i < imagesNumPointsVP.size(); ++i){
		imagesOrderMap[imagesNumPointsVP[i].first] = i;
	}
	// The list of subcommands
	std::vector<std::string> aVSubcommands;
  // The list of subcommands dependencies (others subcommands which can NOT run in parallel)
	std::vector<std::vector< int > > aVRelatedSubcommandsIndexes;

	// We generate a subcommand per image
	for(std::size_t imageIndex = 0; imageIndex < imagesNumPointsVP.size(); ++imageIndex){
		 // the SubcommandIndex is the position of the master image in the list of sorted images (imagesNumPointsVP)
		int subcommandIndex = imageIndex;

		// Get the master image name and the list of related images
		const std::string & masterImageName = imagesNumPointsVP[imageIndex].first;
		const std::vector<string> & relatedImages = relatedImagesMap[masterImageName];

    //Create a subcommand configuration structure
		cXml_ParamSubcommandTiepRed aParamSubcommand;
    // Create the list of related subcommands (commands that use as master images images used in the current subcommand)
		std::vector<int> relatedSubcommandsIndexes;

    // Fill-in the subcommand configuration structure
		aParamSubcommand.NumInit() = imagesNumPointsMap[masterImageName];
		aParamSubcommand.NumSubcommands() = static_cast<int>(imagesNumPointsVP.size());
		aParamSubcommand.Images().push_back(masterImageName); // Add master image to config
		relatedSubcommandsIndexes.push_back(static_cast<int>(imageIndex)); // Add the index of the current subcommand
		aParamSubcommand.MaxNumRelated() = maxNumRelated;

		for(std::size_t j = 0; j < relatedImages.size(); ++j){
			const std::string & relatedImageName = relatedImages[j];
			aParamSubcommand.Images().push_back(relatedImageName); // Add related image to config
			const unsigned int & relatedSubcommandIndex =  imagesOrderMap[relatedImageName]; // Find out in which subcommand the related image will be master
			relatedSubcommandsIndexes.push_back(relatedSubcommandIndex);
		}

		// Save the file to XML
		MakeFileXML(aParamSubcommand,NameParamSubcommand(subcommandIndex,false));
		MakeFileXML(aParamSubcommand,NameParamSubcommand(subcommandIndex,true));
		// Generate the command line to process this image
		std::string aSubcommand = GlobArcArgv + " SubcommandIndex=" + ToString(subcommandIndex);
		// add to list to be executed
		aVSubcommands.push_back(aSubcommand);
		aVRelatedSubcommandsIndexes.push_back(relatedSubcommandsIndexes);
	}


	if (mExpSubCom == false){
		// Execute the subcommands sequentially
		std::list<std::string> aLSubcommand(aVSubcommands.begin(), aVSubcommands.end());
		cEl_GPAO::DoComInSerie(aLSubcommand);
	}
	else ExportSubcommands(aVSubcommands, aVRelatedSubcommandsIndexes); // Export them to run them in parallel using Noodles
}


void  cAppliTiepRed::Exe()
{
   if (mCallBack) DoReduce(); //If this is a child, we execute the reducing algorithm
   else GenerateSubcommands(); //If this is the parent, we generate the subcommands
}


int RedTieP_main(int argc,char **argv){
  // Create the instance of the tool and executes it
	cAppliTiepRed * anAppli = new cAppliTiepRed(argc,argv);
	anAppli->Exe();
	return EXIT_SUCCESS;
}


#else
int RedTieP_main(int argc,char **argv){
   return EXIT_SUCCESS;
}
#endif
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
