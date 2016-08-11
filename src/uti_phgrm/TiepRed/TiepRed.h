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

// Generate XML API
// ======================
//
// once you've added your class to the include/XML_GEN/ParamChantierPhotogram.xml file you need to execute this command (from micmac dir)
//
// make -f Makefile-XML2CPP  all
//
// it will generate all the necessary c++ definitions & declarations;
#ifndef _TiepRed_H_
#define _TiepRed_H_

#define BUG_PUSH_XML_TIEP 0

#include "StdAfx.h"

#if (!BUG_PUSH_XML_TIEP)

class cImageTiepRed;
class cPMulTiepRed;
class cAppliTiepRed;
class cLnk2ImTiepRed;
class cImageGrid;

// DEBUT MPD
class cXml_ParamSubcommandTiepRed;
/*
{
};
*/
// END MPD

/******************************************************************************/
/*                                                                            */
/*               Declaration of classes used by the RedTieP tool              */
/*                                                                            */
/******************************************************************************/

typedef cVarSizeMergeTieP<Pt2df,cCMT_NoVal>  tMerge; // Class for the merging to produce one raw multi-tie-point
typedef cStructMergeTieP<tMerge>  tMergeStr; // Class for the merging structure to produce all the raw multi-tie-points

// Class to store multi-tie-points in the grid in which we divide the image-space of an image
// for each multi-tie-point we use the position of the related tie-point in the current image
class cImageGrid {
	public:
		cImageGrid(cImageTiepRed * aImage, Box2dr & aBox , int aNumCellsX, int aNumCellsY, int aNumImages); // Constructor. Initializes the grid
		cImageTiepRed &  Image() {return *mImage;} // Gets image
		void Add(cPMulTiepRed * aMultiTiePoint); // Adds a multi-tie-point in the cell where it lays (actually the position in current image of the related tie-point)
		void SortCells(); // Sort the multi-tie-points in the cells according to their Gain
		int CellIndex(const Pt2df & aPoint); // Gets the cell index for a given image point
		int NumPointsCell(int aCellIndex); // Gets the number of points in a cell given its cellIndex
		int NumPointsCellImage(int aCellIndex, int aImageId); // Gets the number of points in a cell for a certain related image index given its cellIndex
		std::vector<cPMulTiepRed *> & CellPoints(int aCellIndex); // Gets the vector of (point to) multi-tie-points in a cell given its cellIndex
		void Remove(cPMulTiepRed * aMultiTiePoint, int aCellIndex); // Remove the multi-tie-point from the cell index
	private:
		cImageTiepRed * mImage; // Image
		Box2dr mBox; // Image-space of the image (always p0=(-1,-1) p1=(1,1))
		int mNumCellsX; // Number of cells in x dimension of image-space
		int mNumCellsY; // Number of cells in y dimension of image-space
		std::vector<std::vector<cPMulTiepRed *> > mCellsPoints; // vector of grid cells. For each cell we have the vector of multi-tie-points in it.
		std::vector<std::vector<int> > mCellsImageCounters; // vector of grid cells. For each cell we have a vector of counters that count how many points come from which image
};


// Class to handle an image pair
class cLnk2ImTiepRed {
	public :
		cLnk2ImTiepRed(cImageTiepRed * aImage1,cImageTiepRed * aImage2, CamStenope * aCam1, CamStenope * aCam2); // Constructor from the two images and two cameras
		cImageTiepRed & Image1(); // Gets image 1
		cImageTiepRed & Image2(); // Gets image 2
		CamStenope & Cam1(); // Gets camera 1
		CamStenope & Cam2(); // Gets camera 2
		std::vector<Pt2df>&  VP1(); // Gets vector of tie-points in image 1
		std::vector<Pt2df>&  VP2(); // Gets vector of tie-points in image 2
	private :
		cImageTiepRed *    mImage1; //image 1
		cImageTiepRed *    mImage2; //image 2
		CamStenope * mCam1; // camera 1
		CamStenope * mCam2; // camera 2
		std::vector<Pt2df> mVP1; //vector of tie-points in image 1
		std::vector<Pt2df> mVP2; //vector of tie-points in image 2
};

// Class to handle an image
class cImageTiepRed {
	public :
		cImageTiepRed(const std::string & aImageName); // Constructor. From image name
		const std::string ImageName() const; // Get the image name
		void SetNbPtsHom2Im(int aNbPtsHom2Im); // Sets the total number of tie-points of this iamge
		const int & NbPtsHom2Im() const; // Get the total number of tie-points of this image
		void SetImageId(int aImageId); // Sets the image identifier
		const int & ImageId() const; // Gets the image identifier
		void SetImageBeenMaster(bool aImageBeenMaster); // Sets the ImageBeenMaster
		const bool & ImageBeenMaster() const; // Gets the ImageBeenMaster
	private :
		cImageTiepRed(const cImageTiepRed &); // Not Implemented
		std::string mImageName; // Image name
		int mNbPtsHom2Im; // Total number of tie-points that this image shares with other images (including duplicates)
		int mImageId; // Image identifier (within current subcommand/task)
		bool mImageBeenMaster; // Indicates if image has been master before
};

// Class to handle a multi-tie-point
class cPMulTiepRed {
	public :
		cPMulTiepRed(tMerge * aMultiTiePointRaw, cAppliTiepRed & anAppli); // Constructor. From the raw multi-tie-point
		const double  & Acc() const {return mAcc;} // Gets the accuracy of the multi-tie-point
		const double  & Gain() const {return mGain;} // Gets the Gain
		double  & Gain() {return mGain;} // Gets the Gain
		tMerge * MultiTiePointRaw() {return mMultiTiePointRaw;} // Gets the raw multi-tie-point
		void InitGain(cAppliTiepRed & anAppli); // Sets the Gain value
		bool Removed() const; // Gets bool indicating if this multi-tie-point is selected for removal
		void Remove(); // Sets this multi-tie-point for removal
	private :
		tMerge * mMultiTiePointRaw; // Raw multi-tie-point
		double mAcc; // Accuracy of the multi-tie-point (worst accuracy of the related tie-points)
		double mGain;  // Gain of select this multi-tie-point (takes into account multiplicity and accuracy, which are weighted using WeightAccGain)
		bool mRemoved; //Indicates if this multi-tie-point is selected for removal
};

// Main class for the RedTieP tool (reduce tie-points between image pairs)
class cAppliTiepRed {
	public :
		cAppliTiepRed(int argc,char **argv); // Constructor
		void Exe(); // Executes the tool
		const double & WeightAccGain() const {return mWeightAccGain;} // Get the weight of multi-tie-point accuracy (with respect to the multiplicity) in the Gain formula (only used by children)
		std::map<pair<int, int>, cLnk2ImTiepRed * > & ImagePairsMap() {return mImagePairsMap;} // Gets the map of image pairs. The key is the pair of image ids within the current subcommand/task (only used by children)
		int  & GainMode() {return mGainMode;} // Get the Gain mode (0 is WeightAccGain; 1 otherwise). If Gain mode is 0, then Gain is multiplicity (only used by children)
		double & StdAcc() {return mStdAcc;}  // Gets the median of the accuracy of all the multi-tie-points in the current subcommand being executed (only used by children)
	private :

		/* Generates the list of subcommands/tasks. For each subcommand/task it generates a configuration file.
		* For each image we have a subcommand. Each subcommand deals with the image associated to it, which is called master image,
		* and the images that share tie poins with the master image  (related images)
		* (only used by parent)
		*/
		void GenerateSubcommands();

		/*
		* Export the subcommands to a JSON file
		* (only used by parent)
		*/
		void ExportSubcommands(std::vector<std::string> & aVSubcommands , std::vector<std::vector< int > > & aVRelatedSubcommandsIndexes);

		/* Reduce the tie-points in the image pairs of a master image and its related images
		* (only used by children)
		*/
		void DoReduce();

		/*
		* Load the tie-points in image pairs between the master images and the related images
		* (only used by children)
		*/
		bool DoLoadTiePoints();

		/*
		* Dumps to disk the list of reduced tie-points, i.e. the ones not selected for removal.
		* (only used by children)
		*/
		void DoExport();

		cAppliTiepRed(const cAppliTiepRed & anAppli); // N.I.

		/*
		* Simple getters
		*/
		cVirtInterf_NewO_NameManager & NM(){ return *mNM ;} // Folder names manager
		const cXml_ParamSubcommandTiepRed & ParamSubcommand() const {return mXmlParamSubcommand;} //Structure to handle the subcommands/tasks information from XML (only used by children)

		/*
		* Getters
		*/
		std::string NameHomol(const std::string &aName1,const std::string &aName2) const; // Gets the name of file that contains tie-points of a image pair given by the two images names
		std::string NameHomolTemp(const std::string &aName1,const std::string &aName2) const; // Gets the name of file that contains tie-points of a image pair in the format use by this tool (the coordinates are normalized)
		std::string DirOneImage(const std::string & aName) const; //Gets the output folder for the tie-points related to an image
		std::string DirOneImageTemp(const std::string & aName) const; //Gets the temporal folder for the tie-points related to an image
		std::string NameParamSubcommand(int aK, bool Bin) const; //Gets the name of a subcommand configuration file

		// Variables
		static const std::string TempFolderName; //Name of the temporal folder for the subcommands temporal data
		static const std::string OutputFolderName; //Name of the output folder
		static const std::string SubComFileName; //Name of the file with the subcommands
		std::string mDir; //Parent directory of the images
		std::string mPatImage; // Pattern of images to use
		int mNumCellsX; // Target number of tie-points (related to each image pair) per image in X
		int mNumCellsY; // Target number of tie-points (related to each image pair) per image in Y
		bool mAdaptive; // Indicates if grids are adaptive (for image-pairs with less points grids with more cells are used)
		double mWeightAccGain; // Weight of multi-tie-point accuracy (with respect to the multiplicity) in the Gain formula
		cVirtInterf_NewO_NameManager * mNM ; // Folder name manager
		const std::vector<std::string> * mImagesNames; // Image names
		std::vector<cImageTiepRed *> mImages; // Images
		bool mCallBack; //Indicates if current running process is a subcommand
		bool mExpSubCom; //Indicates if the user wishes to export the subcommands instead of executing them (this is required to run Noodles)
		bool mExpTxt; // Indicates if the user wishes to export to TXT instead of binary
		bool mSortByNum; // Indicates if the user wishes to sort the images by number of tie-points instead of by file name
		bool mDesc; //Indicates if the user wishes to use descending order in sorting the images, instead of ascending
		int mSubcommandIndex; // Subcommand index
		int mGainMode; // Mode to compute the Gain of a multi-tie-point
		int mMinNumHomol; // Minimum number of homol points for a pair not to be excluded
		int mMaxNumRelated; // Maximum number of related images for an image
		int	mMaxNumHomol; // Maximum number of tie-points in a image-pair (used for adaptive grids)
		int mNumInit; // Initial number of tie-points in the master image
		int mNumDeleted; // Number of deleted multi-tie-points
		cXml_ParamSubcommandTiepRed mXmlParamSubcommand; // Structure with subcommand configuration info
		std::map<pair<int, int>, cLnk2ImTiepRed * > mImagePairsMap; //Image pairs map
		tMergeStr * mMergeStruct; //Merging structure to obtain the list of raw multi-tie-points
		const std::list<tMerge *> * mMultiTiePointsRaw;//List of raw multi-tie-points (without gain and removing flag)
		std::vector<cPMulTiepRed* > mMultiTiePoints; // List of multi-tie-points (with gain and removing flag)
		double mStdAcc; // Median of the accuracy of all the multi-tie-points in the current subcommand being executed
};

#endif //
#endif // _TiepRed_H_
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
footer-MicMac-eLiSe-25/06/2007*/
