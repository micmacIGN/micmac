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

typedef cVarSizeMergeTieP<Pt2df>  tMerge; // Class to store a multi-point
typedef cStructMergeTieP<tMerge>  tMergeStr; // Class to store a bunch of multi-points

// Class to store multi-points pointers in the grid in which we divide the image-space of an image
// for each multi-point we use its position in the current image
class cImageGrid {
	public:
		cImageGrid(cImageTiepRed * aImage, Box2dr & aBox , int aNumCellsX, int aNumCellsY, int aNumImages); // Constructor. Initializes the grid
		cImageTiepRed &     Image(); // Gets reference to image
		void Add(cPMulTiepRed * aMultiPoint); // Adds a (point of) multi-point in the cell where it lays (actually the position in current image of the multi-point)
		void SortCells(); // Sort the multi-points in the cells according to their gain
		int CellIndex(const Pt2df & aPoint); // Gets the cell index for a given aPoint
		int NumPointsCell(int aCellIndex); // Gets the number of points in a cell given its cellIndex
		int NumPointsCellImage(int aCellIndex, int aImageId); // Gets the number of points in a cell for a certain related image index given its cellIndex
		std::vector<cPMulTiepRed *> & CellPoints(int aCellIndex); // Gets the vector of (point to) multi-points in a cell given its cellIndex
		void Remove(cPMulTiepRed * aMultiPoint, int aCellIndex); // Remove the multipoint from the cell index
	private:
		cImageTiepRed *                           mImage; // Image
		Box2dr                                    mBox; // Image-space of the image (always p0=(-1,-1) p1=(1,1))
		int                                       mNumCellsX; // Number of cells in x dimension of image-space
		int                                       mNumCellsY; // Number of cells in y dimension of image-space
		std::vector<std::vector<cPMulTiepRed *> > mCellsPoints; // vector of grid cells. For each cell we have the vector of (pointers to) multi-points in it.
		std::vector<std::vector<int> > mCellsImageCounters; // vector of grid cells. For each cell we have a vector of counters that count how many points come from which image
};


// Class to handle the list of tie points between two images (image pair)
class cLnk2ImTiepRed
{
     public :
		cLnk2ImTiepRed(cImageTiepRed * aImage1,cImageTiepRed * aImage2, CamStenope * aCam1, CamStenope * aCam2); // Constructor from pointers to the two images and two cameras
		cImageTiepRed &     Image1(); // Gets reference to image 1
		cImageTiepRed &     Image2(); // Gets reference to image 2
		CamStenope &     Cam1(); // Gets reference to camera 1
		CamStenope &     Cam2(); // Gets reference to camera 2
        std::vector<Pt2df>&  VP1(); // Gets reference to vector of tie points in image 1
        std::vector<Pt2df>&  VP2(); // Gets reference to vector of tie points in image 2
//        std::vector<double>& Acc(); // Gets the reference to vector of accuracies
     private :
        cImageTiepRed *    mImage1; //reference to image 1
        cImageTiepRed *    mImage2; //reference to image 2
        CamStenope * mCam1; // reference to camera 1
        CamStenope * mCam2; // reference to camera 2
        std::vector<Pt2df> mVP1; //vector of tie points in image 1
        std::vector<Pt2df> mVP2; //vector of tie points in image 2
//        std::vector<double> mAcc; //vector of accuracy of the tie-points of this image pair
};

// Class to handle an image
class cImageTiepRed
{
    public :
		cImageTiepRed(const std::string & aImageName); // Constructor. From reference of image name
        const std::string ImageName() const; // Get the image name
        void SetNbPtsHom2Im(int aNbPtsHom2Im); // Sets the number of tie points
        const int &   NbPtsHom2Im() const; // Get the number of tie points with this image
        void SetImageId(int aImageId); // Sets the image identifier
        const int & ImageId() const; // Gets the image identifier
        void SetImageBeenMaster(bool aImageBeenMaster); // Sets the ImageBeenMaster
        const bool & ImageBeenMaster() const; // Gets the ImageBeenMaster
    private :
        cImageTiepRed(const cImageTiepRed &); // Not Implemented
        std::string     mImageName; // Image name
        int             mNbPtsHom2Im; // Number of tie points with this image
        int             mImageId; // Image identifier (within current subset)
        bool 		    mImageBeenMaster; // Indicates if image has been master before
};

// Class to handle a multi-point
class cPMulTiepRed
{
     public :
	   cPMulTiepRed(tMerge * aMergedHomolPoints, cAppliTiepRed & anAppli); // Constructor. From the merged tie points structure pointer
	   const double  & Acc() const {return mAcc;}
       const double  & Gain() const {return mGain;} // Gets the (const) Gain
       double  & Gain() {return mGain;} // Gets the Gain
       tMerge * MergedHomolPoints() {return mMergedHomolPoints;} // Gets the pointer to the merged tie points structure
       void InitGain(cAppliTiepRed & anAppli);
       bool Removed() const; // Gets bool indicating if the point is selected for removal
       void Remove(); // Sets this point for its removal
     private :
       tMerge * mMergedHomolPoints; // Reference to the merged tie points structure
       double   mAcc; // Accuracy of the multi-tie-point (worst accuracy of the related tie-points)
       double   mGain;  // Gain to select this tie points (takes into account multiplicity and precision)
       bool     mRemoved; //Indicates if multi-point is selected for removal
};

// Class for the application to reduce tie points between image pairs
class cAppliTiepRed
{
     public :
          cAppliTiepRed(int argc,char **argv); // Constructor
          void Exe(); // Executes the application
          cVirtInterf_NewO_NameManager & NM(); //Folder names manager
          const cXml_ParamSubcommandTiepRed & ParamSubcommand() const; //Structure to handle the Subcommand information from XML (only by children)
          const double & ThresholdAccMult() const;
          //void AddLnk(cLnk2ImTiepRed * imagePairLink); // Adds an image pair link
          std::map<pair<int, int>, cLnk2ImTiepRed * > & ImagePairsMap() {return mImagePairsMap;}
          int  & GainMode() {return mGainMode;}
          double & StdAcc() {return mStdAcc;}
          std::string NameHomol(const std::string &aName1,const std::string &aName2) const; //Gets the name of file that contains tie points of the given images names
          std::string NameHomolTemp(const std::string &aName1,const std::string &aName2) const; //Gets the name of file that contains tie points of the given images names in the format use by this app
     private :

          /* Generates the list of subcommands. For each subcommand it generates a configuration file.
           * For each image we have a subcommand. Each subcommand deals with the image associated to it, which is called master image,
           * and the images that share tie poins with the master image  (related images)
           * Executed by the parent process
           */
          void GenerateSubcommands();

          /*
           * Export the subcommands to a JSON file
           * Executed by the parent process
           */
          void ExportSubcommands(std::vector<std::string> & aVSubcommands , std::vector<std::vector< int > > & aVRelatedSubcommandsIndexes);

          /* Reduce the tie points of a master image and its related images
           * Executed within a subcommand.
           */
          void DoReduce();

          /*
           * Load the tie points between the master images and the related images
           * Executed within a subcommand.
           */
          bool DoLoadTiePoints();

          /*
           * Dumps to disk the list of reduced points, i.e. the ones not selected for removal.
           * Executed within a subcommand.
           */
          void DoExport();

          cAppliTiepRed(const cAppliTiepRed & anAppli); // N.I.

          /*
           * Simple getters
           */
          std::string DirOneImage(const std::string & aName) const; //Gets the output folder for the tie points of an image
          std::string DirOneImageTemp(const std::string & aName) const; //Gets the temporal folder for the tie points of an image
          std::string NameParamSubcommand(int aK, bool Bin) const; //Gets the name of a subcommand configuration file

          // VARIABLES OF THE APPLICATION
          static const std::string         TempFolderName; //Name of the temporal folder for the subcommands temporal data
          static const std::string         OutputFolderName; //Name of the output folder
          static const std::string         SubComFileName; //Name of the file with the subcommand
          std::string                      mDir; //Parent directory of the images
          std::string                      mPatImage; // Pattern of images to use
          int 							   mNumCellsX; // Target number of tie point per image in X
          int 							   mNumCellsY; // Target number of tie point per image in Y
          bool 							   mAdaptive; // Indicates if grids are adaptive (for image-pairs with less points grids with more cells are used)
          double mThresholdAccMult; // Threshold on accuracy for multiple points
          cVirtInterf_NewO_NameManager *   mNM ; // Folder name manager
          const std::vector<std::string> * mImagesNames; // Image names
          std::vector<cImageTiepRed *>     mImages; // Images
          bool                             mCallBack; //Indicates if current running process is a subcommand
          bool 							   mExpSubCom; //Indicates if the user wishes to export the subcommands instead of executing them
          bool 							   mExpTxt; // Indicates if the user wishes to export to TXT instead of binary
          bool 							   mSortByNum; // Indicates if the user wishes to sort the images by number of tie points instead of by file name
          bool							   mDesc; //Indicates if the user wishes to use descending order in sorting the images, instead of ascending
          int                              mSubcommandIndex; // Subcommand index (used by the subcommands only)
          int 							   mGainMode; // mode to compute the gain of a multi-tie-point
          int 							   mMaxNumRelated; // Maximum number of related images
          int							   mMaxNumHomol; // Maximum number of homol-points in a image-pair
          int 							   mNumInit; // Initial number of tie points in the master image (used by the subcommands only)
          int 							   mNumDeleted; // Number of deleted multi-tie-points (used by the subcommands only)
          cXml_ParamSubcommandTiepRed      mXmlParamSubcommand; // Structure with subcommand configuration info
          //std::list<cLnk2ImTiepRed *>      mImagePairsLinks; //Image pairs
          std::map<pair<int, int>, cLnk2ImTiepRed * > mImagePairsMap; //Image pairs map
          tMergeStr *                      mMergeStruct; //Merged structure to handle all tie points and image pairs
          const std::list<tMerge *> *      mMergedHomolPointss;//List of multi-tie-points (without gain and removing flag)
          std::list<cPMulTiepRed* >        mMultiPoints; // List of multi-tie-points (with gain and removing flag)
          double                           mStdAcc;
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

