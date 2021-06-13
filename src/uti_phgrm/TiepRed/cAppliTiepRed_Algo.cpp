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


/******************************************************************************/
/*                                                                            */
/*                           cAppliTiepRed       (child  methods)             */
/*                                                                            */
/******************************************************************************/
// These methods are the ones executed by the child/children when executing a subcommand/task
// Each Subcommand has a master image and a list of related images (share tie-points with the master)

bool cAppliTiepRed::DoLoadTiePoints(){
	//For all images in the current Subcommand
	for (std::size_t i = 0 ; i< mImagesNames->size() ; i++) {
	   const std::string & imageName = (*mImagesNames)[i];
	   //Create instance of image class
	   cImageTiepRed * aImage= new cImageTiepRed(imageName);
	   aImage->SetImageId(i); //image id is the position of the image in the image subset of the current task
	   // Put them in vector
	   mImages.push_back(aImage);
	}

	// Keep reference of master image (first image in the image set of a task is the master image)
	cImageTiepRed & masterImage = *(mImages[0]);
	masterImage.SetImageBeenMaster(false);

	// We differentiate the related images by the whether they have been master in previously executed subccomands steps or not
	std::vector<std::string> homolImagesBeenMaster;
	std::vector<std::string> homolImagesNotBeenMaster;
	// If all related images have been already master before, all possible tie-point reduction has been done, in which case we do not have to do anything else
	bool stopProcessing = true;
	for (std::size_t i = 1 ; i<mImages.size() ; i++){
		cImageTiepRed & homolImage = *(mImages[i]);
		std::string aName = NameHomolTemp(masterImage.ImageName(), homolImage.ImageName());
		if (ELISE_fp::exist_file(aName)){ // If this file exists it means the homol image was a master before, so its related subcommand has been already executed
			homolImage.SetImageBeenMaster(true);
			homolImagesBeenMaster.push_back(homolImage.ImageName());
		}else{
			stopProcessing = false; // It at least one image has not been master before it means there is still tie-point to be deleted!
			homolImage.SetImageBeenMaster(false);
			homolImagesNotBeenMaster.push_back(homolImage.ImageName());
		}
	}

	// Log the images names
	std::cout << "Master Image: " << masterImage.ImageName() << endl;
	std::cout << "Related Images already processed: ";
	for (std::size_t i = 0 ; i<homolImagesBeenMaster.size() ; i++){
		std::cout << homolImagesBeenMaster[i] << " ";
	}
	std::cout << endl;
	std::cout << "Related Images not yet processed: ";
	for (std::size_t i = 0 ; i<homolImagesNotBeenMaster.size() ; i++){
		std::cout << homolImagesNotBeenMaster[i] << " ";
	}
	std::cout << endl;

	// initialize to 0 the variable that holds the maximum number of tie-points in an image pair
	mMaxNumHomol = 0;

	if (stopProcessing == false){
		// Load tie-points. For each image that is not the master image we load its shared tie-points with the master
		for (std::size_t i = 1 ; i<mImages.size() ; i++){
			// Get related image name
			cImageTiepRed & homolImage = *(mImages[i]);
			// Get the camera pair
			cLnk2ImTiepRed * imagePairLink ;
			if (masterImage.ImageName() > homolImage.ImageName()){
				std::pair<CamStenope*,CamStenope*> cameraPair = NM().CamOriRel(homolImage.ImageName(),masterImage.ImageName());
				// Create a image pair instance with initially no tie-points
				imagePairLink = new cLnk2ImTiepRed(&masterImage, &homolImage, &(*(cameraPair.second)), &(*(cameraPair.first)));
			}else{
				std::pair<CamStenope*,CamStenope*> cameraPair = NM().CamOriRel(masterImage.ImageName(),homolImage.ImageName());
				// Create a image pair instance with initially no tie-points
				imagePairLink = new cLnk2ImTiepRed(&masterImage, &homolImage, &(*(cameraPair.first)), &(*(cameraPair.second)));
			}

			// Get references to the empty list of tie-points in both images
			std::vector<Pt2df> & masterImageTiePoints = imagePairLink->VP1();
			std::vector<Pt2df> & homolImageTiePoints = imagePairLink->VP2();
			if (homolImage.ImageBeenMaster() == true){
				// If the homol image has been already a master image before we load the point from the result of a previous step
				std::string aName = NameHomolTemp(masterImage.ImageName(), homolImage.ImageName());
				ElPackHomologue aPack = ElPackHomologue::FromFile(aName);
				for (ElPackHomologue::iterator itPt = aPack.begin(); itPt != aPack.end(); itPt++ ){
					masterImageTiePoints.push_back(ToPt2df(itPt->P1()));
					homolImageTiePoints.push_back(ToPt2df(itPt->P2()));
				}
			}else{
				// Load tie-points from Homol folder
				NM().LoadHomFloats(masterImage.ImageName(), homolImage.ImageName(), &masterImageTiePoints, &homolImageTiePoints);
			}


			if (masterImageTiePoints.size() > (size_t)mMinNumHomol){
				// Update counter of tie-points in master
				masterImage.SetNbPtsHom2Im(masterImage.NbPtsHom2Im() + masterImageTiePoints.size());
				// Set number of tie-points for this homol image
				homolImage.SetNbPtsHom2Im(masterImageTiePoints.size());

				// Set the maximum number of homol points between a image-pair
				if (homolImage.NbPtsHom2Im() > mMaxNumHomol) mMaxNumHomol = homolImage.NbPtsHom2Im();

				// Add to image pair map
				mImagePairsMap.insert(make_pair(make_pair(masterImage.ImageId(), homolImage.ImageId()), imagePairLink));
			}
		}
	}else{
		std::cout << "#InitialHomolPoints:" << mNumInit << ". All related images already processed, nothing else to be done here!" << endl;
	}

	return stopProcessing;
}

void Verif(Pt2df aPf){
   Pt2dr aPd = ToPt2dr(aPf);
   if (std_isnan(aPd.x) || std_isnan(aPd.y))
   {
       std::cout << "PB PTS " << aPf << " => " << aPd << "\n";
       ELISE_ASSERT(false,"PB PTS in Verif");
   }
}

void cAppliTiepRed::DoExport(){
	// Counter for number of stored multi-tie-points
	int numMultiTiePoints = 0;
	// Number of images in the current task
	std::size_t aNbImage = mImages.size();
	// Create and fill a vector of calibration cameras (to convert tie-point coordinates from normalized to relative to each image)
	std::vector<CamStenope *> calibrationCameras;
	for (std::size_t i = 0 ; i< aNbImage ; i++) {
		const std::string & imageName = (*mImagesNames)[i];
		CamStenope * calibrationCamera = NM().CalibrationCamera(imageName);
		calibrationCameras.push_back(calibrationCamera);
	}

	// For each image pair we store the list of tie-points
	// We have to do it in two formats, one where the tie-points will be stored
	// in normalized format and the other in which they willl be stored in a coordinate system relative to each image
	std::vector<std::vector<ElPackHomologue> > aVVH (aNbImage, std::vector<ElPackHomologue>(aNbImage));
	std::vector<std::vector<ElPackHomologue> > aVVHTemp (aNbImage, std::vector<ElPackHomologue>(aNbImage));

	// Store the related tie-points of the multi-tie-points that have not been selected for removal
	std::cout << "Storing related tie-points of multi-tie-points with Gain:";
	for (int aKP=0 ; aKP<int(mMultiTiePoints.size()) ; aKP++){
		if (mMultiTiePoints[aKP]->Removed() == false){
			// If the multi-tie-point has not been selected for removal
			std::cout << ' ' << mMultiTiePoints[aKP]->Gain();
			// We get the edges. There is an edge for each image pair where a related tie-point was present
			// Note that only the image pairs where one of the images is the master are considered (the rest where not loaded)
      tMerge * aMerge = mMultiTiePoints[aKP]->MultiTiePointRaw();
      const std::vector<Pt2di> &  aVE = aMerge->Edges();
			// For each image pair where the multi-tie-point has a tie-point we add the tie-point to the list of tie-points of the imge pair
      for (std::size_t i=0 ; i<aVE.size() ; i++){
				// Get the ids of the images in this image pair
				int aKImage1 = aVE[i].x;
				int aKImage2 = aVE[i].y;
				// Get the coordinates of the tie-point in the image of the image pair (both positions are stored in two formats, float and real)
				Pt2df aP1 = aMerge->GetVal(aKImage1);
				Pt2df aP2 = aMerge->GetVal(aKImage2);
				Pt2dr aP1r = ToPt2dr(aP1);
				Pt2dr aP2r = ToPt2dr(aP2);

				// Verify the tie-point
				Verif(aP1);
				Verif(aP2);

				// We use the calibration cameras to convert the coordinates of the tie-point from normalized to each image coordinate system
				CamStenope * calibrationCamera1 = calibrationCameras[aKImage1];
				CamStenope * calibrationCamera2 = calibrationCameras[aKImage2];
				Pt2dr pointImage1 = calibrationCamera1->Radian2Pixel(aP1r);
				Pt2dr pointImage2 = calibrationCamera2->Radian2Pixel(aP2r);

				// Add the tie-point (converted) in the list of tie-points of the image pair
				// note that we store it in both Image1-Image2 and Image2-Image1 lists
				aVVH[aKImage1][aKImage2].Cple_Add(ElCplePtsHomologues(pointImage1,pointImage2));
				aVVH[aKImage2][aKImage1].Cple_Add(ElCplePtsHomologues(pointImage2,pointImage1));
				// Add the tie-point (normalized) in the list of tie-points of the image pair
				// note that we store it in both Image1-Image2 and Image2-Image1 lists
				aVVHTemp[aKImage1][aKImage2].Cple_Add(ElCplePtsHomologues(aP1r,aP2r));
				aVVHTemp[aKImage2][aKImage1].Cple_Add(ElCplePtsHomologues(aP2r,aP1r));

				// Increase the counter
				numMultiTiePoints++;
      }
		}
	}

	std::cout << endl;

	// For all the image pairs between the master image and the related ones we dump to files the list of tie-points
	for (std::size_t aKImage1=1 ; aKImage1<aNbImage ; aKImage1++){
		if (mImages[aKImage1]->ImageBeenMaster() == false){
			// Related images that were masters before can skip this step, because we know for sure no multi-tie-point has been deleted
			// (it is forbidden by the algorithm to delete a multi-tie-point that is present in an image that has already been master)

			// We get in both formats (normalized and relative) the image pairs between
			// the master and the related image in both orders (Master-Related and Related-Master) since both files need to be written
			const ElPackHomologue & aPack1 = aVVH[0][aKImage1];
			const ElPackHomologue & aPack2 = aVVH[aKImage1][0];
			const ElPackHomologue & aPack1Temp = aVVHTemp[0][aKImage1];
			const ElPackHomologue & aPack2Temp = aVVHTemp[aKImage1][0];

			// If there are tie-points we write to the related files
			if (aPack1.size()) aPack1.StdPutInFile(NameHomol(mImages[0]->ImageName(),mImages[aKImage1]->ImageName()));
			if (aPack2.size()) aPack2.StdPutInFile(NameHomol(mImages[aKImage1]->ImageName(),mImages[0]->ImageName()));
			// We always write the temp files, so next task knows these were processed already
			aPack1Temp.StdPutInFile(NameHomolTemp(mImages[0]->ImageName(),mImages[aKImage1]->ImageName()));
			aPack2Temp.StdPutInFile(NameHomolTemp(mImages[aKImage1]->ImageName(),mImages[0]->ImageName()));
		}
	}
	// Log the reduction
	std::cout << "#InitialHomolPoints:" << mNumInit << " #HomolPoints:" << mImages[0]->NbPtsHom2Im() <<  "(" <<  mMultiTiePointsRaw->size() << ")=>" << numMultiTiePoints << "(" << (mMultiTiePoints.size()-mNumDeleted) << ")\n";
}


void cAppliTiepRed::DoReduce(){
	// Load the tie-points
	std::cout << "Loading tie-points..." << endl;
	bool stopProcessing = DoLoadTiePoints();
	//stop processing indicates if all the related images have been already master in earlier executed tasks, in which case we can exit

	if (stopProcessing == false){
		// In this moment all the tie-points are loaded
		// Now we need to create the multi-tie-points, topologically merging the tie-points of all image pairs
		std::cout << "Creating merged structures..." << endl;
		// Create an empty merging struct
		mMergeStruct = new tMergeStr(mImages.size(),true);
		// For each image pairs add all the tie-points to the merging structure
		typedef std::map<pair<int, int>, cLnk2ImTiepRed * >::const_iterator it_type;
		for(it_type iterator = mImagePairsMap.begin(); iterator != mImagePairsMap.end(); iterator++) {
			// Get image pair
			cLnk2ImTiepRed * imagePairLink = iterator->second;
			// Get indentifiers of the images in the image pair
			int aKImage1 =  (imagePairLink->Image1()).ImageId();
			int aKImage2 =  (imagePairLink->Image2()).ImageId();
			// Get the list of tie-points of both images
			std::vector<Pt2df>& vP1 = imagePairLink->VP1();
			std::vector<Pt2df>& vP2 = imagePairLink->VP2();
			// Add all tie-points to merging structure
			for (std::size_t aKP=0 ; aKP<vP1.size() ; aKP++){
				// Add elementary connection
				mMergeStruct->AddArc(vP1[aKP],aKImage1,vP2[aKP],aKImage2,cCMT_NoVal());
			}
		}

		// "Compile" to make structure usable
		mMergeStruct->DoExport();
		// Get the list of raw multi-tie-points
		mMultiTiePointsRaw =  & mMergeStruct->ListMerged();
		// The raw multi-tie-point contains the positions of the related tie-points in the different images where they are present

		// Initialize the grids
		std::cout << "Initializing grids..." << endl;
		std::size_t numCells = mNumCellsX * mNumCellsY;
		std::vector<cImageGrid*> imageGridVec;
		// All the tie-points loaded are normalized in a unit box, so our grid spce is also the same box
		Box2dr box = Box2dr(Pt2dr( -1, -1), Pt2dr( 1, 1));
		int nX = mNumCellsX;
		int nY = mNumCellsY;
		// Images with few related images will use grids with more cells (max factor is 2)
		if (mAdaptive == true){
			double ft = std::min(sqrt((double)(mMaxNumRelated / (mImages.size()-1))), 2.0);
			nX = std::floor(ft * mNumCellsX);
			nY = std::floor(ft * mNumCellsY);
		}
		// For each image we create a grid
		for (std::size_t i = 0 ; i<mImages.size() ; i++){
			int ngX = nX;
			int ngY = nY;
			if (mAdaptive == true && i > 0){
				cImageTiepRed & homolImage = *(mImages[i]);
				double fr = std::min(sqrt((double)(mMaxNumHomol / homolImage.NbPtsHom2Im())),2.0);
				ngX = std::floor(fr * nX);
				ngY = std::floor(fr * nY);
			}
			cImageGrid* ig = new cImageGrid(mImages[i], box, ngX, ngY, mImages.size());
			imageGridVec.push_back(ig);
		}

		// The list of accuracies of the multi-tie-points
		std::vector<double> aVAcc;

		// We reserve space for the multi-tie-points (same size as the raw multi-tie-points)
		mMultiTiePoints.reserve(mMultiTiePointsRaw->size());

		// Iterate over all the raw multi-tie-points, creating a multi-tie-point
		// for each raw multi-tie-point and adding it in the grids of the images (where related tie-points are present)
		std::cout << "Filling grids..." << endl;
		for (std::list<tMerge *>::const_iterator itMultiTiePointRaw=mMultiTiePointsRaw->begin() ; itMultiTiePointRaw!=mMultiTiePointsRaw->end() ; itMultiTiePointRaw++){
			// Create the multi-tie-point instance
			cPMulTiepRed * aPM = new cPMulTiepRed(*itMultiTiePointRaw, *this);
			// Add it to th elist of multi-tie-points
			mMultiTiePoints.push_back(aPM);
			// Add the accuracy of this multi-tie-point to the lsit with all accuracies
			aVAcc.push_back(aPM->Acc());
			// Get the indices of the images where this multi-tie-point is present
			const std::vector<cPairIntType<Pt2df> > & vecInd = (*itMultiTiePointRaw)->VecIT();
			for (std::size_t i = 0; i < vecInd.size(); i++){
				// For each image where the point is present we add it into the grid (Add method adds it already in the proper cell of the grid)
				imageGridVec.at(vecInd[i].mNum)->Add(aPM);
			}
		}

		// Compute the median accuracy
		if (aVAcc.size() == 0) {
		  return;
		}
		mStdAcc = MedianeSup(aVAcc);

		// The gain can be computed once we know the standard accuracy
		for (int aKP=0 ; aKP<int(mMultiTiePoints.size()) ; aKP++){
			 mMultiTiePoints[aKP]->InitGain(*this);
		}

		//For the master image we sort points in grid cell. They are sorted according to gain
		cImageGrid * masterImageGrid = imageGridVec[0];
		std::cout << "Sorting master image grid..." << endl;
		masterImageGrid->SortCells();
		std::cout << "Reducing multi-tie-points from master image grid cells..." << endl;
		mNumDeleted = 0;
		// We iterate over all the cells of the grid of the master image
		for (std::size_t cellIndex = 0; cellIndex < numCells ; cellIndex++){
			// Ideally we want to remove all the points in the cell except the one with the highest Gain
			// If the cells only has a point we are done
			if (masterImageGrid->NumPointsCell(cellIndex) > 1){
				// Get the points in the cell
				std::vector<cPMulTiepRed *> & cellPoints = masterImageGrid->CellPoints(cellIndex);
				// For all the points except the first (the one with highest Gain) and we start with the last
				for (std::size_t j = cellPoints.size()-1u; j > 0; j--){
					// Get the multi-tie-point
					cPMulTiepRed * mp = cellPoints[j];
					// We check in which images this point is present. It should be visible in 0 (the master image) and at least anther image
					const std::vector<cPairIntType<Pt2df> > & vecInd = mp->MultiTiePointRaw()->VecIT();
					// We initialize removable to true
					bool removable = true;
					// For all the images where the point is present
					for (std::size_t k = 0; k < vecInd.size(); k++){
						INT imageIndex = vecInd[k].mNum;
						//We do not have to check for the master image, only for the others
						if (imageIndex != 0){
							// Get the grid of the related image
							cImageGrid * relatedImageGrid = imageGridVec[imageIndex];
							// We can only remove points if the related image has not been master before
							if (relatedImageGrid->Image().ImageBeenMaster() == false){
								// Get how many tie-points are in the grid cell of the image where the related tie-point of this multi-tie-point is present
								int numPointsRelImageCell = relatedImageGrid->NumPointsCell(relatedImageGrid->CellIndex(mp->MultiTiePointRaw()->GetVal(imageIndex)));
								// Get the number of points shared with the image specified by imageIndex in the current cell of the master image grid
								int numPointsMasterImageCellRelImage = masterImageGrid->NumPointsCellImage(cellIndex, imageIndex);
								if (numPointsRelImageCell == 1 || numPointsMasterImageCellRelImage == 1){
									// If the cell in the related image where the point lays only has 1 point
									// or if the number of points in the master image cell related to the related image is only 1
									// -> we should not delete it, so we do not need to check other images and we can go to next point
									removable = false;
									break;
								}
							}else{
								// The related image has been master before
								removable = false;
								break;
							}
						}
					}
					// Remove the point if this is ok, i.e. if for all the images where the point is present, by removing the point we do not leave any grid cell empty (which was not already empty)
					if (removable == true){
						masterImageGrid->Remove(mp, cellIndex);
						mNumDeleted++;
					}
				}
			}
		}
		// We can remove the grids
		for (std::size_t i = 0 ; i<mImages.size() ; i++){
			delete imageGridVec[i];
		}

		// Export the points. Write a new Homol-Red folder with the selected homol points between the master image and its related images
		DoExport();
	}
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
