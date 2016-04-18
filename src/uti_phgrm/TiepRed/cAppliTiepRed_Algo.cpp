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

#include "TiepRed.h"

#if (!BUG_PUSH_XML_TIEP)


/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed       (Subcommand methods)     */
/*                                                                    */
/**********************************************************************/

// These methods are the ones executed by the Subcommand
// Each Subcommand has a master image and a list of images that tie homol points with the master

bool cAppliTiepRed::DoLoadTiePoints()
{
	//For all images in the current Subcommand
	for (std::size_t i = 0 ; i< mImagesNames->size() ; i++) {
	   const std::string & imageName = (*mImagesNames)[i];
	   //Create instance of image class
	   cImageTiepRed * aImage= new cImageTiepRed(imageName);
	   aImage->SetImageId(i); //image id is the position of the image in the current image subset
	   // Put them in vector
	   mImages.push_back(aImage);
	}

	// Keep reference of master image
	cImageTiepRed & masterImage = *(mImages[0]);
	masterImage.SetImageBeenMaster(false);

	// We differentiate the homol images by the whether they have been master in previously executed subccomands steps or not
	std::vector<std::string> homolImagesBeenMaster;
	std::vector<std::string> homolImagesNotBeenMaster;
	// If all homol images have been already master before, all possible tie point reduction has been done, in which case we do not have to do anything else
	bool stopProcessing = true;

	for (std::size_t i = 1 ; i<mImages.size() ; i++){
		cImageTiepRed & homolImage = *(mImages[i]);
		std::string aName = NameHomolTemp(masterImage.ImageName(), homolImage.ImageName());
		if (ELISE_fp::exist_file(aName)){ // If this file exists it means the homol image was a master before, so its related subcommand has been already executed
			homolImage.SetImageBeenMaster(true);
			homolImagesBeenMaster.push_back(homolImage.ImageName());
		}else{
			stopProcessing = false; // It at least one image has not been master before it means there is still tie point to be deleted!
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

	mMaxNumHomol = 0;

	if (stopProcessing == false){
		// Load Tie Points. For each image that is not the master image we load its shared tie points with the master
		for (std::size_t i = 1 ; i<mImages.size() ; i++){
			cImageTiepRed & homolImage = *(mImages[i]);

			// Get the camera pair
			std::pair<CamStenope*,CamStenope*> cameraPair = NM().CamOriRel(masterImage.ImageName(),homolImage.ImageName());

			// Create a connection with initially no tie points
			cLnk2ImTiepRed * imagePairLink = new cLnk2ImTiepRed(&masterImage, &homolImage, &(*(cameraPair.first)), &(*(cameraPair.second)));

			// Get references to the empty list of tie points in both images
			std::vector<Pt2df> & masterImageTiePoints = imagePairLink->VP1();
			std::vector<Pt2df> & homolImageTiePoints = imagePairLink->VP2();
			//std::vector<double> & imagePairAccuacies = imagePairLink->Acc();
			if (homolImage.ImageBeenMaster() == true){ // If the homol image has been already a master image before we load the point from the result of a previous step
				std::string aName = NameHomolTemp(masterImage.ImageName(), homolImage.ImageName());
				ElPackHomologue aPack = ElPackHomologue::FromFile(aName);
				for (ElPackHomologue::iterator itPt = aPack.begin(); itPt != aPack.end(); itPt++ ){
					masterImageTiePoints.push_back(ToPt2df(itPt->P1()));
					homolImageTiePoints.push_back(ToPt2df(itPt->P2()));
				}
			}else{
				// Load tie points from Homol folder
				NM().LoadHomFloats(masterImage.ImageName(), homolImage.ImageName(), &masterImageTiePoints, &homolImageTiePoints);
			}

//			for (std::size_t j = 0; j<numImagePairTiePoints; j++){
//				double acc;
//				cameraPair.first->PseudoInterPixPrec(ToPt2dr(masterImageTiePoints[j]),*(cameraPair.second),ToPt2dr(homolImageTiePoints[j]),acc);
//				imagePairAccuacies.push_back(acc);
//			}

			// Update counter of tie-points in master
			masterImage.SetNbPtsHom2Im(masterImage.NbPtsHom2Im() + masterImageTiePoints.size());
			// Set number of tie-points for this homol image
			homolImage.SetNbPtsHom2Im(masterImageTiePoints.size());

			// Set the maximum number of homol points between a image-pair
			if (homolImage.NbPtsHom2Im() > mMaxNumHomol) mMaxNumHomol = homolImage.NbPtsHom2Im();

			// Add to image pair map
			mImagePairsMap.insert(make_pair(make_pair(masterImage.ImageId(), homolImage.ImageId()), imagePairLink));
		}
	}else{
		std::cout << "#InitialHomolPoints:" << mNumInit << ". All related images already processed, nothing else to be done here!" << endl;
	}

	return stopProcessing;
}


//void cAppliTiepRed::AddLnk(cLnk2ImTiepRed * imagePairLink){
//	mImagePairsLinks.push_back(imagePairLink);
//}

void Verif(Pt2df aPf)
{
   Pt2dr aPd = ToPt2dr(aPf);
   if (std_isnan(aPd.x) || std_isnan(aPd.y))
   {
       std::cout << "PB PTS " << aPf << " => " << aPd << "\n";
       ELISE_ASSERT(false,"PB PTS in Verif");
   }
}


void cAppliTiepRed::DoReduce()
{
    std::cout << "Loading tie points..." << endl;
    bool stopProcessing = DoLoadTiePoints();

    if (stopProcessing == false){
		// In this moment all the tie points are loaded and stored in the mImagePairsLinks
		// Now we need to create the multi-points, topologically merging the tie points of all image pairs
		std::cout << "Creating merged structures..." << endl;
		// Create an empty merging struct
		mMergeStruct = new tMergeStr(mImages.size(),true);
		// For each image pairs add all the tie points to the merging structure
		typedef std::map<pair<int, int>, cLnk2ImTiepRed * >::const_iterator it_type;
		for(it_type iterator = mImagePairsMap.begin(); iterator != mImagePairsMap.end(); iterator++) {

			cLnk2ImTiepRed * imagePairLink = iterator->second;
			// Get refernces to image indentifiers and tie points vectors
			int aKImage1 =  (imagePairLink->Image1()).ImageId();
			int aKImage2 =  (imagePairLink->Image2()).ImageId();
			std::vector<Pt2df>& vP1 = imagePairLink->VP1();
			std::vector<Pt2df>& vP2 = imagePairLink->VP2();
			// Add all tie points to merging structure
			for (std::size_t aKP=0 ; aKP<vP1.size() ; aKP++){
				// Add elementary connection
				mMergeStruct->AddArc(vP1[aKP],aKImage1,vP2[aKP],aKImage2);
			}
		}
		/*for (std::list<cLnk2ImTiepRed *>::const_iterator itImagePairLink=mImagePairsLinks.begin() ; itImagePairLink!=mImagePairsLinks.end() ; itImagePairLink++){
			// Get refernces to image indentifiers and tie points vectors
			int aKImage1 =  ((*itImagePairLink)->Image1()).ImageId();
			int aKImage2 =  ((*itImagePairLink)->Image2()).ImageId();
			std::vector<Pt2df>& vP1 = (*itImagePairLink)->VP1();
			std::vector<Pt2df>& vP2 = (*itImagePairLink)->VP2();
			// Add all tie points to merging structure
			for (std::size_t aKP=0 ; aKP<vP1.size() ; aKP++){
				// Add elementary connection
				mMergeStruct->AddArc(vP1[aKP],aKImage1,vP2[aKP],aKImage2);
			}
		}*/
		mMergeStruct->DoExport();                  // "Compile" to make the point usable
		mMergedHomolPointss =  & mMergeStruct->ListMerged();    // Get the merged multiple points
		// A multi-point is a structure that contains the positions of the point in the different images where it is present

		std::cout << "Initializing grids..." << endl;
		std::size_t numCells = mNumCellsX * mNumCellsY;
		std::vector<cImageGrid*> imageGridVec;
        // All the tie-points are normalized in a unit box
		Box2dr box = Box2dr(Pt2dr( -1, -1), Pt2dr( 1, 1));

		int nX = mNumCellsX;
		int nY = mNumCellsY;
		// Images with few related images will use grids with more cells (max factor is 2)
		if (mAdaptive == true){
			double ft = std::min(sqrt(mMaxNumRelated / (mImages.size()-1)), 2.0);
			nX = std::floor(ft * mNumCellsX);
			nY = std::floor(ft * mNumCellsY);
		}

		//Initialize the grids. We only keep counters for the first one, i.e. the master
		for (std::size_t i = 0 ; i<mImages.size() ; i++){
			int ngX = nX;
			int ngY = nY;
			if (mAdaptive == true && i > 0){
				cImageTiepRed & homolImage = *(mImages[i]);
				double fr = std::min(sqrt(mMaxNumHomol / homolImage.NbPtsHom2Im()),2.0);
				ngX = std::floor(fr * nX);
				ngY = std::floor(fr * nY);
			}
			cImageGrid* ig = new cImageGrid(mImages[i], box, ngX, ngY, mImages.size());
			imageGridVec.push_back(ig);
		}

		std::vector<double> aVAcc;

		std::cout << "Filling grids..." << endl;
		// Put points in each image in a grid
		// We iterate over all the multi-points
		for (std::list<tMerge *>::const_iterator itMergedHomolPoints=mMergedHomolPointss->begin() ; itMergedHomolPoints!=mMergedHomolPointss->end() ; itMergedHomolPoints++){
			// Create the multi-point instance
			cPMulTiepRed * aPM = new cPMulTiepRed(*itMergedHomolPoints, *this);
			mMultiPoints.push_back(aPM);

			aVAcc.push_back(aPM->Acc());

			// Get the indices of the images where this multi-point is present
			const std::vector<U_INT2> & vecInd = (*itMergedHomolPoints)->VecInd();
			for (std::size_t i = 0; i < vecInd.size(); i++){
				// For each image where the point is present we add it into the grid (Add method adds it already in the proper cell of the grid)
				imageGridVec.at(vecInd[i])->Add(aPM);
			}
		}

		if (aVAcc.size() ==0)
		{
		  return;
		}
		mStdAcc = MedianeSup(aVAcc);

		// The gain can be computed once we know the standard accuracy
		for (std::list<cPMulTiepRed *>::const_iterator itP=mMultiPoints.begin(); itP!=mMultiPoints.end();  itP++){
			(*itP)->InitGain(*this);
		}

		//For the master image we sort points in grid cell. They are sorted according to gain
		cImageGrid * masterImageGrid = imageGridVec[0];
		std::cout << "Sorting master image grid..." << endl;
		masterImageGrid->SortCells();
		std::cout << "Reducing multi-tie-points from master image grid cells..." << endl;
		mNumDeleted = 0;
		// We iterate over all the cells of the grid of the master image
		for (std::size_t cellIndex = 0; cellIndex < numCells ; cellIndex++){
			// Ideally we want to remove all the points in the cell except the one which is most important (the one which appers in most images)
			if (masterImageGrid->NumPointsCell(cellIndex) > 1){ // If the cells only has a point we are done
				std::vector<cPMulTiepRed *> & cellPoints = masterImageGrid->CellPoints(cellIndex); // Get the points in the cell
				for (std::size_t j = cellPoints.size()-1u; j > 0; j--){ // For all the points except the first (the most important) and we start with the least important point
					cPMulTiepRed * mp = cellPoints[j];
					const std::vector<U_INT2> & vecInd = mp->MergedHomolPoints()->VecInd(); // We check in which images this point is present. It should be visible in 0 (the master image) and at least anther image
					bool removable = true;
					for (std::size_t k = 0; k < vecInd.size(); k++){ // For all the images where the point is present
						U_INT2 imageIndex = vecInd[k];
						if (imageIndex != 0){ //We do not have to check for the master image, only for the others
							// Get how many points are in the grid cell where the point is present in the image
							cImageGrid * relatedImageGrid = imageGridVec[imageIndex];
							if (relatedImageGrid->Image().ImageBeenMaster() == false){ // We can only remove points if the related image has not been master before
								int numPointsRelImageCell = relatedImageGrid->NumPointsCell(relatedImageGrid->CellIndex(mp->MergedHomolPoints()->GetVal(imageIndex)));
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

void cAppliTiepRed::DoExport()
{

	int numHomolPoints = 0;

	std::size_t aNbImage = mImages.size();

	std::vector<CamStenope *> calibrationCameras;
	for (std::size_t i = 0 ; i< aNbImage ; i++) {
		   const std::string & imageName = (*mImagesNames)[i];
		   CamStenope * calibrationCamera = NM().CalibrationCamera(imageName);
		   calibrationCameras.push_back(calibrationCamera);
	}

    std::vector<std::vector<ElPackHomologue> > aVVH (aNbImage,std::vector<ElPackHomologue>(aNbImage));
    std::vector<std::vector<ElPackHomologue> > aVVHTemp (aNbImage,std::vector<ElPackHomologue>(aNbImage));

    std::cout << "Storing multi-tie-points with gain:";

    for (std::list<cPMulTiepRed *>::const_iterator itP=mMultiPoints.begin(); itP!=mMultiPoints.end();  itP++)
    {
    	if ((*itP)->Removed() == false){

    		std::cout << ' ' << (*itP)->Gain();

            tMerge * aMerge = (*itP)->MergedHomolPoints();
            const std::vector<Pt2dUi2> &  aVE = aMerge->Edges();
            for (std::size_t i=0 ; i<aVE.size() ; i++)
            {
                 int aKImage1 = aVE[i].x;
                 int aKImage2 = aVE[i].y;

                 Pt2df aP1 = aMerge->GetVal(aKImage1);
                 Pt2df aP2 = aMerge->GetVal(aKImage2);
                 Pt2dr aP1r = ToPt2dr(aP1);
                 Pt2dr aP2r = ToPt2dr(aP2);

                 Verif(aP1);
                 Verif(aP2);

                 CamStenope * calibrationCamera1 = calibrationCameras[aKImage1];
                 CamStenope * calibrationCamera2 = calibrationCameras[aKImage2];

                 Pt2dr pointImage1 = calibrationCamera1->Radian2Pixel(aP1r);
                 Pt2dr pointImage2 = calibrationCamera2->Radian2Pixel(aP2r);

                 aVVH[aKImage1][aKImage2].Cple_Add(ElCplePtsHomologues(pointImage1,pointImage2));
                 aVVH[aKImage2][aKImage1].Cple_Add(ElCplePtsHomologues(pointImage2,pointImage1));

                 aVVHTemp[aKImage1][aKImage2].Cple_Add(ElCplePtsHomologues(aP1r,aP2r));
                 aVVHTemp[aKImage2][aKImage1].Cple_Add(ElCplePtsHomologues(aP2r,aP1r));

                 numHomolPoints++;
            }
    	}
    }

    std::cout <<endl;

    for (std::size_t aKImage1=1 ; aKImage1<aNbImage ; aKImage1++)
    {
		if (mImages[aKImage1]->ImageBeenMaster() == false){

			 const ElPackHomologue & aPack1 = aVVH[0][aKImage1];
			 const ElPackHomologue & aPack2 = aVVH[aKImage1][0];
			 const ElPackHomologue & aPack1Temp = aVVHTemp[0][aKImage1];
			 const ElPackHomologue & aPack2Temp = aVVHTemp[aKImage1][0];

			 if (aPack1.size())
			 {
				  aPack1.StdPutInFile(NameHomol(mImages[0]->ImageName(),mImages[aKImage1]->ImageName()));
			 }
			 if (aPack2.size())
			 {
				  aPack2.StdPutInFile(NameHomol(mImages[aKImage1]->ImageName(),mImages[0]->ImageName()));
			 }
			 if (aPack1Temp.size())
			 {
				  aPack1Temp.StdPutInFile(NameHomolTemp(mImages[0]->ImageName(),mImages[aKImage1]->ImageName()));
			 }
			 if (aPack2Temp.size())
			 {
				  aPack2Temp.StdPutInFile(NameHomolTemp(mImages[aKImage1]->ImageName(),mImages[0]->ImageName()));
			 }
		}

    }


    std::cout << "#InitialHomolPoints:" << mNumInit << " #HomolPoints:" << mImages[0]->NbPtsHom2Im() <<  "(" <<  mMergedHomolPointss->size() << ")=>" << numHomolPoints << "(" << (mMultiPoints.size()-mNumDeleted) << ")\n";
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
