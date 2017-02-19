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


/**********************************************************************/
/*                                                                    */
/*                            cPMulTiepRed                            */
/*                                                                    */
/**********************************************************************/

cPMulTiepRed::cPMulTiepRed(tMerge * aMultiTiePointRaw, cAppliTiepRed & anAppli)  :
	mMultiTiePointRaw (aMultiTiePointRaw),
	mAcc(0),
	mGain(0),
	mRemoved (false)
{
	// If Gain is 1, we need to compute the accuracy of this multi-tie-point
	// The accuracy of a multi-tie-point is computed as the worse (highest) accuracy of the related tie-points.
	// And the accuracy of a related tie-point is computed from the relative orientation of the image pair
  if (anAppli.GainMode() == 1){
		// Create list of accuracies for the related tie-points
		std::vector<double> accuracies;
		// Get the list of images where the multi-tie-point has related tie-points
		const std::vector<cPairIntType<Pt2df> >  &  aVecInd = mMultiTiePointRaw->VecIT() ;
		// we get accuracy for all the image pais between the master and each of the images where the multi-tie-point has related tie-points
		for (int i=0 ; i<int(aVecInd.size()) ; i++){
			if (aVecInd[i].mNum != 0){
				double acc;
				cLnk2ImTiepRed * imagePair = anAppli.ImagePairsMap()[std::make_pair(0,aVecInd[i].mNum)];
				// if (&(imagePair->Cam1())==0){
				//    ELISE_ASSERT(false,"NUL CAMERA POINTER");
				// }
				(imagePair->Cam1()).PseudoInterPixPrec(ToPt2dr(mMultiTiePointRaw->GetVal(0)),imagePair->Cam2(),ToPt2dr(mMultiTiePointRaw->GetVal(aVecInd[i].mNum)),acc);
				accuracies.push_back(acc);
			}
		}
		mAcc = *(std::max_element(accuracies.begin(), accuracies.end()));
	}
}


void  cPMulTiepRed::InitGain(cAppliTiepRed & anAppli){
	if (anAppli.GainMode() == 0){
		mGain = mMultiTiePointRaw->NbArc();
	}else{
		mGain = mMultiTiePointRaw->NbArc() * (1.0 /(1.0 + ElSquare((anAppli.WeightAccGain() * mAcc)/anAppli.StdAcc())));
	}
}


bool cPMulTiepRed::Removed() const
{
   return mRemoved;
}


void cPMulTiepRed::Remove()
{
    mRemoved = true;
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
