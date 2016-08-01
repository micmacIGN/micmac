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


#include "../../uti_phgrm/NewOri/NewOri.h"
#include "CorrelMesh.h"
#include "vector"
#include "algorithm"

CorrelMesh:: CorrelMesh(InitOutil *aChain)
{
    mChain = aChain;
    reloadTriandPic();
}

void CorrelMesh::reloadTriandPic()
{
    mPtrListPic = mChain->getmPtrListPic();
    mPtrListTri = mChain->getmPtrListTri();
}

pic* CorrelMesh::chooseImgMaitre(bool assum1er = true)
{
    if (assum1er)
        mPicMaitre = mPtrListPic[0];
    for (uint i=1; i<mPtrListPic.size(); i++)
        mListPic2nd.push_back(mPtrListPic[i]);
    return mPicMaitre;
}


//Pt2dr CorrelMesh::correlPtsInteretInImaget(Pt2dr ptInt1,
//                                           ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
//                                           matAffine & affine,
//                                           bool & foundMaxCorr,
//                                           double seuil_corel = 0.9)
//{
//    /*
//        |--------x-i--|
//        |-------------|
//        y-------------|
//        |-------------|
//        j-------------|
//    */
//    foundMaxCorr = false;
//    Pt2dr ptInt2;
//    cCorrelImage * P1 = new cCorrelImage();
//    P1->getFromIm(mPicMaitre->mPic_Im2D, ptInt1.x, ptInt1.y);
//    cCorrelImage * P2 = new cCorrelImage();
//    vector<double> corelScore;
//    int counter=0;
//    for (uint i=0; i<imgetMaitre.szW-P1->mSzW; i++)
//    {
//        counter ++;
//        for (uint j=0; j<imgetMaitre.szW-P1->mSzW; j++)
//        {
//            P2->getFromIm(imget2nd.imaget->getIm(), i, j);
//            double score = P1->CrossCorrelation(P2);
//            corelScore.push_back(score);
//        }
//    }
//    auto max_corel = std::max_element(std::begin(corelScore), std::end(corelScore));
//    int ind = distance(std::begin(corelScore), max_corel);
//    Pt2dr max_corel_coor(ind/counter ,  ind%counter);
//    /*std::cout << "Max element is " << *biggest
//        << " at position " << std::distance(std::begin(v), biggest) << std::endl;*/
//    if (*max_corel > seuil_corel)
//    {
//        ptInt2 = ApplyAffine(imgetMaitre.ptOriginImaget.operator +(max_corel_coor));
//        foundMaxCorr = true;
//    }
//    else
//    {
//        foundMaxCorr = false;
//    }
//    return ptInt2;
//}


///*=====Search for homol in specific triangle - search b/w PicM & all others pic========
// * indTri: index du triangle 3D sur le mesh
// * 1. reproject tri3D => 2 tri2D on image
// * 2. cherche les pack homol init => des couple imgM + imgs2nd
// * 2. Pour chaque couple :choisir un image maitraisse
// * 3. prendre imagette imget1 autour tri2D d'image maitraisse
// * 4. chercher affine b/w 2 tri2D
// * 5. prendre imagette imget2 autour tri2D d'image 2nd par affine
// * 6. correlation pour chaque pts d'interet dans imget1 avec pts dans imget2
// * 7. sort -> prendre pts correlation plus fort
//*/
//void CorrelMesh::correlInTri(int indTri)
//{
//    mSzW=3; //size correlation each pts interest
//    triangle * aTri = mPtrListTri[indTri];
//    this->chooseImgMaitre(true);    //assume 1er image est image maitre
//    Tri2d tri2DMaitre = *aTri->getReprSurImg()[mPicMaitre->mIndex];
//    ImgetOfTri imgetMaitre = aTri->create_Imagette_autour_triangle(mPicMaitre);
//    //creat display mPicMaitre with bord of imgetMaitre + triangle 2D
//    if (mPicMaitre->mListPtsInterestFAST.size() == 0) //detector pt interest if not detected yet
//    {
//        Detector aDetectImgM(mChain, mPicMaitre, pic2nd);      //now using HOMOLINIT
//        aDetectImgM.detect();
//        aDetectImgM.saveToPicTypeVector(mPicMaitre);
//    }
//    vector<Pt2dr> ptsInThisTri = mPicMaitre->getPtsHomoInThisTri(aTri);
//    //creat display imgetMaitre + pts interet inside

//    vector<Pt2dr> ptsCorrelInTri2nd;
//    for (uint i=0; i<mListPic2nd.size(); i++)
//    {
//        pic * pic2nd = mListPic2nd[i];
//        Tri2d tri2D2nd    = *aTri->getReprSurImg()[pic2nd->mIndex];
//        if (tri2DMaitre.insidePic && tri2D2nd.insidePic)
//        {
//            matAffine affineM_2ND = aTri->CalAffine(mPicMaitre, pic2nd);
//            bool getImaget2ndSucces;
//            ImgetOfTri imget2nd = aTri->get_Imagette_by_affine_n(imgetMaitre, pic2nd, affineM_2ND, getImaget2ndSucces);
//            double score_glob = imgetMaitre.imaget->CrossCorrelation(*imget2nd.imaget);
//            cCorrelImage::setSzW(mSzW); //for correlation each point interest inside imaget
//            for (uint j=0; j<ptsInThisTri.size(); j++)
//            {
//                bool foundMaxCorr = false;
//                Pt2dr P1 = ptsInThisTri[j];
//                Pt2dr P2 = this->correlPtsInteretInImaget(P1, imgetMaitre, imget2nd, affineM_2ND, foundMaxCorr);
//                if (!foundMaxCorr)
//                    P2=Pt2dr(-1,-1);
//                ptsCorrelInTri2nd.push_back(P2);
//            }
//        }
//    }
//}





