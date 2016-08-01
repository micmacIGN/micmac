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
#include "Pic.h"

//using namespace cv;


pic::pic(const string *nameImg, string nameOri, cInterfChantierNameManipulateur * aICNM, int indexInListPic)
{
    mICNM = aICNM;
    /*
    std::string keyOri = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+ this->mOri, aNameImg3, true);
    CamStenope * aCam3 = CamOrientGenFromFile(aOri3 , aICNM);
    */
    mNameImg = nameImg;
    mIndex = indexInListPic;
    mOriPic = mICNM->StdCamOfNames(*nameImg , nameOri);
    mPicTiff = new Tiff_Im ( Tiff_Im::StdConvGen(mICNM->Dir()+*mNameImg,1,false));
    mImgSz = mPicTiff->sz();
    mPic_TIm2D = new TIm2D<U_INT1,INT4> (mPicTiff->sz());
    ELISE_COPY(mPic_TIm2D->all_pts(), mPicTiff->in(), mPic_TIm2D->out());
    mPic_Im2D = new Im2D<U_INT1,INT4> (mPic_TIm2D->_the_im);
}

void pic::AddPtsToPack(pic* Pic2nd, const Pt2dr & Pts1, const Pt2dr& Pts2)
{
    cout<<" - Add to Pack";
    mPackHomoWithAnotherPic[Pic2nd->mIndex].aPack.Cple_Add(ElCplePtsHomologues(Pts1, Pts2)); //ERROR ? mIndex se melange ?
}


bool pic::checkInSide(Pt2dr aPoint)
{
    bool result;
    //Pt2di size = mOriPic->Sz();
    Pt2di size = mImgSz;
    if (
         (aPoint.x >= 0) && (aPoint.y >= 0) &&
         (aPoint.x <= size.x) && (aPoint.y <= size.y)
        )
        {result=true;}
    else
        {result = false;}
    return result;
}

vector<Pt2dr> pic::getPtsHomoInThisTri(triangle* aTri)
{
    vector<Pt2dr> result;
    if (this->mListPtsInterestFAST.size() == 0)
        cout<<"+++ WARN +++ : pic don't have pts interest saved";
    else
    {
        for (uint i=0; i<this->mListPtsInterestFAST.size(); i++)
        {
            bool in = aTri->check_inside_triangle(mListPtsInterestFAST[i],
                                                  *aTri->getReprSurImg()[this->mIndex]);
            if (in)
                result.push_back(mListPtsInterestFAST[i]);
        }
    }
    return result;
}

