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



pic::pic(const string *nameImg, string nameOri, cInterfChantierNameManipulateur * aICNM, int indexInListPic)
{
    mICNM = aICNM;
    /*
    std::string keyOri = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+ this->mOri, aNameImg3, true);
    CamStenope * aCam3 = CamOrientGenFromFile(aOri3 , aICNM);
    */
    mNameImg = nameImg;
    mIndex = indexInListPic;
    if (nameOri != "NONE")
    {
        mOriPic = mICNM->StdCamStenOfNames(*nameImg , nameOri);
    }
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
void pic::AddVectorPtsToPack(pic* Pic2nd, vector<Pt2dr> & Pts1, vector<Pt2dr> & Pts2)
{
    //Si P2 = (-1,-1) => not added
    int count = 0;
    for (uint i=0; i<Pts1.size(); i++)
    {
        if ( (Pts2[i].x != -1) && (Pts2[i].y != -1) )
        {
            cout<<" - Add to Pack : ";
            mPackHomoWithAnotherPic[Pic2nd->mIndex].aPack.Cple_Add(ElCplePtsHomologues(Pts1[i], Pts2[i]));
            cout<< Pts1[i]<< Pts2[i]<<endl;
            count++;
        }
    }
    cout<<" - Total "<<count<<" pts / "<<Pts1.size()<<" interet added"<<endl;
}
void pic::AddVectorCplHomoToPack(pic* Pic2nd, vector<ElCplePtsHomologues> aHomo)
{
    int count = 0;
    for (uint i=0; i<aHomo.size(); i++)
    {
        cout<<" - Add to Pack : ";
        mPackHomoWithAnotherPic[Pic2nd->mIndex].aPack.Cple_Add(aHomo[i]);
        cout<< aHomo[i].P1()<< aHomo[i].P2()<<endl;
        count++;
    }
    cout<<" - Total "<<count<<" pts added"<<endl;
}


bool pic::checkInSide(Pt2dr aPoint,int aRab) //default aRab=0
{
    return    (aPoint.x - aRab >= 0)
           && (aPoint.y - aRab >= 0)
           && (aPoint.x + aRab < this->mImgSz.x)
           && (aPoint.y + aRab < this->mImgSz.y);
/*
    bool result;
    //Pt2di size = mOriPic->Sz();
    Pt2di size = mImgSz;
    if (
         (aPoint.x >= aRab) && (aPoint.y >= aRab) &&
         (aPoint.x < size.x -aRab) && (aPoint.y < size.y -aRab)
        )
        {result=true;}
    else
        {result = false;}
    return result;    
*/
}

void pic::getPtsHomoInThisTri(triangle* aTri , vector<Pt2dr> & lstPtsInteret, vector<Pt2dr> & result)
{
    if (lstPtsInteret.size() == 0)
        cout<<"+++ WARN +++ : don't have pts interest";
    else
    {
        for (uint i=0; i<lstPtsInteret.size(); i++)
        {
            bool in = aTri->check_inside_triangle_A2016(lstPtsInteret[i],
                                                  *aTri->getReprSurImg()[this->mIndex]);
            if (in)
                result.push_back(lstPtsInteret[i]);
        }
    }
}

void pic::getPtsHomoOfTriInPackExist( string aKHIn
                                     ,triangle * aTri, pic * pic1st, pic* pic2nd ,
                                     vector<ElCplePtsHomologues> & result)
{
    string HomoIn = mICNM->Assoc1To2(aKHIn,
                                    *pic1st->mNameImg,
                                    *pic2nd->mNameImg,true);
    bool Exist = ELISE_fp::exist_file(HomoIn);
    ElPackHomologue apackInit;
    bool Exist1 = 0;
    if (!Exist)
    {
        StdCorrecNameHomol_G(HomoIn,mICNM->Dir());
        Exist1 = ELISE_fp::exist_file(HomoIn);
    }
    if (Exist || Exist1)
    {
        apackInit =  ElPackHomologue::FromFile(HomoIn);
        for (ElPackHomologue::const_iterator itP=apackInit.begin(); itP!=apackInit.end() ; itP++)
        {
            Pt2dr aP1 = itP->P1();
            Tri2d aTri2d = *aTri->getReprSurImg()[pic1st->mIndex];
            bool in = aTri->check_inside_triangle_A2016(aP1, aTri2d);
            if (in)
            {
                ElCplePtsHomologues aCpl(itP->P1(), itP->P2());
                result.push_back(aCpl);
            }
        }
        cout<<HomoIn<<" - "<<apackInit.size()<<"/"<<result.size()<<endl;
    }
}

double pic::calAngleViewToTri(triangle *aTri)
{
    CamStenope * aCamPic = this->mOriPic;
    Pt3dr centre_cam = aCamPic->VraiOpticalCenter();
    //Tri2d aTri2d = *aTri->getReprSurImg()[this->mIndex];
    Pt3dr centre_geo = (aTri->getSommet(0) + aTri->getSommet(1) + aTri->getSommet(2))/ 3;
    Pt3dr Vec1 = centre_cam - centre_geo;
    Pt3dr aVecNor = aTri->CalVecNormal(centre_geo, 0.05);
    Pt3dr Vec2 = aVecNor - centre_geo;
    //bool devant = aCamPic->Devant(centre_geo);
    double angle_deg = (aTri->calAngle(Vec1, Vec2))*180/PI;
    return angle_deg;
}

void pic::getTriVisible(vector<triangle*> & lstTri, double angleF, bool Zbuf)
{
    CamStenope * aCamPic = this->mOriPic;
    Pt3dr centre_cam = aCamPic->VraiOpticalCenter();
    if (this->triVisible.size() == 0)
    {
        for (uint j=0; j<lstTri.size(); j++)
        {
            triangle * aTri = lstTri[j];
            Tri2d aTri2d = *aTri->getReprSurImg()[this->mIndex];
            Pt3dr centre_geo = (aTri->getSommet(0) + aTri->getSommet(1) + aTri->getSommet(2))/ 3;
            Pt3dr Vec1 = centre_cam - centre_geo;
            Pt3dr aVecNor = aTri->CalVecNormal(centre_geo, 0.05);
            Pt3dr Vec2 = aVecNor - centre_geo;
            bool devant = aCamPic->Devant(centre_geo);
            double angle_deg = (aTri->calAngle(Vec1, Vec2))*180/PI;
            cout<<angle_deg<<endl;
            if ( (angle_deg<angleF) && devant &&  aTri2d.insidePic)
            {
                triVisible.push_back(aTri);
                triVisibleInd.push_back(aTri->mIndex);
            }
        }
        if (Zbuf)
        {
            for (uint i=0; i<triVisible.size(); i++)
            {
                bool found = false;
                triangle * aTri = triVisible[i];
                Tri2d aTri2D = *aTri->getReprSurImg()[this->mIndex];
                Pt2dr ctr_geo2D = (aTri2D.sommet1[0] + aTri2D.sommet1[1] + aTri2D.sommet1[2])/3;
                vector<triangle*> triCollision;
                whichTrianglecontainPt(ctr_geo2D, triVisible, triCollision, found);
                if (found)
                    cout<<"Tri "<<aTri->mIndex<<" has "<<triCollision.size()<<" triangles collision "<<endl;
            }
        }
    }
}

double pic::distDuTriauCtrOpt(triangle * aTri)
{
    Pt3dr opticCentre = this->mOriPic->VraiOpticalCenter();
    Pt3dr triCentre = (aTri->getSommet(0) + aTri->getSommet(1) + aTri->getSommet(2))/3;
    Pt3dr temp = triCentre - opticCentre;
    double dist = sqrt(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
    return dist;
}

void pic::whichTrianglecontainPt(Pt2dr aPt, vector<triangle*>lstTri, vector<triangle*> result, bool & found)
{
    for (uint i=0; i<lstTri.size(); i++)
    {
        triangle * aTri = lstTri[i];
        bool in = aTri->check_inside_triangle_A2016(aPt, *aTri->getReprSurImg()[this->mIndex]);
        if (in)
        {
            result.push_back(aTri);
            found = true;
        }
    }
}

void pic::getTriVisibleWithPic(vector<triangle*> & lstTri, double angleF,
                               pic * pic2, vector<triangle*> & triVisblEnsmbl, bool Zbuf)
{
    if (this->triVisible.size() == 0)
        this->getTriVisible(lstTri, angleF , Zbuf);
    if (pic2->triVisible.size() == 0)
        pic2->getTriVisible(lstTri, angleF, Zbuf);
    vector<double> intersect;
    std::set_intersection(  this->triVisibleInd.begin(), this->triVisibleInd.end(),
                            pic2->triVisibleInd.begin(), pic2->triVisibleInd.end(),
                            std::back_inserter(intersect) );
    for (uint i=0; i<intersect.size(); i++)
    {
        triVisblEnsmbl.push_back(lstTri[intersect[i]]);
    }
}

triangle * pic::whichTriangle(Pt2dr & ptClick, bool & found)
{
    found = false;
    triangle * result = NULL;
    vector<triangle*> lstTriCollision;
    if (this->triVisible.size() > 0)
    {
        for (uint i=0; i<triVisible.size(); i++)
        {
            triangle * aTri = triVisible[i];
            //check if Pt in this tri or not
            bool in = aTri->check_inside_triangle_A2016(ptClick, *aTri->getReprSurImg()[this->mIndex]);
            if (in)
            {
                result =  aTri;
                lstTriCollision.push_back(aTri);
                found = true;
                //break;
            }
        }
    }
    else
    {
        cout<<"No triangle visible for this img"<<endl;
        result = NULL;
    }
    cout<<"There is "<<lstTriCollision.size()<<" triangle collision"<<endl;
    return result;
}

void pic::roundPtInteret()
{
    for (uint i=0; i<this->mListPtsInterestFAST.size(); i++)
    {
        this->mListPtsInterestFAST[i].x = round(this->mListPtsInterestFAST[i].x);
        this->mListPtsInterestFAST[i].y = round(this->mListPtsInterestFAST[i].y);
    }
}



