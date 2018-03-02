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


#include "CorrelMesh.h"
#include "PHO_MI.h"
#include "vector"
#include "algorithm"
#include "iterator"
#include "../../uti_phgrm/NewOri/NewOri.h"

CorrelMesh::CorrelMesh(InitOutil *aChain)
{
    mChain = aChain;
    this->countPts = 0;
    this->countCplOrg = 0;
    reloadTri();
    reloadPic();
}

void CorrelMesh::reloadTri()
{
    mPtrListTri = mChain->getmPtrListTri();
}
void CorrelMesh::reloadPic()
{
    mPtrListPic = mChain->getmPtrListPic();
}

pic* CorrelMesh::chooseImgMaitre(int indTri, double & angleReturn, double angleVisibleLimit, bool assume1er)
{
    triangle * aTri = mPtrListTri[indTri];
    mListPic2nd.clear();
    if (assume1er)
    {
        mPicMaitre = mPtrListPic[0];
        for (uint i=1; i<mPtrListPic.size(); i++)
        {
            mListPic2nd.push_back(mPtrListPic[i]);
        }
        angleReturn = aTri->angleToVecNormalImg(mPicMaitre);
    }
    else
    {
        vector<pic*> lstPic = mPtrListPic;
        vector<Pt2dr> cplAngleIndpic;
        for (uint i=0; i<lstPic.size(); i++)
        {
            Pt2dr aAngPic;
            pic * aPic = lstPic[i];
            aAngPic.x = aTri->angleToVecNormalImg(aPic);
            aAngPic.y = i;
            cplAngleIndpic.push_back(aAngPic);
        }
        this->sortDescend(cplAngleIndpic);
        mPicMaitre = mPtrListPic[cplAngleIndpic[cplAngleIndpic.size()-1].y];
        for (uint i=0; i<cplAngleIndpic.size()-1; i++)
        {
            if (cplAngleIndpic[i].x < angleVisibleLimit)
            {
                mListPic2nd.push_back(mPtrListPic[cplAngleIndpic[i].y]);
            }
        }
        angleReturn = cplAngleIndpic[cplAngleIndpic.size()-1].x;
    }
    return mPicMaitre;
}


extern bool comparatorPt2dr ( Pt2dr const &l,  Pt2dr const &r)
   { return l.x > r.x; }

extern bool comparatorPt2drAsc ( Pt2dr const &l,  Pt2dr const &r)
   { return l.x < r.x; }

extern bool comparatorPt2drY ( Pt2dr const &l,  Pt2dr const &r)
   { return l.y > r.y; }

extern bool comparatorPt2diY ( Pt2di const &l,  Pt2di const &r)
   { return l.y > r.y; }

extern bool comparatorPt2drYAsc ( Pt2dr const &l,  Pt2dr const &r)
   { return l.y < r.y; }


void CorrelMesh::sortDescend(vector<Pt2dr> & input)
{
   sort(input.begin(), input.end(), static_cast<dsPt2drCompFunc>(&comparatorPt2dr));
}

extern void sortDescendPt2drX(vector<Pt2dr> & input)
{
   sort(input.begin(), input.end(), static_cast<dsPt2drCompFunc>(&comparatorPt2dr));
}

extern void sortAscendPt2drX(vector<Pt2dr> & input)
{
   sort(input.begin(), input.end(), static_cast<dsPt2drCompFunc>(&comparatorPt2drAsc));
}

extern void sortDescendPt2drY(vector<Pt2dr> & input)
{
   sort(input.begin(), input.end(), static_cast<dsPt2drCompFunc>(&comparatorPt2drY));
}

extern void sortDescendPt2diY(vector<Pt2di> & input)
{
   sort(input.begin(), input.end(), static_cast<dsPt2diCompFunc>(&comparatorPt2diY));
}


extern void sortAscendPt2drY(vector<Pt2dr> & input)
{
   sort(input.begin(), input.end(), static_cast<dsPt2drCompFunc>(&comparatorPt2drYAsc));
}

vector<ElCplePtsHomologues> CorrelMesh::choosePtsHomoFinal(vector<Pt2dr>&scorePtsInTri,triangle* aTri,
                                 vector<ElCplePtsHomologues>&P1P2Correl)
{
//    vector<ElCplePtsHomologues> result;
//    if (scorePtsInTri.size() < 2)
//    {
//        result = P1P2Correl;
//    }
//    else if (scorePtsInTri[0].x-scorePtsInTri[1].x > 0.2)
//    {
//        for (uint i=0; i<3; i++)
//        {
//            int ind = scorePtsInTri[i].y;
//            result.push_back(P1P2Correl[ind]);
//        }
//    }
//    else
//    {
//        cout<<" "<<result.size()<<" pts/"<<P1P2Correl.size()<<endl;
//    }
//    return result;

//    vector<ElCplePtsHomologues> result;
//    if (scorePtsInTri.size() < 3)
//        result = P1P2Correl;
//    else
//    {
//        for (uint i=0; i<3; i++)
//        {
//            int ind = scorePtsInTri[i].y;
//            result.push_back(P1P2Correl[ind]);
//        }
//    }
//    return result;

    vector<ElCplePtsHomologues> result;
    if (P1P2Correl.size()<=12)
    {
        result = P1P2Correl;
    }
    else
    {
        //creat 3 triangle from bariCtr-Sommet
        Tri2d thisTri = *aTri->getReprSurImg()[this->mPicMaitre->mIndex];
        Pt2dr bariCtr = (thisTri.sommet1[0]+thisTri.sommet1[1]+thisTri.sommet1[2])/3;
        Tri2d tri1, tri2, tri3;
        tri1.sommet1[0] = thisTri.sommet1[0];
        tri1.sommet1[1] = thisTri.sommet1[1];
        tri1.sommet1[2] = Pt2dr(bariCtr);
        tri2.sommet1[0] = thisTri.sommet1[0];
        tri2.sommet1[1] = Pt2dr(bariCtr);
        tri2.sommet1[2] = thisTri.sommet1[2];
        tri3.sommet1[0] = Pt2dr(bariCtr);
        tri3.sommet1[1] = thisTri.sommet1[1];
        tri3.sommet1[2] = thisTri.sommet1[2];
        //sort pts in each triangle
        vector<ElCplePtsHomologues>ptInTri1;
        vector<ElCplePtsHomologues>ptInTri2;
        vector<ElCplePtsHomologues>ptInTri3;
        uint nbPtInTriS = round(P1P2Correl.size()/6);
        for (uint i=0; i<P1P2Correl.size(); i++)
        {
            ElCplePtsHomologues aCpl = P1P2Correl[i];
            if (aTri->check_inside_triangle_A2016(aCpl.P1(), tri1))
                ptInTri1.push_back(aCpl);
            else if (aTri->check_inside_triangle_A2016(aCpl.P1(), tri2))
                ptInTri2.push_back(aCpl);
            else
                ptInTri3.push_back(aCpl);
        }
        //cout<<"Tri 1: "<<ptInTri1.size()<<" Tri 2: "<<ptInTri2.size()<<" Tri 3: "<<ptInTri3.size()
        //   <<" NbCorrel:"<<P1P2Correl.size()<<" NbPtO: "<<nbPtInTriS<<endl;
        if (ptInTri1.size() >= 1 && ptInTri1.size()<=nbPtInTriS)
            result.insert(result.end(), ptInTri1.begin(), ptInTri1.end());
        if (ptInTri2.size() >= 1 && ptInTri2.size()<=nbPtInTriS)
            result.insert(result.end(), ptInTri2.begin(), ptInTri2.end());
        if (ptInTri3.size() >= 1 && ptInTri3.size()<=nbPtInTriS)
            result.insert(result.end(), ptInTri3.begin(), ptInTri3.end());
        if (ptInTri1.size() > nbPtInTriS)
        {
            for (uint i=0; i<nbPtInTriS; i++)
                result.push_back(ptInTri1[i]);
        }
        if (ptInTri2.size() > nbPtInTriS)
        {
            for (uint i=0; i<nbPtInTriS; i++)
                result.push_back(ptInTri2[i]);
        }
        if (ptInTri3.size() > nbPtInTriS)
        {
            for (uint i=0; i<nbPtInTriS; i++)
                result.push_back(ptInTri3[i]);
        }
    }
    return result;
}

Pt2dr CorrelMesh::correlPtsInteretInImaget(Pt2dr ptInt1,
                                           ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                           matAffine & affine,
                                           bool & foundMaxCorr,
                                           double & scoreR,
                                           double seuil_corel = 0.9)
{
    /*
        |--------x-i--|
        |-------------|
        y-------------|
        |-------------|
        j-------------|
    */
    Pt2dr ptInt2(0,0);
    int areaSearch = mChain->mSzAreaCorr;
    int szPtCorr = mChain->mSzPtCorr;
    cCorrelImage::setSzW(szPtCorr);    //taille fenetre correlation pour chaque pt = 3*3
    Pt2dr P1_rlt = ptInt1 - imgetMaitre.ptOriginImaget;
    cCorrelImage * P1 = new cCorrelImage();
    cCorrelImage * P2 = new cCorrelImage();
    P1->getFromIm(imgetMaitre.imaget->getIm(), P1_rlt.x, P1_rlt.y);
    //--determiner la region de correlation autour de Pt interet
    Pt2di PtHG;
    Pt2di PtBD;
    if (P1_rlt.x - areaSearch >= 0)
        PtHG.x=(int)(P1_rlt.x - areaSearch);
    else
        PtHG.x=0;
    if (P1_rlt.y - areaSearch >= 0)
        PtHG.y=(int)(P1_rlt.y - areaSearch);
    else
        PtHG.y=0;
    if (P1_rlt.x + areaSearch <= imget2nd.szW*2+1)
        PtBD.x=(int)(P1_rlt.x + areaSearch);
    else
        PtBD.x=imget2nd.szW*2+1;
    if (P1_rlt.y + areaSearch <= imget2nd.szW*2+1)
        PtBD.y=(int)(P1_rlt.y + areaSearch);
    else
        PtBD.y=imget2nd.szW*2+1;
    //--parcourir la region de correlation---
    vector<double> corelScore;
    int counter=0;
    for (int i=PtHG.x + szPtCorr; i<=PtBD.x-szPtCorr; i++)
    {
        counter ++;
        for (int j=PtHG.y + szPtCorr; j<=PtBD.y-szPtCorr; j++)
        {
            P2->getFromIm(imget2nd.imaget->getIm(), i, j);
            double score = P1->CrossCorrelation(*P2);
            corelScore.push_back(score);
        }
    }
    vector<double>::iterator max_corel = max_element(corelScore.begin(), corelScore.end());
    int ind = distance(corelScore.begin(), max_corel);
    point_temp = Pt2dr(ind/counter + PtHG.x,  ind%counter + PtHG.y);
    Pt2dr max_corel_coor = Pt2dr(ind/counter + PtHG.x,  ind%counter + PtHG.y) + imgetMaitre.ptOriginImaget + Pt2dr(szPtCorr,szPtCorr);
    if (corelScore[ind] > seuil_corel)
    {
        ptInt2 = ApplyAffine(max_corel_coor, affine);
        foundMaxCorr = true;
        scoreR = corelScore[ind];
    }
    else
    {
        foundMaxCorr = false;
    }
    delete P1; delete P2;
    return ptInt2;
}

Pt2dr CorrelMesh::correlPtsInteretInImaget_NEW(Pt2dr ptInt1,
                                                ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                                matAffine & affine,
                                                bool & foundMaxCorr,
                                                double & scoreR, bool dbgByClick,
                                                double seuil_corel)
{
    /*
        |--------x-i--|
        |-------------|
        y----NEW------|
        |-------------|
        j-------------|
    */
    Pt2dr ptInt2(0,0);
    double areaSearch = double(mChain->mSzAreaCorr);
    double szPtCorr = double(mChain->mSzPtCorr);
    Pt2dr P1_rlt = ptInt1 - imgetMaitre.ptOriginImaget;
    //--determiner la region de correlation autour de Pt interet
    Pt2dr PtHG = P1_rlt - Pt2dr(areaSearch,areaSearch);
    Pt2dr PtBD = P1_rlt + Pt2dr(areaSearch,areaSearch);
    if (PtHG.x < 0)
    {
        areaSearch = areaSearch - (0 - PtHG.x);
        PtHG = P1_rlt - Pt2dr(areaSearch,areaSearch);
        PtBD = P1_rlt + Pt2dr(areaSearch,areaSearch);
    }
    if (PtBD.y > imgetMaitre.imaget->getIm()->sz().y)
    {
        areaSearch = areaSearch - (PtBD.y - imgetMaitre.imaget->getIm()->sz().y);
        PtHG = P1_rlt - Pt2dr(areaSearch,areaSearch);
        PtBD = P1_rlt + Pt2dr(areaSearch,areaSearch);
    }
    //int nbCorrelation = (areaSearch-szPtCorr)*2+1;
    if(areaSearch >= szPtCorr)
    {
        //--parcourir la region de correlation---
        cCorrelImage::setSzW(szPtCorr);    //taille fenetre correlation pour chaque pt = 3*3
        cCorrelImage * P1 = new cCorrelImage();
        cCorrelImage * P2 = new cCorrelImage();
        P1->getFromIm(imgetMaitre.imaget->getIm(), P1_rlt.x, P1_rlt.y);
        vector<double> corelScore;
        L2SysSurResol aSys1(6);
        Pt2dr ptSub(szPtCorr,szPtCorr);
        Pt2dr ptScrSurfc(0,0);
        ptSub = ptSub + imgetMaitre.ptOriginImaget + PtHG;
        int counter=0;
        for (int i=PtHG.x + szPtCorr; i<=PtBD.x-szPtCorr; i++)
        {
            counter ++;
            ptScrSurfc.y = 0;
            for (int j=PtHG.y + szPtCorr; j<=PtBD.y-szPtCorr; j++)
            {
                P2->getFromIm(imget2nd.imaget->getIm(), i, j);
                double score = P1->CrossCorrelation(*P2);
                corelScore.push_back(score);
                double aEq1[6] = {1, ptSub.x, ptSub.y, ptSub.x*ptSub.x, ptSub.x*ptSub.y, ptSub.y*ptSub.y};
                aSys1.AddEquation(1, aEq1, score);
                ptScrSurfc.y++;
                ptSub.y++;
            }
            ptScrSurfc.x++;
            ptSub.x++;
        }
        vector<double>::iterator max_corel = max_element(corelScore.begin(), corelScore.end());
        int ind = distance(corelScore.begin(), max_corel);
        point_temp = Pt2dr(ind/counter + PtHG.x,  ind%counter + PtHG.y) + Pt2dr(szPtCorr,szPtCorr);
        Pt2dr max_corel_coor = Pt2dr(ind/counter + PtHG.x,  ind%counter + PtHG.y) + imgetMaitre.ptOriginImaget +Pt2dr(szPtCorr,szPtCorr);
        if (corelScore[ind] > seuil_corel)
        {
            ptInt2 = ApplyAffine(max_corel_coor, affine);
            foundMaxCorr = true;
            scoreR = corelScore[ind];
        }
        else
            foundMaxCorr = false;

        delete P1; delete P2;
    }
    else
        foundMaxCorr = false;
    return ptInt2;
}

Pt2dr CorrelMesh::correlSubPixelPtsIntInImaget(Pt2dr ptInt1, ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                               matAffine & affine,
                                               bool & foundMaxCorr,
                                               double & scoreR,
                                               bool dbgByClick,
                                               double seuil_corel)
{
    Pt2dr ptInt2(0,0);
    double areaSearch = double(mChain->mSzAreaCorr);
    double szPtCorr = double(mChain->mSzPtCorr);
    double pas = double(mChain->mPas);
    Pt2dr P1_rlt = ptInt1 - imgetMaitre.ptOriginImaget;
    //--determiner la region de correlation autour de Pt interet
    Pt2dr PtHG = P1_rlt - Pt2dr(areaSearch,areaSearch);
    Pt2dr PtBD = P1_rlt + Pt2dr(areaSearch,areaSearch);
    if (PtHG.x < 0)
    {
        areaSearch = areaSearch - (0 - PtHG.x);
        PtHG = P1_rlt - Pt2dr(areaSearch,areaSearch);
        PtBD = P1_rlt + Pt2dr(areaSearch,areaSearch);
    }
    if (PtBD.y > imgetMaitre.imaget->getIm()->sz().y)
    {
        areaSearch = areaSearch - (PtBD.y - imgetMaitre.imaget->getIm()->sz().y);
        PtHG = P1_rlt - Pt2dr(areaSearch,areaSearch);
        PtBD = P1_rlt + Pt2dr(areaSearch,areaSearch);
    }
    if (areaSearch > szPtCorr)
    {
//    int nbCorrelation = ((areaSearch-szPtCorr)*2*(1/pas)+1);
//    int szAreaCor = nbCorrelation;

    //==========test========
    cCorrelImage::setSzW(szPtCorr);
    cCorrelImage * P1 = new cCorrelImage();
    cCorrelImage * P2 = new cCorrelImage();
    P1->getFromIm(imgetMaitre.imaget->getIm(), P1_rlt.x, P1_rlt.y);
    vector<double> corelScore;
    int counter=0;
    Pt2di ptIndScr(0,0);
    for (double i=PtHG.x+szPtCorr; i<=PtBD.x-szPtCorr; i=i+pas)
    {
        counter ++;
        ptIndScr.y=0;
        for (double j=PtHG.y+szPtCorr; j<=PtBD.y-szPtCorr; j=j+pas)
        {
            P2->getFromIm(imget2nd.imaget->getIm(), i, j);
            double score = P1->CrossCorrelation(*P2);
            corelScore.push_back(score);
            ptIndScr.y++;
        }
        ptIndScr.x++;
    }
    //--chercher max correlation score pixel entier
    vector<double>::iterator max_corel = max_element(corelScore.begin(), corelScore.end());
    int ind = distance(corelScore.begin(), max_corel);
    point_temp = Pt2dr( (ind/counter)*pas + PtHG.x,  (ind%counter)*pas + PtHG.y) + Pt2dr(szPtCorr,szPtCorr);
    Pt2dr max_corel_coor = Pt2dr((ind/counter)*pas  + PtHG.x,  (ind%counter)*pas + PtHG.y) + imgetMaitre.ptOriginImaget + Pt2dr(szPtCorr,szPtCorr);
    if (corelScore[ind] > seuil_corel)
    {
        ptInt2 = ApplyAffine(max_corel_coor, affine);
        foundMaxCorr = true;
        scoreR = corelScore[ind];
    }
    else
    {
        foundMaxCorr = false;
    }
    //cout<<" Correl: P1 : "<<ptInt1<<" - P2 CorEnt:"<<max_corel_coor<<" -Scr:"<<corelScore[ind]<<"-OK:"<< foundMaxCorr<<"- int: "<<Pt2dr((ind/counter)*pas, (ind%counter)*pas)<<endl;
    delete P1;
    delete P2;
    }
    else
    {
        foundMaxCorr = false;
    }
    return ptInt2;
}



/*=====Search for homol in specific triangle - search b/w PicM & all others pic========
 * indTri: index du triangle 3D sur le mesh
 * 1. reproject tri3D => 2 tri2D on image
 * 2. cherche les pack homol init => des couple imgM + imgs2nd
 * 2. Pour chaque couple :choisir un image maitraisse
 * 3. prendre imagette imget1 autour tri2D d'image maitraisse
 * 4. chercher affine b/w 2 tri2D
 * 5. prendre imagette imget2 autour tri2D d'image 2nd par affine
 * 6. correlation pour chaque pts d'interet dans imget1 avec pts dans imget2
 * 7. sort -> prendre pts correlation plus fort
*/
void CorrelMesh::correlInTri(int indTri)
{
    cout<<"Tri "<<indTri<<endl;
    triangle * aTri = mPtrListTri[indTri];
    double angle_deg;
    this->chooseImgMaitre(indTri, angle_deg, 90, this->mChain->mAssume1er);
        cout<<" ++ PicM: "<<mPicMaitre->getNameImgInStr()<<" - "<<angle_deg<<"°"<<endl;
    Tri2d tri2DMaitre = *aTri->getReprSurImg()[mPicMaitre->mIndex];
        cout<<" ++ SommetTriM:"<<tri2DMaitre.sommet1[0]<<tri2DMaitre.sommet1[1]<<tri2DMaitre.sommet1[2]<<endl;
        cout<<" "<<mListPic2nd.size()<<" pic2nd"<<endl;

    ImgetOfTri imgetMaitre = aTri->create_Imagette_autour_triangle_A2016(mPicMaitre);
    if (imgetMaitre.imaget != NULL)
    {
    for (uint i=0; i<mListPic2nd.size(); i++)
    {
        pic * pic2nd = mListPic2nd[i];
        Tri2d tri2D2nd = *aTri->getReprSurImg()[pic2nd->mIndex];
            cout<<" ++ Pic2nd: "<<pic2nd->getNameImgInStr()<<endl;
            cout<<" ++ SommetTri2nd:"<<tri2D2nd.sommet1[0]<<tri2D2nd.sommet1[1]<<tri2D2nd.sommet1[2]<<endl;
        if (mPicMaitre->mListPtsInterestFAST.size() == 0) //detector pt interest if not detected yet
        {
            Detector * aDetectImgM;
            if (mChain->getPrivMember("mTypeD") == "HOMOLINIT")
                aDetectImgM = new Detector(mChain, mPicMaitre, pic2nd);
            else
                aDetectImgM = new Detector(   mChain->getPrivMember("mTypeD"),
                                              mChain->getParamD(),
                                              mPicMaitre,
                                              mChain
                                            );
            aDetectImgM->detect();
            aDetectImgM->saveToPicTypeVector(mPicMaitre);
            delete aDetectImgM;
        }
        if (tri2DMaitre.insidePic && tri2D2nd.insidePic)
        {
            vector<Pt2dr> ptsInThisTri;
            mPicMaitre->getPtsHomoInThisTri(aTri, mPicMaitre->mListPtsInterestFAST, ptsInThisTri);
            cout<<" ++ "<<ptsInThisTri.size()<<" pts in this tri"<<endl;
            if (ptsInThisTri.size() == 0)
                mTriHavePtInteret.push_back(aTri->mIndex);
            vector<ElCplePtsHomologues> P1P2Correl;
            vector<Pt2dr> scorePtsInTri;
            bool affineResolu;
            matAffine affineM_2ND = aTri->CalAffine(mPicMaitre, pic2nd, affineResolu);
            if (ptsInThisTri.size() > 0 && affineResolu)
            {
                bool getImaget2ndSucces;
                ImgetOfTri imget2nd = aTri->get_Imagette_by_affine_n(imgetMaitre,
                                                                     pic2nd,
                                                                     affineM_2ND,
                                                                     getImaget2ndSucces);


                double score_glob = imgetMaitre.imaget->CrossCorrelation(*imget2nd.imaget);
                cout<<" -   ScGlob: "<<score_glob<<endl;
                if (score_glob > mChain->mCorl_seuil_glob)
                {
                    mTriCorrelSuper.push_back(aTri->mIndex);
                    cCorrelImage::setSzW(mChain->mSzPtCorr); //for correlation each point interest inside imaget
                    int ind = 0;
                    for (uint j=0; j<ptsInThisTri.size(); j++)
                    {
                        bool foundMaxCorr = false;
                        double score;
                        Pt2dr P1 = ptsInThisTri[j];
                        Pt2dr P2 = this->correlPtsInteretInImaget(P1, imgetMaitre,
                                                                  imget2nd, affineM_2ND, foundMaxCorr,
                                                                  score,
                                                                  mChain->mCorl_seuil_pt);
                        if (foundMaxCorr == true)
                        {
                            ElCplePtsHomologues aCpl(P1, P2);
                            P1P2Correl.push_back(aCpl);
                            Pt2dr scorePt;
                            scorePt.x = score; scorePt.y = ind;
                            scorePtsInTri.push_back(scorePt);
                            ind++;
                        }
                    }
                   this->sortDescend(scorePtsInTri);
                   vector<ElCplePtsHomologues> P1P2CorrelF = this->choosePtsHomoFinal(scorePtsInTri, aTri, P1P2Correl);
                   this->countPts = this->countPts + P1P2CorrelF.size();
                   mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2CorrelF,
                                               mChain->getPrivMember("mHomolOutput"));
                   mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2CorrelF,
                                               mChain->getPrivMember("mHomolOutput"), true);
                   cout<<"  "<<P1P2CorrelF.size()<<" pts OK/"<<P1P2Correl.size()<<" pts match/"<<ptsInThisTri.size()<<" pts in Tri"<<endl;

                }
                delete imget2nd.imaget;
            }
        }
    }
    }
    delete imgetMaitre.imaget;
}

void CorrelMesh::correlInTriWithViewAngle(int indTri, double angleF, bool debugByClick)
{
    cout<<"Tri "<<indTri<<endl;
    triangle * aTri = mPtrListTri[indTri];
    double angle_deg;
    this->chooseImgMaitre(indTri, angle_deg, angleF, this->mChain->mAssume1er);
        cout<<" ++ PicM: "<<mPicMaitre->getNameImgInStr()<<" - "<<angle_deg<<"°"<<endl;
    if (angle_deg<angleF)
    {
    Tri2d tri2DMaitre = *aTri->getReprSurImg()[mPicMaitre->mIndex];
        //cout<<" ++ SommetTriM:"<<tri2DMaitre.sommet1[0]<<tri2DMaitre.sommet1[1]<<tri2DMaitre.sommet1[2]<<endl;
        //cout<<" "<<mListPic2nd.size()<<" pic2nd"<<endl;

    ImgetOfTri imgetMaitre = aTri->create_Imagette_autour_triangle_A2016(mPicMaitre);

    if (imgetMaitre.imaget != NULL)
    {
        for (uint i=0; i<mListPic2nd.size(); i++)
        {
            pic * pic2nd = mListPic2nd[i];
                Tri2d tri2D2nd = *aTri->getReprSurImg()[pic2nd->mIndex];
                cout<<" ++ Pic2nd: "<<pic2nd->getNameImgInStr()<<endl;
                //cout<<" ++ SommetTri2nd:"<<tri2D2nd.sommet1[0]<<tri2D2nd.sommet1[1]<<tri2D2nd.sommet1[2]<<endl;
                if (mChain->getPrivMember("mTypeD") != "HOMOLINIT")
                {
                    if(mPicMaitre->mListPtsInterestFAST.size() == 0) //detector pt interest if not detected yet
                    {
                        Detector * aDetectImgM;
                        aDetectImgM = new Detector(   mChain->getPrivMember("mTypeD"),
                                                      mChain->getParamD(),
                                                      mPicMaitre,
                                                      mChain
                                                      );
                        aDetectImgM->detect();
                        aDetectImgM->saveToPicTypeVector(mPicMaitre);
                        delete aDetectImgM;
                    }
                }
                if (mChain->getPrivMember("mTypeD") == "HOMOLINIT")
                {
                    Detector * aDetectImgM;
                    mPicMaitre->mListPtsInterestFAST.clear();
                    aDetectImgM = new Detector(mChain, mPicMaitre, pic2nd);
                    aDetectImgM->detect();
                    aDetectImgM->saveToPicTypeVector(mPicMaitre);
                    delete aDetectImgM;
                }
                if (tri2DMaitre.insidePic && tri2D2nd.insidePic)
                {
                    vector<Pt2dr> ptsInThisTri;
                    mPicMaitre->getPtsHomoInThisTri(aTri,mPicMaitre->mListPtsInterestFAST, ptsInThisTri);
                    //cout<<" ++ "<<ptsInThisTri.size()<<" pts in this tri"<<endl;
                    bool getPtPckHomoDirect = false;
                    if (ptsInThisTri.size() == 0)
                    {
                        getPtPckHomoDirect = true;
                        this->mTriHavePtInteret.push_back(indTri);
                    }
                    vector<ElCplePtsHomologues> P1P2Correl;
                    vector<Pt2dr> scorePtsInTri;
                    bool affineResolu;
                    matAffine affineM_2ND = aTri->CalAffine(mPicMaitre, pic2nd, affineResolu);
                    if (ptsInThisTri.size() > 0 && affineResolu && !getPtPckHomoDirect)
                    {
                        bool getImaget2ndSucces;
                        ImgetOfTri imget2nd = aTri->get_Imagette_by_affine_n(imgetMaitre,
                                                                             pic2nd,
                                                                             affineM_2ND,
                                                                             getImaget2ndSucces);
                        //double score_glob = imgetMaitre.imaget->CrossCorrelation(*imget2nd.imaget);
                        //cout<<" -   ScGlob: "<<score_glob<<endl;
                        vector<Pt2dr>Pdisp2;
                        mTriCorrelSuper.push_back(aTri->mIndex);
                        cCorrelImage::setSzW(mChain->mSzPtCorr); //for correlation each point interest inside imaget
                        int ind = 0;
                        for (uint j=0; j<ptsInThisTri.size(); j++)
                        {
                            bool foundMaxCorr = false;
                            double score;
                            Pt2dr P1 = ptsInThisTri[j];
//                            Pt2dr P2 = this->correlPtsInteretInImaget(P1, imgetMaitre,
//                                                                      imget2nd, affineM_2ND, foundMaxCorr,
//                                                                      score,
//                                                                      mChain->mCorl_seuil_pt);
//                            Pt2dr P2 = this->correlPtsInteretInImaget_NEW(P1, imgetMaitre,
//                                                                      imget2nd, affineM_2ND, foundMaxCorr,
//                                                                      score, debugByClick,
//                                                                      mChain->mCorl_seuil_pt);
                            Pt2dr P2 = this->correlSubPixelPtsIntInImaget(  P1, imgetMaitre,
                                                                            imget2nd, affineM_2ND, foundMaxCorr,
                                                                            score,
                                                                            debugByClick,
                                                                            mChain->mCorl_seuil_pt);

                            if (foundMaxCorr == true)
                            {
                                ElCplePtsHomologues aCpl(P1, P2);
                                P1P2Correl.push_back(aCpl);
                                Pt2dr scorePt;
                                scorePt.x = score; scorePt.y = ind;
                                scorePtsInTri.push_back(scorePt);
                                ind++;
                                Pdisp2.push_back(this->point_temp);
                            }
                        }
                        this->sortDescend(scorePtsInTri);
                        vector<ElCplePtsHomologues> P1P2CorrelF = this->choosePtsHomoFinal(scorePtsInTri, aTri, P1P2Correl);
                        //vector<ElCplePtsHomologues> P1P2CorrelF = P1P2Correl;
//                        if (P1P2CorrelF.size() == 0)
//                        {
//                            mPicMaitre->getPtsHomoOfTriInPackExist(mChain->getPrivMember("mKHIn"),
//                                                                   aTri, mPicMaitre, pic2nd, P1P2CorrelF);
//                        }
                        this->countPts = this->countPts + P1P2CorrelF.size();
                        mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2CorrelF,
                                                    mChain->getPrivMember("mHomolOutput"));
                        mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2CorrelF,
                                                    mChain->getPrivMember("mHomolOutput"), true);
                        //cout<<"  "<<P1P2CorrelF.size()<<" pts OK/"<<P1P2Correl.size()<<" pts match/"<<ptsInThisTri.size()<<" pts in Tri"<<endl;
                        delete imget2nd.imaget;
                    }
//                    else
//                    {
//                        cout<<" Get from Pack Tapioca: ";
//                        vector<ElCplePtsHomologues> P1P2CorrelF;
//                        mPicMaitre->getPtsHomoOfTriInPackExist(mChain->getPrivMember("mKHIn"),
//                                                               aTri, mPicMaitre, pic2nd, P1P2CorrelF);
//                        mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2CorrelF,
//                                                    mChain->getPrivMember("mHomolOutput"));
//                        mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2CorrelF,
//                                                    mChain->getPrivMember("mHomolOutput"), true);
//                        cout<<"  "<<P1P2CorrelF.size()<<" pts get"<<endl;
//                        this->countCplOrg = this->countCplOrg + P1P2CorrelF.size();
//                    }
                }
        }
    }
    else
        cout<<"  Can't Get imagetM"<<endl;
    delete imgetMaitre.imaget; 
    }
    else
        cout<<" °!° - View Angle Img Master pass limit - pass to another triangle"<<endl;
    cout<<"Finish correl - exit"<<endl;
}


void CorrelMesh::homoCplSatisfiyTriangulation(int indTri, double angleF)
{
    //Take 2 image
    pic * aPic1 = mPtrListPic[0];
    pic * aPic2 = mPtrListPic[1];
    vector<ElCplePtsHomologues> P1P2Correl;
    triangle * aTri = mPtrListTri[indTri];
    //=====lire pack homo=====
    string HomoIn = mChain->getPrivmICNM()->Assoc1To2
                    (this->mChain->getPrivMember("mKHIn"),
                     aPic1->getNameImgInStr(), aPic2->getNameImgInStr(),true);
    bool Exist = ELISE_fp::exist_file(HomoIn);
    ElPackHomologue apackInit;
    if (!Exist)
    {
        StdCorrecNameHomol_G(HomoIn,mChain->getPrivmICNM()->Dir());
        Exist = ELISE_fp::exist_file(HomoIn);
        if(Exist)
            apackInit =  ElPackHomologue::FromFile(HomoIn);
    }
    else
        apackInit =  ElPackHomologue::FromFile(HomoIn);
    if (mPicMaitre->mListPtsInterestFAST.size() == 0 && Exist)
    {
        Detector * aDetectImgM;
        aDetectImgM = new Detector(mChain, aPic1, aPic2);
        aDetectImgM->detect();
        aDetectImgM->saveToPicTypeVector(aPic1);
        delete aDetectImgM;
    }
    vector<Pt2dr> ptsInThisTri;
    aPic1->getPtsHomoInThisTri(aTri, aPic1->mListPtsInterestFAST, ptsInThisTri);
    for (uint i=0; i<ptsInThisTri.size(); i++)
    {
        const ElCplePtsHomologues  *  aCpl = apackInit.Cple_Nearest(ptsInThisTri[i]);
        bool inP2 = aTri->check_inside_triangle_A2016(aCpl->P2(),
                                              *aTri->getReprSurImg()[aPic2->mIndex]);
        if (inP2)
        {
            P1P2Correl.push_back(*aCpl);
        }
    }
    this->countPts = this->countPts + P1P2Correl.size();
    if (P1P2Correl.size() > 0)
    {
        mChain->addToExistHomolFile(aPic1, aPic2,  P1P2Correl,
                                    mChain->getPrivMember("mHomolOutput"));
        mChain->addToExistHomolFile(aPic1, aPic2,  P1P2Correl,
                                    mChain->getPrivMember("mHomolOutput"), true);
    }
}

void CorrelMesh::verifCplHomoByTriangulation(int indTri, double angleF)
{
    vector<CplPic> homoExist = this->mChain->getmCplHomolExist();
    if (homoExist.size() <= 0)
        cout<<"WARN : No data homol exist found !"<<endl;
    else
    {
        for (uint i=0; i<homoExist.size(); i++)
        {
            vector<pic*> mPtrListPicT;
            CplPic thisHomoPack = homoExist[i];
            mPtrListPicT.push_back(thisHomoPack.pic1);
            mPtrListPicT.push_back(thisHomoPack.pic2);
            mPtrListPic = mPtrListPicT;
            this->homoCplSatisfiyTriangulation(indTri, angleF);
            reloadPic();
        }
    }
}

void CorrelMesh::correlByCplExist(int indTri)
{
    vector<CplPic> homoExist = this->mChain->getmCplHomolExist();
    if (homoExist.size() <= 0)
        cout<<"WARN : No data homol exist found !"<<endl;
    else
    {
        for (uint i=0; i<homoExist.size(); i++)
        {
            vector<pic*> mPtrListPicT;
            CplPic thisHomoPack = homoExist[i];
            mPtrListPicT.push_back(thisHomoPack.pic1);
            mPtrListPicT.push_back(thisHomoPack.pic2);
            mPtrListPic = mPtrListPicT;
            this->correlInTri(indTri);
            reloadPic();
        }
    }
}
void CorrelMesh::correlByCplExistWithViewAngle(int indTri, double angleF, bool debugByClick)
{
    vector<CplPic> homoExist = this->mChain->getmCplHomolExist();
    if (homoExist.size() <= 0)
        cout<<"WARN : No data homol exist found !"<<endl;
    else
    {
        vector<pic*> mPtrListPicBackUp = mPtrListPic;
        for (uint i=0; i<homoExist.size(); i++)
        {
            vector<pic*> mPtrListPicT;
            CplPic thisHomoPack = homoExist[i];
            mPtrListPicT.push_back(thisHomoPack.pic1);
            mPtrListPicT.push_back(thisHomoPack.pic2);
            mPtrListPic = mPtrListPicT;
            this->correlInTriWithViewAngle(indTri, angleF, debugByClick);
            mPtrListPic = mPtrListPicBackUp;
        }
    }
}

void CorrelMesh::multiCorrel(int indTri, double angleF)
{
    vector<pic*>lstPic = this->mPtrListPic;
    triangle * aTri = this->mPtrListTri[indTri];
    vector<pic*>picVisible;
    //search for all pic visible by triangle
    for (uint i=0; i<lstPic.size(); i++)
    {
        pic * aPic = lstPic[i];
        vector<triangle*> triVisible = aPic->triVisible;
        bool found = std::find(triVisible.begin(), triVisible.end(), aTri) != triVisible.end();
        if (found)
            picVisible.push_back(aPic);
    }
    Pt3dr som1 = aTri->getSommet(0);
    Pt3dr som2 = aTri->getSommet(1);
    Pt3dr som3 = aTri->getSommet(2);
    if (picVisible.size() > 1)
    {
        for (uint i=0; i<picVisible.size(); i++)
        {
        //calcul prof image 1 to centre_geo triangle
        pic * pic0 = picVisible[i];
        Pt3dr bariCtr = (som1+som2+som3)/3;
        CamStenope * camPic0 = pic0->mOriPic;
        double prof = camPic0->GetProfondeur();
        Pt3dr DirK = camPic0->DirK();
        double prof_d = camPic0->ProfInDir(bariCtr,camPic0->DirK());
        cout<<"Tri " <<indTri << " - Pic : "<<pic0->mIndex<<" -Prof : "<<prof<<" -DirK: "<<DirK<<" -Prof_dirK : "<<prof_d<<endl;
        //redresser tout image vers img 1
        //changer prof, cal
        }
    }
}
