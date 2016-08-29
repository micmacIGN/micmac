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
#include "iterator"

CorrelMesh::CorrelMesh(InitOutil *aChain)
{
    mChain = aChain;
    this->countPts = 0;
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
    mListPic2nd.clear();
    for (uint i=1; i<mPtrListPic.size(); i++)
    {
        mListPic2nd.push_back(mPtrListPic[i]);
    }
    return mPicMaitre;
}


Pt2dr CorrelMesh::correlPtsInteretInImaget(Pt2dr ptInt1,
                                           ImgetOfTri imgetMaitre, ImgetOfTri imget2nd,
                                           matAffine & affine,
                                           bool & foundMaxCorr,
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
//    cout<<" + Sz ImagetM:"<<imgetMaitre.imaget->getIm()->sz()
//        <<" - PtOrg: "<<imgetMaitre.ptOriginImaget
//        <<" - Sz ImgetP1:"<<P1->getIm()->sz()<<endl;
//    cout<<" + P1_rtl: "<<P1_rlt;
    //--determiner la region de correlation autour 5pxl de Pt interet
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
//    cout<<PtHG<<PtBD<<endl;
    //--parcourir la region de correlation---
    vector<double> corelScore;
    int counter=0;
    for (int i=PtHG.x + szPtCorr; i<=PtBD.x-szPtCorr; i++)
    {
        counter ++;
        for (int j=PtHG.y + szPtCorr; j<=PtBD.y-szPtCorr; j++)
        {
            //cout<<" + Crl: "<<"["<<i<<","<<j<<"]";
            P2->getFromIm(imget2nd.imaget->getIm(), i, j);
            //cout <<" - Sz ImgetP2:"<<P2->getIm()->sz();
            double score = P1->CrossCorrelation(*P2);
            //cout<<" -Sc: "<<score<<endl;
            corelScore.push_back(score);
        }
    }
    vector<double>::iterator max_corel = max_element(corelScore.begin(), corelScore.end());
    int ind = distance(corelScore.begin(), max_corel);
//    cout<<" + Max Cor of Pt: "<<corelScore[ind]<<" - Posti: "<<ind;
    Pt2dr max_corel_coor = Pt2dr(ind/counter + PtHG.x,  ind%counter + PtHG.y) + imgetMaitre.ptOriginImaget;
//    cout<<" + P1: "<<ptInt1<< " +P2o: "<<max_corel_coor<<" +P2: "<< ApplyAffine(max_corel_coor, affine)<<endl;
    if (corelScore[ind] > seuil_corel)
    {
        ptInt2 = ApplyAffine(max_corel_coor, affine);
        foundMaxCorr = true;
        //cout<<" + "<<"P1: "<<ptInt1<<" - P2: "<<ptInt2<<" - Sc: "<<corelScore[ind]<<endl;
    }
    else
    {
        foundMaxCorr = false;
    }
    delete P1; delete P2;
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
    this->chooseImgMaitre(true);    //assume 1er image est image maitre
        cout<<" ++ PicM: "<<mPicMaitre->getNameImgInStr()<<endl;
    Tri2d tri2DMaitre = *aTri->getReprSurImg()[mPicMaitre->mIndex];
        cout<<" ++ SommetTriM:"<<tri2DMaitre.sommet1[0]<<tri2DMaitre.sommet1[1]<<tri2DMaitre.sommet1[2]<<endl;
        cout<<" "<<mListPic2nd.size()<<" pic2nd"<<endl;

    ImgetOfTri imgetMaitre = aTri->create_Imagette_autour_triangle_A2016(mPicMaitre);

//    VWImg1 = display_image(mPicMaitre->mPic_Im2D,"Img1 - Tri " + intToString(indTri),VWImg1, 0.2);
//    VWImg1 = draw_polygon_onVW(tri2DMaitre, VWImg1);
//    VWImg1 = draw_polygon_onVW(imgetMaitre.ptOriginImaget, imgetMaitre.szW*2+1, VWImg1);

//    Video_Win *VWget1;
//    Im2D<U_INT1,INT4> imgetDisp(300,300);
//    imgetDisp = imgetMaitre.imaget->getIm()->AugmentSizeTo(Pt2di(300,300));
//    VWget1 = display_image(&imgetDisp,"Imget1 - Tri " + intToString(indTri),VWget1, 1);


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
            vector<Pt2dr> ptsInThisTri = mPicMaitre->getPtsHomoInThisTri(aTri);
            cout<<" ++ "<<ptsInThisTri.size()<<" pts in this tri"<<endl;
            if (ptsInThisTri.size() > 0)
                mTriHavePtInteret.push_back(aTri->mIndex);
            vector<ElCplePtsHomologues> P1P2Correl;
            bool affineResolu;
            matAffine affineM_2ND = aTri->CalAffine(mPicMaitre, pic2nd, affineResolu);
            if (ptsInThisTri.size() > 0 && affineResolu)
            {
                bool getImaget2ndSucces;
                ImgetOfTri imget2nd = aTri->get_Imagette_by_affine_n(imgetMaitre,
                                                                     pic2nd,
                                                                     affineM_2ND,
                                                                     getImaget2ndSucces);

//                VWImg2 = display_image(pic2nd->mPic_Im2D, "Img2 - Tri " + intToString(indTri),VWImg2, 0.2);
//                VWImg2 = draw_polygon_onVW(tri2D2nd, VWImg2);

                double score_glob = imgetMaitre.imaget->CrossCorrelation(*imget2nd.imaget);
                cout<<" -   ScGlob: "<<score_glob<<endl;
                if (score_glob > mChain->mCorl_seuil_glob)
                {
                    mTriCorrelSuper.push_back(aTri->mIndex);
                    cCorrelImage::setSzW(mChain->mSzPtCorr); //for correlation each point interest inside imaget
                    for (uint j=0; j<ptsInThisTri.size(); j++)
                    {
                        bool foundMaxCorr = false;
                        Pt2dr P1 = ptsInThisTri[j];
                        Pt2dr P2 = this->correlPtsInteretInImaget(P1, imgetMaitre,
                                                                  imget2nd, affineM_2ND, foundMaxCorr,
                                                                  mChain->mCorl_seuil_pt);
                        if (foundMaxCorr == true)
                        {
                            ElCplePtsHomologues aCpl(P1, P2);
                            P1P2Correl.push_back(aCpl);
                            this->countPts++;
                        }
                    }
                   mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2Correl,
                                               mChain->getPrivMember("mHomolOutput"));
                   mChain->addToExistHomolFile(mPicMaitre, pic2nd,  P1P2Correl,
                                               mChain->getPrivMember("mHomolOutput"), true);
                }
                delete imget2nd.imaget;
            }
        }
    }
    }
    delete imgetMaitre.imaget;
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
            reloadTriandPic();
        }
    }
}





