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

#include "StdAfx.h"

//normalize vignette from [minVal,maxVal] to [rangeMin, rangeMac] so as to affiche (vignette may have negative values)
void normalizeYilin(Im2D<double,double>  & aImSource, Im2D<double,double>  & aImDest, double rangeMin, double rangeMax)
{
    double minVal; //min value of ImSource
    double maxVal; //Max value of ImSource

    //find the Min and Max of ImSource
    ELISE_COPY(aImSource.all_pts(),aImSource.in(),VMax(maxVal)|VMin(minVal));
    cout<< "Avant Min/Max : " << minVal << "/" << maxVal << endl;

    //check if the sizes of ImSource and ImDest are coherent
    ELISE_ASSERT((aImSource.sz().x == aImDest.sz().x && aImSource.sz().y == aImDest.sz().y), "Size not matching for image normalization");

    if (maxVal==minVal)
    {
        //normalize ImSource
        ELISE_COPY(aImSource.all_pts(),(aImSource.in()-minVal)+(rangeMax-rangeMin)/2,aImDest.out());
    }
    else
    {
        //normalize ImSource
        double factor = (rangeMax-rangeMin)/(maxVal-minVal);
        ELISE_COPY(aImSource.all_pts(),(aImSource.in()-minVal)*factor,aImDest.out());
    }

    //find the Min and Max of ImDest
    ELISE_COPY(aImDest.all_pts(),aImDest.in(),VMax(maxVal)|VMin(minVal));
    cout<< "Apres Min/Max : " << minVal << "/" << maxVal << endl;
}

bool IsInside(Pt2di & aPDR, Pt2di & aP)
{
    if (aP.x >= 0 && aP.x <= aPDR.x && aP.y >= 0 && aP.y <= aPDR.y)
        return true;
    else
    {
        cout << "vignette not in the image!" << endl;
        return false;
    }
}

class cParamEsSim
{
public:
    cParamEsSim(string & mDir, string & mNameX, string & mNameY, int & mZoom, int & mNbGrill, int & mSzV);
    string & Dir () {return mDir;}
    string & NameX () {return mNameX;}
    string & NameY () {return mNameY;}
    Tiff_Im & TifX () {return mTifX;}
    Tiff_Im & TifY () {return mTifY;}
    Im2D<double,double> & ImVX() {return mImVX;}
    Im2D<double,double> & ImVY() {return mImVY;}
    int & NbGrill() {return mNbGrill;}
private:
    string mDir;
    string mNameX;
    string mNameY;
    Tiff_Im mTifX;
    Tiff_Im mTifY;
    Im2D<double,double> mImVX;
    Im2D<double,double> mImVY;
    Video_Win * mWX; // Display window for Vignette X
    Video_Win * mWY; // Display window for Vignette Y
    int mZoom;
    int mNbGrill; // Nb of Grill for similitude estimation
    //Pt2di mPtCt;
    int mSzV;
};

cParamEsSim::cParamEsSim(string & aDir, string & aNameX, string & aNameY, int & aZoom, int & aNbGrill, int & aSzV):
    mDir (aDir),
    mNameX (aNameX),
    mNameY (aNameY),
    mTifX (Tiff_Im::UnivConvStd(mDir + mNameX)),
    mTifY (Tiff_Im::UnivConvStd(mDir + mNameY)),
    mImVX  (1,1), //initiation of vignette X
    mImVY  (1,1), //initiation of vignette Y
    mWX (0),
    mWY (0),
    mNbGrill (aNbGrill),
    mZoom (aZoom)
{
    Pt2di aSz = mTifX.sz();
    cout << "Image size = " << aSz << endl;

    //Calculate the center point for each grill
    Pt2di aSzG = Pt2di (int(aSz.x/aNbGrill),int(aSz.y/aNbGrill));

    //vector<Pt2di> aPtCt;
    //Pt2di a[3][3]={Pt2di(),.....};
    //uint aNbPtCt (0);
//    for (uint aK1=0; aK1<NbGrill; aK1++)
//    {
//        for (uint aK2=0; aK2<NbGrill; aK2++)
//        {
//            aPtCt[aNbPtCt] =
//        }
//    }
    Pt2di aPtCt = Pt2di (int(aSzG.x/2),int(aSzG.y/2));

    Pt2di aP0 = aPtCt-Pt2di(aSzV,aSzV);
    Pt2di aP1 = aPtCt+Pt2di(aSzV,aSzV);


    if ( IsInside(aSz,aP0) && IsInside(aSz,aP1) )
    {
        Pt2di aSzVV = Pt2di(2*aSzV+1,2*aSzV+1); // real size of vignette
        mImVX.Resize(aSzVV);
        mImVY.Resize(aSzVV);
        ELISE_COPY(mImVX.all_pts(),trans(mTifX.in(0),aP0),mImVX.out());
        ELISE_COPY(mImVY.all_pts(),trans(mTifY.in(0),aP0),mImVY.out());

        Pt2di aDecal = aPtCt - Pt2di(aSzV,aSzV);

        // Estimation of similitude
        L2SysSurResol aSys(4);
        double * aData1 = NULL;
        for (int aKx=0; aKx < aSzV; aKx++)
        {
            for (int aKy=0; aKy < aSzV; aKy++)
            {
                double coeffX[4] = {double(aKx + aDecal.x), double(-(aKy + aDecal.y)), 1.0, 0.0};
                double coeffY[4] = {double(aKy + aDecal.y), double(aKx + aDecal.x), 0.0, 1.0};
                double delX = ImVX().GetR(Pt2di(aKx, aKy));
                double delY = ImVY().GetR(Pt2di(aKx, aKy));
                aSys.AddEquation(1.0, coeffX, delX);
                aSys.AddEquation(1.0, coeffY, delY);
            }
        }
        bool solveOK = true;
        Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
        aData1 = aResol1.data();
        if (solveOK != false)
            cout << "Estimation : [A B C D] = [" << aData1[0] << " " << aData1[1] << " " << aData1[2] << " " << aData1[3] << "]" <<endl;
        else
            cout<<"Can't estimate."<<endl;

//        if (mWX == 0)
//        {
//             mWX = Video_Win::PtrWStd(mImVX.sz()*aZoom,true,Pt2dr(aZoom,aZoom));
//             mWX = mWX-> PtrChc(Pt2dr(0,0),Pt2dr(aZoom,aZoom),true);
//             std::string aTitle = std::string("Vignette X");
//             mWX->set_title(aTitle.c_str());
//        }
//        if (mWX)
//        {
//            //normalize to affiche
//            Im2D<double,double>  mDisplay;
//            mDisplay.Resize(aSzVV);
//            normalizeYilin(mImVX, mDisplay, 0.0, 255.0);
//            ELISE_COPY(mDisplay.all_pts(),mDisplay.in(),mWX->ogray());
//            //mW->clik_in();
//        }

//        if (mWY == 0)
//        {
//             mWY = Video_Win::PtrWStd(mImVY.sz()*aZoom,true,Pt2dr(aZoom,aZoom));
//             mWY = mWY-> PtrChc(Pt2dr(0,0),Pt2dr(aZoom,aZoom),true);
//             std::string aTitle = std::string("Vignette Y");
//             mWY->set_title(aTitle.c_str());
//        }
//        if (mWY)
//        {
//            //normalize to affiche
//            Im2D<double,double>  mDisplay;
//            mDisplay.Resize(aSzVV);
//            normalizeYilin(mImVY, mDisplay, 0.0, 255.0);
//            ELISE_COPY(mDisplay.all_pts(),mDisplay.in(),mWY->ogray());
//            mWY->clik_in();
//        }


    }
}

int TestYZ_main(int argc, char ** argv)
{
    string aDir;
    string aNameX;
    string aNameY;
    int aZoom (3);
    Pt2di aPtCt (5,5);
    int aNbGrill (10);
    int aSzV (5);

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory")
                    << EAMC(aNameX, "ImgX", eSAM_IsExistFile)
                    << EAMC(aNameY, "ImgY", eSAM_IsExistFile),
        LArgMain()  << EAM(aZoom,"Zoom",true,"Zoom factor for display. Def=3")
                    << EAM(aNbGrill,"NbGrill",true,"Nb of Grill for similitude estimation. Def=10")
                    //<< EAM(aPtCt,"PtCt",true,"Center point of vignette to display. Def=[5,5]")
                    << EAM(aSzV,"SzV",true,"Size of vignette to display. Def=5")
    );

    cParamEsSim * aParam = new cParamEsSim (aDir,aNameX, aNameY, aZoom, aNbGrill, aSzV);



    return EXIT_SUCCESS;
}


// normalize vignette from [minVal,maxVal] to [rangeMin, rangeMac] so as to affiche (vignette may have negative values)
//void normalizeYilin(Im2D<double,double>  & aImSource, Im2D<double,double>  & aImDest, double rangeMin, double rangeMax)
//{
//    double minVal; //min value of ImSource
//    double maxVal; //Max value of ImSource

//    //find the Min and Max of ImSource
//    ELISE_COPY(aImSource.all_pts(),aImSource.in(),VMax(maxVal)|VMin(minVal));
//    cout<<"Avant Min/Max : "<<minVal<<"/"<<maxVal<<endl;

//    //check if the sizes of ImSource and ImDest are coherent
//    ELISE_ASSERT((aImSource.sz().x == aImDest.sz().x && aImSource.sz().y == aImDest.sz().y), "Size not coherent in normalize image");

//    //normalize ImSource
//    double factor = (rangeMax-rangeMin)/(maxVal-minVal);
//    ELISE_COPY(aImSource.all_pts(),(aImSource.in()-minVal)*factor,aImDest.out());

//    //find the Min and Max of ImDest
//    ELISE_COPY(aImDest.all_pts(),aImDest.in(),VMax(maxVal)|VMin(minVal));
//    cout<<"Apres Min/Max : "<<minVal<<"/"<<maxVal<<endl;

//}

//class cAppliTestYZ
//{
//    /*aDir: repertoire
//     *aSzVW: Size of the affiche window
//     *aZoom: factor of zoom*/
//    public:
//        cAppliTestYZ(string & aDir, Pt2di & aSzVW, int aZoom);
//        const std::string & Dir() const {return mDir;}
//        Pt2di & SzW() {return mSzW;}
//        int & Zoom() {return mZoom;}
//    private:
//        string mDir;
//        Pt2di mSzW;
//        int mZoom;

//};

//cAppliTestYZ::cAppliTestYZ(string & aDir, Pt2di & aSzVW, int aZoom):
//    mDir (aDir),
//    mSzW (aSzVW),
//    mZoom (aZoom)
//{}

//class cImgDepl
//{
//    /*cImgDepl: Image of deplacement
//      aImgX: name of the deplacement Image of X-axis
//      aImgY: name of the deplacement Image of Y-axis*/

//    public:
//        cImgDepl(cAppliTestYZ * aAppli, string & aImgX, string & aImgY);
//        void estimeDepl(Pt2dr & aPt, int & aSz); //
//        Tiff_Im & TifX() {return mTifX;}
//        Tiff_Im & TifY() {return mTifY;}
//        cAppliTestYZ * Appli() {return mAppli;}

//    private:
//        cAppliTestYZ * mAppli;
//        Tiff_Im mTifX;
//        Tiff_Im mTifY;
//        Im2D<double,double>     mImDeplX;
//        Im2D<double,double>     mImDeplY;
//};

//get the deplacement images mImDeplX and mImDeplY
//cImgDepl::cImgDepl(cAppliTestYZ * aAppli, string & aImgX, string & aImgY):
//    mAppli (aAppli),
//    mTifX   (Tiff_Im::UnivConvStd(mAppli->Dir() + aImgX)), //read aImgX
//    mTifY   (Tiff_Im::UnivConvStd(mAppli->Dir() + aImgY)), //read aImgY
//    mImDeplX   (1,1), //initiation of vignette of X-axis
//    mImDeplY   (1,1)  //initiation of vignette of Y-axis
//{
//    cout<<"Name :" <<mTifX.name()<<" - Sz : "<<mTifX.sz()<<endl;
//    //read images
//    mImDeplX.Resize(mTifX.sz()); //set the size of mImDeplX
//    mImDeplY.Resize(mTifY.sz()); //set the size of mImDeplY
//    ELISE_COPY(mImDeplX.all_pts(),mTifX.in(),mImDeplX.out()); //copy mTifX to mImDeplX
//    ELISE_COPY(mImDeplY.all_pts(),mTifY.in(),mImDeplY.out()); //copy mTifY to mImDeplY
//}

//class definition of a vignette
//class cVignetteDepl
//{
//    public:
//        /*aImgDepl: the original image of deplacement
//         *aPtCent: the coordinate of the center point of the vignette
//         *aSz: the size of the vignette*/
//        cVignetteDepl(cImgDepl * aImgDepl, Pt2dr & aPtCent, int & aSz);
//    private:
//        cImgDepl * mImgDepl;
//        Im2D<double,double>  mVignetteX;
//        Im2D<double,double>  mVignetteY;
//        Pt2dr mPtCent;
//        Video_Win * mW;
//        Pt2dr mDecalGlob;
//};


//void cImgDepl::estimeDepl(Pt2dr & aPt, int & aSz)
//{
//    cVignetteDepl * aVignetteDepl = new cVignetteDepl (this, aPt, aSz);
//    cout<<aVignetteDepl<<endl;
//}


//get the vignette of aImgDepl at aPtCent of size aSz
//cVignetteDepl::cVignetteDepl(cImgDepl * aImgDepl, Pt2dr & aPtCent, int & aSz):
//    mImgDepl (aImgDepl),
//    mVignetteX (1,1),
//    mVignetteY (1,1),
//    mPtCent   (aPtCent),
//    mW        (0),
//    mDecalGlob (aPtCent - Pt2dr(double(aSz), double(aSz)))
//{
//    //get the vignette at aPtCent of size aSz
//    //Pt2dr aP0 = aPtCent - Pt2dr(double(aSz), double(aSz)); //calculate the original point of the vignette
//    mVignetteX.Resize(Pt2di(aSz*2+1, aSz*2+1)); //reset the size of mVignetteX
//    mVignetteY.Resize(Pt2di(aSz*2+1, aSz*2+1)); //reset the size of mVignetteY
//    ELISE_COPY(mVignetteX.all_pts(),trans(mImgDepl->TifX().in(0),Pt2di(mDecalGlob)),mVignetteX.out()); //get the vignetteX of TifX
//    ELISE_COPY(mVignetteY.all_pts(),trans(mImgDepl->TifY().in(0),Pt2di(mDecalGlob)),mVignetteY.out()); //get the vignetteY of TifY
//    cout<<"Create Vignette : Sz : "<<mVignetteX.sz()<<" - Pt : "<<mDecalGlob<<endl;

//    if (mW == 0)
//    {
//         int aZ = mImgDepl->Appli()->Zoom();
//         mW = Video_Win::PtrWStd(mImgDepl->Appli()->SzW()*aZ,true,Pt2dr(aZ,aZ));
//         mW = mW-> PtrChc(Pt2dr(0,0),Pt2dr(aZ,aZ),true);
//         std::string aTitle = std::string("Mon Vignette");
//         mW->set_title(aTitle.c_str());
//    }
//    if (mW)
//    {
//        //normalize to affiche
//        Im2D<double,double>  mDisplay;
//        mDisplay.Resize(mVignetteX.sz());
//        normalizeYilin(mVignetteX, mDisplay, 0.0, 255.0);
//        ELISE_COPY(mDisplay.all_pts(),mDisplay.in(),mW->ogray());
//        mW->clik_in();
//    }

//    //Estimation similitude
//    L2SysSurResol aSys(4);
//    double* aData1 = NULL;
//    for (int aKx=0; aKx<mVignetteX.sz().x; aKx++)
//    {
//        for (int aKy=0; aKy<mVignetteX.sz().y; aKy++)
//        {
//            double coeffX[4] = {double(aKx + mDecalGlob.x), double(-(aKy + mDecalGlob.y)), 1.0, 0.0};
//            double coeffY[4] = {double(aKy + mDecalGlob.y), double(aKx + mDecalGlob.x), 0.0, 1.0};
//            double delX = mVignetteX.GetR(Pt2di(aKx, aKy));
//            double delY = mVignetteY.GetR(Pt2di(aKx, aKy));
//            aSys.AddEquation(1.0, coeffX, delX);
//            aSys.AddEquation(1.0, coeffY, delY);
//        }
//    }
//    bool solveOK = true;
//    Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
//    aData1 = aResol1.data();
//    if (solveOK != false)
//        cout<<"Estime : A B C D = "<<aData1[0]<<" "<<aData1[1]<<" "<<aData1[2]<<" "<<aData1[3]<<endl;
//    else
//        cout<<"Can't estime"<<endl;
//}

//int TestYZ_main(int argc,char ** argv)
//{

//    string aImgX;
//    string aImgY;
//    string aDir;
//    Pt3di aSzW(5,5,100);
//    Pt2dr aPtCent;
//    int SzV;
//    ElInitArgMain
//            (
//                argc,argv,
//                //mandatory arguments
//                LArgMain()
//                << EAMC(aDir, "Dir", eSAM_None)
//                << EAMC(aImgX, "Img de deplacement X", eSAM_IsExistFile)
//                << EAMC(aImgY, "Img de deplacement Y", eSAM_IsExistFile)
//                << EAMC(aPtCent, "Center point of vignette", eSAM_None)
//                << EAMC(SzV, "Sz of vignette (demi)", eSAM_None),
//                //optional arguments
//                LArgMain()
//                << EAM(aSzW, "SzW", true, "Size Win")
//             );

//    if (MMVisualMode)     return EXIT_SUCCESS;
//    Pt2di aSzWW(aSzW.x, aSzW.y);
//    cAppliTestYZ * anAppli = new cAppliTestYZ(aDir, aSzWW, aSzW.z);
//    cImgDepl * aImgDepl = new cImgDepl(anAppli, aImgX, aImgY);
//    aImgDepl->estimeDepl(aPtCent, SzV);

//            return EXIT_SUCCESS;



//}

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
