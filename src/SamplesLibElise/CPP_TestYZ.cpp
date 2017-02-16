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

struct Entry
{
    string name;
    int number;
};

vector<Entry> phone_book(100);

void normalizeYilin(Im2D<double,double>  & aImSource, Im2D<double,double>  & aImDest, double rangeMin, double rangeMax)
{
    double minVal;
    double maxVal;

    ELISE_COPY(aImSource.all_pts(),aImSource.in(),VMax(maxVal)|VMin(minVal));
    cout<<"Avant Min/Max : "<<minVal<<"/"<<maxVal<<endl;
    ELISE_ASSERT((aImSource.sz().x == aImDest.sz().x && aImSource.sz().y == aImDest.sz().y), "Size not coherent in normalize image");
    // ELISE_COPY(aImSource.all_pts(),aImSource.in(),aImDest.out());


    double factor = (rangeMax-rangeMin)/(maxVal-minVal);
    cout<<"fac "<<factor<<endl;

//    aImDest.substract(minVal);
//    aImDest.multiply(factor);
    ELISE_COPY(aImSource.all_pts(),(aImSource.in()-minVal)*factor,aImDest.out());


    //aImDest.getMinMax(minVal, maxVal);
    ELISE_COPY(aImDest.all_pts(),aImDest.in(),VMax(maxVal)|VMin(minVal));
    cout<<"Apres Min/Max : "<<minVal<<"/"<<maxVal<<endl;

}

class cAppliTestYZ
{
    public:
        cAppliTestYZ(string &aDir, Pt2di & aSzVW, int aZoom);
        const std::string & Dir() const {return mDir;}
        Pt2di & SzW() {return mSzW;}
        int & Zoom() {return mZoom;}
    private:
        string mDir;
        Pt2di mSzW;
        int mZoom;

};

cAppliTestYZ::cAppliTestYZ(string &aDir, Pt2di & aSzVW, int aZoom):
    mDir (aDir),
    mSzW (aSzVW),
    mZoom (aZoom)
{}

class cImgDepl
{
    public:
        cImgDepl(cAppliTestYZ * aAppli, string & aImgX, string & aImgY);
        void estimeDepl(Pt2dr & aPt, int & aSz);
        Tiff_Im & TifX() {return mTifX;}
        Tiff_Im & TifY() {return mTifY;}
        cAppliTestYZ * Appli() {return mAppli;}

    private:
        cAppliTestYZ * mAppli;
        Tiff_Im mTifX;
        Tiff_Im mTifY;
        Im2D<double,double>     mImDeplX;
        Im2D<double,double>     mImDeplY;
};

class cVignetteDepl
{
    public:
        cVignetteDepl(cImgDepl * aImgDepl, Pt2dr & aPtCent, int & aSz);
    private:
        cImgDepl * mImgDepl;
        Im2D<double,double>  mVignetteX;
        Im2D<double,double>  mVignetteY;
        Pt2dr mPtCent;
        Video_Win *mW;
        Pt2dr mDecalGlob;
};

cImgDepl::cImgDepl(cAppliTestYZ * aAppli, string & aImgX, string & aImgY):
    mAppli (aAppli),
    mTifX   (Tiff_Im::UnivConvStd(mAppli->Dir() + aImgX)),
    mTifY   (Tiff_Im::UnivConvStd(mAppli->Dir() + aImgY)),
    mImDeplX   (1,1),
    mImDeplY   (1,1)
{
    cout<<"Name :" <<mTifX.name()<<" - Sz : "<<mTifX.sz()<<endl;
    //lire img
    mImDeplX.Resize(mTifX.sz());
    mImDeplY.Resize(mTifY.sz());
    ELISE_COPY(mImDeplX.all_pts(),mTifX.in(),mImDeplX.out());
    ELISE_COPY(mImDeplY.all_pts(),mTifY.in(),mImDeplY.out());
}

void cImgDepl::estimeDepl(Pt2dr & aPt, int & aSz)
{
    cVignetteDepl * aVignetteDepl = new cVignetteDepl (this, aPt, aSz);
}

cVignetteDepl::cVignetteDepl(cImgDepl *aImgDepl, Pt2dr &aPtCent, int &aSz):
    mImgDepl (aImgDepl),
    mVignetteX (1,1),
    mVignetteY (1,1),
    mPtCent   (aPtCent),
    mW        (0),
    mDecalGlob (aPtCent - Pt2dr(double(aSz), double(aSz)))
{
    //Prendre Imagette
    Pt2dr aP0 = aPtCent - Pt2dr(double(aSz), double(aSz));
    mVignetteX.Resize(Pt2di(aSz*2+1, aSz*2+1));
    mVignetteY.Resize(Pt2di(aSz*2+1, aSz*2+1));
    ELISE_COPY(mVignetteX.all_pts(),trans(mImgDepl->TifX().in(0),Pt2di(mDecalGlob)),mVignetteX.out());
    ELISE_COPY(mVignetteY.all_pts(),trans(mImgDepl->TifY().in(0),Pt2di(mDecalGlob)),mVignetteY.out());
    cout<<"Create Vignette : Sz : "<<mVignetteX.sz()<<" - Pt : "<<mDecalGlob<<endl;

    if (mW == 0)
    {
         int aZ = mImgDepl->Appli()->Zoom();
         mW = Video_Win::PtrWStd(mImgDepl->Appli()->SzW()*aZ,true,Pt2dr(aZ,aZ));
         mW = mW-> PtrChc(Pt2dr(0,0),Pt2dr(aZ,aZ),true);
         std::string aTitle = std::string("Mon Vignette");
         mW->set_title(aTitle.c_str());
    }
    if (mW)
    {
        //normalize pour affichier
        Im2D<double,double>  mDisplay;
        mDisplay.Resize(mVignetteX.sz());
        normalizeYilin(mVignetteX, mDisplay, 0.0, 255.0);
        ELISE_COPY(mDisplay.all_pts(),mDisplay.in(),mW->ogray());
        mW->clik_in();
    }

    //Estimation similitude
    L2SysSurResol aSys(4);
    double* aData1 = NULL;
    for (int aKx=0; aKx<mVignetteX.sz().x; aKx++)
    {
        for (int aKy=0; aKy<mVignetteX.sz().y; aKy++)
        {
            double coeffX[4] = {double(aKx + mDecalGlob.x), double(-(aKy + mDecalGlob.y)), 1.0, 0.0};
            double coeffY[4] = {double(aKy + mDecalGlob.y), double(aKx + mDecalGlob.x), 0.0, 1.0};
            double delX = mVignetteX.GetR(Pt2di(aKx, aKy));
            double delY = mVignetteY.GetR(Pt2di(aKx, aKy));
            aSys.AddEquation(1.0, coeffX, delX);
            aSys.AddEquation(1.0, coeffY, delY);
        }
    }
    bool solveOK = true;
    Im1D_REAL8 aResol1 = aSys.GSSR_Solve(&solveOK);
    aData1 = aResol1.data();
    if (solveOK != false)
        cout<<"Estime : A B C D = "<<aData1[0]<<" "<<aData1[1]<<" "<<aData1[2]<<" "<<aData1[3]<<endl;
    else
        cout<<"Can't estime"<<endl;
}

int TestYZ_main(int argc,char ** argv)
{

    string aImgX;
    string aImgY;
    string aDir;
    Pt3di aSzW(5,5,100);
    Pt2dr aPtCent;
    int SzV;
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aDir, "Dir", eSAM_None)
                << EAMC(aImgX, "Img de deplacement X", eSAM_IsExistFile)
                << EAMC(aImgY, "Img de deplacement Y", eSAM_IsExistFile)
                << EAMC(aPtCent, "Coor vignette", eSAM_None)
                << EAMC(SzV, "Sz vignette (demi)", eSAM_None),
                //optional arguments
                LArgMain()
                << EAM(aSzW, "SzW", true, "Size Win")
             );

    if (MMVisualMode)     return EXIT_SUCCESS;
    Pt2di aSzWW(aSzW.x, aSzW.y);
    cAppliTestYZ * anAppli = new cAppliTestYZ(aDir, aSzWW, aSzW.z);
    cImgDepl * aImgDepl = new cImgDepl(anAppli, aImgX, aImgY);
    aImgDepl->estimeDepl(aPtCent, SzV);

            return EXIT_SUCCESS;



}

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
