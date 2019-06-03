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


static const int TheSeuilC = 196;


class cAppliHomProfPx
{
    public:
        cAppliHomProfPx(int argc,char ** argv);

        void Load();
        void ExportHom();

    private:
        std::string mDir;
        std::string mNameIm2;
        std::string mNameCoordIm1;
        std::string mOri;
        std::string mNuageName;
        std::string mPxName;
        std::string mOut;
        cInterfChantierNameManipulateur* mICNM;

        cElNuage3DMaille * mNuage;
        Pt2dr              mVPxT;
        CamStenope       * mCS2;

        cDicoAppuisFlottant aD; //points in the left image
        
        Im2D_REAL4          mImCorrel;
        Im2D_REAL4          mImProf;
        TIm2D<float,double> mTProf;
        Im2D_REAL4          mImPxT;
        TIm2D<float,double> mTPx;
        Im2D_U_INT1         mMasq;
        TIm2D<U_INT1,INT>   mTMasq;

        int                 mNumCorrel;
        int                 mZoomFinal;
        int                 mNumFinal;

};

cAppliHomProfPx::cAppliHomProfPx(int argc,char ** argv) :
    mDir("./"),
    mOut("HomProfPx.txt"),
    mNuage     (0),
    mImCorrel  (1,1),
    mImProf    (1,1),
    mTProf     (mImProf),
    mImPxT     (1,1),
    mTPx       (mImPxT),
    mMasq      (1,1),
    mTMasq     (mMasq),
    mZoomFinal (2),
    mNumFinal  (11)
{

    
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm2,"Name of the \"second\" image of the pair", eSAM_IsPatFile)
                    << EAMC(mOri,"Input orientation directory", eSAM_IsExistFile)
                    << EAMC(mNuageName,"Path to the NuageImProf file", eSAM_IsExistFile)
                    << EAMC(mPxName,"Path to the Px2 file", eSAM_IsExistFile)
                    << EAMC(mNameCoordIm1,"Input file with pixel positions in the \"first\" image pair (DicoAppuisFlottant format)", eSAM_NoInit),
        LArgMain()  << EAM(mOut,"Out",true,"Output file with correspondences in the image pair", eSAM_NoInit)
                    << EAM(mZoomFinal,"ZoomF", true, "Zoom Final",eSAM_IsPowerOf2)
                    << EAM(mNumFinal,"NumF", true, "Num Final",eSAM_IsPowerOf2)
                    << EAM(mNumCorrel,"NumCor","true","Num Correl", eSAM_NoInit)
    );

    #if (ELISE_windows)
         replace( aImName2.begin(), aImName2.end(), '\\', '/' );
    #endif
    mDir = DirOfFile(mNameIm2);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    StdCorrecNameOrient(mOri,mDir);

}

void cAppliHomProfPx::Load()
{

    //Get orientation of the second image
    std::string aIm2Ori = mDir+mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+mOri,mNameIm2,true);
    mCS2 = CamOrientGenFromFile(aIm2Ori,mICNM);

    // Read the file with Nuage parameters
    mNuage = cElNuage3DMaille::FromFileIm(mDir+mNuageName);
    mVPxT = mNuage->Params().PM3D_ParamSpecifs().DirTrans().Val();

    if (1) std::cout << "PxT:" << mVPxT << "\n";

    mImCorrel.Resize(mNuage->SzUnique());
    mImProf.Resize(mNuage->SzUnique());
    mTProf = TIm2D<float,double>(mImProf);
    mImPxT.Resize(mNuage->SzUnique());
    mTPx = TIm2D<float,double>(mImPxT);
    mMasq.Resize(mNuage->SzUnique());
    mTMasq = TIm2D<U_INT1,INT>(mMasq);

    if (EAMIsInit(&mNumCorrel))
    {
        ELISE_COPY
        (
            mImCorrel.all_pts(),
            Tiff_Im::StdConv(mDir + DirOfFile(mNuageName)+ "Correl_Geom-Im_Num_"+ToString(mNumCorrel)+ ".tif").in(0),
            mImCorrel.out()
        );

        ELISE_COPY(mImCorrel.all_pts(), mImCorrel.in()<TheSeuilC,mMasq.out());
    }

    ELISE_COPY
    (
         mImPxT.all_pts(),
         Tiff_Im::StdConv(mPxName).in(0),
         //Tiff_Im::StdConv(LocPx2FileMatch(mDir + DirOfFile(mNuageName), mNumFinal, mZoomFinal)).in(0),
         mImPxT.out()
    );

    ELISE_COPY(mImProf.all_pts(),mNuage->ImProf()->in(),mImProf.out());

    //Read cooridnates of points in the left image
    aD = StdGetObjFromFile<cDicoAppuisFlottant>
        (
            mNameCoordIm1,
            StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
            "DicoAppuisFlottant",
            "DicoAppuisFlottant"
         );

    
}

void cAppliHomProfPx::ExportHom()
{

    ElPackHomologue aPHom;

    std::list<cOneAppuisDAF>::iterator itA=aD.OneAppuisDAF().begin();
    for( ; itA!=aD.OneAppuisDAF().end(); itA++ )
    {
        Pt2dr aPtIm1(itA->Pt().x,itA->Pt().y);
        
        Pt3dr aPtTer = mNuage->PtOfIndexInterpol(aPtIm1);
        
        Pt2dr aPtIm2 = mCS2->R3toF2(aPtTer);

        Pt2dr aPtIm2Cor = aPtIm2 + mVPxT * mTPx.getr(aPtIm1);
        
        std::cout << "im1=" << aPtIm1 << "  ,im2=" << aPtIm2Cor << "\n";

        ElCplePtsHomologues aCple(aPtIm1,aPtIm2Cor);
        aPHom.Cple_Add(aCple);

    }
   
    aPHom.StdPutInFile(mOut); 

    std::cout << "Saved to: " << mOut << "\n";


}

int HomolFromProfEtPx_main(int argc,char ** argv)
{
    cAppliHomProfPx aApp(argc,argv);
    
    aApp.Load();

    aApp.ExportHom();

    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant \C3  la mise en
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
associés au chargement,  \C3  l'utilisation,  \C3  la modification et/ou au
développement et \C3  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe \C3
manipuler et qui le réserve donc \C3  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités \C3  charger  et  tester  l'adéquation  du
logiciel \C3  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
\C3  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder \C3  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

