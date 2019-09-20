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
        std::string mNameIm1;
        std::string mNameIm2;
        std::string mNameCoordIm1;
        std::string mOri;
        std::string mNuageName;
        std::string mPxName;
        std::string mSH;
        cInterfChantierNameManipulateur* mICNM;

        cElNuage3DMaille * mNuage;
        Pt2dr              mVPxT;
        CamStenope       * mCS1;
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

        int                 mZoomRatio;//zoom ratio between mNameCoordIm1 and mTProf and mTPx

};

cAppliHomProfPx::cAppliHomProfPx(int argc,char ** argv) :
    mDir("./"),
    mSH(""),
    mNuage     (0),
    mImCorrel  (1,1),
    mImProf    (1,1),
    mTProf     (mImProf),
    mImPxT     (1,1),
    mTPx       (mImPxT),
    mMasq      (1,1),
    mTMasq     (mMasq),
    mZoomFinal (2),
    mNumFinal  (11),
    mZoomRatio (1)
{

    
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm1,"Name of the \"first\" image of the pair", eSAM_IsPatFile)
                    << EAMC(mNameIm2,"Name of the \"second\" image of the pair", eSAM_IsPatFile)
                    << EAMC(mOri,"Input orientation directory", eSAM_IsExistFile)
                    << EAMC(mNuageName,"Path to the NuageImProf file", eSAM_IsExistFile)
                    << EAMC(mPxName,"Path to the Px2 file", eSAM_IsExistFile)
                    << EAMC(mNameCoordIm1,"Input file with pixel positions in the \"first\" image pair (DicoAppuisFlottant format)", eSAM_NoInit),
        LArgMain()  << EAM(mSH,"SH",true,"Output homol post-fix", eSAM_NoInit)
                    << EAM(mZoomRatio,"ZoomR", true, "Zoom Ratio",eSAM_IsPowerOf2)
                    << EAM(mZoomFinal,"ZoomF", true, "Zoom Final",eSAM_IsPowerOf2)
                    << EAM(mNumFinal,"NumF", true, "Num Final",eSAM_IsPowerOf2)
                    << EAM(mNumCorrel,"NumCor","true","Num Correl", eSAM_NoInit)
    );

    #if (ELISE_windows)
         replace( mNameIm2.begin(), mNameIm2.end(), '\\', '/' );
    #endif
    mDir = DirOfFile(mNameIm2);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    StdCorrecNameOrient(mOri,mDir);

}

void cAppliHomProfPx::Load()
{

    //Get orientation of the first image
    std::string aIm1Ori = mDir+mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+mOri,mNameIm1,true);
    //mCS1 = CamOrientGenFromFile(aIm1Ori,mICNM);
    mCS1 = BasicCamOrientGenFromFile(aIm1Ori);

    //Get orientation of the second image
    std::string aIm2Ori = mDir+mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+mOri,mNameIm2,true);
    //mCS2 = CamOrientGenFromFile(aIm2Ori,mICNM);
    mCS2 = BasicCamOrientGenFromFile(aIm2Ori);

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
    ElPackHomologue aPHomSym;

    std::list<cOneAppuisDAF>::iterator itA=aD.OneAppuisDAF().begin();
    for( ; itA!=aD.OneAppuisDAF().end(); itA++ )
    {
        Pt2dr aPtIm1(itA->Pt().x,itA->Pt().y);
        Pt2dr aPtIm1Full(double(aPtIm1.x)*mZoomRatio,double(aPtIm1.y)*mZoomRatio);
       

	//get the Z
        Pt3dr aPtTerZ = mNuage->PtOfIndexInterpol(aPtIm1);
	//get the XY
	Pt3dr aPtTer = mCS1->ImEtZ2Terrain(aPtIm1,aPtTerZ.z);
        //std::cout << "aPtIm1=" << aPtIm1 <<", PtTer=" << aPtTer << " " << aPtTer2 << " aPtIm2=" <<  mCS2->R3toF2(aPtTer) << " " <<  mVPxT * mTPx.getr(aPtIm1) * mZoomRatio  << "\n";
         

        if (mCS2->PIsVisibleInImage(aPtTer))
        {
            Pt2dr aPtIm2 = mCS2->R3toF2(aPtTer);
            Pt2dr aPtIm2Cor = aPtIm2 + mVPxT * mTPx.getr(aPtIm1) * mZoomRatio;//a verifier le ratio
           

            //std::cout << "im1=" << aPtIm1 << "  ,im2=" << aPtIm2 << " / " << aPtIm2Cor <<  " : " <<   mVPxT * mTPx.getr(aPtIm1) * mZoomRatio << "\n";
         
            ElCplePtsHomologues aCple(aPtIm1Full,aPtIm2Cor);
            aPHom.Cple_Add(aCple);
            aPHomSym.Cple_Add(ElCplePtsHomologues(aPtIm2Cor,aPtIm1Full));
        }
        else
            std::cout << "not visible : " << aPtIm1Full << "\n"; 

    }
   
    std::string aHomFile = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH+"@txt",mNameIm1,mNameIm2,true);
    std::string aHomSymFile = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH+"@txt",mNameIm2,mNameIm1,true);

    std::cout << "Saved to: \n";
    if (! ELISE_fp::exist_file(aHomFile))
    {
        aPHom.StdPutInFile(aHomFile);
        std::cout << aHomFile << " \n";
    }
    else
        std::cout << aHomFile << " already exists; nothing has been saved." << "\n";

    if (! ELISE_fp::exist_file(aHomSymFile))
    {
        aPHomSym.StdPutInFile(aHomSymFile);
        std::cout << aHomSymFile << "\n";
    }
    else
        std::cout << aHomSymFile << " already exists; nothing has been saved." << "\n";





}

int HomolFromProfEtPx_main(int argc,char ** argv)
{
    cAppliHomProfPx aApp(argc,argv);
    
    aApp.Load();

    aApp.ExportHom();

    return EXIT_SUCCESS;
}


class cAppli_Line2Line  
{
    public:
        cAppli_Line2Line(int argc,char ** argv);


    private:
        cInterfChantierNameManipulateur* mICNM;

        std::string mNameIm1;
        std::string mNameIm2;
        std::string mOri;
        std::string mDir;
        std::string mSH;
        double      mJumpZ;
       
        Pt2dr  mP1; 
        Pt2dr  mP2; 
        int    mX;
        int    mY;
        double mZ;
};

cAppli_Line2Line::cAppli_Line2Line(int argc,char ** argv) : 
    mSH("-L2L")
{
    
    ElPackHomologue aPHom;
    ElPackHomologue aPHomSym;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm1,"Name of the \"first\" image of the pair", eSAM_IsPatFile)
                    << EAMC(mNameIm2,"Name of the \"second\" image of the pair", eSAM_IsPatFile)
                    << EAMC(mOri,"Input orientation directory", eSAM_IsExistFile),
        LArgMain()  << EAM(mX,"X",true,"Column coordinate", eSAM_NoInit)
                    << EAM(mY,"Y",true,"Row coordinate", eSAM_NoInit)
                    << EAM(mP1,"P1",true,"Point1 defining one end of a line", eSAM_NoInit)
                    << EAM(mP2,"P2",true,"Point2 defining other end of a line", eSAM_NoInit)
                    << EAM(mZ,"ZMoy",true,"Mean Z-coordinate", eSAM_NoInit)
                    << EAM(mSH,"SH",true,"Homol postfix. Def=L2L", eSAM_NoInit)
                    << EAM(mJumpZ,"JumpZ",true,"Jump in Z we want to convert in paralx", eSAM_NoInit)
    );

    #if (ELISE_windows)
         replace( mNameIm2.begin(), mNameIm2.end(), '\\', '/' );
    #endif 
    mDir = DirOfFile(mNameIm2);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    StdCorrecNameOrient(mOri,mDir);


    cBasicGeomCap3D * aCam1 =  mICNM->StdCamGenerikOfNames(mOri,mNameIm1);
    cBasicGeomCap3D * aCam2 =  mICNM->StdCamGenerikOfNames(mOri,mNameIm2);
   
    if (!(EAMIsInit(&mZ)))
    {
        mZ = aCam1->GetAltiSol();
        std::cout << "Mean Z-coordinate: " << mZ << "\n";
    }

    if (EAMIsInit(&mJumpZ))
    {
       Pt2dr aP = aCam1->SzBasicCapt3D() / 2.0;
       if (EAMIsInit(&mP1))
          aP = mP1;

        Pt3dr aPt1Ter = aCam1->ImEtZ2Terrain(aP,mZ);
        Pt3dr aPt2Ter = aCam1->ImEtZ2Terrain(aP,mZ + mJumpZ);

        Pt2dr aP1Im2 = aCam2->Ter2Capteur(aPt1Ter);
        Pt2dr aP2Im2 = aCam2->Ter2Capteur(aPt2Ter);

        std::cout << "A Jump of " << mJumpZ << " create a paralax variation of " << aP2Im2 - aP1Im2 << "\n";
    }
    


    if (EAMIsInit(&mP1) && EAMIsInit(&mP2))
    {
        //project Im1 to 3D
        Pt3dr aPt1Ter = aCam1->ImEtZ2Terrain(mP1,mZ);
        Pt3dr aPt2Ter = aCam1->ImEtZ2Terrain(mP2,mZ);

        //back project to Im2
        Pt2dr aP1Im2 = aCam2->Ter2Capteur(aPt1Ter);
        Pt2dr aP2Im2 = aCam2->Ter2Capteur(aPt2Ter);

        aPHom.Cple_Add(ElCplePtsHomologues(mP1,aP1Im2));
        aPHom.Cple_Add(ElCplePtsHomologues(mP2,aP2Im2));
        aPHomSym.Cple_Add(ElCplePtsHomologues(aP1Im2,mP1));
        aPHomSym.Cple_Add(ElCplePtsHomologues(aP2Im2,mP2));

    }
    else if (EAMIsInit(&mX) || EAMIsInit(&mY))
    {
        Pt2di aSz = aCam1->SzBasicCapt3D();
        Pt2di aLineX;
        Pt2di aLineY;
 
 
        if (EAMIsInit(&mX)) 
        {
            aLineX = Pt2di(mX,mX+1);
            aLineY = Pt2di(0,aSz.y);
        }
        else if (EAMIsInit(&mY)) 
        {
            aLineX = Pt2di(0,aSz.x);
            aLineY = Pt2di(mY,mY+1);
        }
 
        std::cout << "Line size=" << aLineX << " " << aLineY << "\n";
 
        for (int aK1=aLineX.x; aK1<aLineX.y; aK1++)
        {
            for (int aK2=aLineY.x; aK2<aLineY.y; aK2++)
            {
                Pt2dr aPIm1(aK1,aK2);
                //project Im1 to 3D
                Pt3dr aPtTer = aCam1->ImEtZ2Terrain(aPIm1,mZ);  
                //back project to Im2
                Pt2dr aPIm2 = aCam2->Ter2Capteur(aPtTer);
 
                aPHom.Cple_Add(ElCplePtsHomologues(aPIm1,aPIm2));
                aPHomSym.Cple_Add(ElCplePtsHomologues(aPIm2,aPIm1));
 
            }
        }
    }
    else 
    {
        if (!EAMIsInit(&mJumpZ))
        {
           ELISE_ASSERT(false,"Insert either X / Y or P1/P2 parameter to indicate column or row coordinate");
        }
    }


    //save
    std::string aHomFile = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH+"@txt",mNameIm1,mNameIm2,true);
    std::string aHomSymFile = mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH+"@txt",mNameIm2,mNameIm1,true);

    std::cout << "Saved to: \n";
    if (! ELISE_fp::exist_file(aHomFile))
    {
        std::cout << aHomFile << " ";
        aPHom.StdPutInFile(aHomFile); 
    }
    if (! ELISE_fp::exist_file(aHomSymFile))
    {
        std::cout << aHomSymFile << "\n";
        aPHomSym.StdPutInFile(aHomSymFile); 
    }

}

int Line2Line_main(int argc,char ** argv)
{

    cAppli_Line2Line aAppli(argc,argv);

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

