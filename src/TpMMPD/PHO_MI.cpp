
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
#include "kugelhupf.h"


#if ELISE_windows
	#define uint unsigned
#endif



/**
 * TestPointHomo: read point homologue entre 2 image après Tapioca
 * Inputs:
 *  - Nom de 2 images
 *  - Fichier de calibration du caméra
 * Output:
 *  - List de coordonné de point homologue
 * */



void StdCorrecNameHomol_G(std::string & aNameH,const std::string & aDir)
{

    int aL = strlen(aNameH.c_str());
    if (aL && (aNameH[aL-1]==ELISE_CAR_DIR))
    {
        aNameH = aNameH.substr(0,aL-1);
    }

    if ((strlen(aNameH.c_str())>=5) && (aNameH.substr(0,5)==std::string("Homol")))
       aNameH = aNameH.substr(5,std::string::npos);

    std::string aTest =  ( isUsingSeparateDirectories()?MMOutputDirectory():aDir ) + "Homol"+aNameH+ ELISE_CAR_DIR;
}


Im2D<U_INT1,INT4> CreatImageZ(Im2D<U_INT1,INT4> origin, int centreX, int centreY, int w, int h)
{
    Im2D<U_INT1,INT4> imageZ(w*2+1,h*2+1);
    ELISE_COPY
    (
        imageZ.all_pts(),                     //List de coordonne on va travailler
        origin.in()[Virgule(FX+centreX-w,FY+centreY-h)],    //entree, FX et FY va parcourir dans la list de coordonne
        imageZ.out()
    );
//    ELISE_COPY
//    (
//        imageZ.all_pts(),
//        imageZ.in()[Virgule(FX,FY)],
//        Tiff_Im(
//            "toto.tif",
//            imageZ.sz(),
//            GenIm::u_int1,
//            Tiff_Im::No_Compr,
//            Tiff_Im::BlackIsZero,
//            Tiff_Im::Empty_ARG ).out()
//    );
    return imageZ;
}


int PHO_MI_main(int argc,char ** argv)
{
    cout<<"*********************"<<endl;
    cout<<"* "<<"P : Points"<<endl;
    cout<<"* "<<"H : Homologues"<<endl;
    cout<<"* "<<"O : Observés sur"<<endl;
    cout<<"* "<<"M : Modèle"<<endl;
    cout<<"* "<<"I : Initial"<<endl;
    cout<<"*********************"<<endl;

    std::string aFullPatternImages = ".*.tif", aOriInput, aNameHomol="Homol/", aHomolOutput="Homol_Filtered/";
    bool ExpTxt = false;
    ElInitArgMain			//initialize Elise, set which is mandantory arg and which is optional arg
    (
    argc,                   //nb d’arguments
    argv,                   //chaines de caracteres contenants tous les arguments
    //mandatory arguments - arg obligatoires
    LArgMain()  << EAMC(aFullPatternImages, "Pattern of images to compute",  eSAM_IsPatFile)
                << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri),
    //optional arguments - arg facultatifs
    LArgMain()  << EAM(aNameHomol, "HomolIn", true, "Name of input Homol foler, Homol par default")
                << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                << EAM(aHomolOutput, "HomolOut" , true, "Output corrected Homologues folder")
    );
    if (MMVisualMode) return EXIT_SUCCESS;

    ELISE_fp::AssertIsDirectory(aNameHomol);

    // Initialize name manipulator & files
    std::string aDirImages, aPatImages;
    SplitDirAndFile(aDirImages,aPatImages,aFullPatternImages);
    StdCorrecNameOrient(aOriInput,aDirImages);//remove "Ori-" if needed

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetImages = *(aICNM->Get(aPatImages));

    ELISE_ASSERT(aSetImages.size()>1,"Number of image must be > 1");
    string aDirImageZ = aDirImages + "Temp_ImageZ/";
 //======================================================================//
    //Lire une image:
//    Tiff_Im mTiffImg(aSetImages[0].c_str());                      //read header of image Tiff
//    cout<<mTiffImg.sz().x<<" x "<<mTiffImg.sz().y<<endl;
//    Im2D<U_INT1,INT4> mImg(mTiffImg.sz().x,mTiffImg.sz().y);      //to read pixel, using Im2D
//    ELISE_COPY(                                                   //read image
//                mTiffImg.all_pts(),
//                mTiffImg.in(),
//                mImg.out()
//              );

//    Im2D<U_INT1,INT4> mImgOut(mTiffImg.sz().x,mTiffImg.sz().y);
//    cout<<mImgOut.sz()<<endl;

//   // Copy une Image Jet
//    int w=200;
//    Im2D<U_INT1,INT4> mImgOut2(w*2+1,w*2+1);
//    float cx=2000,cy=2000;
//    ELISE_COPY
//    (
//        mImgOut2.all_pts(),                     //List de coordonne on va travailler
//        mImg.in()[Virgule(FX+cx-w,FY+cy-w)],    //entrée, FX et FY va parcourir dans la list de coordonne
//        mImgOut2.out()
//    );
//    ELISE_COPY
//    (
//        mImgOut2.all_pts(),
//        mImgOut2.in()[Virgule(FX,FY)],
//        Tiff_Im(
//            "toto.tif",
//            mImgOut2.sz(),
//            GenIm::u_int1,
//            Tiff_Im::No_Compr,
//            Tiff_Im::BlackIsZero,
//            Tiff_Im::Empty_ARG).out()
//    );
//=========================================================================================//
    std::string anExt = ExpTxt ? "txt" : "dat";

    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aNameHomol)
                       +  std::string("@")
                       +  std::string(anExt);
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aHomolOutput)
                        +  std::string("@")
                       +  std::string(anExt);

    int h = 10;    //taille du fenetre de balayage
    int w = 10;
    int extzone = 10; //taile d'extende zone de recherche
    int step = 5;

    ELISE_fp::MkDir(aDirImages+"/temp_balayer");
    for (int aKN1 = 0 ; aKN1<int(aSetImages.size()-1) ; aKN1++)     //test just for 3 images, from point homo b/w 2, verify with 3
    {
        for (int aKN2 = 0 ; aKN2<int(aSetImages.size()-1) ; aKN2++)
        {
             std::string aNameIm1 = aSetImages[aKN1];
             std::string aNameIm2 = aSetImages[aKN2];

             //read image
             Tiff_Im mTiffImg1(aNameIm1.c_str());                      //read header of image Tiff
             Im2D<U_INT1,INT4> mImg1(mTiffImg1.sz().x,mTiffImg1.sz().y);      //to read pixel, using Im2D
             ELISE_COPY(                                                   //read image
                         mTiffImg1.all_pts(),
                         mTiffImg1.in(),
                         mImg1.out()
                       );
             Tiff_Im mTiffImg2(aNameIm2.c_str());
             Im2D<U_INT1,INT4> mImg2(mTiffImg2.sz().x,mTiffImg2.sz().y);
             ELISE_COPY(
                         mTiffImg2.all_pts(),
                         mTiffImg2.in(),
                         mImg2.out()
                       );


             std::string aNameIn = aICNM->Assoc1To2(aKHIn,aNameIm1,aNameIm2,true);  //fichier contient les point homologue à partir des pairs
             StdCorrecNameHomol_G(aNameIn,aDirImages);

             if (ELISE_fp::exist_file(aNameIn))     //check fichier point homo existe
             {
                  ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);    //lire coor de point homo dans images
                  ElPackHomologue aPackOut;
                  cout<<endl<<"There are "<<aPackIn.size()<<" point homologues b/w "<< aNameIm1<<" and "<< aNameIm2<<endl;

                  //   R3 : "reel" coorhttp://stackoverflow.com/questions/5590381/easiest-way-to-convert-int-to-string-in-cdonnee initiale
                  //   L3 : "Locale", apres rotation
                  //   C2 :  camera, avant distortion
                  //   F2 : finale apres Distortion
                  //
                  //       Orientation      Projection      Distortion
                  //   R3 -------------> L3------------>C2------------->F2

                  //get orientation information of 2 images
                  std::string anOrient = "All";
                  std::string aNameOri0 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriInput,aNameIm1,true);
                  std::cout << " ++ For image " << aNameIm1 << " ++ " << aNameOri0  << "\n";
                  CamStenope * aCam0 = CamOrientGenFromFile(aNameOri0 , aICNM);


                  std::string aNameOri1 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriInput,aNameIm2,true);
                  std::cout << " ++ For image " << aNameIm2 << " ++ " << aNameOri1  << "\n";
                  CamStenope * aCam1 = CamOrientGenFromFile(aNameOri1 , aICNM);

                  //le 3eme image pour reprojeter point homo de 1er et 2eme
                  string aNameIm = aSetImages.back();
                  std::string aNameOri3 = aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriInput,aNameIm,true);
                  CamStenope * aCam3 = CamOrientGenFromFile(aNameOri3 , aICNM);

                  Tiff_Im mTiffImg3(aNameIm.c_str());
                  Im2D<U_INT1,INT4> mImg3(mTiffImg3.sz().x,mTiffImg3.sz().y);
                  ELISE_COPY(
                              mTiffImg3.all_pts(),
                              mTiffImg3.in(),
                              mImg3.out()
                            );

                  for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
                  {
                      //lire les point homo
                      Pt2dr aP1 = itP->P1();    //Point 2d REAL
                      Pt2dr aP2 = itP->P2();
                      double d, d1;

                      //calcul coordonné 3D à partir de points homos et orientation 2 camera
                      Pt3dr PInter_Cam0= aCam0->ElCamera::PseudoInter(aP1, *aCam1, aP2, &d);       //partir de cam0

                      //reprojecter à partir de point R3 => F2
                      //Pt2dr aP1verify = aCam0->ElCamera::R3toF2(PInter_Cam0);

                      //reprojeter vers autre camera
                      Pt2dr aP3 = aCam3->ElCamera::R3toF2(PInter_Cam0);

                      //Get imagette1 and imagette 2
                      cCorrelImage::setSzW(w);
                      cCorrelImage Imgette1,Imgette2;
                      Imgette1.getFromIm(&mImg1, aP1.x, aP1.y);
                      Imgette2.getFromIm(&mImg2, aP2.x, aP2.y);

                      //Store all imagette to vector, compute correllation b/w all
                      int startZoneX=aP3.x - w - extzone;
                      int startZoneY=aP3.y - h - extzone;
                      int endZoneX=aP3.x + w + extzone;
                      int endZoneY=aP3.y + h + extzone;

                      //control if slide out of image
                      if (endZoneX > mTiffImg3.sz().x)
                          {endZoneX = mTiffImg3.sz().x;}
                      if (endZoneY > mTiffImg3.sz().y)
                          {endZoneY = mTiffImg3.sz().y;}
                      if (startZoneX <0 )
                          {startZoneX = 0;}
                      if (startZoneY  <0 )
                          {startZoneY = 0;}

                      vector< Im2D<U_INT1,INT4> > Imagette3Autour;                 //vector store all ImageZ
                      vector< double > CorrImagette3_1;
                      vector< double > CorrImagette3_2;
                      for (int ii=startZoneX; ii<endZoneX-w*2; ii=ii+step)
                      {
                          for (int jj=startZoneY; jj<endZoneY-w*2; jj=jj+step)
                          {

                              //creer ImageZ autour point d'interet reprojeter image3
                              cCorrelImage Imgette3;
                              Imgette3.getFromIm(&mImg3, ii+w, jj+h);

//                              ELISE_COPY
//                                    (
//                                          Imgette3.getIm()->all_pts(),
//                                          Imgette3.getIm()->in()[Virgule(FX,FY)],
//                                      Tiff_Im(
//                                          "tototestkukuf_1.tif",
//                                          Imgette3.getmSz(),
//                                          GenIm::u_int1,
//                                          Tiff_Im::No_Compr,
//                                          Tiff_Im::BlackIsZero,
//                                          Tiff_Im::Empty_ARG ).out()
//                                    );
                              Im2D<U_INT1,INT4> Imgette3Courrant (Imgette3.getIm()->sz().x, Imgette3.getIm()->sz().y);
                              ELISE_COPY
                                    (
                                          Imgette3.getIm()->all_pts(),
                                          Imgette3.getIm()->in()[Virgule(FX,FY)],
                                          Imgette3Courrant.out()
                                    );
                              Imagette3Autour.push_back(Imgette3Courrant);
                              CorrImagette3_1.push_back(abs(Imgette1.CrossCorrelation(Imgette3)));
                              CorrImagette3_2.push_back(abs(Imgette2.CrossCorrelation(Imgette3)));
                          }
                      }
                      cout <<"Total "<<Imagette3Autour.size()<<" imagettes +-+-+ ";
                      if (Imagette3Autour.size() > 0)
                      {
                          double max_Corr3_1 = CorrImagette3_1[*max_element(CorrImagette3_1.begin(), CorrImagette3_1.end())];
                          double max_Corr3_2 = CorrImagette3_2[*max_element(CorrImagette3_2.begin(), CorrImagette3_2.end())];
                          cout<<"Max corr value 3-1 = "<<max_Corr3_1<<" ++ 3_2 = "<<max_Corr3_2<<endl;
                      }
                      else
                      {
                          cout<<"Coor reproj sur Img3 = "<<aP3<<" ++StartCoor++ "<<Pt2dr(startZoneX,startZoneY)<<" ++EndCoor++ "<<Pt2dr(endZoneX,endZone)<<endl;
                      }
                  }
             }
        }
    }


/*
    //======================================Lire image et créer les imageZ autour point d'intéret====================//
    int h = 50;    //taille du fenetre de balayage
    int w = 50;
    int extzone = 10; //taile du zone de recherche
    int step = 5;
    ELISE_fp::MkDir(aDirImages+"/temp_balayer");
    Tiff_Im mTiffImg(aSetImages[0].c_str());                      //read header of image Tiff
    cout<<mTiffImg.sz().x<<" x "<<mTiffImg.sz().y<<endl;
    Im2D<U_INT1,INT4> mImg(mTiffImg.sz().x,mTiffImg.sz().y);      //to read pixel, using Im2D
    ELISE_COPY(                                                   //read image
                mTiffImg.all_pts(),
                mTiffImg.in(),
                mImg.out()
              );

        int startZoneX=2000 - w - extzone;
        int startZoneY=1500 - h - extzone;
        int endZoneX=2000 + w + extzone;
        int endZoneY=1500 + h + extzone;

        //control if slide out of image
        if (endZoneX > mTiffImg.sz().x)
            {endZoneX = mTiffImg.sz().x;}
        if (endZoneY > mTiffImg.sz().y)
            {endZoneY = mTiffImg.sz().y;}
        if (startZoneX <0 )
            {startZoneX = 0;}
        if (startZoneY  <0 )
            {startZoneY = 0;}

        vector< Im2D<U_INT1,INT4> > ImageZScan;                 //vector store all ImageZ
        for (int ii=startZoneX; ii<endZoneX-w*2; ii=ii+step)
        {
            for (int jj=startZoneY; jj<endZoneY-w*2; jj=jj+step)
            {
                Im2D<U_INT1,INT4>ImageZ = CreatImageZ(mImg, ii+w, jj+h, w, h);
                ImageZScan.push_back(ImageZ);
            }
        }
        cout <<"Total "<<ImageZScan.size()<<" image Z"<<endl;
        for (uint i=0; i<ImageZScan.size(); i++)
        {
            string count;
            std::stringstream out;
            out << i;
            count = out.str();
            string pathImageZ = aDirImages+"temp_balayer/"+ aSetImages.back()+ "_" + count + "_toto.tif";
            ELISE_COPY
            (
                        ImageZScan[i].all_pts(),
                        ImageZScan[i].in()[Virgule(FX,FY)],
                    Tiff_Im(
                        pathImageZ.c_str(),
                        ImageZScan[i].sz(),
                        GenIm::u_int1,
                        Tiff_Im::No_Compr,
                        Tiff_Im::BlackIsZero,
                        Tiff_Im::Empty_ARG ).out()
            );
        }



    Pt2dr aP1(30,30);
    cCorrelImage::setSzW(10);
    cCorrelImage Imgette0;
    Imgette0.getFromIm(&ImageZScan[5], aP1.x, aP1.y);

    ELISE_COPY
    (
                Imgette0.getIm()->all_pts(),
                Imgette0.getIm()->in()[Virgule(FX,FY)],
            Tiff_Im(
                "tototestkukuf_1.tif",
                Imgette0.getIm()->sz(),
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero,
                Tiff_Im::Empty_ARG ).out()
    );


    Pt2dr aP2(30,30);
    cCorrelImage Imgette1;
    Imgette1.getFromIm(&ImageZScan[6], aP2.x, aP2.y);

    ELISE_COPY
    (
                Imgette1.getIm()->all_pts(),
                Imgette1.getIm()->in()[Virgule(FX,FY)],
            Tiff_Im(
                "tototestkukuf_2.tif",
                Imgette1.getIm()->sz(),
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero,
                Tiff_Im::Empty_ARG ).out()
    );

    cout<<"CrossCorrelation 0-1 = "<<Imgette0.CrossCorrelation(Imgette1)<<endl;
    cout<<"Covariance 0-1 = "<<Imgette0.Covariance(Imgette1)<<endl;
    cout<<"CrossCorrelation 1-0 = "<<Imgette1.CrossCorrelation(Imgette0)<<endl;
    cout<<"Covariance 1-0 = "<<Imgette1.Covariance(Imgette0)<<endl;

*/

return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/


