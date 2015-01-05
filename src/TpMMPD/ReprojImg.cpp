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

/**
 * ReprojImg: reprojects an image into geometry of another
 * Inputs:
 *  - Ori of all images
 *  - correlation directory with reference image DEM
 *  - reference image name
 *  - name of image to reproject
 * 
 * Output:
 *  - image reprojected into reference orientation
 * 
 * */

int ReprojImg_main(int argc,char ** argv)
{
    //just to show args
    /*for (int aK=0 ; aK< argc ; aK++)
            std::cout<<" # "<<argv[aK]<<std::endl;*/

    std::string aOriIn;//Orientation containing all images and calibrations
    std::string aMECIn;//MEC directory with depth image of ref
    std::string aNameRefImage;//reference image name
    std::string aNameRepImage;//name of image to reproject
    
    std::string aCalibRef;//calibration of reference image
    std::string aCalibRep;//calibration of image to reproject
    
    
    ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(aOriIn, "Directory of input orientation",  eSAM_IsExistDirOri)
                << EAMC(aMECIn, "MEC directory with reference image DEM",  eSAM_IsExistDirOri)
                << EAMC(aNameRefImage, "Reference image name",  eSAM_IsExistFile)
                << EAMC(aNameRepImage, "Name of image to reproject",  eSAM_IsExistFile),
    //optional arguments
    LArgMain()  /*<< EAM(aNbXY,"NbXY",true,"Number of points of the grid")
                << EAM(aNbProf,"NbProf",true,"Number of depths of the grid")
                << EAM(aDRMax,"DRMax",true,"Max degree of distorsion (def=10)")*/
    //LArgMain()  //if no optional argument
    );
    
    //get reference orientation file name
    std::string aOriRef=aOriIn+"/Orientation-"+aNameRefImage+".xml";
    std::string aOriRep=aOriIn+"/Orientation-"+aNameRepImage+".xml";
    
    // Initialize name manipulator & files
    std::string aDir;
    std::string aRefImgTmpName;
    SplitDirAndFile(aDir,aRefImgTmpName,aNameRefImage);
    std::cout<<"Dir: "<<aDir<<std::endl;
    std::cout<<"RefImgTmpName: "<<aRefImgTmpName<<std::endl;
    
    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDir);
    CamStenope * aCamRef=CamOrientGenFromFile(aOriRef,aICNM);
    CamStenope * aCamRep=CamOrientGenFromFile(aOriRep,aICNM);
    
    std::string aDepthRefImageName;//reference image DEM file name
    aDepthRefImageName=aMECIn+"/Z_Num8_DeZoom1_STD-MALT.tif";
    std::cout<<"DepthRefImageName: "<<aDepthRefImageName<<std::endl;
    Tiff_Im aRefDepthTiffIm(aDepthRefImageName.c_str());
    
    //load color images
    //use NameFileStd to convert input files into 3-channel tiff files
    //see tiff/tiff_header.cpp:2159
    std::string aNameRefImageTif = NameFileStd(aNameRefImage,3,false,true,true,true);
    std::string aNameRepImageTif = NameFileStd(aNameRepImage,3,false,true,true,true);
    Tiff_Im aRefTiffIm(aNameRefImageTif.c_str());
    Tiff_Im aRepTiffIm(aNameRepImageTif.c_str());
    
    Pt2di aRefImgSz=aRefTiffIm.sz();
    Pt2di aRepImgSz=aRepTiffIm.sz();
    Pt2di aRefDepthImgSz=aRefDepthTiffIm.sz();
    ELISE_ASSERT(aRefImgSz==aRefDepthImgSz,"Depth image is not at DeZoom 1!");
    
    std::cout<<"aRefTiffIm.nb_chan(): "<<aRefTiffIm.nb_chan()<<std::endl;
    std::cout<<"aRepTiffIm.nb_chan(): "<<aRepTiffIm.nb_chan()<<std::endl;
    
    
    //create output image, having the same orientation as reference image
    /*Tiff_Im aOutputTiffIm(aRefTiffIm);//copy from aRefTiffIm to have the same characteristics
    std::cout<<"aOutputTiffIm.nb_chan(): "<<aOutputTiffIm.nb_chan()<<std::endl;*/
    
    //to access to pixels (see ExoMM_CorrelMulImage.cpp:151)
    Im2D_U_INT1 aRepImage(aRepImgSz.x,aRepImgSz.y);
    ELISE_COPY(aRepImage.all_pts(),aRepTiffIm.in(),aRepImage.out());
    TIm2D<U_INT1,INT4> aRepImageT(aRepImage);
    
    Im2D_REAL4 aDepthImage(aRefDepthImgSz.x,aRefDepthImgSz.y);
    ELISE_COPY(aDepthImage.all_pts(),aRefDepthTiffIm.in(),aDepthImage.out());
    TIm2D<REAL4,REAL8> aDepthImageT(aDepthImage);
    
    //Output image
    Im2D_U_INT1 aOutputTiffIm(aRefImgSz.x,aRefImgSz.y);
    
    //access to each pixel value
    U_INT1 ** aOutputTiffImData=aOutputTiffIm.data();
    
    //for each pixel of reference image,
     for (int anY=0 ; anY<aRefImgSz.y ; anY++)
     {
         for (int anX=0 ; anX<aRefImgSz.x ; anX++)
         {
              //create 2D point in Ref image
              Pt2di aPImRef(anX,anY);
              //aOutputTiffImData[anY][anX]= aRepImageData[anY][anX];
              //get depth in aRefDepthTiffIm
              //TODO: use automask
              float aProf=1/aDepthImageT.get(aPImRef);
              //get 3D point
              Pt3dr aPGround=aCamRef->ImEtProf2Terrain((Pt2dr)aPImRef,aProf);
              //project 3D point into Rep image
              Pt2dr aPImRep=aCamRep->R3toF2(aPGround);
              //check that aPImRep is in Rep image
              //std::cout<<aPImRef<<":"<<std::flush;
              //debug
              if ((anX==2115)&&(anY==912))
              {
                std::cout<<"For pixel ("<<anX<<" "<<anY<<"): z="<<aProf<<std::endl;
                std::cout<<"Reprojection is ("<<aPImRep.x<<" "<<aPImRep.y<<")"<<std::endl;
              }
              
              
              //TODO: create output mask?
              if ((aPImRep.x<0) ||(aPImRep.x>=aRepImgSz.x-1) ||(aPImRep.y<0) ||(aPImRep.y>=aRepImgSz.y-1))
              {
                //std::cout<<"x "<<std::flush;
                aOutputTiffImData[anY][anX]=0;
                continue;
              }
              
              //std::cout<<aPImRep<<" "<<std::flush;
              //get color of this point in Rep image
              //U_INT1 value=aRepImageData[(int)aPImRep.y][(int)aPImRep.x];
              U_INT1 value=aRepImageT.getr(aPImRep);
              //copy this color into output image
              aOutputTiffImData[anY][anX]=value;
         }
     }
    
    Tiff_Im::CreateFromIm(aOutputTiffIm,aNameRefImage+"_"+aNameRepImage+".tif");
    //TODO: create image difference!
    //use MM2D for image analysis
    
    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

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
Footer-MicMac-eLiSe-25/06/2007*/
