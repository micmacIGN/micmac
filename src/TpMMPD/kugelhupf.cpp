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
 * Kugelhupf: Automatic fiducial point determination
 * Klics Ubuesques Grandement Evites, Lent, Hasardeux mais Utilisable pour Points Fiduciaux
 * Inputs:
 *  - list of images
 *  - fiducial points on one image
 * 
 * Output:
 *  - fiducial points on all images
 * 
 * Call example:
 *   mm3d Kugelhupf ".*.tif" Ori-InterneScan/MeasuresIm-1987_FR4074_0202.tif.xml SearchIncertitude=10 TargetHalfSize=32
 * 
 * TODO: support 16 bit images?
 *  2 search levels, or dezoom ?
 * */


//----------------------------------------------------------------------------

static const double TheDefCorrel = -2.0;

//Image for correlation class
//all images for correlation have the same size
class cCorrelImage
{
  public :
    cCorrelImage();
    Im2D<REAL4,REAL8> * getIm(){return &mIm;}
    TIm2D<REAL4,REAL8> * getImT(){return &mTIm;}
    double CrossCorrelation(const cCorrelImage & aIm2);
    double Covariance(const cCorrelImage & aIm2);
    static void setSzW(int aSzW);

    void getFromIm(Im2D<U_INT1,INT4> * anIm,double aCenterX,double aCenterY);

  protected:
    void prepare();//prepare for correlation (when mTifIm is set)
    Pt2di mSz;
    static int mSzW;//window size for the correlation
    TIm2D<REAL4,REAL8> mTIm; //the picture
    Im2D<REAL4,REAL8>  mIm;
    TIm2D<REAL4,REAL8> mTImS1; //the sum picture
    Im2D<REAL4,REAL8>  mImS1;
    TIm2D<REAL4,REAL8> mTImS2; //the sum² picture
    Im2D<REAL4,REAL8>  mImS2;
};

int cCorrelImage::mSzW=8;

void cCorrelImage::setSzW(int aSzW)
{
  mSzW=aSzW;
}


cCorrelImage::cCorrelImage() :
  mSz    (Pt2di(mSzW*2+1,mSzW*2+1)),
  mTIm    (mSz),
  mIm     (mTIm._the_im),
  mTImS1  (mSz),
  mImS1   (mTImS1._the_im),
  mTImS2  (mSz),
  mImS2   (mTImS2._the_im)
{}


void cCorrelImage::getFromIm(Im2D<U_INT1,INT4> * anIm,double aCenterX,double aCenterY)
{
  ELISE_COPY
    (
     mIm.all_pts(),
     anIm->in(0)[Virgule(FX+aCenterX-mSzW,FY+aCenterY-mSzW)], //put in (x,y) on destination pic what is in (x+400,y+400) in source pic
     mIm.out()
    );
  //to write to a file:
  //Tiff_Im(
  //          "toto.tif",
  //          Pt2di(400,400),
  //          GenIm::u_int1,
  //          Tiff_Im::No_Compr,
  //          Tiff_Im::BlackIsZero,
  //          Tiff_Im::Empty_ARG ).out()
  prepare();
}

void cCorrelImage::prepare()
{
  ELISE_COPY
    (
     mIm.all_pts(),
     rect_som(mIm.in_proj(),mSzW) / ElSquare(1+2*mSzW),
     mImS1.out()
    );

  ELISE_COPY
    (
     mIm.all_pts(),
     rect_som(Square(mIm.in_proj()),mSzW) / ElSquare(1+2*mSzW),
     mImS2.out()
    );

}


double cCorrelImage::CrossCorrelation( const cCorrelImage & aIm2 )
{
  //if (! InsideW(aPIm1,mSzW)) return TheDefCorrel;

  Pt2di aPIm1(mSzW,mSzW);
  Pt2di aPIm2 = aPIm1;
  //if (! aIm2.InsideW(aPIm2,mSzW)) return TheDefCorrel;

  double aS1 = mTImS1.get(aPIm1);
  double aS2 = aIm2.mTImS1.get(aPIm2);
  //std::cout<<"aS1 "<<aS1<<"   aS2 "<<aS2<<std::endl;


  double aCov = Covariance(aIm2)  -aS1*aS2;
  //std::cout<<"aCov "<<aCov<<std::endl;

  double aVar11 = mTImS2.get(aPIm1) - ElSquare(aS1);
  double aVar22 = aIm2.mTImS2.get(aPIm2) - ElSquare(aS2);
  //std::cout<<"aVar11 "<<aVar11<<"   aVar22 "<<aVar22<<std::endl;

  return aCov / sqrt(ElMax(1e-5,aVar11*aVar22));
}

double cCorrelImage::Covariance( const cCorrelImage & aIm2 )
{
  Pt2di aPIm1(mSzW,mSzW);
  if (1) // A test to check the low level access to data
  {
    float ** aRaw2 = mIm.data();
    float *  aRaw1 = mIm.data_lin();
    ELISE_ASSERT(mTIm.get(aPIm1)==aRaw2[aPIm1.y][aPIm1.x],"iiiii");
    ELISE_ASSERT((aRaw1+aPIm1.y*mSz.x) ==aRaw2[aPIm1.y],"iiiii");
  }
  double aSom =0;
  Pt2di aPIm2 = aPIm1;

  Pt2di aVois;
  /*for (aVois.x=0; aVois.x<=mSzW*2 ; aVois.x++)
    {
    for (aVois.y=0; aVois.y<=mSzW*2 ; aVois.y++)
    {
    aSom +=  mTIm.get(aPIm1+aVois) * aIm2.mTIm.get(aPIm2+aVois);
  //std::cout<<"aPIm1+aVois "<<aPIm1+aVois<<"    mTIm.get(aPIm1+aVois) "<<mTIm.get(aPIm1+aVois)<<"    aIm2.mTIm.get(aPIm2+aVois) "<<aIm2.mTIm.get(aPIm2+aVois)<<std::endl;
  } 
  }*/
  for (aVois.x=-mSzW; aVois.x<=mSzW ; aVois.x++)
  {
    for (aVois.y=-mSzW; aVois.y<=mSzW ; aVois.y++)
    {
      aSom +=  mTIm.get(aPIm1+aVois) * aIm2.mTIm.get(aPIm2+aVois);
    } 
  }

  //std::cout<<"aSom /ElSquare(1+2*mSzW) "<<aSom<<"/"<<ElSquare(1+2*mSzW)<<std::endl;
  return aSom /ElSquare(1+2*mSzW);
}


//----------------------------------------------------------------------------


// ScannedImage class
class cScannedImage
{
  public:
    cScannedImage
      (
       std::string aNameScannedImage,
       cInterfChantierNameManipulateur * aICNM,
       std::string aXmlDir
      );
    void load();
    Pt2di getSize(){return mImgSz;}
    TIm2D<U_INT1,INT4> * getImT(){if (!mIsLoaded) load();return & mImT;}
    Im2D<U_INT1,INT4> * getIm(){if (!mIsLoaded) load();return & mIm;}
    cMesureAppuiFlottant1Im & getAllFP(){return mAllFP;}//all fiducial points
    std::string getName(){return mName;}
    std::string getXmlFileName(){return mXmlFileName;}
    bool isExistingXmlFile(){return ELISE_fp::exist_file(mXmlFileName);}



  protected:
    std::string        mName;
    std::string        mNameImageTif;
    cMesureAppuiFlottant1Im mAllFP;//all fiducial points
    std::string mXmlFileName;
    Tiff_Im            mTiffIm;
    Pt2di              mImgSz;
    TIm2D<U_INT1,INT4> mImT;
    Im2D<U_INT1,INT4>  mIm;
    bool mIsLoaded;
};


cScannedImage::cScannedImage
( std::string aNameScannedImage,
  cInterfChantierNameManipulateur * aICNM,
  std::string aXmlDir  ):
  mName             (aNameScannedImage),
  mNameImageTif     (NameFileStd(mName,1,false,true,true,true)),
  mXmlFileName      (aXmlDir+"MeasuresIm-"+mName+".xml"),
  mTiffIm           (mNameImageTif.c_str()),
  mImgSz            (mTiffIm.sz()),
  mImT              (mImgSz),
  mIm               (mImT._the_im),
  mIsLoaded         (false)
{
  //std::cout<<"ScannedImageName: "<<mName<<std::endl;
  
}

void cScannedImage::load()
{
    ELISE_COPY(mIm.all_pts(),mTiffIm.in(),mIm.out());
    mIsLoaded=true;
}

//----------------------------------------------------------------------------

int Kugelhupf_main(int argc,char ** argv)
{
  std::string aFullPattern;//pattern of all scanned images
  std::string aFiducPtsFileName;//2d fiducial points of 1 image
  int aTargetHalfSzPx=64;//target size in pixel
  int aSearchIncertitudePx=5;//Search incertitude
  double aSearchStepPx=0.5;//Search step
  double aThreshold=0.9;//limit to accept a correlation
 
  bool verbose=false;

  std::cout<<"Kugelhupf (Klics Ubuesques Grandement Evites, Lent, Hasardeux mais Utilisable pour Points Fiduciaux): Automatic fiducial point determination"<<std::endl;
  

  ElInitArgMain
    (
     argc,argv,
     //mandatory arguments
     LArgMain()  << EAMC(aFullPattern, "Pattern of scanned images",  eSAM_IsPatFile)
     << EAMC(aFiducPtsFileName, "2d fiducial points of an image", eSAM_IsExistFile),
     //optional arguments
     LArgMain()  << EAM(aTargetHalfSzPx,"TargetHalfSize",true,"Target half size in pixels (Def=64)")
     << EAM(aSearchIncertitudePx,"SearchIncertitude",true,"Search incertitude in pixels (Def=5)")
     << EAM(aSearchStepPx,"SearchStep",true,"Search step in pixels (Def=0.5)")
     << EAM(aThreshold,"Threshold",true,"Limit to accept a correlation (Def=0.90)")
    );
    
  if (MMVisualMode) return EXIT_SUCCESS;

  std::cout<<"aFiducPtsFileName: "<<aFiducPtsFileName<<std::endl;

  // Initialize name manipulator & files
  std::string aDirXML,aDirImages,aPatIm;
  std::string aFiducPtsFileTmpName;
  SplitDirAndFile(aDirXML,aFiducPtsFileTmpName,aFiducPtsFileName);
  SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
  std::cout<<"Working dir: "<<aDirImages<<std::endl;
  std::cout<<"Images pattern: "<<aPatIm<<std::endl;


  cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
  const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));


  //read xml file
  //see MesureAppuiFlottant1Im definition in include/XML_GEN/ParamChantierPhotogram.xml
  cMesureAppuiFlottant1Im aDico = StdGetFromPCP(aFiducPtsFileName,MesureAppuiFlottant1Im);
  std::list< cOneMesureAF1I > aOneMesureAF1IList= aDico.OneMesureAF1I();
  std::string aMainPictureName=aDico.NameIm();
  cScannedImage aMainImg(aMainPictureName,aICNM,aDirXML);
  std::cout<<"On "<<aMainPictureName<<", found 2d points:\n";

  for (std::list<cOneMesureAF1I>::iterator itP=aOneMesureAF1IList.begin(); itP != aOneMesureAF1IList.end(); itP ++)
  {
    std::cout<<" - "<<itP->NamePt()<<" "<<itP->PtIm()<<"\n";
    aMainImg.getAllFP().OneMesureAF1I().push_back((*itP));
  }


  std::cout<<"Found pictures:\n";
  for (unsigned int i=0;i<aSetIm.size();i++)
  {
    std::cout<<" - "<<aSetIm[i]<<"\n";
  }



  Pt2di aTargetImSz(aTargetHalfSzPx*2+1,aTargetHalfSzPx*2+1);
  //Pt2di aSearchImSz(aTargetHalfSzPx*2+2*aSearchIncertitudePx+1,aTargetHalfSzPx*2+2*aSearchIncertitudePx+1);

  std::cout<<"Create sub pictures..."<<std::endl;

  cCorrelImage::setSzW(aTargetHalfSzPx);
  cCorrelImage aTargetIm;
  cCorrelImage aTargetImSearch;

  std::vector<cScannedImage*> aImgList;
  for (unsigned int i=0;i<aSetIm.size();i++)
  {
    if (aSetIm[i]==aMainPictureName)
      continue;

    std::cout<<"Working on image "<<aSetIm[i]<<std::endl;
    cScannedImage * aImg=new cScannedImage(aSetIm[i],aICNM,aDirXML);
    if (aImg->isExistingXmlFile())
    {
      std::cout<<"  Already has an xml file."<<std::endl;
      continue;
    }

    aImgList.push_back(aImg);
    aImg->getAllFP().NameIm()=aImg->getName();

    for (std::list<cOneMesureAF1I>::iterator itP=aMainImg.getAllFP().OneMesureAF1I().begin(); itP != aMainImg.getAllFP().OneMesureAF1I().end(); itP ++)
    {
      if (verbose)
        std::cout<<"  Target "<<itP->NamePt()<<"  "<<itP->PtIm()<<"\n";
      else
        std::cout<<"."<<std::flush;

      aTargetIm.getFromIm(aMainImg.getIm(),itP->PtIm().x,itP->PtIm().y);

      double aCoefCorrelMax=-1.0;
      double aTmpCoefCorrel;
      double aStepApprox=aTargetHalfSzPx/20.0;
      cOneMesureAF1I aBestPtApprox;
      cOneMesureAF1I aBestPt;
      aBestPt.NamePt()=itP->NamePt();
      aBestPtApprox.NamePt()=itP->NamePt();

      for (double x=-aSearchIncertitudePx;x<=aSearchIncertitudePx;x+=aStepApprox)
      {
        for (double y=-aSearchIncertitudePx;y<=aSearchIncertitudePx;y+=aStepApprox)
        {
          aTargetImSearch.getFromIm(aImg->getIm(),itP->PtIm().x+x,itP->PtIm().y+y);
          if (verbose)
            std::cout<<aTargetIm.CrossCorrelation(aTargetImSearch)<<"     ";
          aTmpCoefCorrel=aTargetIm.CrossCorrelation(aTargetImSearch);
          if (aTmpCoefCorrel>aCoefCorrelMax)
          {
            if (verbose)
              std::cout<<"   new best\n";
            aCoefCorrelMax=aTmpCoefCorrel;
            aBestPtApprox.PtIm()=Pt2dr(itP->PtIm().x+x,itP->PtIm().y+y);
          }
        }
        if (verbose)
          std::cout<<std::endl;
      }
      aCoefCorrelMax=-1.0;
 
      for (double x=-aStepApprox*2;x<=aStepApprox*2;x+=aSearchStepPx)
      {
        for (double y=-aStepApprox*2;y<=aStepApprox*2;y+=aSearchStepPx)
        {
          aTargetImSearch.getFromIm(aImg->getIm(),aBestPtApprox.PtIm().x+x,aBestPtApprox.PtIm().y+y);
          if (verbose)
            std::cout<<aTargetIm.CrossCorrelation(aTargetImSearch)<<"     ";
          aTmpCoefCorrel=aTargetIm.CrossCorrelation(aTargetImSearch);
          if (aTmpCoefCorrel>aCoefCorrelMax)
          {
            aCoefCorrelMax=aTmpCoefCorrel;
            aBestPt.PtIm()=Pt2dr(aBestPtApprox.PtIm().x+x,aBestPtApprox.PtIm().y+y);
          }
        }
        if (verbose)
          std::cout<<std::endl;
      }
      if (verbose)
        std::cout<<"Best: "<<aBestPt.PtIm()<<" ("<<aCoefCorrelMax<<")\n";
      if (aCoefCorrelMax>aThreshold)
      {
        aImg->getAllFP().OneMesureAF1I().push_back(aBestPt);
      }else{
        std::cout<<"Bad match on "<<itP->NamePt()<<": "<<aCoefCorrelMax<<"/"<<aThreshold<<std::endl;
        break;
      }
    }
    std::cout<<"\n";
    //write xml file only if all points found:
    if (aImg->getAllFP().OneMesureAF1I().size()==aMainImg.getAllFP().OneMesureAF1I().size())
    {
      std::cout<<"  Save xml file."<<std::endl;
      MakeFileXML(aImg->getAllFP(),aImg->getXmlFileName());
    }
  }

  return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

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
   Footer-MicMac-eLiSe-25/06/2007/*/
