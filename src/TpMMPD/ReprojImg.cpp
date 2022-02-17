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


cInterfChantierNameManipulateur * aICNM = 0;

U_INT2 LUT8to12bits[256];
U_INT1 LUT12to8bits[4096];

//color value class
class cReprojColor
{
  public:
    cReprojColor(U_INT1 r,U_INT1 g,U_INT1 b):mR(r),mG(g),mB(b){}
    void setR(U_INT1 r){mR=r;}
    void setG(U_INT1 g){mG=g;}
    void setB(U_INT1 b){mB=b;}
    U_INT1 r(){return mR;}
    U_INT1 g(){return mG;}
    U_INT1 b(){return mB;}
  protected:
    U_INT1 mR;
    U_INT1 mG;
    U_INT1 mB;
};

//----------------------------------------------------------------------------

// Image class
class cImReprojImg
{
  public:
    cImReprojImg
    (
      std::string aOriImage,
      std::string aNameImage,
      std::string aDepthImageName, //xml file
      std::string aAutoMaskImageName,
      int imageScale=1
    );
    cImReprojImg(Pt2di sz);
    ~cImReprojImg();
    cReprojColor get(Pt2di pt);
    cReprojColor getr(Pt2dr pt);
    float getDepth(Pt2dr pt); //< pt in full image coord
    float getMask(Pt2di pt); //< pt in full image coord
    void set(Pt2di pt, cReprojColor color);
    void write(std::string filename);

    Pt2di getSize(){return mImgSz;}
    Pt2di getDepthSize(){return mDepthSz;}
    Pt2di getMaskSize(){return mMaskSz;}
    TIm2D<U_INT1,INT4> * getMaskIm(){return mMaskImageT;}
    TIm2D<REAL4,REAL8> * getDepthIm(){return mDepthImageT;}
    CamStenope         * getCam(){return mCam;}
    TIm2D<U_INT1,INT4> * getImgGT(){return mImgGT;}
    Im2D<U_INT1,INT4> * getImgG(){return mImgG;}
    int getDepthScale(){return mDepthScale;}
    int getMaskScale(){return mMaskScale;}
    bool isInside(Pt2dr pt);
  protected:
    std::string        mNameImage;//reference image name
    std::string        mDepthImageName;//reference image DEM file name ("" if unknown)
    std::string        mAutoMaskImageName; //automask image filename ("" if unknown)
    CamStenope         * mCam;
    Pt2di              mImgSz;
    Pt2di              mDepthSz;
    Pt2di              mMaskSz;
    Im2D<U_INT1,INT4>  *mImgR;
    Im2D<U_INT1,INT4>  *mImgG; //only green if b&w
    Im2D<U_INT1,INT4>  *mImgB;
    Im2D<REAL4,REAL8>  *mDepthImage;
    Im2D<U_INT1,INT4>  *mMaskImage;
    TIm2D<U_INT1,INT4> *mImgRT;
    TIm2D<U_INT1,INT4> *mImgGT; //only green if b&w
    TIm2D<U_INT1,INT4> *mImgBT;
    TIm2D<REAL4,REAL8> *mDepthImageT;
    TIm2D<U_INT1,INT4> *mMaskImageT;
    int mDepthScale;
    int mMaskScale;
    double mOrigineAlti;
    double mResolutionAlti;
    int mImageScale;
};

cImReprojImg::cImReprojImg
(std::string aOriImage,
  std::string aNameImage,
  std::string aDepthImageName,
  std::string aAutoMaskImageName,
  int imageScale
) : mNameImage(aNameImage),mDepthImageName(aDepthImageName),
    mAutoMaskImageName(aAutoMaskImageName),mCam(0),mImgSz(0,0),
    mDepthSz(0,0),mMaskSz(0,0),
    mImgR(0),mImgG(0),mImgB(0),mDepthImage(0),mMaskImage(0),
    mImgRT(0),mImgGT(0),mImgBT(0),mDepthImageT(0),mMaskImageT(0),
    mDepthScale(1),mMaskScale(1),mOrigineAlti(0),mResolutionAlti(1),
    mImageScale(imageScale)
{
    std::cout<<"Create image from "<<aNameImage<<"."<<std::endl;

    std::cout<<"Read orientation from "<<aOriImage<<".\n";
    mCam=CamOrientGenFromFile(aOriImage,aICNM);

    Tiff_Im tiffImg(aNameImage.c_str());
    mImgSz.x=tiffImg.sz().x;
    mImgSz.y=tiffImg.sz().y;
    std::cout<<"Image size: "<<mImgSz.x<<"x"<<mImgSz.y<<" (scale "<<mImageScale<<").\n";
    if (tiffImg.nb_chan()==3)
    {
        mImgR=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
        mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
        mImgB=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
        mImgRT=new TIm2D<U_INT1,INT4>(*mImgR);
        mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
        mImgBT=new TIm2D<U_INT1,INT4>(*mImgB);
        ELISE_COPY(mImgR->all_pts(),tiffImg.in(),Virgule(mImgR->out(),mImgG->out(),mImgB->out()));
    }else{
        mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
        mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
        ELISE_COPY(mImgG->all_pts(),tiffImg.in(),mImgG->out());
    }
    if (aDepthImageName!="")
    {
        //read xml
        cFileOriMnt aDepthXML = StdGetFromPCP(aDepthImageName, FileOriMnt);
        mOrigineAlti=aDepthXML.OrigineAlti();
        mResolutionAlti=aDepthXML.ResolutionAlti();
        std::cout<<"Read depth from "<<aDepthXML.NameFileMnt()<<".\n";
        Tiff_Im tiffDepthImg(aDepthXML.NameFileMnt().c_str());
        mDepthSz.x=tiffDepthImg.sz().x;
        mDepthSz.y=tiffDepthImg.sz().y;
        mDepthScale=tiffImg.sz().x/tiffDepthImg.sz().x/mImageScale;
        std::cout<<"DepthScale: "<<mDepthScale<<".\n";
        mDepthImage=new Im2D<REAL4,REAL8>(tiffDepthImg.sz().x,tiffDepthImg.sz().y);
        mDepthImageT=new TIm2D<REAL4,REAL8>(*mDepthImage);
        ELISE_COPY(mDepthImage->all_pts(),tiffDepthImg.in(),mDepthImage->out());
    }
    if (aAutoMaskImageName!="")
    {
        std::cout<<"Read mask from "<<aAutoMaskImageName<<".\n";
        Tiff_Im tiffMaskImg(aAutoMaskImageName.c_str());
        mMaskSz.x=tiffMaskImg.sz().x;
        mMaskSz.y=tiffMaskImg.sz().y;
        mMaskScale=tiffImg.sz().x/tiffMaskImg.sz().x/mImageScale;
        mMaskImage=new Im2D<U_INT1,INT4>(tiffMaskImg.sz().x,tiffMaskImg.sz().y);
        mMaskImageT=new TIm2D<U_INT1,INT4>(*mMaskImage);
        ELISE_COPY(mMaskImage->all_pts(),tiffMaskImg.in(),mMaskImage->out());
    }
}

cImReprojImg::cImReprojImg(Pt2di sz) :
    mNameImage(""),mDepthImageName(""),
    mAutoMaskImageName(""),mCam(0),mImgSz(sz),
    mImgR(0),mImgG(0),mImgB(0),mDepthImage(0),mMaskImage(0),
    mImgRT(0),mImgGT(0),mImgBT(0),mDepthImageT(0),mMaskImageT(0),
    mDepthScale(1),mMaskScale(1),mImageScale(1)
{
    std::cout<<"Create image from size."<<std::endl;
    mImgR=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgG=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgB=new Im2D<U_INT1,INT4>(mImgSz.x,mImgSz.y);
    mImgRT=new TIm2D<U_INT1,INT4>(*mImgR);
    mImgGT=new TIm2D<U_INT1,INT4>(*mImgG);
    mImgBT=new TIm2D<U_INT1,INT4>(*mImgB);
}



cImReprojImg::~cImReprojImg()
{
    if (mImgR) delete mImgR;
    if (mImgG) delete mImgG;
    if (mImgB) delete mImgB;
    if (mImgR) delete mImgRT;
    if (mImgGT) delete mImgGT;
    if (mImgBT) delete mImgBT;
    if (mDepthImageT) delete mDepthImageT;
    if (mMaskImageT) delete mMaskImageT;
}

bool cImReprojImg::isInside(Pt2dr pt)
{
    return !(
                (pt.x<0) ||
                (pt.x*mImageScale>=mImgSz.x-1) ||
                (pt.y<0) ||
                (pt.y*mImageScale>=mImgSz.y-1)
            );
}

float cImReprojImg::getDepth(Pt2dr pt)
{
    if (!mDepthImageT) return NAN;
    Pt2dr aPtDepth(((float)pt.x)/getDepthScale(),
                          ((float)pt.y)/getDepthScale());
    if (aPtDepth.x>getDepthSize().x-1.0001)
        aPtDepth.x=getDepthSize().x-1.0001;
    if (aPtDepth.y>getDepthSize().y-1.0001)
        aPtDepth.y=getDepthSize().y-1.0001;

    return 1.0/(mDepthImageT->getr(aPtDepth)*mResolutionAlti + mOrigineAlti);
}


float cImReprojImg::getMask(Pt2di pt)
{
    if (!mMaskImageT) return NAN;
    Pt2di aPtMask(((float)pt.x)/getMaskScale(),
                          ((float)pt.y)/getMaskScale());

    return mMaskImageT->get(aPtMask);
}

cReprojColor cImReprojImg::get(Pt2di pt)
{
    Pt2di pt2 = pt.mul(mImageScale);
    return cReprojColor(mImgRT->get(pt2),mImgGT->get(pt2),mImgBT->get(pt2));
}

cReprojColor cImReprojImg::getr(Pt2dr pt)
{
    Pt2dr pt2 = pt.mul(mImageScale);
    return cReprojColor(mImgRT->getr(pt2),mImgGT->getr(pt2),mImgBT->getr(pt2));
}

void cImReprojImg::set(Pt2di pt, cReprojColor color)
{
    //no scale there, aImRData has the size of original image
    U_INT1 ** aImRData=mImgR->data();
    U_INT1 ** aImGData=mImgG->data();
    U_INT1 ** aImBData=mImgB->data();
    aImRData[pt.y][pt.x]=color.r();
    aImGData[pt.y][pt.x]=color.g();
    aImBData[pt.y][pt.x]=color.b();
}


void cImReprojImg::write(std::string filename)
{
    if (mImgR)
        ELISE_COPY
        (
            mImgR->all_pts(),
            Virgule( mImgR->in(), mImgG->in(), mImgB->in()) ,
            Tiff_Im(
                filename.c_str(),
                mImgSz,
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::RGB,
                Tiff_Im::Empty_ARG ).out()
        );
    else
        ELISE_COPY
        (
            mImgR->all_pts(),
            mImgG->in(),
            Tiff_Im(
                filename.c_str() ).out()
        );

}



//----------------------------------------------------------------------------



//TODO: use std::vector<Im2DGen *>  aV = aFTmp.ReadVecOfIm();



//----------------------------------------------------------------------------


//----------------------------------------------------------------------------

int ReprojImg_main(int argc,char ** argv)
{
    std::string aOriRefImage;//Orientation of ref image
    std::string aNameRefImage;//reference image name
    std::string aOriRepImage;//Orientation of rep image
    std::string aNameRepImage;//name of image to reproject
    std::string aDepthRefImageName="";//reference image DEM file name
    std::string aDepthRepImageName="";//reference image DEM file name
    std::string aAutoMaskImageName="";//automask image filename
    std::string outFileName="Reproj.tif";//output image name
    int aCoulourImgScale=1;//if color image is bigger than Ori
    bool aKeepGreen=false;//Juste change colors, not luminosity
    bool aUseLutSqrt=false;//if 8 bit values are sqrt of 12bits

    ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(aOriRefImage, "Orientation of reference image (xml)",  eSAM_IsExistDirOri)
                << EAMC(aDepthRefImageName, "Reference DEM filename (xml)", eSAM_IsExistFile)
                << EAMC(aNameRefImage, "Reference image name",  eSAM_IsExistFile)
                << EAMC(aOriRepImage, "Orientation of image to reproject (xml)",  eSAM_IsExistFile)
                << EAMC(aNameRepImage, "Name of image to reproject",  eSAM_IsExistFile),
    //optional arguments
    LArgMain()  << EAM(aAutoMaskImageName,"AutoMask",true,"AutoMask filename", eSAM_IsExistFile)
                << EAM(aDepthRepImageName,"DepthRepImage",true,"Image to reproject DEM file (xml), def=not used", eSAM_IsExistFile)
                << EAM(aKeepGreen,"KeepGreen",true,"Keep original picture green (only for colorization), def=false")
                << EAM(aUseLutSqrt,"LutSqrt",true,"Use LUT sqrt (only for colorization), def=false")
                << EAM(outFileName,"Out",true,"Output image name (tif), def=Reproj.tif")
                << EAM(aCoulourImgScale,"CoulourImgScale",true,"CoulourImgScale (int, if color image is bigger than ori to reproject), def=1")
    );

    if (MMVisualMode) return EXIT_SUCCESS;


    if (aUseLutSqrt)
    {
        //init LUTs
        int offset=16;
        float sqrt_offset=sqrt(offset);
        float kl = 256/(sqrt(4095+offset) - sqrt_offset);
        float val;
        for (unsigned int i=0;i<4096;i++)
        {
            val = (0.5+kl * (sqrt(i + offset) - sqrt_offset));
            if (val>255) val=255;
            LUT12to8bits[i]=val;
        }
        for (unsigned int i=0;i<256;i++)
        {
            val=(i * (sqrt(4095+offset)-sqrt_offset)/256 + sqrt_offset);
            val=val*val-offset;
            if (val>4095) val=4095;
            if (val<0) val=0;
            LUT8to12bits[i]= val;
        }
    }

    // Initialize name manipulator & files
    std::string aDir;
    std::string aRefImgTmpName;
    SplitDirAndFile(aDir,aRefImgTmpName,aNameRefImage);
    std::cout<<"Working dir: "<<aDir<<std::endl;
    std::cout<<"RefImgTmpName: "<<aRefImgTmpName<<std::endl;

    aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDir);

    std::string aNameRefImageTif = NameFileStd(aNameRefImage,1,false,true,true,true);
    cImReprojImg aRefIm(aOriRefImage,aNameRefImageTif,aDepthRefImageName,aAutoMaskImageName);

    std::string aNameRepImageTif = NameFileStd(aNameRepImage,3,false,true,true,true);
    cImReprojImg aRepIm(aOriRepImage,aNameRepImageTif,aDepthRepImageName,"",aCoulourImgScale);

    //Output image
    cImReprojImg aOutputIm(aRefIm.getSize());

    //Mask of reprojected image
    Im2D_U_INT1 aMaskRepIm(aRefIm.getSize().x,aRefIm.getSize().y);
    //access to each pixel value
    U_INT1 ** aMaskRepImData=aMaskRepIm.data();

    cReprojColor black(0,0,0);

    std::cout<<"Reprojecting..."<<std::endl;

    //for each pixel of reference image,
     for (int anY=0 ; anY<aRefIm.getSize().y ; anY++)
     {
         for (int anX=0 ; anX<aRefIm.getSize().x ; anX++)
         {
              //create 2D point in Ref image
              Pt2di aPImRef(anX,anY);

              float originalGreen=aRefIm.getImgGT()->get(aPImRef);
              //check if depth exists
              if ((aRefIm.getMaskIm())&&
                      (aRefIm.getMask(aPImRef)!=1))
              {
                aOutputIm.set(aPImRef,black);
                aMaskRepImData[anY][anX]=0;
                continue;
              }

              //get depth in aRefDepthTiffIm
              float aProf=aRefIm.getDepth((Pt2dr)aPImRef);
              //get 3D point
              Pt3dr aPGround=aRefIm.getCam()->ImEtProfSpherik2Terrain((Pt2dr)aPImRef,aProf);
              //ImEtProfSpherik2Terrain, not ImEtProf2Terrain!! (here aProf is a real cartesian distance, not a deltaH)

              //project 3D point into Rep image
              Pt2dr aPImRep=aRepIm.getCam()->R3toF2(aPGround);
              //check that aPImRep is in Rep image

              //output mask
              if (!aRepIm.isInside(aPImRep))
              {
                  cReprojColor color(originalGreen,originalGreen,originalGreen);
                  aOutputIm.set(aPImRef,color);
                  aMaskRepImData[anY][anX]=0;
                  continue;
              }

              //std::cout<<aPImRef<<" "<<" "<<aProf<<" "<<aPGround<<" "<<aPImRep<<std::endl;
              
              if (aRepIm.getDepthIm())
              {
                  //get depth in aRepDepthTiffIm
                  float anOtherProf=aRepIm.getDepth(aPImRep);

                  //check distance between 3d point and Rep image
                  float sqrDistRep=
                          (aPGround.x-aRefIm.getCam()->VraiOpticalCenter().x)*(aPGround.x-aRefIm.getCam()->VraiOpticalCenter().x)
                          +(aPGround.y-aRefIm.getCam()->VraiOpticalCenter().y)*(aPGround.y-aRefIm.getCam()->VraiOpticalCenter().y)
                          +(aPGround.z-aRefIm.getCam()->VraiOpticalCenter().z)*(aPGround.z-aRefIm.getCam()->VraiOpticalCenter().z);

                  //check of occlusion (DEM in Rep < 90% of real depth)
                  if (anOtherProf*anOtherProf<sqrDistRep*0.9)
                  {
                      cReprojColor color(originalGreen,originalGreen,originalGreen);
                      aOutputIm.set(aPImRef,color);
                      aMaskRepImData[anY][anX]=0;
                      continue;
                  }
              }

              //get color of this point in Rep image
              cReprojColor otherColor=aRepIm.getr(aPImRep);
              
              if (aKeepGreen)
              {
                  if (aUseLutSqrt)
                  {
                      if (otherColor.g()>20)
                      {
                          //if ((anX==2140)&&(anY==820))
                          //    std::cout<<originalGreen<<"   ";
                          originalGreen=LUT8to12bits[(int)originalGreen];
                          float otherRedOnGreen=float(LUT8to12bits[otherColor.r()])/(LUT8to12bits[otherColor.g()]);
                          float otherBlueOnGreen=float(LUT8to12bits[otherColor.b()])/(LUT8to12bits[otherColor.g()]);
                          int newRed=otherRedOnGreen*(originalGreen);
                          if (newRed<0) newRed=0;
                          if (newRed>4095) newRed=4095;
                          int newGreen=originalGreen;
                          int newBlue=otherBlueOnGreen*(originalGreen);
                          if (newBlue<0) newBlue=0;
                          if (newBlue>4095) newBlue=4095;
                          //if ((anX==2140)&&(anY==820))
                          //{
                          //    std::cout<<(int)otherColor.r()<<" "<<(int)otherColor.g()<<" "<<(int)otherColor.b()<<"   ";
                          //    std::cout<<LUT8to12bits[otherColor.r()]<<" "<<LUT8to12bits[otherColor.g()]<<" ";
                          //    std::cout<<LUT8to12bits[otherColor.b()]<<"   "<<originalGreen<<"   ";
                          //    std::cout<<otherRedOnGreen<<" "<<otherBlueOnGreen<<"   ";
                          //    std::cout<<newRed<<" "<<newGreen<<" "<<newBlue<<"   ";
                          //    std::cout<<(int)LUT12to8bits[newRed]<<" "<<(int)LUT12to8bits[newGreen]<<" "<<(int)LUT12to8bits[newBlue]<<"\n";
                          //}
                          otherColor.setR(LUT12to8bits[newRed]);
                          otherColor.setG(LUT12to8bits[newGreen]);
                          otherColor.setB(LUT12to8bits[newBlue]);
                      }else{
                          originalGreen=LUT8to12bits[(int)originalGreen];
                          //std::cout<<(int)otherColor.r()<<" "<<(int)otherColor.g()<<" "<<(int)otherColor.b()<<"   ";
                          //std::cout<<LUT8to12bits[otherColor.r()]<<" "<<LUT8to12bits[otherColor.g()]<<" ";
                          //std::cout<<LUT8to12bits[otherColor.b()]<<"   "<<originalGreen<<"   ";
                          float otherRedOnGreen=LUT8to12bits[otherColor.r()]-LUT8to12bits[otherColor.g()];
                          float otherBlueOnGreen=LUT8to12bits[otherColor.b()]-LUT8to12bits[otherColor.g()];
                          int newRed=otherRedOnGreen+originalGreen;
                          if (newRed<0) newRed=0;
                          if (newRed>4095) newRed=4095;
                          int newGreen=originalGreen;
                          int newBlue=otherBlueOnGreen+originalGreen;
                          if (newBlue<0) newBlue=0;
                          if (newBlue>4095) newBlue=4095;
                          //std::cout<<otherRedOnGreen<<" "<<otherBlueOnGreen<<"   ";
                          //std::cout<<newRed<<" "<<newGreen<<" "<<newBlue<<"   ";
                          //std::cout<<(int)LUT12to8bits[newRed]<<" "<<(int)LUT12to8bits[newGreen]<<" "<<(int)LUT12to8bits[newBlue]<<"\n";
                          otherColor.setR(LUT12to8bits[newRed]);
                          otherColor.setG(LUT12to8bits[newGreen]);
                          otherColor.setB(LUT12to8bits[newBlue]);

                      }
                  }else{
                      if (otherColor.g()<20)
                      {
                          float otherRedOnGreen=otherColor.r()-otherColor.g();
                          float otherBlueOnGreen=otherColor.b()-otherColor.g();
                          float newRed=otherRedOnGreen+originalGreen;
                          if (newRed>255) newRed=255;
                          if (newRed<0) newRed=0;
                          float newBlue=otherBlueOnGreen+originalGreen;
                          if (newBlue>255) newBlue=255;
                          if (newBlue<0) newBlue=0;
                          otherColor.setR(newRed);
                          otherColor.setG(originalGreen);
                          otherColor.setB(newBlue);
                      }else{
                          float otherRedOnGreen=otherColor.r()/(otherColor.g()+0.01);
                          float otherBlueOnGreen=otherColor.b()/(otherColor.g()+0.01);
                          float newRed=otherRedOnGreen*(originalGreen);
                          if (newRed>255) newRed=255;
                          float newGreen=originalGreen;
                          float newBlue=otherBlueOnGreen*(originalGreen);
                          if (newBlue>255) newBlue=255;
                          otherColor.setR(newRed);
                          otherColor.setG(newGreen);
                          otherColor.setB(newBlue);
                      }
                  }
              }
              
              //copy this color into output image
              aOutputIm.set(aPImRef,otherColor);
              aMaskRepImData[anY][anX]=255;
         }
     }

     std::cout<<"Write reproj image: "<<outFileName+".tif"<<std::endl;
     aOutputIm.write(outFileName+".tif");
     Tiff_Im::CreateFromIm(aMaskRepIm,outFileName+"_mask.tif");//TODO: make a xml file? convert to indexed colors?
     //TODO: create image difference!
     //use MM2D for image analysis

     return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systèmes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
