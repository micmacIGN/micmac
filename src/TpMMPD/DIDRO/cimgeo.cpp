#include "cimgeo.h"

cImGeo::cImGeo(std::string aName):
    mName(aName),
    mIm(aName.c_str())
{
   // charge l'image
   mIm = Tiff_Im::StdConvGen(aName,1,true);
   mSzImPix = mIm.sz();

   // charge les donnée geo
   std::vector<double> tfw = loadTFW(aName.substr(0, aName.size()-3) + "tfw");
   mGSD = tfw[0];
   mOrigine = Pt2dr(tfw[1],tfw[2]);
   mXmin = mOrigine.x;
   mXmax = mXmin + mSzImPix.x*mGSD;
   mYmax = mOrigine.y;
   mYmin = mYmax - mSzImPix.y*mGSD;
   mSzImTer=Pt2dr(mXmax-mXmin,mYmax-mYmin);

}

// sorte de copie d'un objet cImGeo
cImGeo::cImGeo(cImGeo * imGeoTemplate,std::string aName):  mIm(Tiff_Im(aName.c_str(),
                                                                 imGeoTemplate->Im().sz(),
                                                                 GenIm::real4,
                                                                 Tiff_Im::No_Compr,
                                                                 Tiff_Im::BlackIsZero))
{
   // copie l'image avec Elise

    mName=aName;

    ELISE_COPY
   (
    mIm.all_pts(),
    imGeoTemplate->Im().in(),
    mIm.out()
   );

   mSzImPix = mIm.sz();
   // copie des autres attributs
   mGSD = imGeoTemplate->mGSD;
   mOrigine = imGeoTemplate->mOrigine;
   mXmin = imGeoTemplate->mOrigine.x;
   mXmax =  imGeoTemplate->mXmax;
   mYmax = imGeoTemplate->mYmax;
   mYmin = imGeoTemplate->mYmin;
   mSzImTer=imGeoTemplate->mSzImTer;

   writeTFW();
}


std::vector<double> cImGeo::loadTFW(std::string aNameTFW)
{
    std::vector<double> result;
    double line;
    // test elise fichier existe ou pas
    ifstream tfwFile(aNameTFW.c_str());
    tfwFile >> line; // resol x
    result.push_back(line);
    tfwFile >> line ; // rotation =0
    tfwFile >> line; // rotation =0
    tfwFile >> line; // resol y
    if (line!=-result[0]) std::cout << "La résolution en X et Y n'est pas identique, pas prévu ça\n";
    // origine, coin supérieur gauche
    tfwFile >> line; // origine X
    result.push_back(line);
    tfwFile >> line; // origine Y
    result.push_back(line);
    tfwFile.close();

    return result;
}


 bool cImGeo::overlap(cImGeo * aIm2)
 {
   bool intersect(false);


   // projeter les bounding-box sur les axes et tester si les segments se recouvrent
   // recouvrement axe horizontal
   //boolean hoverlap = (mXmin<aIm2->Xmin()+w2) && (x2<x1+w1);

   // recouvrement axe vertical
   //boolean voverlap = (y1<y2+h2) && (y2<y1+h1);

   // recouvrement final
   //boolean overlap = hoverlap && voverlap;

    if (((mXmax>=aIm2->Xmax() && aIm2->Xmax() >= mXmin) || (mXmax>=aIm2->Xmin() && aIm2->Xmin()>=mXmin)) && ((mYmax>=aIm2->Ymax() && aIm2->Ymax() >= mYmin) || (mYmax>=aIm2->Ymin() && aIm2->Ymin()>=mYmin))) intersect=true;
   // test im1 est contenue dans im2 ou inversément

   return intersect;

 }

 void cImGeo::Save(const std::string & aName)
 {
     Tiff_Im  aTF
              (
                  aName.c_str(),
                  mIm.sz(),
                  GenIm::real4,
                  Tiff_Im::No_Compr,
                  Tiff_Im::BlackIsZero

              );

     ELISE_COPY(mIm.all_pts(),mIm.in(),aTF.out());

     // save le TFW
     writeTFW(aName);
 }



 Pt2di cImGeo::computeTrans(cImGeo * aIm2)
 {

   Pt2di t(0,0);
   if (this->overlap(aIm2) && GSD() == aIm2->GSD())
   {
    //translation u
    t.x = (aIm2->Xmin()-Xmin())/GSD();
    t.y = -(aIm2->Ymin()-Ymin())/GSD();
   }

   return t;
 }


 void cImGeo::applyTrans(Pt2di aTr)
 {
     //crée une image dans la ram pour appliquer la translation sans foirer son coup (si on écrit l'image sur elle meme cela bug si ty négatif)
     Im2D_REAL4 aImTmp=toRAM();

    // applique la translation au fichier tiff sur le disque
     ELISE_COPY
    (
     aImTmp.all_pts(),
     trans(aImTmp.in(0),aTr),
     Im().out()
    );

    // applique la translation au fichier tfw
     transTFW(aTr);
     writeTFW();
 }

int cImGeo::transTFW(Pt2di aTrPix)
{
    // aTr est une translation en pixels
    mXmax+= aTrPix.x * GSD();
    mXmin+= aTrPix.x * GSD();
    mYmax+= aTrPix.y * -GSD();
    mYmin+= aTrPix.y * -GSD();
    mOrigine.x = mXmin;
    mOrigine.y = mYmax;

    return EXIT_SUCCESS;
}

int cImGeo::writeTFW()
{
    return writeTFW(mName);
}

int cImGeo::writeTFW(std::string aName)
{
    std::string aNameTFW=aName.substr(0, aName.size()-3) + "tfw";
    std::ofstream aTFW(aNameTFW.c_str());
    aTFW.precision(12);
    aTFW << GSD() << "\n" << 0 << "\n";
    aTFW << 0 << "\n" <<  -GSD() << "\n";
    aTFW << OriginePlani().x << "\n" << OriginePlani().y << "\n";
    aTFW.close();
    return EXIT_SUCCESS;
}




Im2D_REAL4 cImGeo::toRAM()
{
    // copy the tiff_im to ram (some processing are only possible with this, or if i want to create a buffer images
    Im2D_REAL4 im(SzUV().x,SzUV().y);

    ELISE_COPY
   (
    this->Im().all_pts(),
    this->Im().in(),
    im.out()
   );
    return im;
}


int cImGeo::updateTiffIm(Im2D_REAL4 * aIm)
{
    ELISE_COPY
   (
    aIm->all_pts(),
    aIm->in(),
    Im().out()
   );

    return EXIT_SUCCESS;
}




