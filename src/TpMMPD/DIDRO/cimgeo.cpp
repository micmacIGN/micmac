#include "cimgeo.h"

/* tfw convention: Xmin and Ymax are the position of the corner of the image, not of the center of the first pixel. xmin and ymax are at the edge of the box, and xmax and ymin are inside the image (center of last pixels)
 * Elise convention: first pixel has position (0,0) in Im2D. last pixel has position (sz.x-1,sz.y-1)
 */

cImGeo::cImGeo(std::string aName):
    mIm(aName.c_str())
{
   // charge l'image
   mIm = Tiff_Im::StdConvGen(aName,1,true);
   mSzImPix = mIm.sz();

   // sépare le nom et le directory
   SplitDirAndFile(mDir,mName,aName);

   // charge les donnée geo
   std::vector<double> tfw = loadTFW(mDir+mName.substr(0, mName.size()-3) + "tfw");
   mGSD = tfw[0];
   mOrigine = Pt2dr(tfw[1],tfw[2]);

   mXmin = mOrigine.x;
   mXmax = mXmin + (mSzImPix.x)*mGSD;
   mYmax = mOrigine.y;
   mYmin = mYmax - (mSzImPix.y)*mGSD;
   mBoxTer= Box2dr(Pt2dr(mXmin,mYmin),Pt2dr(mXmax,mYmax));
   mSzImTer=Pt2dr(mXmax-mXmin,mYmax-mYmin);
   mCentre=Pt2dr(mXmin+mSzImTer.x/2,mYmin+mSzImTer.y/2);

}

cImGeo::cImGeo(std::string aName, std::string aNameTFW ):
    mIm(aName.c_str())
{
   // charge l'image
   mIm = Tiff_Im::StdConvGen(aName,1,true);
   mSzImPix = mIm.sz();

   // sépare le nom et le directory
   SplitDirAndFile(mDir,mName,aName);

   // charge les donnée geo
   std::vector<double> tfw = loadTFW(mDir+aNameTFW);
   mGSD = tfw[0];
   mOrigine = Pt2dr(tfw[1],tfw[2]);

   mXmin = mOrigine.x;
   mXmax = mXmin + (mSzImPix.x)*mGSD; // minus one because the first pixel has the position xmin ymax
   mYmax = mOrigine.y;
   mYmin = mYmax - (mSzImPix.y)*mGSD;
   mBoxTer= Box2dr(Pt2dr(mXmin,mYmin),Pt2dr(mXmax,mYmax));
   mSzImTer=Pt2dr(mXmax-mXmin,mYmax-mYmin);
   mCentre=Pt2dr(mXmin+mSzImTer.x/2,mYmin+mSzImTer.y/2);
}

// si on veux utiliser l'incidence, on doit la charger séparément du constructeur, car on en a pas besoin dans tout les cas
void cImGeo::loadIncid()
{
    std::string aNameIncid;

    aNameIncid=mDir+"Incid" + mName.substr(3, mName.size());
    if (ELISE_fp::exist_file(aNameIncid))
    {
       // lecture et resize de l'incidence
       Tiff_Im aIncid = Tiff_Im::StdConvGen(aNameIncid,1,true);
       // copy to an Im2D
       double aFact(10);
       //mIncid= Im2D_REAL4(aIncid.sz().x*aFact,aIncid.sz().y*aFact );
       mIncid= Im2D_REAL4(mSzImPix.x,mSzImPix.y);
       ELISE_COPY(
                   mIncid.all_pts(),
                   StdFoncChScale(aIncid.in_proj(),Pt2dr(0,0),Pt2dr(1/aFact,1/aFact)),
                   mIncid.oclip()
                   );
       // apply the mask of nd
       ELISE_COPY(
                   select(mIncid.all_pts(),mIm.in()==0),
                   3,// valeur elevée d'incidence
                   mIncid.out()
                   );

    } else {

     std::cout << "cImGeo error, cannot load this incidence tiff image " << aNameIncid << "\n";
    }
}

// sorte de copie d'un objet cImGeo mais avec un autre nom de fichier sinon écrase le fichier existant
cImGeo::cImGeo(cImGeo * imGeoTemplate,std::string aName):  mIm(Tiff_Im(aName.c_str(),
                                                                 imGeoTemplate->Im().sz(),
                                                                 GenIm::real4,
                                                                 Tiff_Im::No_Compr,
                                                                 Tiff_Im::BlackIsZero))
{

    SplitDirAndFile(mDir,mName,aName);
    // copie l'image avec Elise
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
   mCentre=imGeoTemplate->mCentre;

   writeTFW();
}


std::vector<double> loadTFW(std::string aNameTFW)
{
    std::vector<double> result;
    double line;
    // test elise fichier existe ou pas
    if(ELISE_fp::exist_file(aNameTFW))
    {

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
    } else {
        std::cout << "Impossible d'ouvrir le fichier tfw " << aNameTFW << "\n";
        result.push_back(0);
        result.push_back(0);
        result.push_back(0);
        result.push_back(0);
    }
    return result;
}


 bool cImGeo::overlap(cImGeo * aIm2)
 {
   bool intersect(false);
    //if (((mXmax>=aIm2->Xmax() && aIm2->Xmax() >= mXmin) || (mXmax>=aIm2->Xmin() && aIm2->Xmin()>=mXmin)) && ((mYmax>=aIm2->Ymax() && aIm2->Ymax() >= mYmin) || (mYmax>=aIm2->Ymin() && aIm2->Ymin()>=mYmin))) intersect=true;
   // test im1 est contenue dans im2 ou inversément, pas pris en compte ci-dessus
    double h1=mYmax-mYmin;
    double w1=mXmax-mXmin;
    double h2=aIm2->mYmax-aIm2->mYmin;
    double w2=aIm2->mXmax-aIm2->mXmin;

    // projeter les bounding-box sur les axes et tester si les segments se recouvrent
    // recouvrement axe horizontal
    bool hoverlap = (mXmin<aIm2->Xmin()+w2) && (aIm2->Xmin()<mXmin+w1);

    // recouvrement axe vertical
    bool voverlap = (mYmin<aIm2->Ymin()+h2) && (aIm2->Ymin()<mYmin+h1);

    // recouvrement final
    intersect = hoverlap && voverlap;
   return intersect;
 }

 int cImGeo::pixCommun(cImGeo * aIm2)
 {
     Box2dr aBoxTer= this->overlapBox(aIm2);
     Im2D_REAL4 im1=this->clipImTer(aBoxTer);
     Im2D_REAL4 im2=aIm2->clipImTer(aBoxTer);

     int nbPix(0);

     ELISE_COPY(
                 select(im1.all_pts(),im1.in()!=0 && im2.in_proj()!=0),
                 1,
                 sigma(nbPix)
                 );
    return nbPix;
 }

 bool cImGeo::overlap(cImGeo * aIm2,int aRec) //est ce que l'overlap est au moin = au recouvrement donné en argument, en pourcent
 {

   double h1=mYmax-mYmin;
   double w1=mXmax-mXmin;
   double h2=aIm2->mYmax-aIm2->mYmin;
   double w2=aIm2->mXmax-aIm2->mXmin;

   // projeter les bounding-box sur les axes et tester si les segments se recouvrent
   // recouvrement axe horizontal
   bool hoverlap = (mXmin<aIm2->Xmin()+w2) && (aIm2->Xmin()<mXmin+w1);

   // recouvrement axe vertical
   bool voverlap = (mYmin<aIm2->Ymin()+h2) && (aIm2->Ymin()<mYmin+h1);

   // recouvrement final
   bool overlap = hoverlap && voverlap;
   //std::cout << "Calcul overlap entre 2 cimgeo, "<< this->Name() << " and " << aIm2->Name() << ". hoverlap :" << hoverlap << ", voverlap " << voverlap << "\n";
   //calcule le recouvrement et le compare a aRec

   if (overlap)
   {
    Box2dr overBox = this->overlapBox(aIm2);
    int hRec = 100*overBox.hauteur()/h1;
    int vRec = 100*overBox.largeur()/w1;
    if (hRec<aRec || vRec<aRec) overlap=0;
    //std::cout << "overlap  en pourcent v et h :" << vRec << " et " << hRec << "\n";
   }
   return overlap;
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
    t.x = -(aIm2->Xmin()-Xmin())/GSD();
    t.y = +(aIm2->Ymin()-Ymin())/GSD();// c'est pas Ymax plutot?
   }

   return t;
 }


Pt2di cImGeo::computeTrans(Pt2dr aPTer) // compte the translation required for the image to have the left upper corner at this points
 {

   Pt2di t(0,0);

    t.x = (aPTer.x-Xmin())/GSD();// deux signe diff car GSD y = - GSD x
    t.y = -(aPTer.y-Ymax())/GSD();
   return t;
 }



 void cImGeo::applyTrans(Pt2di aTr) // translation en pixel
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
    return writeTFW(mDir+mName);
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

Im2D_REAL4 cImGeo::clipImTer(Pt2dr aMin,Pt2dr aMax)
{
    // compute box in pixel
    int U(0), V(0);
    U = (aMax.x-aMin.x)/GSD();
    V = (aMax.y-aMin.y)/GSD();
    // initialise resulting image
    Im2D_REAL4 im(U,V);

    // inversion min max Y car en geom image l'axe des y pointe dans le sens contraire A FAIRE!! ou supprimer cette fonction

    if (containTer(aMin) && containTer(aMax))
    {
     im=clipImPix(XY2UV(aMin),XY2UV(aMax));

     //std::cout << "la box est contenue dans l'image \n";
     //std::cout << "aMin 2 UV " <<  XY2UV(aMin) << "\n";
     //std::cout << "aMax 2 UV " <<  XY2UV(aMax) << "\n";
    }
    return im;
}

Im2D_REAL4 cImGeo::clipImTer(Box2dr aBox)
{

    // compute box in pixel
    //int szU(0), szV(0);
    //szU = round((aBox.sz().x)/GSD());// again, bug if no round here, very scarce but bug may happend
    //szV = round((aBox.sz().y)/GSD());
    //Pt2di apMaxPix(XY2UV(aPMin)+Pt2di(szU,szV));
    //if(XY2UV(aPMax)!=apMaxPix) to detect bug of casting double to int. round solve the problem

    Im2D_REAL4 im(1,1);
    // inversion min max Y car en geom image l'axe des y pointe dans le sens contraire
    Pt2dr aPMin(aBox._p0.x,aBox._p1.y), aPMax(aBox._p1.x,aBox._p0.y);
    if (containTer(aBox._p0) && containTer(aBox._p1))
    {
     im=clipImPix(XY2UV(aPMin),XY2UV(aPMax));
    }
    else
    {
     std::cout << "Dans l'appel à la fonction cImGeo::clipImTer(Box2dr aBox); nous vous signalons que la box n'est pas comprise dans l'image \n";
    }

    return im;
}


Im2D_REAL4 cImGeo::clipIncidTer(Box2dr aBox)
{
    // initialise resulting image
    Im2D_REAL4 im(1,1);
    // inversion min max Y car en geom image l'axe des y pointe dans le sens contraire
    Pt2dr aPMin(aBox._p0.x,aBox._p1.y), aPMax(aBox._p1.x,aBox._p0.y);

    if (containTer(aBox._p0) && containTer(aBox._p1))
    {
     im=clipIncidPix(XY2UV(aPMin),XY2UV(aPMax));
    }
    else
    {
     std::cout << "Dans l'appel à la fonction cImGeo::clipIncidTer(Box2dr aBox); nous vous signalons que la box n'est pas comprise dans l'image \n";
    }
    return im;
}



Im2D_REAL4 cImGeo::clipImPix(Pt2di aMin,Pt2di aMax)
{
    // initialise l'image retour
    Im2D_REAL4 im(aMax.x-aMin.x,aMax.y-aMin.y);
    //std::cout << " size of clipped Im Pix :" << im.sz() << "\n";
    // translate l'imageGeo et clip en meme temps
    ELISE_COPY
   (
    Im().all_pts(),
    trans(Im().in(0),aMin),
    im.oclip()
   );
    return im;
}


Im2D_REAL4 cImGeo::clipIncidPix(Pt2di aMin,Pt2di aMax)
{
    // initialise l'image retour
    Im2D_REAL4 im(aMax.x-aMin.x,aMax.y-aMin.y);
    // translate l'imageGeo et clip en meme temp
    ELISE_COPY
   (
    Incid()->all_pts(),
    trans(Incid()->in(1),aMin),
    im.oclip()
   );
    return im;
}

bool cImGeo::containTer(Pt2dr pt)
{
  return (pt.x<=mXmax && pt.x>=mXmin && pt.y<= mYmax && pt.y>= mYmin);
}

Pt2di cImGeo::XY2UV(Pt2dr XY)// UV of Elise convention, so first pix is (0,0) not (1,1)
{
  double x=round(-1  + (XY.x-mXmin)/mGSD); // veery wierd bug, if i do not round the result, sometimes a problem occur during the convertion of double to int in the Pt2di() call
  double y=round(-1  + (mYmax-XY.y)/mGSD);
  return Pt2di(x,y);
}

Pt2dr cImGeo::UV2XY(Pt2di UV)// UV of Elise convention, so first pix is (0,0) not (1,1)
{
  double x=mXmin+mGSD*(1+UV.x);
  double y=mYmax-mGSD*(1+UV.y);
  return Pt2dr(x,y);
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

void cImGeo::display()
{
    std::cout << "Image Name ;" << mName << " \n";
    std::cout << "Frame size  ;" << Im().sz() << " \n";
    std::cout << "mXMin mXMax  ;" << mXmin << " ," << mXmax << " \n";
    std::cout << "mYMin mYMax  ;" << mYmin << " ," << mYmax << " \n";
    std::cout << "GSD  ;" << mGSD << " \n";
}

// renvoie la box terrain du recouvrement entre image 1 et image 2
Box2dr cImGeo::overlapBox(cImGeo * aIm2)
{
    Pt2dr aP1(0,0),aP2(0,0);
    if (this->overlap(aIm2) && GSD() == aIm2->GSD())
    {
      // box: pt(xmin, ymin, xmax, ymax)
     aP1.x = max(aIm2->Xmin()+aIm2->GSD(),mXmin+mGSD);// on enlève le xmin ymax qui n'as pas de pixel associé, voir convention tfw
     aP1.y = max(aIm2->Ymin(),mYmin);

     aP2.x = min(aIm2->Xmax(),mXmax);
     aP2.y = min(aIm2->Ymax()-aIm2->GSD(),mYmax-mGSD);
    }
    return Box2dr(aP1,aP2);
}


cImGeo::cImGeo(std::string aName,Im2D_REAL4 * aIm,Box2dr aBoxTer): mName(aName),mIm(Tiff_Im(aName.c_str(),
                                                                             aIm->sz(),
                                                                             GenIm::real4,
                                                                             Tiff_Im::No_Compr,
                                                                             Tiff_Im::BlackIsZero))
{
    ELISE_COPY(aIm->all_pts(),aIm->in(),mIm.out());
    mSzImPix = mIm.sz();
    // copie des autres attributs
    mGSD = (aBoxTer._p1.x-aBoxTer._p0.x)/mSzImPix.x;
    // x min, y max
    mOrigine = Pt2dr(aBoxTer._p0.x,aBoxTer._p1.y);
    mXmin = mOrigine.x;
    mXmax =  aBoxTer._p1.x;
    mYmax = aBoxTer._p1.y;
    mYmin = aBoxTer._p0.y;
    mSzImTer=Pt2dr(aBoxTer._p1.x-aBoxTer._p0.x,  aBoxTer._p1.y-aBoxTer._p1.y);
    writeTFW();
}


Im2D_REAL4 cImGeo::box2Im(Box2dr aBoxTer)
{
    Im2D_REAL4 im((aBoxTer.sz().x/GSD()),(aBoxTer.sz().y/GSD()));
    return im;
}


Box2dr cImGeo::boxEnglob(cImGeo * aIm2)
{
    Pt2dr aP1(0,0),aP2(0,0);
    if (this->overlap(aIm2))
    {
      // box: pt(xmin, ymin, xmax, ymax)
     aP1.x = min(aIm2->Xmin(),mXmin);
     aP1.y = min(aIm2->Ymin(),mYmin);

     aP2.x = max(aIm2->Xmax(),mXmax);
     aP2.y = max(aIm2->Ymax(),mYmax);
    }
    return Box2dr(aP1,aP2);
}

Im2D_REAL4 cImGeo::applyRE(c2DLineModel aMod)
{
    // on charge en mémoire RAM l'image in et out pour pouvoir utiliser elise copy sans se faire insulter rle mode machin
    Im2D_REAL4 tmpOut(SzUV().x, SzUV().y);
    Im2D_REAL4 tmpIn(toRAM());

    ELISE_COPY(
                select(tmpIn.all_pts(), tmpIn.in()!=0 && (aMod.getA() +  aMod.getB()*tmpIn.in()>=0 )),
                (aMod.getA() +  aMod.getB()*tmpIn.in()),
                tmpOut.out()
                );
    // les valeurs corrigées sont plus petites que 0; on les note à 0. Les valeurs de 0: on reste à ND
    // normalement la commande elise ci-dessus est suffisante mais le résultat part en c*** sur les bord, un peu comme si in_proj était appelé
    ELISE_COPY(
                select(tmpIn.all_pts(), tmpIn.in()==0 || tmpOut.in()<=0 ),
                0,
                tmpOut.out()
                );
    return tmpOut;
}

cFileOriMnt TFW2FileOriMnt(std::string aTFWName){
    if (ELISE_fp::exist_file(aTFWName)){
    cFileOriMnt FOM;
    std::vector<double> tfw = loadTFW(aTFWName);
    FOM.ResolutionPlani() = Pt2dr(tfw[0],-tfw[0]);
    FOM.OriginePlani() = Pt2dr(tfw[1],tfw[2]);
    // warn, cannot determine nb of pixel with only the tfw file
    return FOM;
    } else {
     std::cout << "Error : cannot read TFW file " << aTFWName << "\n";
     return cFileOriMnt();
    }
}

