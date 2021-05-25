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

// List  of classes

// Class to store the application of Multi Correlation
class cMCI_Appli;
// Contains the information to store each image : Radiometry & Geometry
class cMCI_Ima;

// classes declaration

class cMCI_Ima
{
    public:
       cMCI_Ima(cMCI_Appli & anAppli,const std::string & aName);

       // A didactic test to see different way to manipulate images
       void TestManipImage();

       Pt2dr ClikIn();
       // Renvoie le saut de prof pour avoir un pixel
       void EstimateStep(cMCI_Ima *);

       void  DrawFaisceaucReproj(cMCI_Ima & aMas,const Pt2dr & aP);
       Video_Win *  W() {return mW;}
       void InitMemImOrtho(cMCI_Ima *);

       void CalculImOrthoOfProf(double aProf,cMCI_Ima * aMaster);

       Fonc_Num  FCorrel(cMCI_Ima *);
       Fonc_Num  FQuickCorrel(cMCI_Ima *);
       Pt2di Sz(){return mSz;}

    private :
       cMCI_Appli &   mAppli;
       std::string     mName;
       Tiff_Im         mTifIm;
       Pt2di           mSz;
       Im2D_U_INT1     mIm;
       Im2D_U_INT1     mImOrtho;


       Video_Win *     mW;
       std::string     mNameOri;
       CamStenope *    mCam;

};



class cMCI_Appli
{
    public :

        cMCI_Appli(int argc, char** argv);
        const std::string & Dir() const {return mDir;}
        bool ShowArgs() const {return mShowArgs;}
        std::string NameIm2NameOri(const std::string &) const;
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}

        Pt2dr ClikInMaster();

        // Function to make basic test on geometry, seize a point in one image
        // and show its projection in others
        void TestProj();
        void InitGeom();
        void AddEchInv(double aInvProf,double aStep)
        {
            mNbEchInv++;
            mMoyInvProf += aInvProf;
            mStep1Pix   += aStep;
        }
    private :
        cMCI_Appli(const cMCI_Appli &); // To avoid unwanted copies

        void DoShowArgs();



        std::string mFullName;
        std::string mDir;
        std::string mPat;
        std::string mOri;
        std::string mNameMast;
        std::list<std::string> mLFile;
        cInterfChantierNameManipulateur * mICNM;
        std::vector<cMCI_Ima *>          mIms;
        cMCI_Ima *                       mMastIm;
        bool                              mShowArgs;
        int                               mNbEchInv;
        double                            mMoyInvProf;
        double                            mStep1Pix;
};

/********************************************************************/
/*                                                                  */
/*         cMCI_Ima                                                 */
/*                                                                  */
/********************************************************************/

cMCI_Ima::cMCI_Ima(cMCI_Appli & anAppli,const std::string & aName) :
   mAppli  (anAppli),
   mName   (aName),
   mTifIm  (Tiff_Im::StdConvGen(mAppli.Dir() + mName,1,true)),
   mSz     (mTifIm.sz()),
   mIm     (mSz.x,mSz.y),
   mImOrtho (1,1), // Do not know the size for now
   mW      (0),
   mNameOri  (mAppli.NameIm2NameOri(mName)),
   mCam      (CamOrientGenFromFile(mNameOri,mAppli.ICNM()))
{
   ELISE_COPY(mIm.all_pts(),mTifIm.in(),mIm.out());
}

void cMCI_Ima::TestManipImage()
{

       std::cout << mName << mSz << "\n";
       mW = Video_Win::PtrWStd(Pt2di(1200,800));
       mW->set_title(mName.c_str());

       // Load tiff file in image and window
       ELISE_COPY(mW->all_pts(),mTifIm.in(),mW->ogray());
       std::cout << "IMAGE LOADED\n";
       mW->clik_in();

       std::cout << mNameOri
                 << " F=" << mCam->Focale()
                 << " P=" << mCam->GetProfondeur()
                 << " A=" << mCam->GetAltiSol()
                 << "\n";
        // 1- Test of mani at very low level

           // Check that copy of the image share in fact the data
        Im2D<U_INT1,INT> aImAlias = mIm;
        ELISE_ASSERT(aImAlias.data()==mIm.data(),"Data");

           // Test of miroring a part of image
        U_INT1 ** aData = aImAlias.data();
        for (int anY=0 ; anY<ElMin(mSz.y,mSz.x) /2 ; anY++)
        {
            for (int anX=0 ; anX<anY ; anX++)
            {
                 ElSwap(aData[anY][anX],aData[anX][anY]);
            }
        }
        ELISE_COPY(mW->all_pts(),mIm.in(),mW->ogray());
        std::cout << "TEST MIRORING LOW LEVEL\n";
        mW->clik_in();

           // Check that 2D representation en 1D share the same memory space
        U_INT1 * aDataL = aImAlias.data_lin();
        for (int anY=0 ; anY<mSz.y ; anY++)
        {
            ELISE_ASSERT
            (
                aData[anY]==(aDataL+anY*mSz.x),
                "data"
            );
         }
         memset(aDataL+mSz.x*50,128,mSz.x*100);


         // 2- Test with functional approach

          // Reload
         ELISE_COPY
         (
            mIm.all_pts(),
            mTifIm.in(),
            mIm.out() | mW->ogray()
          );

          // Show in window the negative value,
          // 255 is the constant function (a,b,...) -> 255
          //  "-" is an operator on funtion
          //  f1-f2  :  (a,b,...) -> f1(a,b..) -f2(a,b...)
         ELISE_COPY
         (
            disc(Pt2dr(200,200),150),
            255-mIm.in(),
             mW->ogray()
          );
          std::cout << "SHOW NEGATIVE\n";
          mW->clik_in();

         // Use the [] operator which is compositon of function
         // Use the Virgule operator which result is concatenation
         // FX and FY predefined function, return coordinates of point
         //  FX    (a,b,...) -> a
         //  FY    (a,b,...) -> b
         // Virgule(FY,FX)  (a,b,...) -> (b,a)
         //  This call show the negative of the mirror image on a disc
    ELISE_COPY
         (
            disc(Pt2dr(200,200),150),
            255-mIm.in()[Virgule(FY,FX)],
            //
             mW->ogray()
          );
          std::cout << "SHOW NEGATIVE+MIRROR in Functionnal way\n";
          mW->clik_in();


         // rect_som  is an operator (fiter) on function, compute for each
         // pixel the sum of F on rectangular neighborhood
         //  rect_som(F,10) :(a,b)->Sum (x in [a-10,a+10]x[b-10,b+10]) F(x)
         int aSzF = 20;
         ELISE_COPY
         (
            rectangle(Pt2di(0,0),Pt2di(400,500)),
            // rect_som(mIm.in(),20)/ElSquare(1+2*aSzF),
            rect_som(mIm.in_proj(),aSzF)/ElSquare(1+2*aSzF),
            // Output is directed both in window & Im
             mW->ogray()
         );
         std::cout << "Show the result of filtering by averaging\n";
         mW->clik_in();


         // Compute a function with the application of 5 iteration of
         // the average filter
         Fonc_Num aF = mIm.in_proj();
         aSzF=4;
         int aNbIter = 5;
         for (int aK=0 ; aK<aNbIter ; aK++)
              aF = rect_som(aF,aSzF)/ElSquare(1+2*aSzF);

         ELISE_COPY(mIm.all_pts(),aF,mW->ogray());

         std::cout << "5 iteration average, functionnal mode\n";
         mW->clik_in();

        //  ELISE_COPY(mIm.all_pts(),mTifIm.in(),mIm.out());


         ELISE_COPY(mIm.all_pts(),mIm.in(),mW->ogray());

         // 3- Test with Tpl  approach

         Im2D<U_INT1,INT4> aDup(mSz.x,mSz.y);
         TIm2D<U_INT1,INT4> aTplDup(aDup);
         TIm2D<U_INT1,INT4> aTIm(mIm);

         for (int aK=0 ; aK<aNbIter ; aK++)
         {
            for (int anY=0 ; anY<mSz.y ; anY++)
                for (int anX=0 ; anX<mSz.x ; anX++)
                {
                    int aNbVois = ElSquare(1+2*aSzF);
                    int aSom=0;
                    for (int aDx=-aSzF; aDx<=aSzF ; aDx++)
                       for (int aDy=-aSzF; aDy<=aSzF ; aDy++)
                           aSom += aTIm.getproj(Pt2di(anX+aDx,anY+aDy));
                    aTplDup.oset(Pt2di(anX,anY),aSom/aNbVois);
                }

            Pt2di aP;
            for (aP.y=0 ; aP.y<mSz.y ; aP.y++)
                for (aP.x=0 ; aP.x<mSz.x ; aP.x++)
                    aTIm.oset(aP,aTplDup.get(aP));
         }
         ELISE_COPY(mIm.all_pts(),mIm.in(),mW->ogray());

         std::cout << "5 iteration average, pixel manipulation\n";
         mW->clik_in();

}

// For a given depth, compute the rectified image in the geometry of
// Master using the plane parallel to Master;  to speed up the computation,
// a grid of correspondance is first computed, the it is used to interpolate

void cMCI_Ima::CalculImOrthoOfProf(double aProf,cMCI_Ima * aMaster)
{

    TIm2D<U_INT1,INT> aTIm(mIm);  // Initial image
    TIm2D<U_INT1,INT> aTImOrtho(mImOrtho); // image resampled in Master geometry

    // Step of the grid
    int aSsEch = 10;
    Pt2di aSzR = aMaster->mSz/ aSsEch;

    // Store the mappind grid  Phi : Master -> Image a given depth,
    //  Phi(u,v) =  (aImX(u,v),aImY(u,v))
    TIm2D<float,double> aImX(aSzR);
    TIm2D<float,double> aImY(aSzR);

    Pt2di aP;
    for (aP.x=0 ; aP.x<aSzR.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSzR.y; aP.y++)
        {
            // aP = point of the grid, aP*aSsEch = correspond point in Master
            // aPTer = corresponding 3D point in the plane at given depth
            // aPIm = projection of PTer in my geometry
            Pt3dr aPTer = aMaster->mCam->ImEtProf2Terrain(Pt2dr(aP*aSsEch),aProf);
            Pt2dr aPIm = mCam->R3toF2(aPTer);
            // Store the result in grid
            aImX.oset(aP,aPIm.x);
            aImY.oset(aP,aPIm.y);
        }
    }

    for (aP.x=0 ; aP.x<aMaster->mSz.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<aMaster->mSz.y; aP.y++)
        {
            /*  Exact formula, that would be used in case of no grid :
            Pt3dr aPTer = aMaster->mCam->ImEtProf2Terrain(Pt2dr(aP),aProf);
            Pt2dr aPIm0 = mCam->R3toF2(aPTer);
            */
            //  aP : master image point
            // aPInt : float point in the grid
            // aPIm : homologous of aP in my geometry
            // aVal : value of my image at aPIm
            Pt2dr aPInt = Pt2dr(aP) / double(aSsEch);
            Pt2dr aPIm (aImX.getr(aPInt,0),aImY.getr(aPInt,0));
            float aVal = aTIm.getr(aPIm,0);
            aTImOrtho.oset(aP,round_ni(aVal));
        }
    }

    //if ( 0 &&  (mName=="Abbey-IMG_0250.jpg"))
	if (1)
    {
        static Video_Win * aW = Video_Win::PtrWStd(Pt2di(1200,800));
        ELISE_COPY(mImOrtho.all_pts(),mImOrtho.in(),aW->ogray());
    }
}

/*
    Use the formula

      Corr(I1,I2) = [ E(I1*I2) - E(I1)*E(I2)] / sqrt[(E^2(I1) -E(I1^2))*(E^2(I2) -E(I2^2))]
      Here the "expectation" is computed by the average operator :

       E(I) =  rect_som(aF1,aSzW) / (1+2*aNbW) ^2
*/
Fonc_Num  cMCI_Ima::FCorrel(cMCI_Ima *aMaster)
{
    int aSzW = 2;
    double aNbW = ElSquare(1+2*aSzW); // Number of Pixel in window
    Fonc_Num aF1 = mImOrtho.in_proj();
    Fonc_Num aF2 = aMaster->mImOrtho.in_proj();


    Fonc_Num aS1 = rect_som(aF1,aSzW) / aNbW; // E(I1)
    Fonc_Num aS2 = rect_som(aF2,aSzW) / aNbW; // E(I2)

    Fonc_Num aS12 = rect_som(aF1*aF2,aSzW) / aNbW - aS1*aS2; // E(I1*I2) - E(I1)*E(I2)
    Fonc_Num aS11 = rect_som(Square(aF1),aSzW) / aNbW - Square(aS1); // (E^2(I1) -E(I1^2))
    Fonc_Num aS22 = rect_som(Square(aF2),aSzW) / aNbW - Square(aS2); // E(I1^2))*(E^2(I2) -E(I2^2))

   // Corr(I1,I2) : aS11 and aS22 should be >=0 , Take care of null
   // and negative value, possible because of numerical errors
    Fonc_Num aRes = aS12 / sqrt(Max(1e-5,aS11*aS22));


    return aRes;
}

// Optimized computation of correlation using the Symb_FNum, which avoid
// multiple computation of the same values
Fonc_Num  cMCI_Ima::FQuickCorrel(cMCI_Ima * aMaster)
{
   int aSzW=2;
   double aNbW = ElSquare(1+2*aSzW);
   Symb_FNum aS1 (mImOrtho.in_proj());
   Symb_FNum aS2 (aMaster->mImOrtho.in_proj());

   Symb_FNum  aSom =
rect_som(Virgule(aS1,aS2,aS1*aS2,Square(aS1),Square(aS2)),aSzW);

   Symb_FNum aSum1  (aSom.kth_proj(0) / aNbW);
   Symb_FNum aSum2   (aSom.kth_proj(1)  / aNbW);
   Symb_FNum aSum12  (aSom.kth_proj(2)  / aNbW -aSum1 * aSum2);
   Symb_FNum aSum11  (aSom.kth_proj(3)  / aNbW -Square(aSum1));
   Symb_FNum aSum22   (aSom.kth_proj(4)  / aNbW - Square(aSum2)) ;

    Fonc_Num aRes = aSum12 / sqrt(Max(1e-5,aSum11*aSum22));


    return aRes;
}

void cMCI_Ima::InitMemImOrtho(cMCI_Ima * aMas)
{
    mImOrtho.Resize(aMas->mIm.sz());
}

Pt2dr cMCI_Ima::ClikIn()
{
    return mW->clik_in()._pt;
}

void  cMCI_Ima::DrawFaisceaucReproj(cMCI_Ima & aMas,const Pt2dr & aP)
{
    if (! mW) return ;
    double aProfMoy =  aMas.mCam->GetProfondeur(); // Get average depth
    double aCoef = 1.2;

    std::vector<Pt2dr> aVProj;
    // Parse a "big" interval of depth to get the 3D bundle
    for (double aMul = 0.2; aMul < 5; aMul *=aCoef)
    {
         // Get the 3D point correspond to aP in aMas, and with given depth
         Pt3dr aP3d =  aMas.mCam->ImEtProf2Terrain(aP,aProfMoy*aMul);
         // Project in this image
         Pt2dr aPIm = this->mCam->R3toF2(aP3d);

         // stack the result
         aVProj.push_back(aPIm);
    }
    // Show the segments
    for (int aK=0 ; aK<((int) aVProj.size()-1) ; aK++)
        mW->draw_seg(aVProj[aK],aVProj[aK+1],mW->pdisc()(P8COL::red));
}

//  Each image adds its own contribution to the estimation of average depth
// and steps that correspond to a displacement of 1 pixel
//
//  This estimation is done using the tie points resulting from Tapioca

void cMCI_Ima::EstimateStep(cMCI_Ima * aMas)
{
   // Get the name of tie points name using the key
   std::string aKey = "NKS-Assoc-CplIm2Hom@@dat";
   std::string aNameH =   mAppli.Dir()
                        + mAppli.ICNM()->Assoc1To2
                        (
                            aKey,
                            this->mName,
                            aMas->mName,
                            true
                        );
   // Load the tie points
   ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);

   // Axis of master camera in ground geometry
   Pt3dr aDirK = aMas->mCam->DirK();
   for
   (
       ElPackHomologue::iterator iTH = aPack.begin();
       iTH != aPack.end();
       iTH++
   )
   {
       Pt2dr aPInit1 = iTH->P1();
       Pt2dr aPInit2 = iTH->P2();

       // Compute the 3D point from tie point by "pseudo" intersection of
       // bundles
       double aDist;
       Pt3dr aTer = mCam->PseudoInter(aPInit1,*(aMas->mCam),aPInit2,&aDist);

       double aProf2 = aMas->mCam->ProfInDir(aTer,aDirK);

       // Compute the "real" projection (slightly different of aPInit1)
       Pt2dr aProj1 = mCam->R3toF2(aTer);
       Pt2dr aProj2 = aMas->mCam->R3toF2(aTer);

      // std::cout << aMas->mCam->ImEtProf2Terrain(aProj2,aProf2) -aTer << "\n";


       if (0)  // Check if aPInit ~ aProj1
          std::cout << "Ter " << aDist << " " << aProf2
                 << " Pix " << euclid(aPInit1,aProj1)
                 << " Pix " << euclid(aPInit2,aProj2) << "\n";

       // Modify slightly the depth to compute the effect on P1
       double aDeltaProf = aProf2 * 0.0002343;
       Pt3dr aTerPert = aMas->mCam->ImEtProf2Terrain
                      (aProj2,aProf2+aDeltaProf);

       Pt2dr aProjPert1 = mCam->R3toF2(aTerPert);

       // Ratio variation in depth (meters) - variation in projection (pixel)
       double aDelta1Pix = aDeltaProf / euclid(aProj1,aProjPert1);

       // Ratio variation in 1/depth  - variation in projection, use
       // D(1/x) = D(x) / x2
       double aDeltaInv = aDelta1Pix / ElSquare(aProf2);
       // std::cout << "Firts Ecart " << aDelta1Pix << " "<< aDeltaInv  << "\n";

       // accumulate for averaging, invert prof and step in invert prof
       mAppli.AddEchInv(1/aProf2,aDeltaInv);
   }
}

/********************************************************************/
/*                                                                  */
/*         cMCI_Appli                                               */
/*                                                                  */
/********************************************************************/


cMCI_Appli::cMCI_Appli(int argc, char** argv):
    mNbEchInv (0),
    mMoyInvProf (0),
    mStep1Pix    (0)
{
    // Reading parameter : check and  convert strings to low level objects
    mShowArgs=false;
    bool mTestIm = false;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pat)")
                    << EAMC(mNameMast,"Name of Master Image")
                    << EAMC(mOri,"Used orientation"),
        LArgMain()  << EAM(mShowArgs,"Show",true,"Give details on args")
                    << EAM(mTestIm,"TestIm",true,"Test the basic operations on image")
    );

    // Initialize name manipulator & files
    SplitDirAndFile(mDir,mPat,mFullName);
    mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mLFile = mICNM->StdGetListOfFile(mPat);

    StdCorrecNameOrient(mOri,mDir);

    if (mShowArgs) DoShowArgs();

    // Initialize all the images structure
    mMastIm = 0;
    for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
     {
           cMCI_Ima * aNewIm = new  cMCI_Ima(*this,*itS);

           mIms.push_back(aNewIm);
           if (*itS==mNameMast)
           {
               mMastIm = aNewIm;
               if (mTestIm) mMastIm->TestManipImage();
           }
     }

     // Ckeck the master is included in the pattern
     ELISE_ASSERT
     (
        mMastIm!=0,
        "Master image not found in pattern"
     );

     if (mShowArgs)
        TestProj();

     InitGeom();
     Pt2di aSz = mMastIm->Sz();
     Im2D_REAL4 aImCorrel(aSz.x,aSz.y);
     Im2D_REAL4 aImCorrelMax(aSz.x,aSz.y,-10);
     Im2D_INT2  aImPax(aSz.x,aSz.y);


     double aStep = 0.5;

     //  Parse the different depths
     for (int aKPax = -60 ; aKPax <=60 ; aKPax++)
     {
         std::cout << "ORTHO at " << aKPax << "\n";
         // Use the average value to get InvDepth and Depth
         double aInvProf = mMoyInvProf + aKPax * mStep1Pix * aStep;
         double aProf = 1/aInvProf;

         // Compute all ortho photos
         for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
              mIms[aKIm]->CalculImOrthoOfProf(aProf,mMastIm);

         // Compute the global correlation as sum of individual
         Fonc_Num aFCorrel = 0;
         for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
         {
             cMCI_Ima * anIm = mIms[aKIm];
             if (anIm != mMastIm)
                 aFCorrel = aFCorrel+anIm->FQuickCorrel(mMastIm);
         }
         // Store the result in aImCorrel
         ELISE_COPY(aImCorrel.all_pts(),aFCorrel,aImCorrel.out());
         // Udate the Pax for points the increase the current best correl
         ELISE_COPY
         (
            select(aImCorrel.all_pts(),aImCorrel.in()>aImCorrelMax.in()),
            Virgule(aImCorrel.in(),aKPax),
            Virgule(aImCorrelMax.out(),aImPax.out())
         );
     }
     Video_Win aW = Video_Win::WStd(Pt2di(1200,800),true);
     ELISE_COPY(aW.all_pts(),aImPax.in()*6,aW.ocirc());
     aW.clik_in();

}

void cMCI_Appli::InitGeom()
{
    for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
    {
        cMCI_Ima * anIm = mIms[aKIm];
        if (anIm != mMastIm)
        {
            anIm->EstimateStep(mMastIm) ;
        }
        anIm->InitMemImOrtho(mMastIm) ;
    }
    mMoyInvProf /= mNbEchInv;
    mStep1Pix /= mNbEchInv;
}

void cMCI_Appli::TestProj()
{
    if (! mMastIm->W()) return;
    while (1)
    {
        // Get a point
        Pt2dr aP = ClikInMaster();
        for (int aKIm=0 ; aKIm<int(mIms.size()) ; aKIm++)
        {
            // Show the bundle in others
            mIms[aKIm]->DrawFaisceaucReproj(*mMastIm,aP);
        }
    }
}


Pt2dr cMCI_Appli::ClikInMaster()
{
    return mMastIm->ClikIn();
}


std::string cMCI_Appli::NameIm2NameOri(const std::string & aNameIm) const
{
    return mICNM->Assoc1To1
    (
        "NKS-Assoc-Im2Orient@-"+mOri+"@",
        aNameIm,
        true
    );
}

void cMCI_Appli::DoShowArgs()
{
     std::cout << "DIR=" << mDir << " Pat=" << mPat << " Orient=" << mOri<< "\n";
     std::cout << "Nb Files " << mLFile.size() << "\n";
     for (
              std::list<std::string>::iterator itS=mLFile.begin();
              itS!=mLFile.end();
              itS++
              )
      {
              std::cout << "    F=" << *itS << "\n";
      }
}

/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

int ExoMCI_main(int argc, char** argv)
{
   cMCI_Appli anAppli(argc,argv);

   return EXIT_SUCCESS;
}


int ExoMCI_2_main(int argc, char** argv)
{
   std::string aNameFile;
   double D=1.0;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameFile,"Name of GCP file"),
        LArgMain()  << EAM(D,"D",true,"Unused")
    );

    cDicoAppuisFlottant aDico=  StdGetFromPCP(aNameFile,DicoAppuisFlottant);


    std::cout << "Nb Pts " <<  aDico.OneAppuisDAF().size() << "\n";
    std::list<cOneAppuisDAF> & aLGCP =  aDico.OneAppuisDAF();

    for (
             std::list<cOneAppuisDAF>::iterator iT= aLGCP.begin();
             iT != aLGCP.end();
             iT++
    )
    {
        // iT->Pt() equiv a (*iTp).Pt()
        std::cout << iT->NamePt() << " " << iT->Pt() << "\n";
    }

   return EXIT_SUCCESS;
}

int ExoMCI_1_main(int argc, char** argv)
{
   int I,J;
   double D=1.0;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(I,"Left Operand")
                    << EAMC(J,"Right Operand"),
        LArgMain()  << EAM(D,"D",true,"divisor of I+J")
    );

    std::cout << "(I+J)/D = " <<  (I+J)/D << "\n";

   return EXIT_SUCCESS;
}


int ExoMCI_0_main(int argc, char** argv)
{

   for (int aK=0 ; aK<argc ; aK++)
      std::cout << " Argv[" << aK << "]=" << argv[aK] << "\n";

   return EXIT_SUCCESS;
}

// In this exercise, some computation can be made in 3 modes for illustrationb
typedef enum
{
   eMCEM_Func =0,  // Functionnal mode with ELISE_COPY
   eMCEM_Tpl  =1,   // Using the "securized" access to images
   eMCEM_Raw  =2,    // Using raw data
   eMCEM_Raw1D =3
} eModeComputeEM;

int  ExoCorrelEpip_main(int argc,char ** argv)
{
    std::string aNameI1,aNameI2;
    int aPxMax= 199;
    int aSzW = 5;

    int aImode=0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameI1,"Name Image1")
                    << EAMC(aNameI2,"Name Image2"),
        LArgMain()  << EAM(aPxMax,"PxMax",true,"Pax Max")
                    << EAM(aSzW,"SzW",true,"Size of Window, def=5")
                    << EAM(aImode,"Mode",true,"Mode comput 0=Func, 1=Tpl, 2=Raw, 3=Raw1D")
    );
    eModeComputeEM aMode = (eModeComputeEM) aImode;

    // Looding the images from file to memory
    Im2D_U_INT1 aI1 = Im2D_U_INT1::FromFileStd(aNameI1);
    Im2D_U_INT1 aI2 = Im2D_U_INT1::FromFileStd(aNameI2);

    Pt2di aSz1 = aI1.sz();
    Pt2di aSz2 = aI2.sz();
    // Tempory images

    // Image of best score, initialized to "infinity" value
    Im2D_REAL4  aIScoreMin(aSz1.x,aSz1.y,1e10);
    Im2D_REAL4  aIScore(aSz1.x,aSz1.y);   // Image of current score
    Im2D_INT2   aIPaxOpt(aSz1.x,aSz1.y);  // Image giving the best paralax

    // Create object("wraper") for "safe" manipulation of each pixels
    TIm2D<U_INT1,INT> aTI1(aI1);
    TIm2D<U_INT1,INT> aTI2(aI2);
    TIm2D<REAL4,REAL> aTScoreMin(aIScoreMin);
    TIm2D<REAL4,REAL> aTScore(aIScore);
    TIm2D<INT2,INT> aTPaxOpt(aIPaxOpt);

    // Create object("wraper") for "safe" manipulation of each pixels
    U_INT1 ** aDI1 = aI1.data();
    U_INT1 ** aDI2 = aI2.data();

    // For visualization
    Video_Win aW = Video_Win::WStd(Pt2di(1200,800),true);

    for (int aPax = -aPxMax ; aPax <=aPxMax ; aPax++)
    {
        // Exact limit of domain, used for non functionnal mode to avoid
        // acces out of image domains

    int aX0 = aSzW+ElMax(0,-aPax);
        int aX1 = ElMin(aSz2.x+ElMin(0,-aPax),aSz1.x)-aSzW ;
        int aY0 = aSzW;
        int aY1 = ElMin(aSz1.y,aSz2.y) - aSzW;

        std::cout << "PAX tested " << aPax << "\n";
        // Function of image translated
        Fonc_Num aI2Tr = trans(aI2.in_proj(),Pt2di(aPax,0));

        // Compute the similarity by sum of differences on a square window
        // of size [-aSzW,aSzW] x [-aSzW,aSzW]

        // Do 4 times the same computation using different "programming style"
        // just for didactic purpose


        switch (aMode)
        {
            // Using "high level" functionnal mode
            case eMCEM_Func :
               ELISE_COPY
               (
                  aI1.all_pts(),
                  rect_som(Abs(aI1.in_proj()-aI2Tr),aSzW),
                  aIScore.out()
               );
            break;

            // Using explicit access to each pixel , with checking of image limit
            case eMCEM_Tpl :
            {
               Pt2di aP;
               for (aP.x = 0 ; aP.x<aSz1.x ; aP.x++)
               {
                  for (aP.y = 0 ; aP.y<aSz1.y ; aP.y++)
                  {
                      Pt2di aV;
                      double aSomDif=0;
                      for (aV.x = -aSzW ; aV.x<=aSzW ; aV.x++)
                      {
                         for (aV.y = -aSzW ; aV.y<=aSzW ; aV.y++)
                         {
// get proj : "safe" access to images, if pixel is outside image, return
// the value of nearest pixel inside
                            int aV1 = aTI1.getproj(aP+aV);
                            int aV2 = aTI2.getproj(aP+aV+Pt2di(aPax,0));
                aSomDif += ElAbs(aV1-aV2);
                         }
                      }
                      aTScore.oset(aP,aSomDif);
                  }
               }
            }
            break;

            // Using raw access to pixel, relatively "clean" code
            case eMCEM_Raw :
            {
               Pt2di aP;

               for (aP.x = aX0 ; aP.x<aX1 ; aP.x++)
               {
                  for (aP.y = aY0 ; aP.y<aY1 ; aP.y++)
                  {
                      Pt2di aV;
                      double aSomDif=0;
                      for (aV.x = -aSzW ; aV.x<=aSzW ; aV.x++)
                      {
                         for (aV.y = -aSzW ; aV.y<=aSzW ; aV.y++)
                         {
                            Pt2di aQ1 = aP+aV;
                            Pt2di aQ2 = aP+aV+Pt2di(aPax,0);

                            int aV1 = aDI1[aQ1.y][aQ1.x];
                            int aV2 = aDI2[aQ2.y][aQ2.x];
                aSomDif += ElAbs(aV1-aV2);
                         }
                      }
                      aTScore.oset(aP,aSomDif);
                  }
               }
            }
            break;

            // Using optimzed Old C style code (*pt++ ...)
            case eMCEM_Raw1D :
            {
               Pt2di aP;

               for (aP.x = aX0 ; aP.x<aX1 ; aP.x++)
               {
                  for (aP.y = aY0 ; aP.y<aY1 ; aP.y++)
                  {
                      double aSomDif=0;
                      for (int aDy = -aSzW ; aDy<=aSzW ; aDy++)
                      {
                         U_INT1 * aL1 =  aDI1[aP.y+aDy] -aSzW + aP.x;
             U_INT1 * aL2 =  aDI2[aP.y+aDy] -aSzW + aPax+ aP.x;

                         U_INT1 * aL1Sup  =  aL1 + 2 * aSzW+1;
                         while (aL1 != aL1Sup)
                         {
                aSomDif += ElAbs(*(aL1++) -  *(aL2++));
                         }
                      }
                      aTScore.oset(aP,aSomDif);
                  }
               }
            }
            break;


        }
        // Save the paralax and score for pixel that increase the quality of
        // matching

        if (aMode == eMCEM_Func)
        {
           ELISE_COPY
           (
              select(aI1.all_pts(),aIScore.in()<aIScoreMin.in()),
              Virgule(aPax,aIScore.in()),
              Virgule(aIPaxOpt.out(),aIScoreMin.out())
           );
        }
        else
        {
        Pt2di aP;
            for (aP.x = aX0 ; aP.x<aX1 ; aP.x++)
            {
                for (aP.y = aY0 ; aP.y<aY1 ; aP.y++)
                {
                    if (aTScore.get(aP) < aTScoreMin.get(aP))
                    {
                        aTScoreMin.oset(aP,aTScore.get(aP));
                        aTPaxOpt.oset(aP,aPax);
                    }
                }
            }
        }
    }

    ELISE_COPY
    (
        aIPaxOpt.all_pts(),
        aIPaxOpt.in()*3,
        aW.ocirc()
     );
     aW.clik_in();



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
