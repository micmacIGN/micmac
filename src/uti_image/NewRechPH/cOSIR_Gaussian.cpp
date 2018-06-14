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

#include "NewRechPH.h"


// Gaussian   1/(sqrt(2Pi) Sig2)  exp(-X/2 Sig^2)

// PB = > 

// Simga2 =  (2*a) /(ElSquare(1-a)); voir TestSigma2 ds NewRechPH.cpp
// a = ((Simga2+1) + sqrt(((Simga2+1)^2 -1)  ) / sigma2

double Sigma2FromFactExp(double a)
{
   return (2*a) / ElSquare(1-a);
}
double FactExpFromSigma2(double aS2)
{
    return (aS2+1 - sqrt(ElSquare(aS2+1)-ElSquare(aS2))  ) / aS2 ;
}

void TestSigma2(double a)
{
   int aNb = 100 + 100/(1-a);
   double aSom=0;
   double aSomK2=0;
   for (int aK=-aNb ; aK<= aNb ; aK++)
   {
       double aPds = pow(a,ElAbs(aK));
       aSom += aPds;
       aSomK2 += aPds * ElSquare(aK);
   }

   double aSigmaExp =  aSomK2 / aSom;
   // double aSigmaTh = (2*a) /((1+a) * ElSquare(1-a));
   // double aSigmaTh = (2*a) /(ElSquare(1-a));
   double aSigmaTh = Sigma2FromFactExp(a);

   double aATh = FactExpFromSigma2(aSigmaTh);

   std::cout << "TestSigma2 " << aSigmaExp << " EcarRel=" << aSigmaTh/aSigmaExp - 1 << " FactExp=" << a << " Sig2=" << aATh << "\n";
}

double Gauss(double aSig,double aVal)
{
   static double aS2PI = sqrt(1/(2*PI));
   double aS2 = ElSquare(aSig);

   return (aS2PI / aSig) * exp(-ElSquare(aVal)/ (2*aS2));
}
void TestGauss(double aSig)
{
   int aNbIn = 1000;
   int aNbOut = 10;
   int aNbTot = aNbIn * aNbOut;

   double aSomG = 0.0;
   double aSomGX = 0.0;
   double aSomGX2 = 0.0;

   double aStep = aSig / double(aNbIn);

   for (int aK=-aNbTot ; aK<= aNbTot ; aK++)
   {
        double aVal = aK * aStep;

        double aG = Gauss(aSig,aVal) * aStep;
        aSomG   += aG;
        aSomGX  += aG * aVal;
        aSomGX2 += aG * ElSquare(aVal);
   }
   std::cout << "IntegG=" << aSomG   
             <<  " S2=" << aSomGX2 / ElSquare(aSig) 
             << " Moy=" << aSomGX  
             << "\n";
// Gaussian   1/(sqrt(2Pi) Sig2)  exp(-X/2 Sig^2)
}

void TestEcartTypeStd()
{
   TestGauss(0.5);
   TestGauss(1);
   TestGauss(2);
   getchar();
   TestSigma2(0.1);
   TestSigma2(0.5);
   TestSigma2(0.9);
   TestSigma2(0.95);
   TestSigma2(0.99);
   getchar();

}


/*******************************************************************************************/
/*                                                                                         */
/*   classes optimisees pour faire du filtrage gaussien par exp recurs sur image entiere   */
/*                                                                                         */
/*******************************************************************************************/

class cFGGBEBufLigne
{
    public :
      friend class cFilterGaussByExpOpt;
      cFGGBEBufLigne(int aSz,double aFact);
    private :
      // Replace aDataIn by it convulotion with exponential filter .. F^2 F 1 F F^2 ... 
     // with F = mDynF / ( 2^mNBDynF)
      void OneFiterLine(INT4 * aData);

      // Due to side effect, need to normalise sides answers
      void OneFiterLineAndNormalize(INT4 * aData);

      int           mSz;
      double        mFact;
      int           mNBBDynFE;
      int           mDynFE;  // Dyn du facteur exp
      int           mDemiDynFE;  // Dyn du facteur exp
      int           mFactInt;
      Im1D_INT4     mBuf1;  // Filtrage de l'image constante
      INT4 *        mData1; // Data mBuf1
      Im1D_INT4     mBufIm;
      INT4 *        mDataIm;
      Im1D_INT4     mBufForw;
      INT4 *        mDataForw;
      Im1D_INT4     mBufBackw;
      INT4 *        mDataBackw;
};

cFGGBEBufLigne::cFGGBEBufLigne(int aSz,double aFact) :
   mSz        (aSz),
   mFact      (aFact),
   mNBBDynFE  (14),  
   mDynFE     (1<<mNBBDynFE), // +ou 16000
   mDemiDynFE (mDynFE/2),
   mFactInt (round_ni(mDynFE*aFact)),
   mBuf1   (mSz),           // To know how a constant function is modified
   mData1  (mBuf1.data()),  // 
   mBufIm  (mSz),
   mDataIm (mBufIm.data()),
   mBufForw   (mSz),
   mDataForw  (mBufForw.data()),
   mBufBackw  (mSz),
   mDataBackw (mBufBackw.data())
{
   // Compute normalization on constant signal
    for (int aX=0 ; aX < mSz ; aX++) 
        mData1[aX] = mDynFE;
    OneFiterLine(mData1);
}

void cFGGBEBufLigne::OneFiterLineAndNormalize(INT4 * aData)
{
    OneFiterLine(aData);
    for (int aX=0 ; aX < mSz ; aX++) 
    {
        // La formule correspond a (aData[aX] / (mData1[aX] /mDynFE))
        // On divise  donc par la reponse sur un signal constant 1, en tenant compte du
        // fait que mData1 a une dynamique de mDynFE
        aData[aX] = (aData[aX] << mNBBDynFE) /  mData1[aX];
    }
}

void cFGGBEBufLigne::OneFiterLine(INT4 * aDataIn)
{
    //  Forw[4] = I[3] * F + I[2] * F^2 + ..
    //  Forw[5] = I[4] * F + I[3] * F^2 + ..
    //  Forw[5] = F* (I[4] + Forw[4])
    mDataForw[0] = 0;
    for (int aX=1 ; aX < mSz ; aX++) 
    {
        mDataForw[aX] =  ((mDataForw[aX-1] +aDataIn[aX-1]) *mFactInt + mDemiDynFE) >> mNBBDynFE;
    }

    //  BackW [6] =   aDataIn[7] * F + aDataIn[8] * F^2 ..
    //  BackW [5] =   aDataIn[6] * F + aDataIn[7] * F^2 ..
    //  BackW[5]  =  F * (aDataIn[6] +  BackW [6])
    
    mDataBackw[mSz-1] = 0;
    for (int aX=mSz-2 ; aX>=0 ; aX--) 
    {
        mDataBackw[aX] =  ((mDataBackw[aX+1] +aDataIn[aX+1]) *mFactInt + mDemiDynFE) >> mNBBDynFE;
    }

    for (int aX=0 ; aX < mSz ; aX++) 
    {
       aDataIn[aX] += mDataForw[aX] + mDataBackw[aX];
    }
}

class cFilterGaussByExpOpt
{
    public :
        friend class cFGGBEBufLigne;
        cFilterGaussByExpOpt(tImNRPH anIm,double aSigmaN, int aNb);
    public :

        tImNRPH mIm;
        INT2**  mData;
        Pt2di   mSz;
        double  mSigma1;
        double  mFact;
        cFGGBEBufLigne mBufX;
        cFGGBEBufLigne mBufY;
};


cFilterGaussByExpOpt::cFilterGaussByExpOpt(tImNRPH anIm,double aSigmaN, int aNb) :
   mIm      (anIm),
   mData    (mIm.data()),
   mSz      (mIm.sz()),
   mSigma1  (aSigmaN/sqrt(aNb)),
   mFact    (FactExpFromSigma2(ElSquare(mSigma1))),
   mBufX    (mSz.x,mFact),
   mBufY    (mSz.y,mFact)
{
    // Filtrage X
    {
        for (int aY=0 ; aY<mSz.y ;aY++)
        {
           INT2 * aLineI2 = mData[aY];
           INT*   aLineI = mBufX.mDataIm;
           for (int aX=0 ; aX<mSz.x ;aX++)
           {
               aLineI[aX] = aLineI2[aX];
           }
           for (int aK=0 ; aK<aNb ;aK++)
           {
                mBufX.OneFiterLineAndNormalize(aLineI);
           }
           for (int aX=0 ; aX<mSz.x ;aX++)
           {
               aLineI2[aX] = aLineI[aX] ;
           }
        }
    }
    // Filtrage Y
    {
        for (int aX=0 ; aX<mSz.x ;aX++)
        {
           INT*   aLineI = mBufY.mDataIm;
           for (int aY=0 ; aY<mSz.y ;aY++)
           {
               aLineI[aY] = mData[aY][aX];
           }
           for (int aK=0 ; aK<aNb ;aK++)
           {
                mBufY.OneFiterLineAndNormalize(aLineI);
           }
           for (int aY=0 ; aY<mSz.y ;aY++)
           {
               mData[aY][aX] = aLineI[aY];
           }
        }
    }
}

void TestFilterGauss(Pt2di aSz,Flux_Pts aFlux,const std::string & aName,double aSigma,int aNb)
{
    double aSAv,aSApr;
    tImNRPH aIm(aSz.x,aSz.y,0);
    ELISE_COPY(aFlux,30000,aIm.out());
    ELISE_COPY(aIm.all_pts(),aIm.in(),sigma(aSAv));

    cFilterGaussByExpOpt aFG(aIm,aSigma,aNb);
    Tiff_Im::CreateFromIm(aIm,aName);

    ELISE_COPY(aIm.all_pts(),aIm.in(),sigma(aSApr));

    std::cout << "SOM , Av " << aSAv << " " << aSApr/aSAv << "\n";
}

void TestFilterGauss(const std::string & aNameIn,double aSigma,int aNb)
{
    Tiff_Im aTif(aNameIn.c_str());
    
    
}

void TestFilterGauss()
{
     TestFilterGauss(Pt2di(1000,1000),rectangle(Pt2di(500,500),Pt2di(501,501)),"Dirac_20_4.tif",2.0,4);
     TestFilterGauss(Pt2di(1000,1000),rectangle(Pt2di(500,500),Pt2di(501,501)),"Dirac_20_8.tif",2.0,8);
     TestFilterGauss(Pt2di(1000,1000),rectangle(Pt2di(500,500),Pt2di(501,501)),"Dirac_20_1.tif",2.0,1);
     TestFilterGauss(Pt2di(1000,1000),rectangle(Pt2di(500,500),Pt2di(550,550)),"Rect50_20_8.tif",2.0,8);
}
 

//========================================================


template <class T1> void  LocFilterGauss(T1 & anIm, double aSigmaN,int aNbIter)
{
  double aSig0 = aSigmaN / sqrt(aNbIter);
  double aF = FactExpFromSigma2(ElSquare(aSig0));

  Pt2di aSz = anIm.sz();
  Im2D_REAL4 aIP1(aSz.x,aSz.y,1.0);
  FilterExp(aIP1,aF);

  for (int aKIt=0 ; aKIt<aNbIter ; aKIt++)
  {
      FilterExp(anIm,aF);
      ELISE_COPY(anIm.all_pts(),anIm.in()/aIP1.in(),anIm.out());
  }
}


void FilterGaussProgr(tImNRPH anIm,double  aSTarget,double  aSInit,int aNbIter)
{
    aSTarget = ElSquare(aSTarget);
    aSInit = ElSquare(aSInit);
    if (aSTarget > aSInit)
    {
        // LocFilterGauss(anIm,sqrt(aSTarget-aSInit),aNbIter);
        cFilterGaussByExpOpt aFG(anIm,sqrt(aSTarget-aSInit),aNbIter);
    }
    else if (aSTarget==aSInit)
    {
    }
    else
    {
      ELISE_ASSERT(false,"FilterGaussProgr");
    }
}

void TestDist(Pt2di aSz,Fonc_Num aP,double aScale)
{
   Symb_FNum aSP = aP;
   double aSom,aSX,aSY,aSXX,aSYY;
   ELISE_COPY
   (
       rectangle(Pt2di(0,0),aSz),
       Virgule(aSP,aSP*FX,aSP*FY,aSP*FX*FX,aSP*FY*FY),
       Virgule(sigma(aSom),sigma(aSX),sigma(aSY),sigma(aSXX),sigma(aSYY))
   );
   aSX /= aSom;
   aSY /= aSom;
   aSXX /= aSom;
   aSYY /= aSom;
   aSXX -= ElSquare(aSX);
   aSYY -= ElSquare(aSY);

   // Il y a une difficulte, pour les vrais images on fait l'hypothese que a la premiere
   // echelle sigm=1 (on n'en sait rien, ptet + ptet - en fait)
   // mais ici avec un dirac la sigma=0 a ech =1
   // la relation est en 1+ sig ^2 = Ech ^2

   double aSigTh = sqrt(ElSquare(aScale)-1);

   std::cout << "SC= " << aScale  
             << "STAT S=" << aSom << " X=" << aSX << " Y=" << aSY
             << " SigX=" << sqrt(aSXX) / aSigTh 
             << " SigY=" << sqrt(aSYY) / aSigTh << "\n";
}

int Generate_ImagSift(int argc,char ** argv)
{
     Pt2di aSz(1000,1000);
     Im2D_REAL4 aIm(aSz.x,aSz.y);

     for (int aKx=0 ; aKx<10 ; aKx++)
     {
         for (int aKy=0 ; aKy<10 ; aKy++)
         {
             Pt2di aP0(aKx*100,aKy*100);
             Pt2di aP1((aKx+1)*100,(aKy+1)*100);
             Pt2dr aMil = Pt2dr(aP0+aP1) / 2.0;

             double aSigmaX = (0.25*aKx + 1.0*aKy  + 1);
             double aSigmaY = ( 1.0*aKx + 0.25*aKy + 1);
             double aSign = ((aKx+aKy) % 2)   ? 1 : -1;

             ELISE_COPY
             (
                  rectangle(aP0,aP1),
                  128 * (1+aSign * exp(-  ( Square(FX-aMil.x)/Square(aSigmaX) + Square(FY-aMil.y)/ Square(aSigmaY))   )),
                  aIm.out()
             );

         }
     }
     Tiff_Im::CreateFromIm(aIm,"TestSift.tif");
     return EXIT_SUCCESS;
}

int Generate_ImagePer(int argc,char ** argv)
{
   std::string aNameIm;
   std::string aNameOut;
   int aSz = 512;
   int aBord = 64;
   int aNb=4;

   MMD_InitArgcArgv(argc,argv);
   ElInitArgMain
   (
         argc,argv,
         LArgMain()   << EAMC(aNameIm, "Name Image",  eSAM_IsPatFile),
         LArgMain()   << EAM(aNb, "Nb",true,"Nb repetition")
   );

   aNameOut = "Per-" + aNameIm;


   Im2D_U_INT1 aImIn(aSz,aSz);
   Tiff_Im aFile(aNameIm.c_str());

   ELISE_COPY(aImIn.all_pts(),aFile.in(),aImIn.out());
   ELISE_COPY(aImIn.border(aBord),0,aImIn.out());

   //Im2D_U_INT1 aImOut(aTx,aTy);
   Tiff_Im aTifOut
           (
                aNameOut.c_str(),
                Pt2di(aSz*aNb,aSz*aNb),
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::BlackIsZero
           );

   ELISE_COPY
   (
       aTifOut.all_pts(),
       aImIn.in()[Virgule(FX%aSz,FY%aSz)],
       aTifOut.out()
   );

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
aooter-MicMac-eLiSe-25/06/2007*/
