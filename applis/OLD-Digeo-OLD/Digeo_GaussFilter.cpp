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

#include "general/all.h"
#include "private/all.h"
#include "Digeo.h"

namespace NS_ParamDigeo
{

/****************************************/
/*                                      */
/*             ::                       */
/*                                      */
/****************************************/

Im1D_REAL8 MakeSom1(Im1D_REAL8 aIm);

//  K3-C3 = K1-C1 + K2-C2

Im1D_REAL8 DeConvol(int aC2,int aSz2,Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI3,int aC3)
{
   L2SysSurResol aSys(aSz2);
   
   int  aSz1 = aI1.tx();
   int  aSz3 = aI3.tx();

   for (int aK3 =0 ; aK3 < aSz3 ; aK3++)
   {
       std::vector<int> aVInd;
       std::vector<double> aVCoef;
       for (int aK=0; aK < aSz1 ; aK++)
       {
           int aK1 = aK;
           int aK2 = aC2 + (aK3-aC3) - (aK1-aC1);
           if ((aK1>=0)&&(aK1<aSz1)&&(aK2>=0)&&(aK2<aSz2))
           {
               aVInd.push_back(aK2);
               aVCoef.push_back(aI1.data()[aK1]);
           }
       }
       if (aVInd.size()) 
          aSys.GSSR_AddNewEquation_Indexe(aVInd,1.0,&(aVCoef.data()[0]),aI3.data()[aK3]);
   }

   Im1D_REAL8 aRes = aSys.GSSR_Solve(0);
   ELISE_COPY(aRes.all_pts(),Max(aRes.in(),0.0),aRes.out());
   return MakeSom1(aRes);
}

Im1D_REAL8 DeConvol(int aDemISz2,Im1D_REAL8 aI1,Im1D_REAL8 aI3)
{
   ELISE_ASSERT((aI1.tx()%2)&&(aI3.tx()%2),"Parity error in DeConvol");
   return DeConvol(aDemISz2,1+2*aDemISz2,aI1,aI1.tx()/2,aI3,aI3.tx()/2);
}


Im1D_REAL8 Convol(Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI2,int aC2)
{
    Im1D_REAL8 aRes(aI1.tx()+aI2.tx()-1,0.0);

    ELISE_COPY
    (
         rectangle(Pt2di(0,0),Pt2di(aRes.tx(),aRes.tx())),
         aI1.in(0)[FX]*aI2.in(0)[FY-FX],
         aRes.histo(true).chc(FY)
    );

   return aRes;
}

Im1D_REAL8 Convol(Im1D_REAL8 aI1,Im1D_REAL8 aI2)
{
   ELISE_ASSERT((aI1.tx()%2)&&(aI2.tx()%2),"Parity error in Convol");
   return Convol(aI1,aI1.tx()/2,aI2,aI2.tx()/2);
}




Im1D_REAL8 MakeSom(Im1D_REAL8 aIm,double aSomCible)
{
    double aSomActuelle;
    Im1D_REAL8 aRes(aIm.tx());
    ELISE_COPY(aIm.all_pts(),aIm.in(),sigma(aSomActuelle));
    ELISE_COPY(aIm.all_pts(),aIm.in()*(aSomCible/aSomActuelle),aRes.out());
    return aRes;
}

Im1D_REAL8 MakeSom1(Im1D_REAL8 aIm)
{
    return MakeSom(aIm,1.0);
}
 

Im1D_REAL8  GaussianKernel(double aSigma,int aNb,int aSurEch)
{
   Im1D_REAL8 aRes(2*aNb+1);

   for (int aK=0 ; aK<=aNb ; aK++)
   {
        double aSom = 0;
        for (int aKE =-aSurEch ; aKE<=aSurEch ; aKE++)
        {
            double aX = aK-aNb + aKE/double(2*aSurEch+1);
            double aG = exp(-ElSquare(aX/aSigma)/2.0);
            aSom += aG;
        }
        aRes.data()[aK] =  aRes.data()[2*aNb-aK] = aSom;
   }
   

   return MakeSom1(aRes);
}

int NbElemForGausKern(double aSigma,double aResidu)
{
   return round_up( sqrt(-2*log(aResidu))*aSigma);
}

Im1D_REAL8  GaussianKernelFromResidu(double aSigma,double aResidu,int aSurEch)
{
   return GaussianKernel(aSigma,NbElemForGausKern( aSigma,aResidu),aSurEch);
}

// Methode pour avoir la "meilleure" approximation entiere d'une image reelle
// avec somme imposee. Tres sous optimal, mais a priori utilise uniquement sur de
// toute petites images

Im1D_INT4 ToIntegerKernel(Im1D_REAL8 aRK,int aMul,bool aForceSym)
{
    aRK = MakeSom1(aRK);
    int aSz=aRK.tx();
    Im1D_INT4 aIK(aSz);

    int aSom;
    ELISE_COPY(aIK.all_pts(),round_ni(aRK.in()*aMul),aIK.out()|sigma(aSom));

    int *    aDI = aIK.data();
    double * aDR = aRK.data();
    while (aSom != aMul)
    {

        int toAdd = aMul-aSom;
        int aSign = (toAdd>0) ? 1 : -1;
        int aKBest=-1;
        double aDeltaMin = 1e20;

        if (aForceSym && (ElAbs(toAdd)==1) )
        {
            ELISE_ASSERT((aSz%2),"ToIntegerKernel Sym");
            aKBest=  aSz/2;
        }
        else
        {
           for (int aK=0 ; aK<aSz ; aK++)
           {
               double aDelta =  ((aDI[aK]/double(aMul)) -aDR[aK]) * aSign;
               if ((aDelta < aDeltaMin) && (aDI[aK]+aSign >=0))
               {
                  aDeltaMin = aDelta;
                  aKBest= aK;
               }
           }
        }

        ELISE_ASSERT(aKBest!=-1,"Inco(1) in ToIntegerKernel");

        aDI[aKBest] += aSign;
        aSom += aSign;

        if (aForceSym && (aSom!=aMul))
        {
           int aKSym = aSz - aKBest-1;
           if (aKSym!=aKBest)
           {
               aDI[aKSym] += aSign;
               aSom += aSign;
           }
        } 
    }
    return aIK;
}

Im1D_INT4  ToOwnKernel(Im1D_REAL8 aRK,int & aShift,bool aForceSym,int *)
{
     return ToIntegerKernel(aRK,1<<aShift,aForceSym);
}
Im1D_REAL8  ToOwnKernel(Im1D_REAL8 aRK,int & aShift,bool aForceSym,double *)
{
    return aRK;
}

Im1D_REAL8 ToRealKernel(Im1D_INT4 aIK)
{
   Im1D_REAL8 aRK(aIK.tx());
   ELISE_COPY(aIK.all_pts(),aIK.in(),aRK.out());
   return MakeSom1(aRK);
}

Im1D_REAL8 ToRealKernel(Im1D_REAL8 aRK)
{
   return aRK;
}


   // Permt de shifter les entiers (+ rapide que la div) sans rien faire pour
   // les flottants
inline double ShiftDr(const double & aD,const int &) { return aD; }
inline double ShiftG(const double & aD,const int &) { return aD; }
inline double InitFromDiv(double ,double *) { return 0; }

inline int ShiftDr(const int & aD,const int & aShift) { return aD >> aShift; }
inline int ShiftG(const int & aD,const int & aShift) { return aD << aShift; }
inline int InitFromDiv(int aDiv,int *) { return aDiv/2; }



/*

   // Pour utiliser un filtre sur les bord, clip les intervalle
   // pour ne pas deborder et renvoie la somme partielle
template <class tBase> tBase ClipForConvol(int aSz,int aKXY,tBase * aData,int & aDeb,int & aFin)
{
    ElSetMax(aDeb,-aKXY);
    ElSetMin(aFin,aSz-1-aKXY);
*/




   // Pour utiliser un filtre sur les bord, clip les intervalle
   // pour ne pas deborder et renvoie la somme partielle
template <class tBase> tBase ClipForConvol(int aSz,int aKXY,tBase * aData,int & aDeb,int & aFin)
{
    ElSetMax(aDeb,-aKXY);
    ElSetMin(aFin,aSz-1-aKXY);

    tBase aSom = 0;
    for (int aK= aDeb ; aK<=aFin ; aK++)
        aSom += aData[aK];

   return aSom;
}



   // Produit scalaire basique d'un filtre lineaire avec une ligne
   // et une colonne image
template <class Type,class tBase> 
inline tBase CorrelLine(tBase aSom,const Type * aData1,const tBase *  aData2,const int & aDeb,const int & aFin)
{


     for (int aK= aDeb ; aK<=aFin ; aK++)
        aSom += aData1[aK]*aData2[aK];

   return aSom;
}


/****************************************/
/*                                      */
/*             cTplImInMem              */
/*                                      */
/****************************************/

template <class Type> 
void  cTplImInMem<Type>::SetConvolBordX
      (
          Im2D<Type,tBase> aImOut,
          Im2D<Type,tBase> aImIn,
          int anX,
          tBase * aDFilter,int aDebX,int aFinX
      )
{
    tBase aDiv = ClipForConvol(aImOut.tx(),anX,aDFilter,aDebX,aFinX);
    Type ** aDOut = aImOut.data();
    Type ** aDIn = aImIn.data();

    const tBase aSom = InitFromDiv(aDiv,(tBase*)0);

    int aSzY = aImOut.ty();
    for (int anY=0 ; anY<aSzY ; anY++)
    {
        aDOut[anY][anX] = CorrelLine(aSom,aDIn[anY]+anX,aDFilter,aDebX,aFinX) / aDiv;
    }
}


    //  SetConvolSepX(aImIn,aData,-aSzKer,aSzKer,aNbShitXY,aCS);
template <class Type> 
void cTplImInMem<Type>::SetConvolSepX
     (
          Im2D<Type,tBase> aImOut,
          Im2D<Type,tBase> aImIn,
          tBase *  aDFilter,int aDebX,int aFinX,
          int  aNbShitX,
          cConvolSpec<Type> * aCS
     )
{
    ELISE_ASSERT(aImOut.sz()==aImIn.sz(),"Sz in SetConvolSepX");
    int aSzX = aImOut.tx();
    int aSzY = aImOut.ty();
    int aX0 = -aDebX;
    int aX1 = aSzX-aFinX;

    for (int anX = 0 ; anX <aX0 ; anX++)
    {
        SetConvolBordX(aImOut,aImIn,anX,aDFilter,aDebX,aFinX);
    }

    for (int anX =aX1  ; anX <aSzX ; anX++)
    {
        SetConvolBordX(aImOut,aImIn,anX,aDFilter,aDebX,aFinX);
    }
   
    const tBase aSom = InitFromDiv(ShiftG(tBase(1),aNbShitX),(tBase*)0);
    for (int anY=0 ; anY<aSzY ; anY++)
    {
        Type * aDOut = aImOut.data()[anY];
        Type * aDIn =  aImIn.data()[anY];

        if (aCS)
        {
           aCS->Convol(aDOut,aDIn,aX0,aX1);
        }
        else
        {
           for (int anX = aX0; anX<aX1 ; anX++)
           {
               aDOut[anX] =  ShiftDr(CorrelLine(aSom,aDIn+anX,aDFilter,aDebX,aFinX),aNbShitX);
           }
        }
    }
}


template <class Type> 
void cTplImInMem<Type>::SetConvolSepX
     (
          const cTplImInMem<Type> & aImIn,
          tBase *  aDFilter,int aDebX,int aFinX,
          int  aNbShitX,
          cConvolSpec<Type> * aCS
     )
{
      SetConvolSepX
      (
         mIm,aImIn.mIm, 
         aDFilter,aDebX,aFinX,aNbShitX,
         aCS
      );
}


template <class Type> 
void cTplImInMem<Type>::SelfSetConvolSepY
     (
          tBase * aDFilter,int aDebY,int aFinY,
          int  aNbShitY,
          cConvolSpec<Type> * aCS
     )
{
    Im2D<Type,tBase> aBufIn(mSz.y,PackTranspo);
    Im2D<Type,tBase> aBufOut(mSz.y,PackTranspo);

    Type ** aData =  mIm.data();

    for (int anX = 0; anX<mSz.x ; anX+=PackTranspo)
    {
         ELISE_ASSERT(false,"::SelfSetConvolSepY NOT FINISH (bord ...)");
/*
         int aDelta = mSz.x-anX;
         int aDebord = ElMax(0,PackTranspo-anX);
         anX = ElMin(anX,mSz.x-PackTranspo);   POUR EVITER LES DEBORDEMENTS - MARCHE PAS NON PLUS
*/
         Type * aL0 = aBufIn.data()[0];
         Type * aL1 = aBufIn.data()[1];
         Type * aL2 = aBufIn.data()[2];
         Type * aL3 = aBufIn.data()[3];
         for (int aY=0 ; aY<mSz.y ; aY++)
         {
             Type * aL = aData[aY]+anX;
             *(aL0)++ = *(aL++);
             *(aL1)++ = *(aL++);
             *(aL2)++ = *(aL++);
             *(aL3)++ = *(aL++);
         }
         SetConvolSepX
         (
            aBufOut,aBufIn, 
            aDFilter,aDebY,aFinY,aNbShitY,
            aCS
         );

         aL0 = aBufOut.data()[0];
         aL1 = aBufOut.data()[1];
         aL2 = aBufOut.data()[2];
         aL3 = aBufOut.data()[3];

         for (int aY=0 ; aY<mSz.y ; aY++)
         {
             Type * aL = aData[aY]+anX;
             *(aL)++ = *(aL0++);
             *(aL)++ = *(aL1++);
             *(aL)++ = *(aL2++);
             *(aL)++ = *(aL3++);
         }
    }
}


  

template <class Type> 
void cTplImInMem<Type>::SetConvolSepXY
     (
          const cTplImInMem<Type> & aImIn,
          Im1D<tBase,tBase> aKerXY,
          int  aNbShitXY
     )
{

    ELISE_ASSERT(mSz==aImIn.mSz,"Size im diff in ::SetConvolSepXY");
    int aSzKer = aKerXY.tx();
    ELISE_ASSERT(aSzKer%2,"Taille paire pour ::SetConvolSepXY");
    aSzKer /= 2;

    tBase * aData = aKerXY.data() + aSzKer;
    // Parfois il y a "betement" des 0 en fin de ligne ...
    while (aSzKer && (aData[aSzKer]==0) && (aData[-aSzKer]==0))
          aSzKer--;

    if (mAppli.GenereCodeConvol().IsInit())
    {
       MakeClassConvolSpec
       (
           mAppli.FileGGC_H(),
           mAppli.FileGGC_Cpp(),
           aData,
           -aSzKer,
           aSzKer,
           aNbShitXY
       );
       // return;
    }



    if ((mAppli.ShowTimes().Val() > 100) && (mResolGlob==1))
    {
        std::cout << "Nb = " << aSzKer << " ;; " ;
        for (int aK=-aSzKer ; aK<=aSzKer ; aK++)
            std::cout << aData[aK] << " ";
         std::cout << "\n";
    }

    cConvolSpec<Type> * aCS=  mAppli.UseConvolSpec().Val() ?
                              cConvolSpec<Type>::Get(aData,-aSzKer,aSzKer,aNbShitXY,false) :
                              0 ;

    if (mAppli.ShowConvolSpec().Val())
       std::cout << "CS = " << aCS << "\n";
    if (mAppli.ExigeCodeCompile().Val() && (aCS==0))
    {
       ELISE_ASSERT(false,"cannot find code compiled\n");
    }

    ElTimer aChrono;
    SetConvolSepX(aImIn,aData,-aSzKer,aSzKer,aNbShitXY,aCS);
    
    double aTX = aChrono.uval();
    aChrono.reinit();

    SelfSetConvolSepY(aData,-aSzKer,aSzKer,aNbShitXY,aCS);

    double aTY = aChrono.uval();
    aChrono.reinit();

    if (mAppli.ShowTimes().Val() > 100)
    {
         std::cout << "Time convol , X : " << aTX << " , Y : " << aTY <<   " SzK " << aKerXY.tx() << "\n";
    }
}


void TestConvol()
{
   int aT1 = 5;
   int aT2 = 3;
   Im1D_REAL8  aI1 = GaussianKernel(2.0,aT1,10);
   Im1D_REAL8  aI2 = GaussianKernel(1.0,aT2,10);

   // ELISE_COPY(aI1.all_pts(),FX==aT1,aI1.out());
   // ELISE_COPY(aI2.all_pts(),FX==aT2,aI2.out());
  
   Im1D_REAL8  aI3 = Convol(aI1,aI2);

   Im1D_REAL8 aI2B = DeConvol(2,aI1,aI3);

   Im1D_REAL8 aI4 = Convol(aI1,aI2B);


   for (int aK=0 ; aK<ElMax(aI3.tx(),aI4.tx()) ; aK++)
   {
       std::cout 
                  << aK << ":" 
                 << "  " << (aK<aI3.tx() ? ToString(aI3.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI1.tx() ? ToString(aI1.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI2.tx() ? ToString(aI2.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI2B.tx() ? ToString(aI2B.data()[aK]) : " XXXXXX " )
                 << "  " << (aK<aI4.tx() ? ToString(aI4.data()[aK]) : " XXXXXX " )
                 << "\n";
   }
}


/*
template <class Type> 
void cTplImInMem<Type>::MakeConvolInit(double aV)
{
   // TestConvol();
    const cPyramideGaussienne aPG = mAppli.TypePyramide().PyramideGaussienne().Val();
    mNbShift = aPG.NbShift().Val();
    mTFille->mNbShift = mNbShift;
    mKernelTot = GaussianKernelFromResidu
                 (
                   mResolOctaveBase,
                   aPG.EpsilonGauss().Val(),
                   aPG.SurEchIntegralGauss().Val()
                 );
    if (aV==0) return;
    if (aV==-1) aV = mResolOctaveBase;

    ELISE_ASSERT(aV>0,"Bad value in ConvolFirstImage");
    ElTimer aChrono;



    Im1D_REAL8  aKR =  GaussianKernelFromResidu
                       (
                           aV,
                           aPG.EpsilonGauss().Val(),
                           aPG.SurEchIntegralGauss().Val()
                       );
    
    int aShift = aPG.NbShift().Val();
    Im1D<tBase,tBase> aOwnK =  ToOwnKernel(aKR,aShift,true,(tBase *)0);

    if (mAppli.ShowTimes().Val() > 100)
    {
       std::cout << "NB KER GAUSS " << aKR.tx() << "\n";
       for (int aK=0 ; aK<aKR.tx() ; aK++)
       {
            std::cout << " G[" << aK << "]=" << aKR.data()[aK] << " " << aOwnK.data()[aK] << "\n";
       }
    }

    int aKInOct = mTFille->mKInOct;
    mTFille->mKInOct = mKInOct;
    mTFille->SetConvolSepXY(*this,aOwnK,aShift);
    mTFille->mKInOct = aKInOct;

    for (int anY=0;anY<mSz.y; anY++)
    {
       memcpy
       (
          mIm.data()[anY],
          mTFille->mIm.data()[anY],
          mSz.x*sizeof(Type)
       );
    }

    if (mAppli.ShowTimes().Val() > 100)
    {
        std::cout << "Time Convol Init " << aChrono.uval() << "\n";
    }
}
*/


template <class Type> 
void cTplImInMem<Type>::ReduceGaussienne()
{
    const cPyramideGaussienne aPG = mAppli.TypePyramide().PyramideGaussienne().Val();
    int aSurEch = aPG.SurEchIntegralGauss().Val();
    double anEpsilon = aPG.EpsilonGauss().Val();
    mNbShift = aPG.NbShift().Val();

    if (mMere && (RGlob()!=mMere->RGlob()))
    {
       mOrigOct->MakeReduce(*(mMere->Oct().ImBase()),mAppli.PyramideImage().ReducDemiImage().Val());
    }
    Resize(mOrigOct->Sz());

    // Valeur a priori du sigma en delta / au prec

    double aSigTot = mResolOctaveBase;;
    int aNbCTot = NbElemForGausKern(aSigTot,aPG.EpsilonGauss().Val()/10) +1 ;
    Im1D_REAL8 aKerTot=GaussianKernel(aSigTot,aNbCTot,aSurEch);


    bool isIncrem = aPG.ConvolIncrem().Val();
    if (mAppli.ModifGCC())
       isIncrem = mAppli.ModifGCC()->ConvolIncrem();


    if (InitRandom())
    {
       return;
    }

    if (! isIncrem)
    {
       Im1D<tBase,tBase> aIKerTotD =  ToOwnKernel(aKerTot,mNbShift,true,(tBase *)0);
       if (0)
       {
          for (int aK=0 ; aK<aIKerTotD.tx() ; aK++)
          {
             std::cout << "  " << aIKerTotD.data()[aK] ;
          }
          std::cout << "\n";
       }
       SetConvolSepXY(*mOrigOct,aIKerTotD,mNbShift);
       return;
    }

    // ELISE_ASSERT(false,"Pb with convol incrementale\n");



    double aSigmD =  sqrt(ElSquare(aSigTot) - ElSquare(mTMere->mResolOctaveBase));
    int aNbCD= NbElemForGausKern(aSigmD,anEpsilon);
    Im1D_REAL8 aKerD =  DeConvol(aNbCD,mTMere->mKernelTot,aKerTot);

    // Bug ou incoherence dans la deconvol, donne pb
    aKerD = GaussianKernelFromResidu(aSigmD,anEpsilon,aSurEch);
// Im1D_REAL8  GaussianKernelFromResidu(double aSigma,double aResidu,int aSurEch)


    Im1D<tBase,tBase> aIKerD =  ToOwnKernel(aKerD,mNbShift,true,(tBase *)0);
    SetConvolSepXY(*mTMere,aIKerD,mNbShift);

    Im1D_REAL8        aRealKerD =  ToRealKernel(aIKerD);

    mKernelTot = Convol(aRealKerD,mTMere->mKernelTot);
    


    if ((mAppli.ShowTimes().Val() > 100) && (mResolGlob<=2))
    {
         
/*
         std::cout << "       + + + CONVOL + + \n";
         for (int aK=0 ; aK< mKernelTot.tx() ; aK++)
         {
              std::cout <<  mKernelTot.data()[aK]  << ((aK==mKernelTot.tx()/2)? " @@@@@" : "")<< "\n";
         }
         std::cout << "       + + + GLOB + + \n";
         for (int aK=0 ; aK< aKerTot.tx() ; aK++)
         {
              std::cout <<  aKerTot.data()[aK]  << ((aK==aKerTot.tx()/2)? " @@@@@" : "")<< "\n";
         }
*/

         Im1D_REAL8        aSigKer = GaussianKernel(aSigmD,aNbCD,aSurEch);
         std::cout << "---------------------------------------------------------\n";
         std::cout << "  DZ " << mResolGlob << " ; " 
                   << " K  " << mKInOct
                   << " Sig-Delta " << aSigmD << " ; "
                   <<  " NbC-Delta " << aNbCD << " ; "
                   <<  " NbC-Tot " << aNbCTot << " ; "
                   << "\n";


         
         for (int aK=0 ; aK<aKerD.tx() ; aK++)
         {
             std::cout << "  " << aIKerD.data()[aK] ;
         }
         std::cout << "\n";
         // if (mAppli.ShowTimes().Val() > 100)
         if (0)
         {
            for (int aK=0 ; aK<aKerD.tx() ; aK++)
            {
                 if (aK<aKerD.tx()) std::cout << " Cur= " << aKerD.data()[aK] << " ";
                 if (aK<aKerD.tx()) std::cout << " ICur= " << aRealKerD.data()[aK] << " ";
                 if (aK<aKerD.tx()) std::cout << " SCur= " << aSigKer.data()[aK] << " ";
                 std::cout  << "\n";
            }
         }
         //getchar();
    }
    
}


InstantiateClassTplDigeo(cTplImInMem)

/*
template  class cTplImInMem<U_INT1>;
template  class cTplImInMem<U_INT2>;
template  class cTplImInMem<INT>;
template  class cTplImInMem<float>;
*/

 


/****************************************/
/*                                      */
/*             cImInMem                 */
/*                                      */
/****************************************/

/*
void  cImInMem::MakeReduce()
{
    VMakeReduce(*mMere);
}
*/

};



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
Footer-MicMac-eLiSe-25/06/2007*/
