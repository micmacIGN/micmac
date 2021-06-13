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


/*


                            cKernelInterpol1D
                           /        |          \
       cCubicInterpKernel           |           cTabulKernelInterpol
                            cScaledKernelInterpol



template <class TypeEl> class cInterpolateurIm2D;  Classe virtuelle pure
class cIm2DInter;  pour etre renvoye par les Im2Gen

                             cInterpolateurIm2D<TypeEl> __
                            /          |           \   \  \
    cTplCIKTabul<TypeEl,tTabulCIK>     |            \   \  cInterpolBicubique<TypeEl>
                                cInterpolPPV<TypeEl> \   cInterpolBilineaire<TypeEl>
                                                      cInterpolSinusCardinal<TypeEl>


cInterpolPPV<TypeEl>

*/


/******************************************************/
/*                                                    */
/*              cInterpolateurIm2D<TypeEl>            */
/*                                                    */
/******************************************************/


template <class TypeEl>
void  cInterpolateurIm2D<TypeEl>::GetVals
      (
          TypeEl ** aTab,
          const Pt2dr *  aPts,
          double *aVals,
          int Nb
      ) const
{
   for (int aK=0 ; aK<Nb ; aK++)
       aVals[aK] = GetVal(aTab,aPts[aK]);
}


template <class TypeEl>
cInterpolateurIm2D<TypeEl>::~cInterpolateurIm2D()
{
}


template <class TypeEl>
Pt3dr cInterpolateurIm2D<TypeEl>::GetValAndQuickGrad(TypeEl ** aTab,const Pt2dr &  aP) const
{
   //double aV = GetVal(aTab,aP);
   int anX = round_down(aP.x);
   int anY = round_down(aP.y);
    
   return  Pt3dr
           (
               aTab[anY][anX+1]-aTab[anY][anX],
               aTab[anY+1][anX]-aTab[anY][anX],
               GetVal(aTab,aP)
           );
}

template <class TypeEl>
 Pt3dr cInterpolateurIm2D<TypeEl>::GetValDer(TypeEl ** aTab,const Pt2dr &  aP) const
{
   ELISE_ASSERT(false,"no ::GetValDer");
   return Pt3dr(0,0,0);
}



/******************************************************/
/*                                                    */
/*             cTabIM2D_FromIm2D<TypeEl>              */
/*                                                    */
/******************************************************/

template <class TypeEl> cTabIM2D_FromIm2D<TypeEl>::cTabIM2D_FromIm2D(const cKernelInterpol1D * aK0, int NbDisc1,bool mPrecBil):
    mK0  (aK0),
    mK1 (aK0,NbDisc1,mPrecBil),
    mSzK (round_up(mK1.SzKernel()))
{
}

template <class TypeEl> cTabIM2D_FromIm2D<TypeEl>::~cTabIM2D_FromIm2D()
{
}

template <class TypeEl> int cTabIM2D_FromIm2D<TypeEl>::SzKernel() const
{
   return mSzK;
}


template <class TypeEl> double cTabIM2D_FromIm2D<TypeEl>::GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
    int aCI = round_down(aP.x);
    int aCJ = round_down(aP.y);
    double aFracX =  aP.x - aCI;
    double aFracY =  aP.y - aCJ;

    double aSomGlob = 0.0;
    for (int aJ= -mSzK+1 ; aJ<= mSzK ; aJ++)
    {
        TypeEl * aLine  = aTab[aJ+aCJ] + aCI;
        const double *  anAdr = mK1.AdrDisc2Real(-aFracX+(-mSzK+1));
        double aSomLine = 0;
        for (int aI= -mSzK+1 ; aI<= mSzK ; aI++)
        {
            ///   aSomLine += aLine[aI] + mK0->Value(aI-aFracX);
            aSomLine += aLine[aI] * (*anAdr);
            // aSomLine +=  (*anAdr);
            anAdr +=  mK1.NbDisc1();
        }
        aSomGlob += aSomLine * mK1.Value(aJ-aFracY);
    }

    return aSomGlob;
}

template <class TypeEl> Pt3dr cTabIM2D_FromIm2D<TypeEl>::GetValDer(TypeEl ** aTab,const Pt2dr &  aP) const
{
    int aCI = round_down(aP.x);
    int aCJ = round_down(aP.y);
    double aFracX =  aP.x - aCI;
    double aFracY =  aP.y - aCJ;

    double aSomVal = 0.0;
    double aSomDerX= 0.0;
    double aSomDerY= 0.0;
    for (int aJ= -mSzK+1 ; aJ<= mSzK ; aJ++)
    {
        TypeEl * aLine  = aTab[aJ+aCJ] + aCI;
        const double *  anAdr = mK1.AdrDisc2Real(-aFracX+(-mSzK+1));
        const double *  aDxAdr = mK1.DerAdrDisc2Real(-aFracX+(-mSzK+1));
        double aSomLine = 0;
        double aDxSomLine = 0;
        for (int aI= -mSzK+1 ; aI<= mSzK ; aI++)
        {
            ///   aSomLine += aLine[aI] + mK0->Value(aI-aFracX);
            aSomLine += aLine[aI] * (*anAdr);
            aDxSomLine += aLine[aI] * (*aDxAdr);
            // aSomLine +=  (*anAdr);
            anAdr +=  mK1.NbDisc1();
            aDxAdr +=  mK1.NbDisc1();
        }
        double aCoefY = mK1.Value(aJ-aFracY);
        aSomVal += aSomLine * aCoefY;
        aSomDerX += aDxSomLine * aCoefY;
        aSomDerY +=  aSomLine * mK1.ValueDer(aJ-aFracY);
    }

    // return aSomGlob;
   return Pt3dr(-aSomDerX,-aSomDerY,aSomVal);
}


/*

template <class TypeEl>
 Pt3dr cTabIM2D_FromIm2D<TypeEl>::GetValDer(TypeEl ** aTab,const Pt2dr &  aP) const
{
   ELISE_ASSERT(false,"no ::GetValDer");
   return Pt3dr(0,0,0);
}
*/


/******************************************************/
/*                                                    */
/*             cInterpolPPV<TypeEl>            */
/*                                                    */
/******************************************************/

template <class TypeEl>
int cInterpolPPV<TypeEl>::SzKernel() const
{
   return 1;
}

template <class TypeEl>
double cInterpolPPV<TypeEl>::IL_GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
    return aTab[round_ni(aP.y)][round_ni(aP.x)];
}

template <class TypeEl>
double cInterpolPPV<TypeEl>::GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
    return IL_GetVal(aTab,aP);
}

template <class TypeEl>
void  cInterpolPPV<TypeEl>::GetVals
      (
          TypeEl ** aTab,
          const Pt2dr *  aPts,
          double *aVals,
          int Nb
      ) const
{
   for (int aK=0 ; aK<Nb ; aK++)
       aVals[aK] = IL_GetVal(aTab,aPts[aK]);
}


/******************************************************/
/*                                                    */
/*             cInterpolBilineaire<TypeEl>            */
/*                                                    */
/******************************************************/

template <class TypeEl>
int cInterpolBilineaire<TypeEl>::SzKernel() const
{
   return 1;
}

template <class TypeEl>
double cInterpolBilineaire<TypeEl>::IL_GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
    INT xo   = round_down(aP.x) ;
    INT yo   = round_down(aP.y) ;
    REAL  px1 = aP.x - xo;
    REAL  py1 = aP.y - yo;
    REAL  px0 =  1.0 -px1;
    REAL  py0 =  1.0 -py1;
    TypeEl * l0 = aTab[yo]+xo;
    TypeEl * l1 = aTab[yo+1]+xo;

    return
    (
          py0 * (px0*l0[0]+ px1*l0[1])
        + py1 * (px0*l1[0]+ px1*l1[1])
    );
}

template <class TypeEl>
Pt3dr cInterpolBilineaire<TypeEl>::GetValDer(TypeEl ** aTab,const Pt2dr &  aP) const
{
    INT xo   = round_down(aP.x) ;
    INT yo   = round_down(aP.y) ;
    REAL  px1 = aP.x - xo;
    REAL  py1 = aP.y - yo;
    REAL  px0 =  1.0 -px1;
    REAL  py0 =  1.0 -py1;
    TypeEl * l0 = aTab[yo]+xo;
    TypeEl * l1 = aTab[yo+1]+xo;

    return Pt3dr
           (
                  py0 * (l0[1]-l0[0]) +  py1 * (l1[1]-l1[0]),
                  px0 * (l1[0]-l0[0]) +  px1 * (l1[1]-l0[1]),
                  py0 * (px0*l0[0]+ px1*l0[1])
                + py1 * (px0*l1[0]+ px1*l1[1])
            );
}




template <class TypeEl>
double cInterpolBilineaire<TypeEl>::GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
    return IL_GetVal(aTab,aP);
}

template <class TypeEl>
void  cInterpolBilineaire<TypeEl>::GetVals
      (
          TypeEl ** aTab,
          const Pt2dr *  aPts,
          double *aVals,
          int Nb
      ) const
{
   for (int aK=0 ; aK<Nb ; aK++)
       aVals[aK] = IL_GetVal(aTab,aPts[aK]);
}


/******************************************************/
/*                                                    */
/*           cTplCIKTabul<TypeEl>                     */
/*                                                    */
/******************************************************/

void MPD_Coeff(double   anX,double &A,double & B,double &C,double &D)
{
   bool toSwap = false;
   if (anX>0.5)
   {
       toSwap = true;
       anX = 1.0-anX;
   }

   D=0.0;

   //  Les equations sont
   //   A+B+C =1 
   //   -A + C = x  => Barrycentre
   //
   //  Donc forme parametrique :
   //    A
   //    1-x-2A
   //    x+A

    // VERSION ON MAINTIENT l'ECART MOYEN
   //  L'ecar moyen vaut
   //  (1+x)A + x B + (1-x) C
   // Cet ecart vaut 0.5 lorque x vaut 0.5 si on
   // se limite a deux pixels

   // donc A = (0.5 - 2x (1-x)) / (2-2x)


    A = (0.5 -2.0 * anX * (1.0-anX)) / (2.0-2.0*anX);
    B = 1.0 -anX  - 2.0 * A;
    C = anX + A;


   if (toSwap)
   {
      ElSwap(A,D);
      ElSwap(B,C);
   }
}


template <class TypeEl,class tTabulCIK>
cTplCIKTabul<TypeEl,tTabulCIK>::cTplCIKTabul(INT aNBBVal,INT aNBBResol,REAL aA,eModeInterTabul aMode) :
    mNbResol  (1<<aNBBResol),
    mNbVal    (1<<aNBBVal),
    mNbVal2   (ElSquare(REAL(mNbVal))),
    mTV       (4,1+mNbResol),
    mTD       (4,1+mNbResol)
{
     mDV =  mTV.data();
     mDD =  mTD.data();

     cCubicInterpKernel aCIK(aA);


     for (INT indX = 0 ; indX <=mNbResol ; indX++)
     {
         REAL x = XofInd(indX);

         double A=0,B=0,C=0,D=0;

         if (aMode==eTabulBicub)
         {
              A = aCIK.Value(-x-1);
              B = aCIK.Value(-x);
              C = aCIK.Value(1-x);
              D = aCIK.Value(2-x);
         }
         else if (aMode==eTabulMPD_EcartMoyen)
         {
              MPD_Coeff(x,A,B,C,D);
//std::cout << "MPD_Coeff \n";
         }
         else if (aMode==eTabul_Bilin)
         {
              A = 0.0;
              B = 1-x;
              C = x;
              D = 0 ;
         }
         else
         {
             ELISE_ASSERT(false,"Bad mode in ::cTplCIKTabul");
         }
      
	 mDV[indX][0] = ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * A);
	 mDV[indX][1] = ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * B);
	 mDV[indX][2] = ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * C);
	 mDV[indX][3] = ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * D);

     // Pour assurer une somme rigoureusement egale a 1 (cas entier) on modifie le + gros
         if (x < 0.5)
         {
              mDV[indX][1] = mNbVal - (mDV[indX][0] + mDV[indX][2] + mDV[indX][3]);
         }
         else
         {
              mDV[indX][2] = mNbVal - (mDV[indX][0] + mDV[indX][1] + mDV[indX][3]);
         }

/*
	 mDV[indX][2] = ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * aCIK.Value(1-x));
     // Pour assurer une somme rigoureusement egale a 1
	 mDV[indX][3] = mNbVal - (mDV[indX][0] + mDV[indX][1] + mDV[indX][2]);
*/


	 mDD[indX][0] = - ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * aCIK.Derivee(-x-1));
	 mDD[indX][1] = - ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * aCIK.Derivee(-x));
	 mDD[indX][2] = - ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * aCIK.Derivee(1-x));
	 mDD[indX][3] = - ElStdTypeScal<tTabulCIK>::RtoT(mNbVal * aCIK.Derivee(2-x));

     }

     // Pour assurer une somme rigoureusement egale a 1
}

template <class TypeEl,class tTabulCIK>
tTabulCIK   cTplCIKTabul<TypeEl,tTabulCIK>::InterpolateVal(TypeEl * aTab,INT Frac) const
{
	tTabulCIK * aD = mDV[Frac];
	return     aTab[-1]*aD[0] + aTab[0]*aD[1] + aTab[1]*aD[2] + aTab[2]*aD[3];
}

template <class TypeEl,class tTabulCIK>
tTabulCIK cTplCIKTabul<TypeEl,tTabulCIK>::InterpolateDer(TypeEl * aTab,INT Frac) const
{
	tTabulCIK * aD = mDD[Frac];
	return     aTab[-1]*aD[0] + aTab[0]*aD[1] + aTab[1]*aD[2] + aTab[2]*aD[3];
}


template <class TypeEl,class tTabulCIK>
REAL    cTplCIKTabul<TypeEl,tTabulCIK>::BicubValue(TypeEl ** aTab,const Pt2dr & aP) const
{
      INT iX = round_down(aP.x);
      INT fX = round_down((aP.x-iX)*mNbResol);

      INT iY = round_down(aP.y);
      INT fY = round_down((aP.y-iY)*mNbResol);

      tTabulCIK  Vm1 = InterpolateVal(aTab[iY-1]+iX,fX);
      tTabulCIK  V0  = InterpolateVal(aTab[iY]+iX,fX);
      tTabulCIK  V1  = InterpolateVal(aTab[iY+1]+iX,fX);
      tTabulCIK  V2  = InterpolateVal(aTab[iY+2]+iX,fX);

      tTabulCIK * aD = mDV[fY];

      return (Vm1*aD[0] + V0*aD[1] + V1*aD[2] + V2*aD[3]) / mNbVal2;
}


template <class TypeEl,class tTabulCIK>
Pt3dr    cTplCIKTabul<TypeEl,tTabulCIK>::BicubValueAndDer(TypeEl ** aTab,const Pt2dr & aP) const
{
      INT iX = round_down(aP.x);
      INT fX = round_down((aP.x-iX)*mNbResol);

      INT iY = round_down(aP.y);
      INT fY = round_down((aP.y-iY)*mNbResol);

      tTabulCIK  Vm1 = InterpolateVal(aTab[iY-1]+iX,fX);
      tTabulCIK  V0  = InterpolateVal(aTab[iY]+iX,fX);
      tTabulCIK  V1  = InterpolateVal(aTab[iY+1]+iX,fX);
      tTabulCIK  V2  = InterpolateVal(aTab[iY+2]+iX,fX);

      tTabulCIK * aDV = mDV[fY];

      REAL V = (Vm1*aDV[0] + V0*aDV[1] + V1*aDV[2] + V2*aDV[3]) / mNbVal2;


      tTabulCIK  aDxm1 = InterpolateDer(aTab[iY-1]+iX,fX);
      tTabulCIK  aDx0  = InterpolateDer(aTab[iY]+iX,fX);
      tTabulCIK  aDx1  = InterpolateDer(aTab[iY+1]+iX,fX);
      tTabulCIK  aDx2  = InterpolateDer(aTab[iY+2]+iX,fX);


      REAL aDx = (aDxm1*aDV[0] + aDx0*aDV[1] + aDx1*aDV[2] + aDx2*aDV[3]) / mNbVal2;



      tTabulCIK * aDD = mDD[fY];
      REAL aDy = (Vm1*aDD[0] + V0*aDD[1] + V1*aDD[2] + V2*aDD[3]) / mNbVal2;

      return Pt3dr(aDx,aDy,V);
}

template <class TypeEl,class tTabulCIK>
bool cTplCIKTabul<TypeEl,tTabulCIK>::OkForInterp(Pt2di aSz,Pt2dr aP) const
{
   return (aP.x>2) && (aP.y>2) && (aP.x<aSz.x-3) && (aP.y<aSz.y-3);
}


template <class TypeEl,class tTabulCIK>
double cTplCIKTabul<TypeEl,tTabulCIK>::GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
   return BicubValue(aTab,aP);
}

template <class TypeEl,class tTabulCIK>
int cTplCIKTabul<TypeEl,tTabulCIK>::SzKernel() const
{
   return 2;
}




/******************************************************/
/*                                                    */
/*              cKernelInterpol                       */
/*                                                    */
/******************************************************/


     // ================  cKernelInterpol1D   =============

cKernelInterpol1D::cKernelInterpol1D (double mSzKernel) :
   mSzKernel (mSzKernel),
   mVecPX    (2*(2+round_up(mSzKernel))),
   // mDataPX   (mVecPX.data()), because MacOS
   mDataPX   (VData(mVecPX)),
   mVecPY    (mVecPX.size()),
   // mDataPY   (mVecPY.data())
   mDataPY   (VData(mVecPY))
{
}

cKernelInterpol1D::~cKernelInterpol1D()
{
}

double cKernelInterpol1D::Interpole(const cFoncI2D & aFonc,const double & x,const double & y)
{
    Box2di aBox = aFonc.BoxDef();

    int aX0 = ElMax(aBox._p0.x,round_up(x-mSzKernel));
    int aX1 = ElMin(aBox._p1.x,round_up(x+mSzKernel));
    
    int aY0 = ElMax(aBox._p0.y,round_up(y-mSzKernel));
    int aY1 = ElMin(aBox._p1.y,round_up(y+mSzKernel));

    double * aDx0 = mDataPX-aX0;
    double * aDy0 = mDataPY-aY0;

 //  Pre-calcule des poids en x et y
    double aSomPx = 0.0;  // Pour normaliser
    for (int anIx=aX0 ; anIx<aX1 ; anIx++)
    {
          double aV = Value(x-anIx);
          aDx0[anIx] = aV;
          aSomPx += aV;
    }
    double aSomPy = 0.0;  // Pour normaliser
    for (int anIy=aY0 ; anIy<aY1 ; anIy++)
    {
          double aV = Value(y-anIy);
          aDy0[anIy] = aV;
          aSomPy += aV;
    }

    double aSomV = 0.0;
    for (int anIx=aX0 ; anIx<aX1 ; anIx++)
    {
       for (int anIy=aY0 ; anIy<aY1 ; anIy++)
       {
            aSomV += aDx0[anIx]*aDy0[anIy]*aFonc.Val(anIx,anIy);
       }
    }

    return aSomV/ ElMax(1e-9,(aSomPx*aSomPy));
}


cKernelInterpol1D  * cKernelInterpol1D::StdInterpCHC(double aScale,int  aNbTab)
{
   double aParam = 0;
   if (aScale<1.0)
      aParam = -0.5;
   else if (aScale<1.5)
      aParam = aScale -1.5;


   cKernelInterpol1D * aKern = new cCubicInterpKernel(aParam);

   if (aScale==1) return aKern;

   aKern = new cScaledKernelInterpol(aKern,ElMax(1.0,aScale));
   if (aNbTab>=0)
   {
      aKern = new cTabulKernelInterpol(aKern,aNbTab,false);
   }
   return aKern;
   
}
     // ================  cSinCardApodInterpol1D   =============

cSinCardApodInterpol1D::cSinCardApodInterpol1D
(
          eModeApod aModeApod,
          double aSzK, 
          double aSzApod,
          double aEpsilon,
          bool   OnlyApod
) :
     cKernelInterpol1D(aSzK),
     mModeApod         (aModeApod),
     mOnlyApod         (OnlyApod),
     mSzApod          (ElMin(mSzKernel,aSzApod)),
     mEpsilon         (aEpsilon)
{
}


double  cSinCardApodInterpol1D::Value(double x) const
{
   double aPiX = PI * x;
   double aNorm = ElAbs(x);

   double aSinC =  ( aNorm < mEpsilon) ?  (1 - ElSquare(aPiX)/6.0) : (sin(aPiX)/aPiX);

   double aPdsApod =  0.0;

   if(aNorm<mSzKernel)
   {
      aPdsApod = 1.0;

      if (mModeApod==eTukeyApod)
      {

         if (aNorm > (mSzKernel-mSzApod))
         {
             double aDist = ElAbs(mSzKernel-aNorm);
             aDist = (aDist /mSzApod) * (PI/2);

             aPdsApod  = ElSquare(sin(aDist));
         }
      }
      else if (mModeApod==eModePorte)
      {
      }
   }


   if (mOnlyApod)
      return  aPdsApod;
   return aSinC  * aPdsApod;
}


     // ================  cCubicInterpKernel   =============

cCubicInterpKernel::cCubicInterpKernel(REAL aA) :
    cKernelInterpol1D((aA==0.0) ? 1.0 : 2.0),
    mA    (aA)
{
    if ((mA>0.0) || (mA<-3.0))
    {
        static bool First = true;
        if (First)
        {
           std::cout << "Warn cCubicInterpKernel Val " << mA << "\n";
           getchar();
           getchar();
        }
        First = false;
    }
/*
*/
}

REAL cCubicInterpKernel::Value(REAL x) const
{
     x = ElAbs(x);
     REAL x2 = x * x;
     REAL x3 = x2 * x;

     if (x <=1.0)
        return (mA+2) * x3-(mA+3)*x2+1;
     if (x <=2.0)
        return mA*x3 - 5*mA * x2 + 8* mA * x -4 * mA;
     return 0.0;
}

REAL cCubicInterpKernel::Derivee(REAL x) const
{
     INT sign = (x>0) ? 1 : -1;
     x *= sign;
     REAL x2 = x * x;

     if (x <=1.0)
        return sign * (3* (mA+2) * x2- 2*(mA+3)*x);
     if (x <=2.0)
        return sign * (3*mA * x2 - 10*mA * x + 8* mA);
     return 0.0;
}

void cCubicInterpKernel::ValAndDerivee(REAL x,REAL &V,REAL &D) const
{
     INT sign = (x>0) ? 1 : -1;
     x *= sign;
     REAL x2 = x * x;
     REAL x3 = x2 * x;

     if (x <=1.0)
     {
          D = sign * (3* (mA+2) * x2- 2*(mA+3)*x);
          V =  (mA+2) * x3-(mA+3)*x2+1;
     }
     else if (x <=2.0)
     {
          D = sign * (3*mA * x2 - 10*mA * x + 8* mA);
          V = mA*x3 - 5*mA * x2 + 8* mA * x -4 * mA;;
     }
     else
     {
         D=V=0;
     }
}


     // ================  cScaledKernelInterpol   =============

cScaledKernelInterpol::cScaledKernelInterpol
(
    const cKernelInterpol1D * aKer0,
    double ascale
) :
  cKernelInterpol1D(ascale*aKer0->SzKernel()),
  mKer0 (aKer0),
  mScale(ascale),
  m1SurS (1.0/mScale)
{
}

cScaledKernelInterpol::~cScaledKernelInterpol()
{
   delete mKer0;
}

double  cScaledKernelInterpol::Value(double x) const
{
   return mKer0->Value(x*m1SurS)*m1SurS;
}

     // ================  cTabulKernelInterpol   =============

cTabulKernelInterpol::cTabulKernelInterpol
(
    const cKernelInterpol1D * aKer0,
    int NbDisc1,
    bool mPrecBil
) :
   cKernelInterpol1D (aKer0->SzKernel()),
   // mKer0             (aKer0),
   mNbDisc1          (NbDisc1),
   mNbValPos         (round_up(NbDisc1*mSzKernel)),
   mSzTab            (1+2*mNbValPos),
   mImTab            (mSzTab),
   mTab              (mImTab.data()),
   mImDer            (mSzTab),
   mDer              (mImDer.data())
{
    for (int anX=0 ;  anX<mSzTab ; anX++)
       mTab[anX] = aKer0->Value(Disc2Real(anX));

    // Normalisation a somme 1
    for (int anX=0 ; anX<mNbDisc1 ; anX++)
    {
        double aSom=0.0;

        for (int anY=anX ; anY<mSzTab ; anY+=mNbDisc1)
        {
            aSom += mTab[anY];
        }

        for (int anY=anX ; anY<mSzTab ; anY+=mNbDisc1)
        {
            mTab[anY] /= aSom;
        }
    }


    for (int anX=0 ; anX<mSzTab ; anX++)
    {
        int aXm1 = ElMax(0,anX-1);
        int aXp1 = ElMin(mSzTab-1,anX+1);

        double aDif = mTab[aXp1]-mTab[aXm1];
        mDer[anX] = (aDif/2.0) * mNbDisc1;
    }
}

double cTabulKernelInterpol::Disc2Real(double aX) const
{
    return (aX-mNbValPos) / mNbDisc1;
}

int cTabulKernelInterpol::Real2Disc(double aX) const
{
    return round_ni(mNbValPos+aX*mNbDisc1);
}

cTabulKernelInterpol::~cTabulKernelInterpol()
{
   // delete mKer0;
}

double  cTabulKernelInterpol::Value(double x) const
{
    int aK = Real2Disc(x);
    if (aK<0) return 0;
    if (aK>=mSzTab) return 0;
    return mTab[aK];
}

double  cTabulKernelInterpol::ValueDer(double x) const
{
    int aK = Real2Disc(x);
    if (aK<0) return 0;
    if (aK>=mSzTab) return 0;
    return mDer[aK];
}


const double * cTabulKernelInterpol::AdrDisc2Real(double  aX) const
{
    return mTab +  Real2Disc(aX);
}

const double * cTabulKernelInterpol::DerAdrDisc2Real(double  aX) const
{
    return mDer +  Real2Disc(aX);
}


/******************************************************/
/*                                                    */
/*                cTplElemBicub                       */
/*                                                    */
/******************************************************/
    //===========================

class cTplElemBicub
{
    public :

      inline cTplElemBicub(const cCubicInterpKernel & aCIK,REAL anX)
      {
          mI = round_down(anX);
          mFrac = anX-mI;
          aCIK.ValAndDerivee(-1-mFrac , mPdsValM1 , mPdsDerM1 );
          aCIK.ValAndDerivee(  -mFrac , mPdsVal0  , mPdsDer0  );
          aCIK.ValAndDerivee(1 -mFrac , mPdsVal1  , mPdsDer1  );
          aCIK.ValAndDerivee(2 -mFrac , mPdsVal2  , mPdsDer2  );
      }

      template <class Type > inline REAL GetVal(Type * aTab)
      {
           return    mPdsValM1 * aTab[-1]
                  +  mPdsVal0  * aTab[ 0]
                  +  mPdsVal1  * aTab[ 1]
                  +  mPdsVal2  * aTab[ 2] ;
      }
      template <class Type > inline REAL GetDer(Type * aTab)
      {
           return    mPdsDerM1 * aTab[-1]
                  +  mPdsDer0  * aTab[ 0]
                  +  mPdsDer1  * aTab[ 1]
                  +  mPdsDer2  * aTab[ 2] ;
      }

      INT  mI;
      REAL mFrac;

      REAL mPdsValM1;
      REAL mPdsVal0;
      REAL mPdsVal1;
      REAL mPdsVal2;

      REAL mPdsDerM1;
      REAL mPdsDer0;
      REAL mPdsDer1;
      REAL mPdsDer2;

};





template <class Type>
Pt3dr BicubicInterpol(const cCubicInterpKernel & aCIK,Type ** data,Pt2dr aP)
{
    cTplElemBicub aElemX(aCIK,aP.x);
    cTplElemBicub aElemY(aCIK,aP.y);


    Type * aLm1 = data[aElemY.mI-1] + aElemX.mI;
    Type * aL0  = data[aElemY.mI]   + aElemX.mI;
    Type * aL1  = data[aElemY.mI+1] + aElemX.mI;
    Type * aL2  = data[aElemY.mI+2] + aElemX.mI;

    static REAL aTabVX[4];
    aTabVX[0] =  aElemX.GetVal(aLm1);
    aTabVX[1] =  aElemX.GetVal(aL0);
    aTabVX[2] =  aElemX.GetVal(aL1);
    aTabVX[3] =  aElemX.GetVal(aL2);

    static REAL aTabDX[4];
    aTabDX[0] =  aElemX.GetDer(aLm1);
    aTabDX[1] =  aElemX.GetDer(aL0);
    aTabDX[2] =  aElemX.GetDer(aL1);
    aTabDX[3] =  aElemX.GetDer(aL2);

    return Pt3dr
           (
               - aElemY.GetVal(aTabDX+1),
               - aElemY.GetDer(aTabVX+1),
               aElemY.GetVal(aTabVX+1)
           );
}

class cTplElemBicubVal
{
    public :

      inline cTplElemBicubVal(const cCubicInterpKernel & aCIK,REAL anX)
      {
          mI = round_down(anX);
          mFrac = anX-mI;
          mPdsValM1 = aCIK.Value(-1-mFrac );
          mPdsVal0  = aCIK.Value(  -mFrac );
          mPdsVal1  = aCIK.Value(1 -mFrac );
          mPdsVal2  = aCIK.Value(2 -mFrac );
      }

      template <class Type > inline REAL GetVal(Type * aTab)
      {
           return    mPdsValM1 * aTab[-1]
                  +  mPdsVal0  * aTab[ 0]
                  +  mPdsVal1  * aTab[ 1]
                  +  mPdsVal2  * aTab[ 2] ;
      }

      INT  mI;
      REAL mFrac;

      REAL mPdsValM1;
      REAL mPdsVal0;
      REAL mPdsVal1;
      REAL mPdsVal2;
};

template <class Type>
REAL  BicubicInterpolVal(const cCubicInterpKernel & aCIK,Type ** data,Pt2dr aP)
{
    cTplElemBicubVal aElemX(aCIK,aP.x);
    cTplElemBicubVal aElemY(aCIK,aP.y);


    static REAL aTabVX[4];
    aTabVX[0] =  aElemX.GetVal(data[aElemY.mI-1] + aElemX.mI);
    aTabVX[1] =  aElemX.GetVal(data[aElemY.mI]   + aElemX.mI);
    aTabVX[2] =  aElemX.GetVal(data[aElemY.mI+1] + aElemX.mI);
    aTabVX[3] =  aElemX.GetVal(data[aElemY.mI+2] + aElemX.mI);

    return aElemY.GetVal(aTabVX+1) ;
}


/******************************************************/
/*                                                    */
/*             cInterpolBicubique<TypeEl>             */
/*                                                    */
/******************************************************/

template <class TypeEl>
int cInterpolBicubique<TypeEl>::SzKernel() const
{
   return 2;
}

template <class TypeEl>
double cInterpolBicubique<TypeEl>::GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
    return BicubicInterpolVal(mCIK,aTab,aP);

}
template <class TypeEl>
cInterpolBicubique<TypeEl>::cInterpolBicubique(double aVal) :
  mCIK(aVal)
{
}


/******************************************************/
/*                                                    */
/*             cInterpolSinusCardinal<TypeEl>             */
/*                                                    */
/******************************************************/

double gaussian(double x, double sigma)
{
	return exp(- x*x / (2*sigma*sigma));
}

void computeTab(REAL *tab, unsigned int size, REAL frac, bool apodise)
{
	if (fabs(frac) < 1e-6)
	{
		unsigned int doublesize = size * 2;
		for (unsigned int i=0; i<doublesize; ++i)
			tab[i] = 0;
		tab[size] = 1.0;
		return;
	}
  REAL pos, val;
  unsigned int doublesize = size * 2;
  for (unsigned int i=0; i<doublesize; ++i)
    {
    	pos = i - size - frac;
	    val = pos * PI;
		tab[i] = sin(val) / val;
		if (apodise)
			tab[i] *= gaussian(pos, size / 2.);
			/// On ne se pose pas trop de questions de normalisation, 
			/// parce qu'on le gère explicitement dans la suite...
    }
  /// Renormalisation pour que l'integrale vaille 1
  REAL sum = 0;
  for (unsigned int i=0; i<doublesize; ++i)
    {
      sum += tab[i];
    }
  if (sum != 1.)
  	for (unsigned int i=0; i<doublesize; ++i)
    	{
      		tab[i] /= sum;
    	}
}

template <class TypeEl>
double computeVal(REAL *tab, TypeEl *data, int size)
{
  REAL val = 0;
  unsigned int doublesize = size * 2;
  REAL* curTab = tab;
  REAL* finTab = tab + doublesize;
  TypeEl* curData = data;
  for (; curTab != finTab; ++curTab, ++curData)
  	val += (*curData) * (*curData);

  return val;
}

template <class TypeEl>
double cInterpolSinusCardinal<TypeEl>::GetVal(TypeEl ** aTab,const Pt2dr &  aP) const
{
  INT Ix = round_down(aP.x), Iy = round_down(aP.y);
  REAL fracX = aP.x-Ix, fracY = aP.y-Iy;
//  std::cout << "fracX : " << fracX << std::endl;
//  std::cout << "fracY : " << fracY << std::endl;
  computeTab(m_tabX, m_sizeOfWindow, fracX, m_apodise);
  computeTab(m_tabY, m_sizeOfWindow, fracY, m_apodise);
  unsigned int doublesize = m_sizeOfWindow * 2;
  for (unsigned int i=0; i<doublesize; ++i)
	{
		m_tabTemp[i] = computeVal(m_tabX, aTab[Iy - m_sizeOfWindow + i] + Ix - m_sizeOfWindow, m_sizeOfWindow);
//std::cout << i << " : " << m_tabTemp[i] << std::endl;
	}
  return computeVal(m_tabY, m_tabTemp, m_sizeOfWindow);
}

template <class TypeEl>
int cInterpolSinusCardinal<TypeEl>::SzKernel() const
{
  return m_sizeOfWindow + 1; // Modif MPD
}

template <class TypeEl>
cInterpolSinusCardinal<TypeEl>::cInterpolSinusCardinal(int sizeOfWindow, bool apodise)
  :  m_apodise(apodise), m_sizeOfWindow(sizeOfWindow)
{
  m_tabX = new REAL[m_sizeOfWindow*2];
  m_tabY = new REAL[m_sizeOfWindow*2];
  m_tabTemp = new REAL[m_sizeOfWindow*2];
}

template <class TypeEl>
cInterpolSinusCardinal<TypeEl>::~cInterpolSinusCardinal()
{
  delete m_tabX;
  delete m_tabY;
  delete m_tabTemp;
}


#include "general/bitm.h"

void TestSinc()
{
	cInterpolBicubique<U_INT1> bicu(-0.5);
	cInterpolSinusCardinal<U_INT1> sinc(4, false);

	U_INT1 *data = new U_INT1[100];
	U_INT1 **ligne = new U_INT1*[10];
	for (unsigned int i=0; i<10; ++i)
		ligne[i] = data + i*10;

//	for (unsigned int j=0; j<10;++j)
//		ligne[0][j] = j*10;

	for (unsigned int i=0; i<10; ++i)
		for (unsigned int j=0; j<10;++j)
			ligne[i][j] = i*10 +j;

	Pt2dr pos;
	pos.y = 5.;
	for (pos.x = 4.; pos.x < 5.1; pos.x += 0.1)
	{
	double vb = bicu.GetVal(ligne, pos);
	std::cout << "bic : " << vb << std::endl;
	double vs = sinc.GetVal(ligne, pos);
	std::cout << "sin : " << vs << std::endl;
	}
}

/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*                                                                          */
/****************************************************************************/

class cTestCubic
{
	public :
            cTestCubic() :
              mW (Video_Win::WStd(Pt2di(600,400),1.0))
	    {
	    }

	     void DrawKernel(REAL aA);

	     Pt2dr ToPVal(Pt2dr aP) {  aP.y *= -100;
		                       aP.x *=  100;
		                       return aP+Pt2dr(300.0,200.0);
	                            }
	     Pt2dr ToPDer(Pt2dr aP) {  aP.y *= -100;
		                       aP.x *=  100;
		                       return aP+Pt2dr(300.0,200.0);
	                            }
	private  :
              Video_Win mW;
};
void cTestCubic::DrawKernel(REAL aA)
{
     cout << "TEST NOYAU BI-CUB FOR : " << aA << "\n";
     mW.clear();
     cCubicInterpKernel aCI(aA);
     REAL aStep = 0.1;
     mW.draw_seg(ToPVal(Pt2dr(-4,0)),ToPVal(Pt2dr(4,0)),mW.pdisc()(P8COL::white));
     mW.draw_seg(ToPVal(Pt2dr(0,-4)),ToPVal(Pt2dr(0,4)),mW.pdisc()(P8COL::white));

    for (int aX=-3; aX<=3 ; aX++)
    {
         mW.draw_seg(ToPVal(Pt2dr(aX,-0.2)),ToPVal(Pt2dr(aX,0.2)),mW.pdisc()(P8COL::white));
    }

     for (REAL x1= -3; x1<=3 ; x1 += aStep)
     {
          REAL x2 = x1+aStep;
          REAL aV1 = aCI.Value(x1);
          REAL aD1 = aCI.Derivee(x1);
          REAL aV2,aD2;
          aCI.ValAndDerivee(x1,aV2,aD2);
          ELISE_ASSERT(ElAbs(aV1-aV2)<1e-6,"cTestCubic::DrawKernel");
          ELISE_ASSERT(ElAbs(aD1-aD2)<1e-6,"cTestCubic::DrawKernel");
          mW.draw_seg
          (
               ToPVal(Pt2dr(x1,aCI.Value(x1))),
               ToPVal(Pt2dr(x2,aCI.Value(x2))),
               mW.pdisc()(P8COL::red)
          );
          mW.draw_seg
          (
               ToPDer(Pt2dr(x1,aCI.Derivee(x1))),
               ToPDer(Pt2dr(x2,aCI.Derivee(x2))),
               mW.pdisc()(P8COL::blue)
          );
     }
}

static void Verif
            (
                 const cCubicInterpKernel & aKer,
                 cTplCIKTabul<U_INT1,int> & aCIKT,
                 Im2D_U_INT1 aIm,
                 Pt2dr aP
            )
{
     Pt3dr  DxDyV = aCIKT.BicubValueAndDer(aIm.data(),aP);
     Pt3dr  aPex = BicubicInterpol(aKer,aIm.data(),aP);
     REAL   aVex = BicubicInterpolVal(aKer,aIm.data(),aP);
     REAL V2 =    5.0 *aP.x
                + 4.0 * aP.y
                + 3.0 * ElSquare(aP.x-5)
                + 2.0 * ElSquare(aP.y-5)
                + aP.x * aP.y;

     REAL Dx = 5.0 + 6.0*(aP.x-5) + aP.y;
     REAL Dy = 4.0 + 4.0*(aP.y-5) + aP.x;

     cout << "------  " << aPex.z << " " << aVex << "\n";
     cout << DxDyV << aPex << "\n";
     cout << " VALS " << DxDyV.z << " " << V2
          << " Dx   " << DxDyV.x << " " << Dx
          << " Dy   " << DxDyV.y << " " << Dy
	  << "\n";
}

void TestCubic()
{

        REAL aVA = -0.5;
	cTplCIKTabul<U_INT1,int> aCIKT(8,8,-0.5);
        cCubicInterpKernel aKer(aVA);
	Im2D_U_INT1 aIm(10,10);

	ELISE_COPY
        (
	      aIm.all_pts(),
	      5*FX +4*FY + 3* Square(FX-5) + 2 *  Square(FY-5) + FX * FY,
	      aIm.out()
	 );

	// ELISE_COPY(aIm.all_pts(),FX+FY,aIm.out());

	Verif(aKer,aCIKT,aIm,Pt2dr(5.0,5.25));
	Verif(aKer,aCIKT,aIm,Pt2dr(7.125,6.75));
	Verif(aKer,aCIKT,aIm,Pt2dr(4.75,2.375));

	/*
	cout << "Bi Cube " << aCIKT.BicubValue(aIm.data(),Pt2dr(5.0,5.25)) << "\n";
	cout << "Bi Cube " << aCIKT.BicubValue(aIm.data(),Pt2dr(5.5,5.0)) << "\n";
	cout << "Bi Cube " << aCIKT.BicubValue(aIm.data(),Pt2dr(5.5,5.125)) << "\n";
	cout << "Bi Cube " << aCIKT.BicubValue(aIm.data(),Pt2dr(6.0,5.125)) << "\n";
	*/

	cTestCubic aTC;



        while (1)
        {
            std::cout << "ENTER BICUB VALUE \n";
            double aV;
            std::cin >> aV;
	    aTC.DrawKernel(aV);
        }

	aTC.DrawKernel(0.0);
	getchar();


	aTC.DrawKernel(-0.5);
	getchar();

	aTC.DrawKernel(-(48/pow(PI,4.0)));
	getchar();

	aTC.DrawKernel(-0.75);
	getchar();

	aTC.DrawKernel(-1.0);
	getchar();

	aTC.DrawKernel(-3.0);
	getchar();
}



#define IntantIntepr(aType,aTbase)\
template class cTabIM2D_FromIm2D<aType>;\
template class cInterpolateurIm2D<aType>;\
template class cInterpolBilineaire<aType>;\
template class cInterpolPPV<aType>;\
template class cTplCIKTabul<aType,aTbase>;\
template Pt3dr BicubicInterpol(const cCubicInterpKernel & aCIK,aType ** data,Pt2dr aP);\
template REAL BicubicInterpolVal(const cCubicInterpKernel & aCIK,aType ** data,Pt2dr aP);\
template class cInterpolBicubique<aType>;\
template class cInterpolSinusCardinal<aType>;

IntantIntepr(U_INT1,INT);
IntantIntepr(INT1,INT);
IntantIntepr(U_INT2,INT);
IntantIntepr(INT2,INT);
IntantIntepr(INT,INT);
IntantIntepr(REAL4,REAL8);
IntantIntepr(REAL8,REAL8);
IntantIntepr(REAL16,REAL16);





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
