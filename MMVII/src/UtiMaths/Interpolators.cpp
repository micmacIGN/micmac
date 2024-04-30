#include "MMVII_Interpolators.h"

namespace MMVII
{

class cInterpolator1D : public cMemCheck, public cFctrRR
{
      public :
        cInterpolator1D(const tREAL8 & aSzKernel);
        virtual ~cInterpolator1D();

        // virtual tREAL8  F(const tREAL8 & anX) const = 0;
	const tREAL8 SzKernel() const;  // accessor
      protected :
	tREAL8 mSzKernel;
};

class cDiffInterpolator1D : public cInterpolator1D
{
       public :
            cDiffInterpolator1D(tREAL8 aSzK);
            virtual tREAL8  DF(tREAL8  anX) const =0;
};



class cBilinInterpolator1D : public cInterpolator1D
{
      public :
        cBilinInterpolator1D();
        tREAL8  F(tREAL8  anX) const override ;
};


class cIntepolatorCubic : public cDiffInterpolator1D
{
	public :
            // mA = valeur de la derivee en  ?  ( 1?)
            // si vaut -0.5, reconstitue parfaitement une droite
            // doit etre comprise en 0 et -3
             cIntepolatorCubic(tREAL8 aParam);
             tREAL8  F(tREAL8  anX) const override ;
             tREAL8  DF(tREAL8  anX) const override ;
       private :
	     tREAL8 mA;

};

class cTabulatedInterpolator : public cInterpolator1D
{
      public :
          cTabulatedInterpolator(const cInterpolator1D &,int aNbTabul,bool IsBilin);

          tREAL8  F(tREAL8  anX) const override ;
      private :

	  cTabulFonc1D mTabul;
};

class cEpsDiffFctr : public cDiffInterpolator1D
{
      public :
          cEpsDiffFctr(const cInterpolator1D & anInt,tREAL8 aEps) ;
          tREAL8  F(tREAL8  anX) const override ;
          tREAL8  DF(tREAL8  anX) const override;
      private :
	   const cInterpolator1D & mInt;
	   tREAL8    mEps;
};

cEpsDiffFctr::cEpsDiffFctr(const cInterpolator1D & anInt,tREAL8 aEps) :
    cDiffInterpolator1D (anInt.SzKernel()),
    mInt (anInt),
    mEps (aEps)
{
}


tREAL8  cEpsDiffFctr::F(tREAL8  anX) const  {return mInt.F(anX);}
tREAL8  cEpsDiffFctr::DF(tREAL8  anX) const  {return (mInt.F(anX+mEps)-mInt.F(anX-mEps)) / (2*mEps) ;}


/*
class cTabulatedDiffInterpolator : public cDiffInterpolator1D
{
      public :
      private :
             tREAL8  F(tREAL8  anX) const override ;
};
*/

/* *************************************************** */
/*                                                     */
/*           cDiffInterpolator1D                       */
/*                                                     */
/* *************************************************** */

cDiffInterpolator1D::cDiffInterpolator1D(tREAL8 aSzK) :
      cInterpolator1D (aSzK)
{
}

/* *************************************************** */
/*                                                     */
/*           cEpsDiffFctr                              */
/*                                                     */
/* *************************************************** */





/* *************************************************** */
/*                                                     */
/*           cIntepolatorCubic                         */
/*                                                     */
/* *************************************************** */

tREAL8  cIntepolatorCubic::F(tREAL8  x) const
{
     x = std::abs(x);
     tREAL8 x2 = x * x;
     tREAL8 x3 = x2 * x;

     if (x <=1.0)
        return (mA+2) * x3-(mA+3)*x2+1;   // f(0) = 1,  f(1)=a+2 -(a+3) + 1 = 0  , f'(1) = 3(A+2) -2(A+3) = A
     if (x <=2.0)
        return mA*(x3 - 5 * x2 + 8* x -4); 
        // f(1) =A(1-5+8-4)=0 , f'(1)= 3a-10A +8A = A  ,
        // f(2) = 8(A -20 +16-4) = 0   f'(2)=A( 12 -20 +8) = 0
     return 0.0;
}

cIntepolatorCubic::cIntepolatorCubic(tREAL8 aParam) :
   cDiffInterpolator1D((aParam==0.0) ? 1.0 : 2.0),  // when A=0, the kernel is [-1,1] 
   mA (aParam)
{
}

tREAL8  cIntepolatorCubic::DF(tREAL8  x) const
{
     int aS = (x>=0) ? 1 : -1;
     x = std::abs(x);
     tREAL8 x2 = x * x;

     if (x <=1.0)
        return  aS* (3*(mA+2) * x2 - 2*(mA+3) * x);
     if (x <=2.0)
        return aS* mA*(3*x2 - 10 * x + 8);
     return 0.0;
}

/* *************************************************** */
/*                                                     */
/*           cTabulatedInterpolator                    */
/*                                                     */
/* *************************************************** */

cTabulatedInterpolator::cTabulatedInterpolator(const cInterpolator1D &anInt,int aNbTabul,bool IsBilin) :
     cInterpolator1D  (anInt.SzKernel()),
     mTabul   (anInt,0.0,anInt.SzKernel(),round_up(anInt.SzKernel()*aNbTabul),IsBilin)
{
}

tREAL8  cTabulatedInterpolator::F(tREAL8  anX) const 
{
   return mTabul.F(std::abs(anX));
}

/* *************************************************** */
/*                                                     */
/*           cInterpolator1D                           */
/*                                                     */
/* *************************************************** */

cInterpolator1D::cInterpolator1D(const tREAL8 & aSzKernel) :
	mSzKernel (aSzKernel)
{
}

cInterpolator1D::~cInterpolator1D()
{
}

const tREAL8 cInterpolator1D::SzKernel() const {return mSzKernel;}

/* *************************************************** */
/*                                                     */
/*           cBilinInterpolator1D                      */
/*                                                     */
/* *************************************************** */

cBilinInterpolator1D::cBilinInterpolator1D() :
       cInterpolator1D (1.0)
{
}

tREAL8  cBilinInterpolator1D::F(tREAL8  anX) const 
{
      return std::max(0.0,1.0-std::abs(anX));
}

/* *************************************************** */
/*                                                     */
/*           IM2D/IM2D                                 */
/*                                                     */
/* *************************************************** */

template <> inline  bool cPixBox<2>::InsideInterpolator(const cInterpolator1D & anInterpol,const cPtxd<double,2> & aP) const
{
    tREAL8 aSzK = anInterpol.SzKernel();
    // tREAL8 aSzKm1 = aSzK-1.0;

    // StdOut()  << " IIII " << aP << " XX=" << aP.x()-aSzKm1 << "\n";


    return   ( round_Uup(aP.x()-aSzK) >= tBox::mP0.x()) &&  (round_Ddown(aP.x()+aSzK) <  tBox::mP1.x())
          && ( round_Uup(aP.y()-aSzK) >= tBox::mP0.y()) &&  (round_Ddown(aP.y()+aSzK) <  tBox::mP1.y())
    ;
}

static std::vector<tREAL8>  TheBufCoeffX;  // store the value on 1 line, as they are separable
template <class Type>  tREAL8 cDataIm2D<Type>::GetValueInterpol(const cPt2dr & aP,const cInterpolator1D & anInterpol) const 
{
    TheBufCoeffX.clear();
    tREAL8 aSzK = anInterpol.SzKernel();

    tREAL8 aRealY = aP.y();
    int aY0 = round_Uup(aRealY-aSzK);  
    int aY1 = round_Ddown(aRealY+aSzK);

    tREAL8 aRealX = aP.x();
    int aX0 = round_Uup(aRealX-aSzK); 
    int aX1 = round_Ddown(aRealX+aSzK);
    int aNbX =  aX1-aX0+1;

    tREAL8 aSomWX=0;
    for (int aIntX=aX0 ; aIntX<=aX1 ; aIntX++)
    {
        tREAL8 aWX = anInterpol.F(aIntX-aRealX);
        TheBufCoeffX.push_back(aWX);
	aSomWX += aWX;
    }
    const tREAL8 *  aLineW0  = TheBufCoeffX.data();

    tREAL8 aSomWIxy = 0.0;
    tREAL8 aSomWY=0;
    for (int aIntY=aY0 ; aIntY<=aY1 ; aIntY++)
    {
        tREAL8 aWY = anInterpol.F(aIntY-aRealY);
	aSomWY += aWY;
	const Type *  aLineIm = mRawData2D[aIntY] + aX0;
	const tREAL8 *  aCurWX  = aLineW0;
	tREAL8 aSomWIx = 0.0;

	// for (int aKX=0 ; aKX< aNbX ; aKX++)
        int aKx= aNbX;
	while (aKx--)
            aSomWIx += *(aLineIm++)  *  *(aCurWX++) ;
        aSomWIxy  += aSomWIx * aWY;
    }

    return aSomWIxy / (aSomWX * aSomWY) ;
}

template class cDataIm2D<tU_INT1>;
template class cDataIm2D<tREAL4>;


/**  Basic test on bilinear interpolator, compare "hand crafted" GetVBL with :
 *     - cBilinInterpolator1D
 *     - Tabulated-raw
 *     - Tabulated-bilin
 *     
 */
template <class Type>  void  TplBenchInterpol_1(const cPt2di & aSz)
{
     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();

     cBilinInterpolator1D aBil1;
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));
     cTabulatedInterpolator aTabBil1(aBil1,1e5,false);

     cTabulatedInterpolator aTabBiBil(aBil1,1e2,true);  // the tabulation is itself bilin

     int aNbTest = 0;
     while (aNbTest<1000)
     {
         cPt2dr  aPt = aBoxR.GeneratePointInside();
	 if (aDIm.InsideBL(aPt))
	 {
             tREAL8 aV1 = aDIm.GetVBL(aPt);
             tREAL8 aV2 = aDIm.GetValueInterpol(aPt,aBil1);
             tREAL8 aV3 = aDIm.GetValueInterpol(aPt,aTabBil1);
             tREAL8 aV4 = aDIm.GetValueInterpol(aPt,aTabBiBil);
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV1-aV2)<1e-5,"Interpol ");

	     tREAL8 aDif = std::abs(aV1-aV3) / tNumTrait<Type>::AmplRandomValueCenter();
	     MMVII_INTERNAL_ASSERT_bench(aDif<1e-4,"Interpol ");

	     aDif = std::abs(aV1-aV4);
	     MMVII_INTERNAL_ASSERT_bench(aDif<1e-4,"Interpol ");
             aNbTest ++;
	 }
     }
}

// template <class Type>  void  TplBenchInterpol_2(const cPt2di & aSz)

template <class Type>  void  TplBenchInterpol_2(const cPt2di & aSz)
{
     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));

     tREAL8 A = -2;
     tREAL8 B = 3;
     tREAL8 C = -123;

     for (const auto & aP : aDIm)
     {
         aDIm.SetV(aP,A*aP.x()+B*aP.y() + C);
     }

     // bicub that interpolate exactly linear function
     cIntepolatorCubic aI3(-0.5);
     cTabulatedInterpolator aTabI3(aI3,1000,true);
     int aNbTest = 0;
     while (aNbTest<5000)
     {
         cPt2dr  aPt = aBoxR.GeneratePointInside();
         if (aDIm.InsideInterpolator(aI3,aPt))
         {
             aNbTest++;

	     tREAL8 aV0 = A*aPt.x()+B*aPt.y() + C;
	     tREAL8 aV1 = aDIm.GetValueInterpol(aPt,aI3);
	     tREAL8 aV2 = aDIm.GetValueInterpol(aPt,aTabI3);

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV0-aV1)<1e-6,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV0-aV2)<1e-6,"Interpol ");
         }
     }
}

void  BenchInterpol(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Interpol")) return;


     //  Bench on bicub interpola
     //
     for (const auto & aP : {0.0,1.0,2.0,3.0})
     {
         cIntepolatorCubic  anI3(aP);

	 MMVII_INTERNAL_ASSERT_bench(std::abs(1-anI3.F(0))<1e-8,"Interpol ");   // F(0)=1
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.F(1))<1e-8,"Interpol ");     // F(1)=0
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.F(2))<1e-8,"Interpol ");     // F(2)=0
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.F(2.1))<1e-8,"Interpol ");    // F(X>2) = 0
										 //
	 for (const auto aEps : {1e-2,1e-3,1e-4})
	 {
             // Derivate in 2 = 0
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.F(2-aEps))<4*Square(aEps) ,"Interpol ");
	     // diffrentuale in 1  dF/dx+ = dF/dx-
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.F(1-aEps) + anI3.F(1+aEps))<4*Square(aEps) ,"Interpol ");
	 }
	 // compare  "analytical derivatives" with finite diff
         cEpsDiffFctr aEpsI3(anI3,1e-4);

	 for (int aK=0 ; aK<100 ; aK++)
	 {
             tREAL8 anX = RandInInterval(-3,3);

	     tREAL8 aD1 = anI3.DF(anX);
	     tREAL8 aD2 = aEpsI3.DF(anX);
	     tREAL8 aF1 = anI3.F(anX);
	     tREAL8 aF2 = aEpsI3.F(anX);

	     // StdOut() << "DDDD X=" << anX << " "  << aD1 << " DifF=" << aD2 << " Dif=" << aD1 -aD2 << "\n";
	     // StdOut() << "FFFF X=" << anX << " "  << aF1 << " DifF=" << aF2 << " Dif=" << aF1 -aF2 << "\n\n";

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aD2)<1e-5 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF1-aF2)<1e-10 ,"Interpol ");
	 }
     }

     TplBenchInterpol_1<tU_INT1>(cPt2di(230,127));
     TplBenchInterpol_1<tREAL4>(cPt2di(144,201));


     TplBenchInterpol_2<tINT2>(cPt2di(199,147));
     TplBenchInterpol_2<tREAL8>(cPt2di(100,188));


     aParam.EndBench();
}




/* *************************************************** */
/*                                                     */
/*                ::                                   */
/*                                                     */
/* *************************************************** */


tREAL8 CubAppGaussVal(const tREAL8& aV)
{
   tREAL8 aAbsV = std::abs(aV);
   if (aAbsV>1.0) return 0.0;

   tREAL8 aAV2 = Square(aAbsV);

   return 1.0 + 2.0*aAbsV*aAV2 - 3.0*aAV2;
}

/**   Compute the weighting for ressampling one pixel of an image with a mapping M.
 *  Formalisation :
 *
 *      - we have pixel out  Co
 *      - we have an image weighing arround Co  W(P) = BiCub((P-Co)/aSzK)
 *      - let S be the support of W(P) we compute the box of M-1(S)
 *
 */

cRessampleWeigth  cRessampleWeigth::GaussBiCub(const cPt2dr & aCenterOut,const cAff2D_r & aMapO2I, double aSzK)
{
     cRessampleWeigth aRes;

     // [1] compute the box in input image space 
     cPt2dr aSzW = cPt2dr::PCste(aSzK);
     cBox2dr aBoxOut(aCenterOut-aSzW,aCenterOut+aSzW);
     cBox2di aBoxIn =  ImageOfBox(aMapO2I,aBoxOut).Dilate(1).ToI();

     cAff2D_r  aMapI2O = aMapO2I.MapInverse();

     double aSomW = 0.0;
     for (const auto & aPixIn : cRect2(aBoxIn))
     {
         cPt2dr aPixOut = aMapI2O.Value(ToR(aPixIn));
         double aW =  CubAppGaussVal(Norm2(aPixOut-aCenterOut)/aSzK);
         if (aW>0)
         {
            aRes.mVPts.push_back(aPixIn);
            aRes.mVWeight.push_back(aW);
            aSomW += aW;
         }
     }

     // if not empty  , som W = 1
     if (aSomW>0)
     {
        for (auto & aW : aRes.mVWeight)
        {
            aW /= aSomW;
        }
     }

     return aRes;
}

};

