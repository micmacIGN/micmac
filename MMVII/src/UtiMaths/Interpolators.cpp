#include "MMVII_Interpolators.h"

namespace MMVII
{

class cInterpolator1D ;
class cDiffInterpolator1D ;
class cLinearInterpolator ;
class cCubicInterpolator ;
class cSinCApodInterpolator ;

class cTabulatedInterpolator ;
class cTabulatedDiffInterpolator ;
class cEpsDiffFctr ;



class cInterpolator1D : public cMemCheck
{
      public :
        cInterpolator1D(const tREAL8 & aSzKernel);
        virtual ~cInterpolator1D();

        virtual tREAL8  Weight(tREAL8  anX) const = 0;
	const tREAL8 SzKernel() const;  // accessor
      protected :
	tREAL8 mSzKernel;
};

class cDiffInterpolator1D : public cInterpolator1D
{
       public :
            cDiffInterpolator1D(tREAL8 aSzK);
            virtual tREAL8  DiffWeight(tREAL8  anX) const =0;
	    /// Sometime its more optimized to compute both value simultaneously, default calls F&DF
            virtual std::pair<tREAL8,tREAL8>  WAndDiff(tREAL8  anX) const ;
};


/** Linear interpolator, not very usefull as  function GetVBL do the jobs
 * inline and faster, but used in unitary test
 *
 */

class cLinearInterpolator : public cInterpolator1D
{
      public :
        cLinearInterpolator();
        tREAL8  Weight(tREAL8  anX) const override ;
};

/** Cubic  interpolator, we make it differentiable essentially for 
 * unitary test, because pratically the tabulated is probably more efficient
 *
 */

class cCubicInterpolator : public cDiffInterpolator1D
{
	public :
            // mA = valeur de la derivee en  ?  ( 1?)
            // si vaut -0.5, reconstitue parfaitement une droite
            // doit etre comprise en 0 et -3
             cCubicInterpolator(tREAL8 aParam);
             tREAL8  Weight(tREAL8  anX) const override ;
             tREAL8  DiffWeight(tREAL8  anX) const override ;
             std::pair<tREAL8,tREAL8>   WAndDiff(tREAL8  anX) const override;
       private :
	     tREAL8 mA;

};

class cSinCApodInterpolator : public cInterpolator1D
{
       public :
            cSinCApodInterpolator(tREAL8 aSzSinC,tREAL8 aSzAppod);
            tREAL8  Weight(tREAL8  anX) const override ;
       public :
            tREAL8 mSzSinC;
            tREAL8 mSzAppod;
};

cSinCApodInterpolator::cSinCApodInterpolator(tREAL8 aSzSinC,tREAL8 aSzAppod) :
    cInterpolator1D  (aSzSinC+aSzAppod),
    mSzSinC          (aSzSinC),
    mSzAppod         (aSzAppod)
{
}

tREAL8  cSinCApodInterpolator::Weight(tREAL8  anX) const
{
    anX = std::abs(anX);  // classicaly even function
    if (anX> mSzKernel)  // classicaly 0 out of kernel
       return 0.0;

    tREAL8 aSinC = sinC(anX*M_PI); // pure sinus cardinal

    if (anX<mSzSinC)  // before apodisation window  , no apodisation
       return aSinC;

    // Apod coef : 1 in mSzSinc , 0 in SzKernel
    anX = (mSzKernel - anX) / mSzAppod;
    //  use CubAppGaussVal to make apodisation a differentiable function
    // return CubAppGaussVal(anX) * aSinC;
    // strangely seems better interpol with trapeze window ??
    return anX * aSinC;
}





class cTabulatedInterpolator : public cInterpolator1D
{
      public :
          friend class cTabulatedDiffInterpolator;

          cTabulatedInterpolator(const cInterpolator1D &,int aNbTabul,bool IsBilin,bool DoNorm=true);
          tREAL8  Weight(tREAL8  anX) const override ;
      private :
	  void SetDiff(const cTabulatedInterpolator & anInt);
	  void DoNormalize(bool ForDeriv);
	  bool               mIsBilin;
	  int                mNbTabul;
	  int                mSzTot;
          cIm1D<double>      mIm;
          cDataIm1D<double>* mDIm;

};


class cEpsDiffFctr : public cDiffInterpolator1D
{
      public :
          cEpsDiffFctr(const cInterpolator1D & anInt,tREAL8 aEps) ;
          tREAL8  Weight(tREAL8  anX) const override ;
          tREAL8  DiffWeight(tREAL8  anX) const override;
      private :
	   const cInterpolator1D & mInt;
	   tREAL8    mEps;
};

class cTabulatedDiffInterpolator : public cDiffInterpolator1D
{
      public :
          cTabulatedDiffInterpolator(const cInterpolator1D &,int aNbTabul);

          tREAL8  Weight(tREAL8  anX) const override ;
          tREAL8  DiffWeight(tREAL8  anX) const override;
          std::pair<tREAL8,tREAL8>   WAndDiff(tREAL8  anX) const override;
      private :
	  cTabulatedInterpolator  mTabW;
	  cTabulatedInterpolator  mTabDifW;
	  int                     mNbTabul;
	  int                     mSzTot;
          const tREAL8 *          mRawW; 
          const tREAL8 *          mRawDifW; 

};



/* *************************************************** */
/*                                                     */
/*           cEpsDiffFctr                              */
/*                                                     */
/* *************************************************** */

cEpsDiffFctr::cEpsDiffFctr(const cInterpolator1D & anInt,tREAL8 aEps) :
    cDiffInterpolator1D (anInt.SzKernel()),
    mInt (anInt),
    mEps (aEps)
{
}


tREAL8  cEpsDiffFctr::Weight(tREAL8  anX) const  {return mInt.Weight(anX);}
tREAL8  cEpsDiffFctr::DiffWeight(tREAL8  anX) const  {return (mInt.Weight(anX+mEps)-mInt.Weight(anX-mEps)) / (2*mEps) ;}



/* *************************************************** */
/*                                                     */
/*           cDiffInterpolator1D                       */
/*                                                     */
/* *************************************************** */

cDiffInterpolator1D::cDiffInterpolator1D(tREAL8 aSzK) :
      cInterpolator1D (aSzK)
{
}

std::pair<tREAL8,tREAL8>   cDiffInterpolator1D::WAndDiff(tREAL8  anX) const
{
    return std::pair<tREAL8,tREAL8> (Weight(anX),DiffWeight(anX));
}


/* *************************************************** */
/*                                                     */
/*           cCubicInterpolator                         */
/*                                                     */
/* *************************************************** */

cCubicInterpolator::cCubicInterpolator(tREAL8 aParam) :
   cDiffInterpolator1D((aParam==0.0) ? 1.0 : 2.0),  // when A=0, the kernel is [-1,1] 
   mA (aParam)
{
}

tREAL8  cCubicInterpolator::Weight(tREAL8  x) const
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

tREAL8  cCubicInterpolator::DiffWeight(tREAL8  x) const
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

std::pair<tREAL8,tREAL8>   cCubicInterpolator::WAndDiff(tREAL8  x) const
{
     int aS = (x>=0) ? 1 : -1;
     x = std::abs(x);
     tREAL8 x2 = x * x;
     tREAL8 x3 = x2 * x;

     if (x <=1.0)
     {
        tREAL8 aAP2 = mA+2;
        tREAL8 aAP3 = mA+3;
        return std::pair<tREAL8,tREAL8>(   aAP2*x3-aAP3*x2+1,   aS*(3*aAP2*x2-2*aAP3*x));
     }
     if (x <=2.0)
        return std::pair(mA*(x3 - 5 * x2 + 8* x -4), aS*mA*(3*x2 - 10 * x + 8));
     return std::pair<tREAL8,tREAL8> (0.0,0);
}

/* *************************************************** */
/*                                                     */
/*           cTabulatedInterpolator                    */
/*                                                     */
/* *************************************************** */

cTabulatedInterpolator::cTabulatedInterpolator(const cInterpolator1D &anInt,int aNbTabul,bool IsBilin,bool DoNorm) :
     cInterpolator1D  (anInt.SzKernel()),
     mIsBilin         (IsBilin),
     mNbTabul         (aNbTabul),
     mSzTot           (round_up(anInt.SzKernel()*mNbTabul)),
     mIm              (mSzTot+1),
     mDIm             (&mIm.DIm())
{
      // [0]  initialisation of weight
      for (int aK=0 ; aK<mDIm->Sz() ; aK++)
          mDIm->SetV(aK,anInt.Weight(aK/tREAL8(mNbTabul)));
      mDIm->SetV(mSzTot,0.0);

      if (DoNorm) 
          DoNormalize(false);
}

void cTabulatedInterpolator::SetDiff(const cTabulatedInterpolator & anInt)
{
     mDIm->SetV(0,0.0);
     mDIm->SetV(mSzTot,0.0);

     tREAL8 a2Eps = 2.0/mNbTabul;
     for (int aK=1 ; aK<mSzTot ; aK++)
         mDIm->SetV(aK,(anInt.mDIm->GetV(aK+1)-anInt.mDIm->GetV(aK-1)) / a2Eps);
}

void cTabulatedInterpolator::DoNormalize(bool ForDerive)
{
      tREAL8 aSomWDif = 0; // observation of inital deviation
      for (int aK=0 ; aK< mNbTabul ; aK++)
      {
	  // [1] retract to first (negative) value of same phase
	  int aK1 = aK;
	  while ((aK1-mNbTabul) >= -mSzTot) 
                aK1-= mNbTabul;

	  // [2]  compute the sum of all value same phase
          tREAL8 aSumV=0.0;
	  int aNb=0;
          for (; aK1<=mSzTot  ; aK1+=mNbTabul)
	  {
               tREAL8 aV = mDIm->GetV(std::abs(aK1)); 
	       if (ForDerive) aV *= SignSupEq0(aK1);
               aSumV += aV;
	       aNb++;
	  }

	  if (ForDerive)   // if not Sum1, then Sum0 -> this is the avg we substract
             aSumV /= aNb;

          for (aK1=aK; aK1<=mSzTot  ; aK1+=mNbTabul)
	  {
	       if (ForDerive)
                  mDIm->SetV(aK1,mDIm->GetV(aK1)-aSumV);
	       else
                  mDIm->SetV(aK1,mDIm->GetV(aK1)/aSumV);
	  }

	  aSomWDif +=  ForDerive ?  std::abs(aSumV) : std::abs(aSumV-1) ;
      }

      if (0)
      {
          aSomWDif /= mNbTabul;
          StdOut() << "SSS= " << aSomWDif  << " NbT=" << mNbTabul << "\n";
          getchar();
      }
}


tREAL8  cTabulatedInterpolator::Weight(tREAL8  anX) const 
{
   tREAL8 aRK = std::abs(anX) * mNbTabul;
   if (aRK>= mSzTot) 
      return 0.0;

   if (mIsBilin)
      return mDIm->GetVBL(aRK);
   else
      return mDIm->GetV(round_ni(aRK));
}

/* *************************************************** */
/*                                                     */
/*           cTabulatedDiffInterpolator                */
/*                                                     */
/* *************************************************** */

cTabulatedDiffInterpolator::cTabulatedDiffInterpolator(const cInterpolator1D &anInt,int aNbTabul) :
	cDiffInterpolator1D (anInt.SzKernel()),
	mTabW     (anInt,aNbTabul,true,true),
	mTabDifW  (anInt,aNbTabul,true,false), // we dont normalize, btw coeff are not up to date
        mNbTabul  (mTabW.mNbTabul),
	mSzTot    (mTabW.mSzTot),
	mRawW     (mTabW.mDIm->RawDataLin()),
	mRawDifW  (mTabDifW.mDIm->RawDataLin())
{
	mTabDifW.SetDiff(mTabW);   // put in DifW the difference of W
	mTabDifW.DoNormalize(true); // normalize by sum 0
}

tREAL8  cTabulatedDiffInterpolator::Weight(tREAL8  anX) const {return mTabW.Weight(anX);}
tREAL8  cTabulatedDiffInterpolator::DiffWeight(tREAL8  anX) const 
{
	return SignSupEq0(anX) * mTabDifW.Weight(anX);
}

std::pair<tREAL8,tREAL8>   cTabulatedDiffInterpolator::WAndDiff(tREAL8  anX) const 
{
   tREAL8 aRK = std::abs(anX) * mNbTabul;
   if (aRK>= mSzTot) 
      return std::pair<tREAL8,tREAL8>(0,0);

   int aIk = round_down(aRK);
   tREAL8 aWeight1 = aRK-aIk;
   tREAL8 aWeight0 = 1-aWeight1;
   
   const tREAL8  * aDataW    = mRawW+aIk;
   const tREAL8  * aDataDifW = mRawDifW+aIk;

   return std::pair<tREAL8,tREAL8>
	 ( 
	      aWeight0*aDataW[0] + aWeight1*aDataW[1] ,  
	      (aWeight0*aDataDifW[0] + aWeight1*aDataDifW[1]) * SignSupEq0(anX)
	 );
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
/*           cLinearInterpolator                      */
/*                                                     */
/* *************************************************** */

cLinearInterpolator::cLinearInterpolator() :
       cInterpolator1D (1.0)
{
}

tREAL8  cLinearInterpolator::Weight(tREAL8  anX) const 
{
      return std::max(0.0,1.0-std::abs(anX));
}

/* *************************************************** */
/*                                                     */
/*           IM2D/IM2D                                 */
/*                                                     */
/* *************************************************** */

template <> inline  bool cPixBox<2>::InsideInterpolator(const cInterpolator1D & anInterpol,const cPtxd<double,2> & aP,tREAL8 aMargin) const
{
    tREAL8 aSzK = anInterpol.SzKernel() + aMargin;
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
        tREAL8 aWX = anInterpol.Weight(aIntX-aRealX);
        TheBufCoeffX.push_back(aWX);
	aSomWX += aWX;
    }
    const tREAL8 *  aLineW0  = TheBufCoeffX.data();

    tREAL8 aSomWIxy = 0.0;
    tREAL8 aSomWY=0;
    for (int aIntY=aY0 ; aIntY<=aY1 ; aIntY++)
    {
        tREAL8 aWY = anInterpol.Weight(aIntY-aRealY);
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

/*
 *     d(Im @ Int) / dx = Im @ d(Int)/dx
 */

static std::vector<tREAL8>  TheBufCoeffDerX;  // store the value on 1 line, as they are separable
					   
template <class Type>  cPt3dr cDataIm2D<Type>::GetValueAndDerInterpol(const cPt2dr & aP,const cDiffInterpolator1D & anInterpol) const 
{
    TheBufCoeffX.clear();
    TheBufCoeffDerX.clear();
    tREAL8 aSzK = anInterpol.SzKernel();

    tREAL8 aRealY = aP.y();
    int aY0 = round_Uup(aRealY-aSzK);  
    int aY1 = round_Ddown(aRealY+aSzK);

    tREAL8 aRealX = aP.x();
    int aX0 = round_Uup(aRealX-aSzK); 
    int aX1 = round_Ddown(aRealX+aSzK);
    int aNbX =  aX1-aX0+1;

    for (int aIntX=aX0 ; aIntX<=aX1 ; aIntX++)
    {
        auto [aWX,aDerX]  = anInterpol.WAndDiff(aIntX-aRealX);
        TheBufCoeffX.push_back(aWX);
        TheBufCoeffDerX.push_back(aDerX);
    }
    /*
FakeUseIt(aNbX);
FakeUseIt(aY0);
FakeUseIt(aY1);
*/

    const tREAL8 *  aLineW0  = TheBufCoeffX.data();
    const tREAL8 *  aDerLineW0  = TheBufCoeffDerX.data();

    tREAL8 aSomWxyI = 0.0;
    tREAL8 aSomWI_Dx = 0.0;
    tREAL8 aSomWI_Dy = 0.0;

    for (int aIntY=aY0 ; aIntY<=aY1 ; aIntY++)
    {
	const Type *  aLineIm = mRawData2D[aIntY] + aX0;

	tREAL8 aSomWIx = 0.0;
	tREAL8 aSomDerWIx = 0.0;
	for (int aKX=0 ; aKX< aNbX ; aKX++)
	{
            aSomWIx    += aLineIm[aKX]  *  aLineW0 [aKX];
            aSomDerWIx += aLineIm[aKX]  *  aDerLineW0[aKX];
	}

        auto [aWY,aDerY] = anInterpol.WAndDiff(aIntY-aRealY);
	aSomWxyI  +=  aSomWIx    * aWY;
	aSomWI_Dx +=  aSomDerWIx * aWY;
	aSomWI_Dy +=  aSomWIx    * aDerY;
    }

    return cPt3dr(-aSomWI_Dx,-aSomWI_Dy,aSomWxyI);
}


template class cDataIm2D<tU_INT1>;
template class cDataIm2D<tREAL4>;


/**  Basic test on bilinear interpolator, compare "hand crafted" GetVBL with :
 *     - cLinearInterpolator
 *     - Tabulated-raw
 *     - Tabulated-bilin
 *     
 */
template <class Type>  void  TplBenchInterpol_1(const cPt2di & aSz)
{
     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();

     cLinearInterpolator aBil1;
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


/*  Test GetValueInterpol with a not so basic interolator : cubic
 *  we use Cubic(0.5) as for this value the interpolation of a linear
 *  function is exact, so we have a ground truth
 *
 *  The test is made with analytic and tabulated interpolator
 *
 */
template <class Type>  void  TplBenchInterpol_2(const cPt2di & aSz)
{
	// ----- [0]  parameters for linear function "Ax+By+C"
     tREAL8 A = -2;
     tREAL8 B = 3;
     tREAL8 C = -123;

	// -----  [1]   Generate the linear function  in the image
     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();

     for (const auto & aP : aDIm)
     {
         aDIm.SetV(aP,A*aP.x()+B*aP.y() + C);
     }

	// -----  [2]   Generate the linear function  in the image
     // bicub that interpolate exactly linear function
     cCubicInterpolator aI3(-0.5);
     //  tabulated interpolator
     cTabulatedInterpolator aTabI3(aI3,1000,true);


	// -----  [3]   Test itsefl
     int aNbTest = 0;
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));
     while (aNbTest<5000)
     {
         cPt2dr  aPt = aBoxR.GeneratePointInside();
         if (aDIm.InsideInterpolator(aI3,aPt))
         {
             aNbTest++;

	     tREAL8 aV0 = A*aPt.x()+B*aPt.y() + C;   // Analytical value
	     tREAL8 aV1 = aDIm.GetValueInterpol(aPt,aI3);  // interpolation with analytical interpolator
	     tREAL8 aV2 = aDIm.GetValueInterpol(aPt,aTabI3);  // interpolation with tabulated interpolator

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV0-aV1)<1e-6,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aV0-aV2)<1e-6,"Interpol ");
         }
     }
}


template <class Type>  void  TplBenchInterpol_3(const cPt2di & aSz,tREAL8 aCoeff)
{
     tREAL8 aPerX = 10.0;
     tREAL8 aPerY = 15.0;

     cIm2D<Type> anIm(aSz,nullptr,eModeInitImage::eMIA_RandCenter);
     cDataIm2D<Type> & aDIm = anIm.DIm();
     cBox2dr  aBoxR(cPt2dr(0,0),ToR(aSz));

     for (const auto & aP : aDIm)
     {
         tREAL8 aVal = std::sin(aP.x()/aPerX) *  std::cos(aP.y()/aPerY) ;
	 aDIm.SetV(aP,aVal);
     }

     cCubicInterpolator aI3(aCoeff);

     cTabulatedDiffInterpolator aTabI3(aI3,1000);

     cSinCApodInterpolator  aSinCInt(5.0,5.0);
     cTabulatedDiffInterpolator aTabSinC(aSinCInt,1000);

     int aNbTest = 0;
     tREAL8 aSumDifSinC = 0;
     while (aNbTest<5000)
     {
         tREAL8 aEpsDif = 1e-4;
         cPt2dr  aPt = aBoxR.GeneratePointInside();
         if (aDIm.InsideInterpolator(aSinCInt,aPt,aEpsDif*1.01))
         {
             aNbTest++;

             cPt3dr aVAndD = aDIm.GetValueAndDerInterpol(aPt,aI3);
	     tREAL8 aV0 = aDIm.GetValueInterpol(aPt,aI3);

	     cPt2dr aEpsX(aEpsDif,0);
	     tREAL8 aVDx = (aDIm.GetValueInterpol(aPt+aEpsX,aI3)-aDIm.GetValueInterpol(aPt-aEpsX,aI3)) / (2*aEpsDif);

	     cPt2dr aEpsY(0,aEpsDif);
	     tREAL8 aVDy = (aDIm.GetValueInterpol(aPt+aEpsY,aI3)-aDIm.GetValueInterpol(aPt-aEpsY,aI3)) / (2*aEpsDif);


	     MMVII_INTERNAL_ASSERT_bench(std::abs(aVAndD.z()  - aV0)<1e-6,"Interpol VAndDer");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aVAndD.x() - aVDx)<1e-5,"Interpol VAndDer X");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aVAndD.y() - aVDy)<1e-5,"Interpol VAndDer Y");

             cPt3dr aTI3VD = aDIm.GetValueAndDerInterpol(aPt,aTabI3);
	     // StdOut () << aVAndD - aTI3VD << aPt << aSz << "\n";
	     cPt3dr aDif = aVAndD - aTI3VD;
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aDif.x() )<1e-4,"Interpol VAndDer Tabulated");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aDif.y() )<1e-4,"Interpol VAndDer Tabulated");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aDif.z() )<1e-4,"Interpol VAndDer Tabulated");

             cPt3dr aTSinC = aDIm.GetValueAndDerInterpol(aPt,aTabSinC);
             tREAL8 aVTh = std::sin(aPt.x()/aPerX) *  std::cos(aPt.y()/aPerY) ;
             // aTSinC = aDIm.GetValueAndDerInterpol(aPt,aSinCInt);
	     aSumDifSinC  += std::abs(aTSinC.z()-aVTh);
	     if (0)
	     {
                  tREAL8 aVThX = std::cos(aPt.x()/aPerX) *  std::cos(aPt.y()/aPerY)  / aPerX;
                  tREAL8 aVThY = std::sin(aPt.x()/aPerX) *  -std::sin(aPt.y()/aPerY)  / aPerY;

	          StdOut() << "Dv=" << aVAndD 
			  << " Difd" <<  aVAndD.z()- aVTh  << " " << aVAndD.x()- aVThX << " " << aVAndD.y()- aVThY
			  << " DifSinC" <<  aTSinC.z()- aVTh  << " " << aTSinC.x()- aVThX << " " << aTSinC.y()- aVThY
		          << "\n";
	     }
	 }
     }
     if (0)
        StdOut() << "AVG DIF SINC =" << aSumDifSinC / aNbTest << "\n";
}



void  BenchInterpol(cParamExeBench & aParam)
{
     if (! aParam.NewBench("Interpol")) return;


     //  Bench on bicub interpola
     //
     for (const auto & aP : {0.0,1.0,2.0,3.0})
     {
         cCubicInterpolator  anI3(aP);

	 MMVII_INTERNAL_ASSERT_bench(std::abs(1-anI3.Weight(0))<1e-8,"Interpol ");   // F(0)=1
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(1))<1e-8,"Interpol ");     // F(1)=0
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(2))<1e-8,"Interpol ");     // F(2)=0
	 MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(2.1))<1e-8,"Interpol ");    // F(X>2) = 0
										 //
	 for (const auto aEps : {1e-2,1e-3,1e-4})
	 {
             // Derivate in 2 = 0
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(2-aEps))<4*Square(aEps) ,"Interpol ");
	     // diffrentuale in 1  dF/dx+ = dF/dx-
	     MMVII_INTERNAL_ASSERT_bench(std::abs(anI3.Weight(1-aEps) + anI3.Weight(1+aEps))<4*Square(aEps) ,"Interpol ");
	 }
	 // compare  "analytical derivatives" with finite diff
         cEpsDiffFctr aEpsI3(anI3,1e-5);

	 cTabulatedDiffInterpolator aTabDif(anI3,10000);

	 for (int aK=0 ; aK<1000 ; aK++)
	 {
             tREAL8 anX = RandInInterval(-3,3);

	     tREAL8 aD1 = anI3.DiffWeight(anX);
	     tREAL8 aF1 = anI3.Weight(anX);

	     tREAL8 aD2 = aEpsI3.DiffWeight(anX);
	     tREAL8 aF2 = aEpsI3.Weight(anX);

	     auto [aF3,aD3] = anI3.WAndDiff(anX);

	     // StdOut() << "DDDD X=" << anX << " "  << aD1 << " DifF=" << aD3 << " Dif=" << aD1 -aD3 << "\n";
	     // StdOut() << "FFFF X=" << anX << " "  << aF1 << " DifF=" << aF3 << " Dif=" << aF1 -aF3 << "\n\n";

	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aD2)<1e-4 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF1-aF2)<1e-8 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD1-aD3)<1e-4 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF1-aF3)<1e-8 ,"Interpol ");


	     tREAL8 aD4 = aTabDif.DiffWeight(anX);
	     tREAL8 aF4 = aTabDif.Weight(anX);
	     auto [aF5,aD5] = aTabDif.WAndDiff(anX);

	     //  these one should be very clos if not ==0, because they are same bilin formula
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF4-aF5)<1e-8 ,"Interpol TabDif ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD4-aD5)<1e-8 ,"Interpol  TabDif");

	     // StdOut() << "D455  " <<  aD3-aD5  << " " << aF3 - aF5 << "\n";
	     // Not so close : analyticall formula & bilin tab
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aD3-aD5)<1e-3 ,"Interpol ");
	     MMVII_INTERNAL_ASSERT_bench(std::abs(aF3-aF5)<1e-3 ,"Interpol ");
	 }
     }

     TplBenchInterpol_1<tU_INT1>(cPt2di(230,127));
     TplBenchInterpol_1<tREAL4>(cPt2di(144,201));


     TplBenchInterpol_2<tINT2>(cPt2di(199,147));
     TplBenchInterpol_2<tREAL8>(cPt2di(100,188));

     TplBenchInterpol_3<tREAL8>(cPt2di(300,400),-0.5);

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
   // cubic formula , return (mA+2) * x3-(mA+3)*x2+1;   for A=0 -> 2x3-3x2+1  . 
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

