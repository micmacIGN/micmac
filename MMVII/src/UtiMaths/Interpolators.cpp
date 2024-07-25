#include "MMVII_Interpolators.h"

namespace MMVII
{

/* *************************************************** */
/*                                                     */
/*           cInterpolator1D                           */
/*                                                     */
/* *************************************************** */

cInterpolator1D::cInterpolator1D(const tREAL8 & aSzKernel,const std::vector<std::string> & aVNames) :
	mSzKernel (aSzKernel),
	mVNames   (aVNames)
{
}

cInterpolator1D::~cInterpolator1D()
{
}
const tREAL8 & cInterpolator1D::SzKernel() const {return mSzKernel;}
const std::vector<std::string> & cInterpolator1D::VNames() const {return mVNames;}

cInterpolator1D *  cInterpolator1D::TabulatedInterp(const cInterpolator1D & anInt,int aNbTabul,bool BilinInterp)
{
	return new cTabulatedInterpolator(anInt,aNbTabul,BilinInterp);
}

void  cInterpolator1D::SetSzKernel(tREAL8  aSzK) 
{
      mSzKernel = aSzK;
}


/* *************************************************** */
/*                                                     */
/*           cDiffInterpolator1D                       */
/*                                                     */
/* *************************************************** */

cDiffInterpolator1D::cDiffInterpolator1D(tREAL8 aSzK,const std::vector<std::string> & aVNames) :
      cInterpolator1D (aSzK,aVNames)
{
}

std::pair<tREAL8,tREAL8>   cDiffInterpolator1D::WAndDiff(tREAL8  anX) const
{
    // default value, simply calls 2 elementary methods
    return std::pair<tREAL8,tREAL8> (Weight(anX),DiffWeight(anX));
}

const std::string & cDiffInterpolator1D::Get(const std::vector<std::string> & aVName,size_t aK0)
{
    if (aK0>=aVName.size())
    {
        StdOut() << "while parsing " << aVName << "\n";
	MMVII_UnclasseUsEr("Coulnd extract elem " + ToStr(aK0) +" while parsing interpolator vect-string");
    }
    return aVName.at(aK0);
}

void cDiffInterpolator1D::AssertEndParse(const std::vector<std::string> & aVName,size_t aK0)
{
    if ((aK0+1) != aVName.size())
	MMVII_UnclasseUsEr("Too many string in parsing interpolator vect-string");
}


cDiffInterpolator1D * cDiffInterpolator1D::AllocFromNames(const std::vector<std::string> & aVName)
{
	return AllocFromNames(aVName,0);
}

cDiffInterpolator1D * cDiffInterpolator1D::AllocFromNames(const std::vector<std::string> & aVName,size_t aK)
{
     const std::string & aN0 = Get(aVName,aK);
     if (aN0== cTabulatedDiffInterpolator::TheNameInterpol)
     {
	 int aNbTabul =  cStrIO<int>::FromStr(Get(aVName,aK+1));
	 cDiffInterpolator1D * anInt = AllocFromNames(aVName,aK+2);
	 cTabulatedDiffInterpolator * aRes = new cTabulatedDiffInterpolator(*anInt,aNbTabul);
	 delete anInt;
	 return aRes;
     }
     if (aN0== cLinearInterpolator::TheNameInterpol)
     {
         AssertEndParse(aVName,aK);
         return new cLinearInterpolator;
     }
     if (aN0== cMMVII2Inperpol::TheNameInterpol)
     {
         AssertEndParse(aVName,aK);
         return new cMMVII2Inperpol;
     }

     if (aN0== cCubicInterpolator::TheNameInterpol)
     {
	 tREAL8 aParam =  cStrIO<tREAL8>::FromStr(Get(aVName,aK+1));
         AssertEndParse(aVName,aK+1);
         return new cCubicInterpolator(aParam);
     }
     if (aN0== cMMVIIKInterpol::TheNameInterpol)
     {
	 tREAL8 aParam =  cStrIO<tREAL8>::FromStr(Get(aVName,aK+1));
         AssertEndParse(aVName,aK+1);
         return new cMMVIIKInterpol(aParam);
     }
     if (aN0== cSinCApodInterpolator::TheNameInterpol)
     {
	 tREAL8 aSzK    =  cStrIO<tREAL8>::FromStr(Get(aVName,aK+1));
	 tREAL8 aSzApod =  cStrIO<tREAL8>::FromStr(Get(aVName,aK+2));
         AssertEndParse(aVName,aK+2);
	 return new cSinCApodInterpolator(aSzK,aSzApod);
     }


     MMVII_UnclasseUsEr("Cannot interpret [" + aN0 + "] while  parsing interpolator vect-string");
     return nullptr;
}


cDiffInterpolator1D *  cDiffInterpolator1D::TabulatedInterp(const cInterpolator1D & anInt,int aNbTabul)
{
    return new cTabulatedDiffInterpolator(anInt,aNbTabul);
}

cDiffInterpolator1D *  cDiffInterpolator1D::TabulatedInterp(cInterpolator1D * anInt,int aNbTabul)
{
    return new cTabulatedDiffInterpolator(anInt,aNbTabul);
}




/* *************************************************** */
/*                                                     */
/*           cLinearInterpolator                      */
/*                                                     */
/* *************************************************** */

cLinearInterpolator::cLinearInterpolator() :
       cDiffInterpolator1D (1.0,{TheNameInterpol}) // kernel defined on [-1,1]
{
}

tREAL8  cLinearInterpolator::Weight(tREAL8  anX) const 
{
      return std::max(0.0,1.0-std::abs(anX)); // classical formula
}

const std::string cLinearInterpolator::TheNameInterpol="Linear";

tREAL8  cLinearInterpolator::DiffWeight(tREAL8  anX) const
{
     if (anX<-1)  return 0.0;
     if (anX==-1) return 0.5;
     if (anX<0)   return 1.0;
     if (anX==0)  return 0.0;
     if (anX<1)   return -1.0;
     if (anX==1)  return -0.5;
     return 0.0;
}

/* *************************************************** */
/*                                                     */
/*           cCubicInterpolator                         */
/*                                                     */
/* *************************************************** */

cCubicInterpolator::cCubicInterpolator(tREAL8 aParam) :
   cDiffInterpolator1D((aParam==0.0) ? 1.0 : 2.0,{TheNameInterpol,ToStr(aParam)}),  // when A=0, the kernel is [-1,1] , else [-2,2]
   mA (aParam)
{
}

const std::string cCubicInterpolator::TheNameInterpol="Cubic";

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

/*  Optimized version, some computation are shared between value and derivates,
 *  rather a minor gain here, it's more to illustrate the way it works
 */

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
/*           cSinCApodInterpolator                     */
/*                                                     */
/* *************************************************** */

cSinCApodInterpolator::cSinCApodInterpolator(tREAL8 aSzSinC,tREAL8 aSzAppod) :
    cDiffInterpolator1D  (aSzSinC+aSzAppod,{TheNameInterpol,ToStr(aSzSinC),ToStr(aSzAppod)}),
    mSzSinC              (aSzSinC),
    mSzAppod             (aSzAppod)
{
}

tREAL8  cSinCApodInterpolator::DiffWeight(tREAL8  anX) const 
{
    MMVII_INTERNAL_ERROR("No DiffWeight for cSinCApodInterpolator");
    return 0;
}

tREAL8  cSinCApodInterpolator::Weight(tREAL8  anX) const
{
    anX = std::abs(anX);  // classicaly even function
    if (anX> mSzKernel)  // classicaly 0 out of kernel
       return 0.0;

    tREAL8 aSinC = sinC(anX*M_PI); // pure sinus cardinal

    if (anX<mSzSinC)  // before apodisation window  : no apodisation is done
       return aSinC;

    // Apod coeff compute so tha value =  1 in mSzSinc , 0 in SzKernel
    anX = (mSzKernel - anX) / mSzAppod;
    return anX * aSinC;

    //  Tentative to   use CubAppGaussVal to make apodisation a differentiable function
    // return CubAppGaussVal(anX) * aSinC;
    // strangely seems better interpol with trapeze window ?? to investigate later ??
}


const std::string cSinCApodInterpolator::TheNameInterpol = "SinCApod";

/* *************************************************** */
/*                                                     */
/*           cMMVII2Inperpol                           */
/*                                                     */
/* *************************************************** */

/* The "MPD" interpolator comes from a compromize for trying to limit
 * the defautlt of current interpolator :
 *
 *     - linear is biased because for the smoothin effect varies with phase "Ph" of interpolation:*
 *         - if "Ph"=0.5  the weight is [0.5,0.5] having a smoothing effect (low frequence filter)
 *         - whlie if "Ph=0" the weight is [0,1,0], we take the exact value
 *
 *     - sinus card is  potentially  slow if taken with too big kernel
 *     
 *     - other like bicubic  are some in between, but not completely satisfafying vs bias
 *
 *  The specification of MPD is :
 *    
 *      - like linear take as few pixel as possible,
 *      - when "Ph"=0.5 , you cannot do much better than weigth [0.5,0.5], if you want to limit the size and
 *        process  equivalently the two neighboors
 *      - when "Ph in [-0.5,0.5]" :
 *
 *            * we limit to 3 pixel {-1,0,1}
 *            * we compute the weight in such way that the frequence filtering effect is the same than with
 *             weigth [0.5,0.5];  this effect is defined as the variance of the distribution
 *
 *   Then let "{a,b,c}" be the weight of {-1,0,1}, and "x=Ph" we have 3 equation :
 *
 *   (1)    "a+b+c =1"   => they are weigthing
 *   (2)    "-a+c  =x"   =>  barycenter must be equal to phase
 *   (3)    "a(1+x)^2 + b x^2 + c(1-x) = 1/4 " => have the same variance than for "[0.5,0.5]"
 *
 *   Solving this equation, we use "b=1-a-c" in (3) :
 *
 *          "a(1+2x) + c (1-2x) = 1/4-x^2"    then using (2) to substitute "c" by "x+a"  we get
 *          "a=1/2(x-1/2)^2"                  by symetry x->1-x, we get for c
 *          "c=1/2(x+1/2)^2"                  and finaly for b we get
 *          "b=1/2(3/2-2x^2)"
 *
 *  Some check, for x=0.5  a,b,c={0,0.5,0.5}  , for x=-0.5 a,b,c={0.5,0.5,0} , for x=0 a,b,c={1/8,3/4,1/8}
 *
 *  Regarding the kernel we have :
 *
 *   For |x|<=0.5  (then F(x)= b with previous) we have :
 *    "F=F-(X) = 1/2 (3/2-2x^2)"  
 *   For 0.5<= |x|<=1.5  ( for ex if x>0.5,  F(x) = a with Ph=x-1
 *    "F=F+(X) = 1/2(x-3/2)^2"
 *   Else F=0
 *
 *  We can check that F is continous and differentianle
 *
 *    F-(1/2) = 1/2  = F+(1/2)
 *    dF-/dx = -2x  dF-/dx(1/2) = -1
 *    dF+/dx = x-3/2 dF+/dx(1/2)= -1
 *    dF+/dx(3/2) = 0
 */

cMMVII2Inperpol::cMMVII2Inperpol():
    cDiffInterpolator1D (1.5,{TheNameInterpol})
{
}

const std::string cMMVII2Inperpol::TheNameInterpol = "MMVII";

tREAL8  cMMVII2Inperpol::Weight(tREAL8  anX) const 
{
    anX = std::abs(anX) ;
    if (anX<=0.5)  return 0.5 *(1.5-2*Square(anX));
    if (anX<=1.5)  return 0.5 * Square(anX-1.5) ;

    return 0.0;
}

tREAL8  cMMVII2Inperpol::DiffWeight(tREAL8  anX) const
{
    int aS = SignSupEq0(anX);
    tREAL8 aXAbs = aS * anX;

    if (aXAbs<=0.5)  
       return -2*anX;

    if (aXAbs<=1.5)  
       return aS * (aXAbs-1.5);
    return 0.0;
}

/* *************************************************** */
/*                                                     */
/*           cMMVIIKInterpol                             */
/*                                                     */
/* *************************************************** */

cMMVIIKInterpol::cMMVIIKInterpol(tREAL8 anExp) :
    cDiffInterpolator1D(1.5,{TheNameInterpol,ToStr(anExp)}),
    mExp  (anExp)
{
}

const std::string cMMVIIKInterpol::TheNameInterpol = "MMVIIK";

tREAL8  cMMVIIKInterpol::DiffWeight(tREAL8  anX) const 
{
    MMVII_INTERNAL_ERROR("No DiffWeight for cMMVIIKInterpol");
    return 0;
}

tREAL8  cMMVIIKInterpol::Weight(tREAL8  anX) const
{
    anX = std::abs(anX);
    if (anX>= mSzKernel) return 0.0;

    tREAL8 aPhX = anX;
    if ((anX>0.5) && (anX<=1.0))
       aPhX = 1.0 - anX;
    if ((anX>1.0))
       aPhX =  anX -1.0;

    cPt3dr aL0(1,1,1);  //  A+B+C = 1
    cPt3dr aL1(-1,0,1);  // -A+C = 1
    cPt3dr aL2(std::pow(1+aPhX,mExp),std::pow(aPhX,mExp),std::pow(1-aPhX,mExp));
			

    cDenseMatrix aM = M3x3FromLines(aL0,aL1,aL2);

    cPt3dr aCol (1,aPhX,std::pow(0.5,mExp));

    cPt3dr aABC = SolveCol(aM,aCol);

    if (anX<0.5)
      return aABC.y();

    if (anX<1.0)
      return aABC.z();
       
    return aABC.x();
}

/* *************************************************** */
/*                                                     */
/*           cEpsDiffFctr                              */
/*                                                     */
/* *************************************************** */

cEpsDiffFctr::cEpsDiffFctr(const cInterpolator1D & anInt,tREAL8 aEps) :
    cDiffInterpolator1D (anInt.SzKernel(),Append(   {"EpdDif",ToStr(aEps)},anInt.VNames() )),
    mInt (anInt),
    mEps (aEps)
{
}


tREAL8  cEpsDiffFctr::Weight(tREAL8  anX) const  {return mInt.Weight(anX);}
tREAL8  cEpsDiffFctr::DiffWeight(tREAL8  anX) const  {return (mInt.Weight(anX+mEps)-mInt.Weight(anX-mEps)) / (2*mEps) ;}


/* *************************************************** */
/*                                                     */
/*           cTabulatedInterpolator                    */
/*                                                     */
/* *************************************************** */

cTabulatedInterpolator::cTabulatedInterpolator(const cInterpolator1D &anInt,int aNbTabul,bool mInterpolTab,bool DoNorm) :
     cInterpolator1D  (anInt.SzKernel(),Append(   {"NDTabul",ToStr(aNbTabul)}, anInt.VNames()  )),
     mInterpolTab     (mInterpolTab),
     mNbTabul         (aNbTabul),
     mSzTot           (round_up(anInt.SzKernel()*mNbTabul)),
     mIm              (mSzTot+1),
     mDIm             (&mIm.DIm())
{

      // [0]  initialisation of weight
      for (int aK=0 ; aK<mDIm->Sz() ; aK++)
          mDIm->SetV(aK,anInt.Weight(aK/tREAL8(mNbTabul)));
      mDIm->SetV(mSzTot,0.0);

      // [1] if we do the normalization, do it in mode "non derivative"
      if (DoNorm) 
          DoNormalize(false);
}

cTabulatedInterpolator::cTabulatedInterpolator(const cInterpolator1D &anInt,int aNbTabul,bool mInterpolTab)  :
	cTabulatedInterpolator(anInt,aNbTabul,mInterpolTab,true)
{
}

void cTabulatedInterpolator::SetDiff(const cTabulatedInterpolator & anInt)
{
    // for low and high bounds, specific fixing the value
     mDIm->SetV(0,0.0);       // even function so : dW/dx(0) =0
     // mDIm->SetV(mSzTot,0.0);  // bounded function, 0 out of kernel support

     tREAL8 a2Eps = 2.0/mNbTabul;

     mDIm->SetV(mSzTot,(anInt.mDIm->GetV(mSzTot)-anInt.mDIm->GetV(mSzTot-1)) / a2Eps);

     // for other compute the value by finite difference
     for (int aK=1 ; aK<mSzTot ; aK++)
         mDIm->SetV(aK,(anInt.mDIm->GetV(aK+1)-anInt.mDIm->GetV(aK-1)) / a2Eps);
}


void cTabulatedInterpolator::DoNormalize(bool ForDerive)
{
      // Can't make a partion of unity if kernel too small
      MMVII_INTERNAL_ASSERT_bench(mSzKernel>0.5,"Kernel too small in cTabulatedInterpolator");

      tREAL8 aSomWDif = 0; // observation of inital deviation, for eventual show/debug
      tREAL8 aCheckS=0.0;   // Check sum, useful 4 derive
			   
      int aAmpl = (mSzTot/mNbTabul+3) * mNbTabul;
      // Avoid  parse twice the phase
      for (int aKPhase=0 ; 2*aKPhase<= mNbTabul ; aKPhase++)
      {
	  // [2]  compute the sum/average of all value same phase
          tREAL8 aSumV=0.0;
	  int aNb=0;  // count number for average
          for (int aKSigned=aKPhase-aAmpl ; aKSigned<aAmpl ; aKSigned+=mNbTabul)
	  {
               tREAL8 aV = mDIm->DefGetV(std::abs(aKSigned),0.0); 
	       if (ForDerive) 
                   aV *= SignSupEq0(aKSigned); // even func => dF/dx (-X) = - dF/dx (X)
               aSumV += aV;
	       aNb++;
	  }

	  if (ForDerive)   // if not Sum1, then Sum0 -> this is the avg we substract
             aSumV /= aNb;


	  // divide/substratc to all value same phase
          for (int aKSigned=aKPhase-aAmpl ; aKSigned<aAmpl ; aKSigned+=mNbTabul)
	  {
               int aKAbs =  std::abs(aKSigned);
               if (mDIm->Inside(cPt1di(aKAbs)))
	       {
			      
	           if (ForDerive)
		   {
                      mDIm->SetV(aKAbs,mDIm->GetV(aKAbs)-aSumV);
                      aCheckS += mDIm->GetV(aKAbs);
		   }
	           else
		   {
                      // if the 2*phase = mNbTabul we dont want to divide twice the result
                      if (  ((2*aKPhase)!=mNbTabul) || (aKSigned<=0))
		      {
                           mDIm->SetV(aKAbs,mDIm->GetV(aKAbs)/aSumV);
		      }
                      aCheckS += mDIm->GetV(aKAbs);
		   }
	       }
	  }

	  aSomWDif +=  ForDerive ?  std::abs(aSumV) : std::abs(aSumV-1) ;
      }

      if (0)
      {
          aSomWDif /= mNbTabul;
          StdOut() << "SSS= " << aSomWDif  << " NbT=" << mNbTabul   << " CheckS=" << aCheckS << "\n";
          getchar();
      }
}


tREAL8  cTabulatedInterpolator::Weight(tREAL8  anX) const 
{
   tREAL8 aRK = std::abs(anX) * mNbTabul;  // compute the real index in tab

   if (aRK>= mSzTot) // out of kernel ->0
      return 0.0;


   if (mInterpolTab)
      return mDIm->GetVBL(aRK);  // case linear interpol
   else
      return mDIm->GetV(round_ni(aRK)); // case integer value
}

/* *************************************************** */
/*                                                     */
/*           cTabulatedDiffInterpolator                */
/*                                                     */
/* *************************************************** */

cTabulatedDiffInterpolator::cTabulatedDiffInterpolator(const cInterpolator1D &anInt,int aNbTabul) :
	cDiffInterpolator1D (anInt.SzKernel(), Append(   {TheNameInterpol,ToStr(aNbTabul)}, anInt.VNames()  )),
	mTabW     (anInt,aNbTabul,true,true),    // true -> linear interpol, true ->we normalize value
	mTabDifW  (anInt,aNbTabul,true,false),   // we dont normalize, btw coeff are not up to date and would divide by 0
        mNbTabul  (mTabW.mNbTabul),              // fast direct access
	mSzTot    (mTabW.mSzTot),                // fast direct access
	mRawW     (mTabW.mDIm->RawDataLin()),     // raw data for efficiency
	mRawDifW  (mTabDifW.mDIm->RawDataLin())   // raw data for efficiency
{
	mTabDifW.SetDiff(mTabW);   // put in DifW the difference of W
	mTabDifW.DoNormalize(true); // normalize by sum 0
}

cTabulatedDiffInterpolator::cTabulatedDiffInterpolator(cInterpolator1D * anInt,int aNbTabul) :
     cTabulatedDiffInterpolator (*anInt,aNbTabul)
{
     delete anInt;
}

const std::string cTabulatedDiffInterpolator::TheNameInterpol = "Tabul";

// just call the weight of tabulated values
tREAL8  cTabulatedDiffInterpolator::Weight(tREAL8  anX) const 
{
	return mTabW.Weight(anX);
}

// call the weight of tab value; adjust sign to take account " F'(-X) = - F'(X) "
tREAL8  cTabulatedDiffInterpolator::DiffWeight(tREAL8  anX) const 
{
	return SignSupEq0(anX) * mTabDifW.Weight(anX);
}


/* Optimized version, avoid multiple computation of indexes & linear weighting */

std::pair<tREAL8,tREAL8>   cTabulatedDiffInterpolator::WAndDiff(tREAL8  anX) const 
{
   tREAL8 aRK = std::abs(anX) * mNbTabul;  // real indexe
					  
   if (aRK>= mSzTot)   // out kernel : value and derivative = 0
      return std::pair<tREAL8,tREAL8>(0,0);

   int aIk = round_down(aRK);  // for interpol Ik <= aRK < Ik+1
   tREAL8 aWeight1 = aRK-aIk;     // if Rk=Ik+E   W(Ik+1)=E and  W(Ik) = 1-E
   tREAL8 aWeight0 = 1-aWeight1;
   
   const tREAL8  * aDataW    = mRawW+aIk;      // raw data value shifted from IK
   const tREAL8  * aDataDifW = mRawDifW+aIk;   // raw data derivative shifted from IK

   // weighted linear interpolation, as before note the sign in derivative for "F'(-X)=-F'(X)"
   return std::pair<tREAL8,tREAL8>
	 ( 
	      aWeight0*aDataW[0] + aWeight1*aDataW[1] ,  
	      (aWeight0*aDataDifW[0] + aWeight1*aDataDifW[1]) * SignSupEq0(anX)
	 );
}


/* *********************************************** */
/*                                                 */
/*             cScaledInterpolator                 */
/*                                                 */
/* *********************************************** */

/* class for constructing an scaled version of an existing interpolator from scaled version,
 *
 *    +  usefull essentially for image ressampling to create a specific blurring
 *    +  !!!  its not a partition unit, so like "cSinCApodInterpolator" it must be used to generate a tabulated versions
 *
 * */

class cScaledInterpolator : public cInterpolator1D
{
      public :
            cScaledInterpolator(cInterpolator1D *,tREAL8 aScale,bool ToDelete=false);

	    virtual ~ cScaledInterpolator();
	    tREAL8  Weight(tREAL8  anX) const override;
	    static  cTabulatedDiffInterpolator * AllocTab(const cInterpolator1D &,tREAL8 aScale,int aNbTabul);


      private :

	    cInterpolator1D  * mInterp;
	    tREAL8             mScale;
	    bool               mToDelete;
};


cScaledInterpolator::cScaledInterpolator
(
        cInterpolator1D * aInterp,
	tREAL8 aScale,
	bool   isToDelete
)  :
	cInterpolator1D
	(
	      aInterp->SzKernel() * aScale,
	      Append(std::vector<std::string>{"Scale",ToStr(aScale)},aInterp->VNames())
	),
	mInterp     (aInterp),
	mScale      (aScale),
        mToDelete   (isToDelete)
{
}

cScaledInterpolator::~cScaledInterpolator()
{
   if (mToDelete)
      delete mInterp;
}


tREAL8  cScaledInterpolator::Weight(tREAL8  anX) const 
{
      return mInterp->Weight(anX/mScale);
}

cTabulatedDiffInterpolator * cScaledInterpolator::AllocTab(const cInterpolator1D & anI,tREAL8 aScale,int aNbTabul) 
{
	cScaledInterpolator aScalI(const_cast<cInterpolator1D*>(&anI),aScale,false);

	return new cTabulatedDiffInterpolator(aScalI,aNbTabul);
}

/* *********************************************** */
/*                                                 */
/*             cMultiScaledInterpolator            */
/*                                                 */
/* *********************************************** */

class cMultiScaledInterpolator : public cDiffInterpolator1D
{
	public :
             cMultiScaledInterpolator
             (
                  const cInterpolator1D & anI,
		  const tREAL8 aScale0, 
		  const tREAL8 aScale1, 
		  int   aNbScale,
		  int   aNbTabul
	     );
	     virtual ~cMultiScaledInterpolator();
	     tREAL8  UnBoundedIndex2Scale(tREAL8 anIndex) const;
	     tREAL8  UnBoundedScale2Index(tREAL8 aScale) const;

	     tREAL8  Weight(tREAL8  anX) const override;
	     tREAL8  DiffWeight(tREAL8  anX) const override;

	     void SetScale(tREAL8 aScale);
	private :

	     tREAL8  mSzKInit;
	     tREAL8  mCurScale;
	     tREAL8  mScale0;
	     tREAL8  mScale1;
	     tREAL8  mRatio;
	     tREAL8  mLogRatio;
	     int     mNbScale;

	     std::vector<cTabulatedDiffInterpolator*> mTabInterps;
	     std::vector<cTabulatedDiffInterpolator*> mICur;
	     std::vector<tREAL8>                  mWCur;
};


cMultiScaledInterpolator::cMultiScaledInterpolator
(
     const cInterpolator1D & anI,
     const tREAL8 aScale0, 
     const tREAL8 aScale1, 
     int   aNbScale,
     int   aNbTabul
) :
     cDiffInterpolator1D (1.0,{}),
     mSzKInit  (anI.SzKernel()),
     mCurScale (-1),
     mScale0   (aScale0),
     mScale1   (aScale1),
     mRatio    (mScale1/mScale0),
     mLogRatio (std::log(mRatio)),
     mNbScale  (aNbScale)
{
	for (int aK=0 ; aK<=mNbScale ; aK++)
	{
            tREAL8 aScale = UnBoundedIndex2Scale(aK);
            mTabInterps.push_back(cScaledInterpolator::AllocTab(anI,aScale,aNbTabul));
	}

	SetScale(std::sqrt(mScale0*mScale1));
}

cMultiScaledInterpolator::~cMultiScaledInterpolator()
{
    DeleteAllAndClear(mTabInterps);
}

void cMultiScaledInterpolator::SetScale(tREAL8 aScale)
{
    mICur.clear();
    mWCur.clear();

    tREAL8  aRIndex = UnBoundedScale2Index(aScale);

    if (aRIndex<=0)
    {
        mICur.push_back(mTabInterps[0]);
	mWCur.push_back(1.0);
    }
    else if (aRIndex>=mNbScale)
    {
        mICur.push_back(mTabInterps.back());
	mWCur.push_back(1.0);
    }
    else
    {
        int anInd = round_down(aRIndex);
	tREAL8 aW1 = aRIndex-anInd;

        mICur.push_back(mTabInterps.at(anInd));
	mWCur.push_back(1-aW1);

        mICur.push_back(mTabInterps.at(anInd+1));
	mWCur.push_back(aW1);
    }
    SetSzKernel(std::max(mICur[0]->SzKernel(),mICur.back()->SzKernel()));

}



tREAL8  cMultiScaledInterpolator::UnBoundedIndex2Scale(tREAL8 anIndex) const
{
      return  mScale0 * std::pow(mRatio,anIndex/mNbScale);
}

tREAL8  cMultiScaledInterpolator::UnBoundedScale2Index(tREAL8 aScale) const
{
	return std::log(aScale/mScale0) * (mNbScale / mLogRatio) ;
}

tREAL8  cMultiScaledInterpolator::Weight(tREAL8  anX) const 
{ 
     tREAL8 aRes =  mICur[0]->Weight(anX) *  mWCur[0];
     if (mICur.size() >= 2)
         aRes +=  mICur[1]->Weight(anX) *  mWCur[1];

     return aRes;
}

tREAL8  cMultiScaledInterpolator::DiffWeight(tREAL8  anX) const 
{ 
     tREAL8 aRes =  mICur[0]->DiffWeight(anX) *  mWCur[0];
     if (mICur.size() >= 2)
         aRes +=  mICur[1]->DiffWeight(anX) *  mWCur[1];

     return aRes;
}


// tREAL8  cMultiScaledInterpolator::DiffWeight(tREAL8  anX) const { return 0.0; }

void Bench_cMultiScaledInterpolator()
{
     for (int aK=0 ; aK<10 ; aK++)
     {
          tREAL8 aS0 = 0.7 + RandUnif_0_1() * 2;
          tREAL8 aS1 = aS0 * std::pow(100.0, RandUnif_0_1());
          int aNbInd = 10;
          cMultiScaledInterpolator  aMSI(cCubicInterpolator(0.0),aS0,aS1,aNbInd,50);

          for (int aK=0 ; aK<10 ; aK++)
          {
              tREAL8 aInd = RandInInterval(-3,aNbInd+3);
              tREAL8 aS = aMSI.UnBoundedIndex2Scale(aInd) ;
              tREAL8 aInd2 = aMSI.UnBoundedScale2Index(aS) ;
              MMVII_INTERNAL_ASSERT_bench(std::abs(aInd-aInd2)<1e-5,"Bench_cMultiScaledInterpolator");

              aMSI.SetScale(aS);
          }
     }
}




};
