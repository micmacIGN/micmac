#include "MMVII_Interpolators.h"

namespace MMVII
{

/* *************************************************** */
/*                                                     */
/*           IM2D/IM2D                                 */
/*                                                     */
/* *************************************************** */

template <> bool cPixBox<2>::InsideInterpolator(const cInterpolator1D & anInterpol,const cPtxd<double,2> & aP,tREAL8 aMargin) const
{
    tREAL8 aSzK = anInterpol.SzKernel() + aMargin;
    // tREAL8 aSzKm1 = aSzK-1.0;

    // StdOut()  << " IIII " << aP << " XX=" << aP.x()-aSzKm1 << "\n";


    return   ( round_up(aP.x()-aSzK) >= tBox::mP0.x()) &&  (round_down(aP.x()+aSzK) <  tBox::mP1.x())
          && ( round_up(aP.y()-aSzK) >= tBox::mP0.y()) &&  (round_down(aP.y()+aSzK) <  tBox::mP1.y())
    ;
}
/*
*/

/*   Compute the linear interpolation of an image, using the separability property W(X,Y) = W(X)W(Y).
 *   Let :
 *      - I be an image, note I[i,j] = I[j][i] the integer values of I
 *      - x,y be real coordinates
 *      - W an interpolator assimilated to its weighting function
 *      - we want to compute I(x,y,W) the value of x,y in using using interpolator W
 *
 *      I(x,y,W) 
 *   =  Sum{i,j} I[i,j]   W(x-i,y-j)
 *   = Sum{j}  W(y-j)  Sum{i}  I[i,j] W(x-i)
 *
 *   For optimizatin we pre-compute the values "W(x-i)" in a buffer, and for a given j,
 *   we compute once "W(y-j)"  and then  access to the line of image I[j]+i
 *
 */


static std::vector<tREAL8>  TheBufCoeffX;  // store the value on 1 line, as they are separable
template <class Type>  tREAL8 cDataIm2D<Type>::GetValueInterpol(const cInterpolator1D & anInterpol,const cPt2dr & aP) const 
{

    MMVII_INTERNAL_ASSERT_tiny(this->InsideInterpolator(anInterpol,aP,0.0),"Outside interpolator in GetValueInterpol");

    TheBufCoeffX.clear(); // purge the buffer
    tREAL8 aSzK = anInterpol.SzKernel();

    // [0]  compute the bounding in X and Y
    tREAL8 aRealY = aP.y();
    //  round_Uup ->  y=4 SzK=2  ->   3, because value 2 is useless (Kernel is 0 outside support, and continuous)
    int aY0 = round_up(aRealY-aSzK);  
    int aY1 = round_down(aRealY+aSzK);  // idem  Uup

    tREAL8 aRealX = aP.x();
    int aX0 = round_up(aRealX-aSzK); 
    int aX1 = round_down(aRealX+aSzK);
    int aNbX =  aX1-aX0+1;

    // [1] memorize the weights W(x)
    for (int aIntX=aX0 ; aIntX<=aX1 ; aIntX++)
    {
        tREAL8 aWX = anInterpol.Weight(aIntX-aRealX);
        TheBufCoeffX.push_back(aWX);
    }
    const tREAL8 *  aLineWX  = TheBufCoeffX.data();

    // [2] compute the  Sum{i,j} I[i,j]   W(x-i) W(y-j)
    tREAL8 aSomWIxy = 0.0;
    for (int aIntY=aY0 ; aIntY<=aY1 ; aIntY++)
    {
	const Type *  aLineIm = mRawData2D[aIntY] + aX0;
	// const tREAL8 *  aCurWX  = aLineWXInit;
	tREAL8 aSomWIx = 0.0;

	for (int aKX=0 ; aKX< aNbX ; aKX++) // doc  SCALARPRODUCT
            aSomWIx += aLineIm[aKX]  * aLineWX[aKX] ;

	/*  to see if this old style optim is any faster ? which I doubt ...
        int aKx= aNbX;
	while (aKx--)
            aSomWIx += *(aLineIm++)  *  *(aCurWX++) ;
	    */

        aSomWIxy  += aSomWIx * anInterpol.Weight(aIntY-aRealY);
    }
    return aSomWIxy ;
}


template <class Type>  
    tREAL8 cDataIm2D<Type>::ClipedGetValueInterpol(const cInterpolator1D & anInterpol,const cPt2dr & aP,double  aDefVal,bool * Ok) const 
{
    TheBufCoeffX.clear(); // purge the buffer
    tREAL8 aSzK = anInterpol.SzKernel();

    // [0]  compute the bounding in X and Y
    tREAL8 aRealY = aP.y();
    //  round_Uup ->  y=4 SzK=2  ->   3, because value 2 is useless (Kernel is 0 outside support, and continuous)
    int aY0 = std::max(0,round_up(aRealY-aSzK));  
    int aY1 = std::min(SzY()-1,round_down(aRealY+aSzK));  // idem  Uup

    tREAL8 aRealX = aP.x();
    int aX0 = std::max(0,round_up(aRealX-aSzK)); 
    int aX1 = std::min(SzX()-1,round_down(aRealX+aSzK));

    tREAL8 aSWX = 0.0;
    for (int aIntX=aX0 ; aIntX<=aX1 ; aIntX++)
    {
        tREAL8 aWX = anInterpol.Weight(aIntX-aRealX);
        TheBufCoeffX.push_back(aWX);
	aSWX += aWX;
    }
    int aNbX =  aX1-aX0+1;
    const tREAL8 *  aLineWX  = TheBufCoeffX.data();

    tREAL8 aSWY = 0.0;
    // [2] compute the  Sum{i,j} I[i,j]   W(x-i) W(y-j)
    tREAL8 aSomWIxy = 0.0;
    for (int aIntY=aY0 ; aIntY<=aY1 ; aIntY++)
    {
	const Type *  aLineIm = mRawData2D[aIntY] + aX0;
	// const tREAL8 *  aCurWX  = aLineWXInit;
	tREAL8 aSomWIx = 0.0;

	for (int aKX=0 ; aKX< aNbX ; aKX++) // doc  SCALARPRODUCT
            aSomWIx += aLineIm[aKX]  * aLineWX[aKX] ;

	tREAL8 aWY = anInterpol.Weight(aIntY-aRealY);

        aSomWIxy  += aSomWIx * aWY;
	aSWY += aWY;
    }

    if ((aSWX==0.0) || (aSWY==0.0))
    {
        if (Ok)
        {
           *Ok = false;
           return aDefVal;	
        }
        MMVII_INTERNAL_ERROR("No point for ClipedGetValueInterpol");
    }
    if (Ok) *Ok = true;

    return aSomWIxy  / (aSWX *  aSWY);
}



/*   Compute the linear interpolation and its derivative of an image,  noting @ the convultion operator we can
 *   write  using the separability of W :
 *
 *      I(x,y,W)  = I @ Wx @ Wy
 *
 *    Then :
 *
 *      d I(x,y,W)/ dx  =  I @  d(Wx)/dx @ Wy 
 *      d I(x,y,W)/ dy =  I @  d(Wx)/dx @ Wy 
 *      I(x,y,W)  = I @ Wx @ Wy
 *
 *   Using the same structure than "GetValueInterpol" we compute simultaneously the 3 value that share
 *   much in common .
 *
 */

static std::vector<tREAL8>  TheBufCoeffDerX;  // store the value on 1 line, as they are separable
					   
template <class Type>  
    std::pair<tREAL8,cPt2dr> 
           cDataIm2D<Type>::GetValueAndGradInterpol
           (
		const cDiffInterpolator1D & anInterpol,
	        const cPt2dr & aP
           ) const 
{
    MMVII_INTERNAL_ASSERT_tiny(this->InsideInterpolator(anInterpol,aP,0.0),"Outside interpolator in GetValueAndDerInterpol");
    // free the two buffers 
    TheBufCoeffX.clear(); 
    TheBufCoeffDerX.clear();
    tREAL8 aSzK = anInterpol.SzKernel();

    // [0]  compute the bounds
    tREAL8 aRealY = aP.y();
    int aY0 = round_up(aRealY-aSzK);  
    int aY1 = round_down(aRealY+aSzK);

    tREAL8 aRealX = aP.x();
    int aX0 = round_up(aRealX-aSzK); 
    int aX1 = round_down(aRealX+aSzK);
    int aNbX =  aX1-aX0+1;

    // [1]  pre-compute in buffers the weights on x and it derivative
    for (int aIntX=aX0 ; aIntX<=aX1 ; aIntX++)
    {
        auto [aWX,aDerX]  = anInterpol.WAndDiff(aRealX-aIntX);
        TheBufCoeffX.push_back(aWX);
        TheBufCoeffDerX.push_back(aDerX);
    }

    const tREAL8 *  aLineWX  = TheBufCoeffX.data();
    const tREAL8 *  aLineDerWX  = TheBufCoeffDerX.data();

    // [2]  compute aSomWxyI= I @ Wx @ Wy  , aSomWI_Dx = I @  d(Wx)/dx @ Wy , aSomWI_Dy = I @  d(Wx)/dx @ Wy 
    tREAL8 aSomWxyI = 0.0;
    tREAL8 aSomWI_Dx = 0.0;
    tREAL8 aSomWI_Dy = 0.0;

    for (int aIntY=aY0 ; aIntY<=aY1 ; aIntY++)
    {
	const Type *  aLineIm = mRawData2D[aIntY] + aX0;

	tREAL8 aSomWIx = 0.0;
	tREAL8 aSomDerWIx = 0.0;
	for (int aKX=0 ; aKX< aNbX ; aKX++) // doc  SCALARPRODUCT
	{
            aSomWIx    += aLineIm[aKX]  *  aLineWX [aKX];
            aSomDerWIx += aLineIm[aKX]  *  aLineDerWX[aKX];
	}

        auto [aWY,aDerY] = anInterpol.WAndDiff(aRealY-aIntY);
	aSomWxyI  +=  aSomWIx    * aWY;
	aSomWI_Dx +=  aSomDerWIx * aWY;
	aSomWI_Dy +=  aSomWIx    * aDerY;
    }

    return std::pair<tREAL8,cPt2dr> (aSomWxyI,cPt2dr(aSomWI_Dx,aSomWI_Dy));
}

template <class Type>  
       cIm2D<Type>  cIm2D<Type>::Scale(const cInterpolator1D & anInterpol,tREAL8 aFX,tREAL8 aFY) const
{

     // By Defaut SzY==SzX
     if (aFY<0) aFY = aFX;

     cPt2dr aPScale(aFX,aFY);
     cPt2di aSzOut =  Pt_round_up(DivCByC(  ToR(mPIm->Sz()) , aPScale  ));

     cIm2D<Type> aImOut(aSzOut);
     cDataIm2D<Type> & aDImOut = aImOut.DIm();

     for (const auto & aPixOut : aDImOut)
     {
         bool Ok;
         cPt2dr aPixIn = MulCByC(ToR(aPixOut),aPScale);
         tREAL8 aVal = mPIm->ClipedGetValueInterpol(anInterpol,aPixIn,0.0,&Ok);
	 aDImOut.SetVTrunc(aPixOut,aVal);
     }


     return aImOut;
}


template <class Type>  
       cIm2D<Type>  cIm2D<Type>::Scale
                    (
                         tREAL8 aFX,tREAL8 aFY,tREAL8 aSzSinC,tREAL8 aDilateKernel,
                         const std::vector<std::string> & aVNameKernI 
                    ) const
{
     if (aFY<0) aFY = aFX;

     if ((aFX==1.0) && (aFY==1.0))
     {
	     return Dup();
     }

     tREAL8 aF = std::sqrt(aFX*aFY);

     cInterpolator1D * aTabInt = nullptr; // tabulated interpolator


     if (aF>1.0)
     {
           tREAL8 aFactCub = std::min(-0.5 + (aF-1)/2.0,0.0);

	   // aInt = cScaledInterpolator::AllocTab(cCubicInterpolator(aFactCub),aF*aDilateKernel,100);
            cDiffInterpolator1D * aInt = cDiffInterpolator1D::AllocFromNames(aVNameKernI);

	   aTabInt = cScaledInterpolator::AllocTab(*aInt,aF*aDilateKernel,1000);
           delete aInt;

           if (0)
           {
                StdOut() << "SCALEIT FZoom= " << aF  << " FCub=" << aFactCub 
                         << " D=" << aDilateKernel << " SzK=" << aTabInt->SzKernel()
                         << " \n   ";
                for (int aK=0 ; aK<= aF+2 ; aK++)
                    StdOut() <<  " [K=" << aK  << " I=" << aTabInt->Weight(aK) << "]" ;
                 StdOut()  << "\n";

                 for (tREAL8 aPh=-0.1 ; aPh<aF ; aPh+=0.234)
                 {
                      tREAL8 aSum =0;
                      for (int aK=-20 ; aK<=20 ; aK++)
                          aSum += aTabInt->Weight(aK*aF+aPh);
                      StdOut() << "  S="  << aSum  ;  // << "," << aPh ;
                 }
                 StdOut()  << "\n";
           }
     }
     else // if we enlarge the image, we must use standard interpolation
     {
	 if (aSzSinC>0)   // sincard is theoretically "the best" but costly
         {
	     aTabInt = new cTabulatedInterpolator(cSinCApodInterpolator(aSzSinC,aSzSinC),100,true);
         }
         else  // bi cubic is a compromise
         {
             // for F=1 use the bicub linear -0.5, for F=0.5 use -1 and do go over
	     // a+b=-0.5  a/2+b=-1  ; a/2 =0.5 ; a=1 b=-1.5
             // tREAL8 aFactCub = aF-1.5
             tREAL8 aFactCub = std::max(-1.0,aF-1.5);

             aTabInt = new cTabulatedInterpolator(cCubicInterpolator(aFactCub),100,true);
         }
     }

      cIm2D<Type> aResult =  Scale(*aTabInt,aFX,aFY);

      delete aTabInt;

      return aResult;
}






#define INSTANTIATE_IM(TYPE)\
template class cDataIm2D<TYPE>;\
template class cIm2D<TYPE>;

INSTANTIATE_IM(tU_INT1);
INSTANTIATE_IM(tINT1);
INSTANTIATE_IM(tU_INT2);
INSTANTIATE_IM(tINT2);
INSTANTIATE_IM(tINT4);
INSTANTIATE_IM(tREAL4);
INSTANTIATE_IM(tREAL8);


};

