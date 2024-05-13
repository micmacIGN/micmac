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

	for (int aKX=0 ; aKX< aNbX ; aKX++)
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
	for (int aKX=0 ; aKX< aNbX ; aKX++)
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


template class cDataIm2D<tU_INT1>;
template class cDataIm2D<tINT1>;
template class cDataIm2D<tU_INT2>;
template class cDataIm2D<tINT2>;
template class cDataIm2D<tINT4>;
template class cDataIm2D<tREAL4>;
template class cDataIm2D<tREAL8>;

};

