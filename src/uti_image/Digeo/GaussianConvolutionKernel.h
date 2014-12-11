//----------------------------------------------------------------------
// DigeoGaussianKernel related functions
//----------------------------------------------------------------------

// Methode pour avoir la "meilleure" approximation entiere d'une image reelle
// avec somme imposee. Tres sous optimal, mais a priori utilise uniquement sur de
// toute petites images

Im1D_INT4 ToIntegerKernel(Im1D_REAL8 aRK,int aMul,bool aForceSym)
{
	aRK = MakeSom1(aRK);
	int aSz = aRK.tx();
	Im1D_INT4 aIK(aSz);

	int aSom;
	ELISE_COPY( aIK.all_pts(), round_ni( aRK.in()*aMul ), aIK.out()|sigma(aSom) );

	int    * aDI = aIK.data();
	double * aDR = aRK.data();
	while (aSom != aMul)
	{
		int toAdd = aMul-aSom;
		int aSign = ( toAdd>0 )?1:-1;
		int aKBest = -1;
		double aDeltaMin = 1e20;

		if ( aForceSym && ( ElAbs(toAdd)==1 ) )
		{
			ELISE_ASSERT((aSz%2),"ToIntegerKernel Sym");
			aKBest=  aSz/2;
		}
		else
		{
			for ( int aK=0; aK<aSz; aK++ )
			{
				double aDelta =  ( ( aDI[aK]/double(aMul) )-aDR[aK] )*aSign;
				if ( ( aDelta<aDeltaMin ) && ( aDI[aK]+aSign>=0 ) )
				{
					aDeltaMin = aDelta;
					aKBest= aK;
				}
			}
		}

		ELISE_ASSERT( aKBest!=-1, "Inco(1) in ToIntegerKernel" );

		aDI[aKBest] += aSign;
		aSom += aSign;

		if ( aForceSym && (aSom!=aMul) )
		{
			int aKSym = aSz-aKBest-1;
			if ( aKSym!=aKBest )
			{
				aDI[aKSym] += aSign;
				aSom += aSign;
			}
		} 
	}
	return aIK;
}

Im1D_INT4 ToOwnKernel( Im1D_REAL8 aRK, int & aShift, bool aForceSym, int * )
{
	return ToIntegerKernel(aRK,1<<aShift,aForceSym);
}

Im1D_REAL8 ToOwnKernel( Im1D_REAL8 aRK, int & aShift, bool aForceSym, double * )
{
	return aRK;
}

Im1D_REAL8 ToRealKernel( Im1D_INT4 aIK )
{
	Im1D_REAL8 aRK( aIK.tx() );
	ELISE_COPY( aIK.all_pts(), aIK.in(), aRK.out() );
	return MakeSom1(aRK);
}

Im1D_REAL8 ToRealKernel( Im1D_REAL8 aRK ) { return aRK; }

Im1D_REAL8 MakeSom( Im1D_REAL8 &i_vector, double i_dstSum )
{
	double aSomActuelle;
	Im1D_REAL8 aRes( i_vector.tx() );
	ELISE_COPY( i_vector.all_pts(), i_vector.in(), sigma(aSomActuelle) );
	ELISE_COPY( i_vector.all_pts(), i_vector.in()*( i_dstSum/aSomActuelle ), aRes.out() );
	return aRes;
}

Im1D_REAL8 MakeSom1( Im1D_REAL8 &i_vector ){ return MakeSom( i_vector, 1.0 ); }

Im1D_REAL8 DigeoGaussianKernel( double aSigma, int aNb, int aSurEch )
{
	Im1D_REAL8 aRes( 2*aNb+1 );

	for ( int aK=0; aK<=aNb; aK++ )
	{
		double aSom = 0;
		for ( int aKE =-aSurEch; aKE<=aSurEch; aKE++ )
		{
			double aX = aK - aNb + aKE/double( 2*aSurEch+1 );
			double aG = exp( -ElSquare( aX/aSigma )/2.0 );
			aSom += aG;
		}
		aRes.data()[aK] = aRes.data()[2*aNb-aK] = aSom;
	}

	return MakeSom1(aRes);
}

int DigeoGaussianKernelNbElements(double aSigma,double aResidu)
{
	return round_up( sqrt(-2*log(aResidu))*aSigma);
}

Im1D_REAL8 DigeoGaussianKernelFromResidue( double aSigma, double aResidu, int aSurEch )
{
	return DigeoGaussianKernel( aSigma, DigeoGaussianKernelNbElements( aSigma, aResidu ), aSurEch );
}

template <class Type>
Im1D<Type,Type> DigeoGaussianKernel( double aSigma, int aNbShift, double aEpsilon, int aSurEch )
{
	Im1D_REAL8 aKerD = DigeoGaussianKernelFromResidue( aSigma, aEpsilon, aSurEch );
	return ToOwnKernel( aKerD, aNbShift, true, (Type *)NULL );
}


//----------------------------------------------------------------------
// SampledGaussianKernel related functions
//----------------------------------------------------------------------

void createGaussianKernel_1d( double i_standardDeviation, vector<float> &o_kernel );

Im1D_REAL8 createSampledGaussianKernel( double aSigma )
{
	vector<float> kernel;
	createGaussianKernel_1d( aSigma, kernel );
	Im1D_REAL8 res( (int)kernel.size() );
	const float *itSrc = kernel.data();
	REAL8 *itDst = res.data();
	size_t i = kernel.size();
	while ( i-- ) *itDst++ = (REAL8)( *itSrc++ );
	return res;
}

template <class Type>
Im1D<Type,Type> SampledGaussianKernel( double aSigma, int aNbShift )
{
	Im1D_REAL8 aKerD = createSampledGaussianKernel(aSigma);
	return ToOwnKernel( aKerD, aNbShift, true, (Type *)NULL );
}

