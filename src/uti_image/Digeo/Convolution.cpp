#include "StdAfx.h"
#include "Convolution.h"

//----------------------------------------------------------------------
// methods of class DigeoConvolution
//----------------------------------------------------------------------

const int PackTranspo = 4;

Im1D_REAL8 MakeSom1( Im1D_REAL8 &i_vector );

//  K3-C3 = K1-C1 + K2-C2

Im1D_REAL8 DeConvol(int aC2,int aSz2,Im1D_REAL8 aI1,int aC1,Im1D_REAL8 aI3,int aC3)
{
   L2SysSurResol aSys(aSz2);
   aSys.SetPhaseEquation(0);
   
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
       {
          aSys.GSSR_AddNewEquation_Indexe
          (
                0,0,0,
                aVInd,
                1.0,
                &(aVCoef.data()[0]),
                aI3.data()[aK3]
          );
       }
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

   // Pour utiliser un filtre sur les bord, clip les intervalle
   // pour ne pas deborder et renvoie la somme partielle
template <class tBase> tBase ClipForConvol( int aSz, int aKXY, tBase *aData, int & aDeb, int &aFin )
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

template <class tData>
int getNbShift( const tData *i_data, size_t i_nbElements )
{
	typedef typename El_CTypeTraits<tData>::tBase tBase;

	ELISE_DEBUG_ERROR( !numeric_limits<tData>::is_integer, "getNbShift", "data type " << El_CTypeTraits<tData>::Name() << " is not integer" );

	tBase sum = (tBase)0;
	while ( i_nbElements-- ) sum += (tBase)( *i_data++ );
	int nbShift = round_ni( log(sum)/log(2) );

	ELISE_DEBUG_ERROR( sum!=(1<<nbShift), "getNbShift", "sum = " << sum << " is not a power of 2" );

	return nbShift;
}

template <class Type,class tBase>
cConvolSpec<Type> * ToCompKer( Im1D<tBase,tBase> aKern )
{
	int aSzKer = aKern.tx();
	int nbShift = ( numeric_limits<Type>::is_integer ? getNbShift( aKern.data(), (size_t)aKern.tx() ) : 0 );

	ELISE_ASSERT( aSzKer%2, "Taille paire pour ::SetConvolSepXY" );
	aSzKer /= 2;

	tBase * aData = aKern.data() + aSzKer;
	while ( aSzKer && (aData[aSzKer]==0) && (aData[-aSzKer]==0) )
		aSzKer--;

	return cConvolSpec<Type>::GetOrCreate( aData, -aSzKer, aSzKer, nbShift, false );
}

// anX must not be lesser than 0
template <class tData> 
void SetConvolBordX
	(
		const tData **i_srcData, const int i_width, const int i_height,
		tData **o_dstData,
		int anX,
		typename El_CTypeTraits<tData>::tBase * aDFilter, int aDebX, int aFinX
	)
{
	typedef typename El_CTypeTraits<tData>::tBase tBase;
	tBase aDiv = ClipForConvol( i_width, anX, aDFilter, aDebX, aFinX );
	const tBase aSom = InitFromDiv( aDiv, (tBase*)0 );

	for (int anY=0 ; anY<i_height ; anY++)
		o_dstData[anY][anX] = CorrelLine( aSom, i_srcData[anY]+anX, aDFilter, aDebX, aFinX )/aDiv;
}


    //  SetConvolSepX(aImIn,aData,-aSzKer,aSzKer,aNbShitXY,aCS);
template <class tData> 
void SetConvolSepX
	(
		const tData **i_srcData, const int i_width, const int i_height,
		tData **i_dstData,
		cConvolSpec<tData> * aCS
	)
{
	typedef typename El_CTypeTraits<tData>::tBase tBase;
	/*
	// __DEL
	Im2D<Type,tBase> src = aImIn;
	if ( aImOut.data_lin()==aImIn.data_lin() )
	{
	 Im2D<Type,tBase> newSrc( aImIn.tx()+PackTranspo, aImIn.ty() );
	 newSrc.Resize( aImIn.sz() );
	 memcpy( newSrc.data_lin(), aImIn.data_lin(), aImIn.tx()*aImIn.ty()*sizeof(Type) );
	 aImIn = newSrc;
	}
	*/

	int aX0 = std::min( -aCS->Deb(), i_width );

	int anX;
	for (anX=0; anX <aX0 ; anX++)
		SetConvolBordX( i_srcData, i_width, i_height, i_dstData, anX, aCS->DataCoeff(), aCS->Deb(), aCS->Fin() );

	int aX1 = std::max( i_width-aCS->Fin(), anX );
	for ( anX=aX1; anX<i_width; anX++ ) // max car aX1 peut être < aX0 voir negatif et faire planter
		SetConvolBordX( i_srcData, i_width, i_height, i_dstData, anX, aCS->DataCoeff(), aCS->Deb(), aCS->Fin() );

	// const tBase aSom = InitFromDiv(ShiftG(tBase(1),aNbShitX),(tBase*)0);
	for (int anY=0 ; anY<i_height ; anY++)
	{
		tData *aDOut = i_dstData[anY];
		const tData *aDIn =  i_srcData[anY];

		aCS->Convol(aDOut,aDIn,aX0,aX1);
	}
}

template <class tData> 
static tData ** __new_lines_data( int i_width, int i_height )
{
	tData *data = new tData[i_width*i_height];
	tData **lines = new tData*[i_height];
	for ( int y=0; y<i_height; y++ )
	{
		lines[y] = data;
		data += i_width;
	}
	return lines;
}

// i_data must have at least one line
template <class tData> 
static void __delete_lines_data( tData **i_data )
{
	delete [] i_data[0];
	delete [] i_data;
}

template <class tData> 
void SelfSetConvolSepY
	(
		tData **i_data, int i_width, int i_height,
		cConvolSpec<tData> *aCS
	)
{
	tData **aBufIn  = __new_lines_data<tData>( i_height, PackTranspo );
	tData **aBufOut = __new_lines_data<tData>( i_height, PackTranspo );
	for ( int anX=0; anX<i_height; anX+=PackTranspo )
	{
		// Il n'y a pas de debordement car les images  sont predementionnee 
		// d'un Rab de PackTranspo;  voir ResizeBasic

		tData * aL0 = aBufIn[0];
		tData * aL1 = aBufIn[1];
		tData * aL2 = aBufIn[2];
		tData * aL3 = aBufIn[3];
		for (int aY=0 ; aY<i_height; aY++)
		{
			const tData * aL = i_data[aY]+anX;
			*(aL0)++ = *(aL++);
			*(aL1)++ = *(aL++);
			*(aL2)++ = *(aL++);
			*(aL3)++ = *(aL++);
		}
		SetConvolSepX( (const tData**)aBufIn, i_height, PackTranspo, aBufOut, aCS );

		aL0 = aBufOut[0];
		aL1 = aBufOut[1];
		aL2 = aBufOut[2];
		aL3 = aBufOut[3];

		for ( int aY=0; aY<i_height; aY++ )
		{
			tData * aL = i_data[aY]+anX;
			*(aL)++ = *(aL0++);
			*(aL)++ = *(aL1++);
			*(aL)++ = *(aL2++);
			*(aL)++ = *(aL3++);
		}
	}

	__delete_lines_data(aBufIn);
	__delete_lines_data(aBufOut);
}

template <class tData>
DigeoConvolution<tData>::DigeoConvolution()
{
	cConvolSpec<tData>::init();
}

template <class tData>
bool DigeoConvolution<tData>::operator ()( const tData **i_srcData, const int i_width, const int i_height, const Im1D<tBase,tBase> &i_kernel, tData **o_dstData ) const
{
	cConvolSpec<tData> *aCS = ToCompKer<tData>(i_kernel);
	SetConvolSepX( i_srcData, i_width, i_height, o_dstData, aCS );
	SelfSetConvolSepY( o_dstData, i_width, i_height, aCS );
	return aCS->IsCompiled();
}

template <class tData>
void DigeoConvolution<tData>::operator ()( const tData **i_srcData, const int i_width, const int i_height, const cConvolSpec<tData> &i_convolSpec, tData **o_dstData ) const
{
	SetConvolSepX( i_srcData, i_width, i_height, o_dstData, i_convolSpec );
	SelfSetConvolSepY( o_dstData, i_width, i_height, i_convolSpec );
}

template <class tData>
bool DigeoConvolution<tData>::operator ()( const Im2D<tData,tBase> &i_src, const Im1D<tBase,tBase> &i_kernel, Im2D<tData,tBase> &o_dst ) const
{
	if ( i_src.tx()!=o_dst.tx() || i_src.ty()!=o_dst.ty() || ( o_dst.linearDataAllocatedSize()<(i_src.tx()+PackTranspo)*i_src.ty() ) )
	{
		o_dst.Resize( Pt2di( i_src.tx()+PackTranspo, i_src.ty() ) );
		o_dst.Resize( i_src.sz() );
	}
	return (*this)( (const tData **)i_src.data(), i_src.tx(), i_src.ty(), i_kernel, o_dst.data() );
}

template <class tData>
inline void DigeoConvolution<tData>::getCompiledKernels( vector<vector<tBase> > &o_kernels ) const
{
	cConvolSpec<tData>::getCompiledKernels(o_kernels);
}

string lower( const string &i_str )
{
	string res = i_str;
	for ( size_t i=0; i<i_str.length(); i++ )
		res[i] = tolower( res[i] );
	return res;
}

template <class tData>
void DigeoConvolution<tData>::generateCode() const
{
	string generatedFilesBasename = string("GenConvolSpec_")+lower( El_CTypeTraits<tData>::Name() );
	cConvolSpec<tData>::generate_classes( generatedFilesBasename+".classes.h" );
	cConvolSpec<tData>::generate_instantiations( generatedFilesBasename+".instantiations.h" );
}


//----------------------------------------------------------------------
// functions of LegacyConvolution
//----------------------------------------------------------------------

template <class tData>
void LegacyConvolution_transpose_real( const tData *i_src, const int i_width, const int i_height, const vector<typename El_CTypeTraits<tData>::tBase> &i_kernel, tData *o_dst )
{
	typedef typename El_CTypeTraits<tData>::tBase tBase;

	ELISE_DEBUG_ERROR( i_kernel.size()%2==0, "LegacyConvolution_transpose<" << El_CTypeTraits<tData>::Name() << ">", "i_kernel.size()%2==0" );

	// convolve along columns, save transpose
	// filter is (2*W+1) by 1
	const int W = ( i_kernel.size()-1 )/2;
	const tBase *filter_pt = i_kernel.data();
	const tData *src_pt = i_src;
	for ( int j=0; j<i_height; ++j )
	{
		for ( int i=0; i<i_width; ++i )
		{
			tBase acc = 0, x;
			const tBase *g = filter_pt;
			const tData *start = src_pt+( i-W ), *stop;

			// beginning
			stop = src_pt ;
			x = *stop ;
			while ( start<=stop )
			{
				acc += (*g++)*x;
				start++;
			}

			// middle
			stop =  src_pt + std::min(i_width-1, i+W) ;
			while ( start<stop )
				acc += (*g++)*(*start++);

			// end
			x  = *start ;
			stop = src_pt + (i+W);
			while( start<=stop ) { acc += (*g++)*x; start++; }

			// save
			*o_dst = (tData)acc;
			o_dst += i_width;
		}
		// next column
		src_pt += i_width;
		o_dst -= i_width*i_height-1;
	}
}

template <class tData>
void LegacyConvolution_real( const tData *i_src, const int i_width, const int i_height, tData *o_tmp, const vector<typename El_CTypeTraits<tData>::tBase> &i_kernel, tData *o_dst )
{
	LegacyConvolution_transpose( i_src, i_width, i_height, i_kernel, o_tmp );
	LegacyConvolution_transpose( o_tmp, i_width, i_height, i_kernel, o_dst );
}

template <class tData>
void LegacyConvolution_transpose_integer( const tData *i_src, const int i_width, const int i_height, const vector<typename El_CTypeTraits<tData>::tBase> &i_kernel, int i_nbShift, tData *o_dst )
{
	typedef typename El_CTypeTraits<tData>::tBase tBase;

	ELISE_DEBUG_ERROR( i_kernel.size()%2==0, "LegacyConvolution_transpose<" << El_CTypeTraits<tData>::Name() << ">", "i_kernel.size()%2==0" );

	// convolve along columns, save transpose
	// filter is (2*W+1) by 1
	const int W = ( i_kernel.size()-1 )/2;
	const tBase *filter_pt = i_kernel.data();
	const tData *src_pt = i_src;
	for ( int j=0; j<i_height; ++j )
	{
		for ( int i=0; i<i_width; ++i )
		{
			tBase acc = 0, x;
			const tBase *g = filter_pt;
			const tData *start = src_pt+( i-W ), *stop;

			// beginning
			stop = src_pt ;
			x = *stop ;
			while ( start<=stop )
			{
				acc += (*g++)*x;
				start++;
			}

			// middle
			stop =  src_pt + std::min(i_width-1, i+W) ;
			while ( start<stop )
				acc += (*g++)*(*start++);

			// end
			x  = *start ;
			stop = src_pt + (i+W);
			while( start<=stop ) { acc += (*g++)*x; start++; }

			// save
			*o_dst = (tData)( acc>>i_nbShift );
			o_dst += i_width;
		}
		// next column
		src_pt += i_width;
		o_dst -= i_width*i_height-1;
	}
}

template <class tData>
void LegacyConvolution_integer( const tData *i_src, const int i_width, const int i_height, tData *o_tmp, const vector<typename El_CTypeTraits<tData>::tBase> &i_kernel, tData *o_dst )
{
	int nbShift = getNbShift( i_kernel.data(), i_kernel.size() );
	LegacyConvolution_transpose_integer( i_src, i_width, i_height, i_kernel, nbShift, o_tmp );
	LegacyConvolution_transpose_integer( o_tmp, i_width, i_height, i_kernel, nbShift, o_dst );
}

template <> void LegacyConvolution( const U_INT1 *i_src, const int i_width, const int i_height, U_INT1 *o_tmp, const vector<INT> &i_kernel, U_INT1 *o_dst )
{
	LegacyConvolution_integer( i_src, i_width, i_height, o_tmp, i_kernel, o_dst );
}

template <> void LegacyConvolution( const U_INT2 *i_src, const int i_width, const int i_height, U_INT2 *o_tmp, const vector<INT> &i_kernel, U_INT2 *o_dst )
{
	LegacyConvolution_integer( i_src, i_width, i_height, o_tmp, i_kernel, o_dst );
}


//----------------------------------------------------------------------
// instantiation
//----------------------------------------------------------------------

template class DigeoConvolution<U_INT1>;
template class DigeoConvolution<U_INT2>;
template class DigeoConvolution<REAL4>;
