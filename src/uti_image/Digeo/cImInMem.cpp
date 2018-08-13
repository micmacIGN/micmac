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

#include "Digeo.h"

//#define __DEBUG_DIGEO_GAUSSIANS_OUTPUT_RAW
//#define __DEBUG_DIGEO_GAUSSIANS_OUTPUT_PGM
//#define __DEBUG_DIGEO_GAUSSIANS_INPUT
//#define __DEBUG_DIGEO_DOG_OUTPUT
//#define __DEBUG_DIGEO_DOG_INPUT
#define __DEBUG_DIGEO_NORMALIZE_FLOAT_OCTAVE

/****************************************/
/*                                      */
/*             cTplImInMem              */
/*                                      */
/****************************************/

template <class Type> 
cTplImInMem<Type>::cTplImInMem
(
        cImDigeo & aIGlob,
        const Pt2di & aSz,
        GenIm::type_el aType,
        cTplOctDig<Type>  & anOct,
        double aResolOctaveBase,
        int    aKInOct,
        int    IndexSigma
) :
     cImInMem(aIGlob,aSz,aType,anOct,aResolOctaveBase,aKInOct,IndexSigma),
     mTOct (anOct),
     mIm(1,1),
     mTIm (mIm),
     mTMere (0),
     mTFille (0),
     // mOrigOct (0),
     mData   (0),
     mCalculateDiff(NULL)
{
	ResizeImage(aSz);
	mNbShift =  mAppli.Params().TypePyramide().PyramideGaussienne().Val().NbShift().Val();
	switch ( mAppli.refinementMethod() ){
	case eRefine3D: mCalculateDiff = &cTplImInMem<Type>::CalculateDiff_3d; break;
	case eRefine2D: mCalculateDiff = &cTplImInMem<Type>::CalculateDiff_2d; break;
	case eRefineNone: mCalculateDiff = &cTplImInMem<Type>::CalculateDiff_none; break;
	}
}

template <class Type> void cTplImInMem<Type>::ResizeBasic(const Pt2di & aSz)
{
   mIm.Resize(aSz);
   mTIm = mIm;
   mSz = aSz;
   mData = mIm.data();

   mN0 = -mSz.x-1;
   mN1 = -mSz.x;
   mN2 = -mSz.x+1;
   mN3 = -1;
   mN4 = 1;
   mN5 = mSz.x-1;
   mN6 = mSz.x;
   mN7 = mSz.x+1;
}

template <class Type> void cTplImInMem<Type>::ResizeImage(const Pt2di & aSz)
{
    ResizeBasic(aSz+Pt2di(PackTranspo,0));
    ResizeBasic(aSz);
}

template <class tData, class tBase>
tData getMaxValue( const Im2D<tData,tBase> &io_image )
{
	tData res = 0;

	const tData *it = io_image.data_lin();
	unsigned int i = (unsigned int)io_image.tx()*(unsigned int)io_image.ty();
	while ( i-- ) ElSetMax( res, *it++ );
}

template <class tData, class tBase>
void changeDynamic_real( Im2D<tData,tBase> &io_image, tData i_srcMax, tData i_dstMax )
{
	if ( i_srcMax==i_dstMax ) return;
	const int width = io_image.tx(), height = io_image.ty();
	const tData  mul = (tData)i_dstMax/i_srcMax;
	tData **data = io_image.data();
	for (int aY=0 ; aY<height ; aY++)
	{
		tData * aL = data[aY];
		for (int aX=0 ; aX<width; aX++)
			aL[aX] *= mul;
	}
}

template <class tData, class tBase>
void changeDynamic_integer( Im2D<tData,tBase> &io_image, tBase i_srcMax, tBase i_dstMax )
{
	if ( i_srcMax==i_dstMax ) return;
	const int width = io_image.tx(), height = io_image.ty();
	tData **data = io_image.data();
	tBase aMul = i_dstMax/i_srcMax;
	for ( int aY=0; aY<height; aY++ )
	{
		tData * aL = data[aY];
		for ( int aX=0; aX<width; aX++ )
			aL[aX] *= aMul;
	}
}

template <class tData, class tBase>
void changeDynamic_integer_safe( Im2D<tData,tBase> &io_image, tBase i_srcMax, tBase i_dstMax )
{
	if ( i_srcMax==i_dstMax ) return;
	const int width = io_image.tx(), height = io_image.ty();
	tBase aMul = (tBase)round_ni( (double)i_dstMax/(double)i_srcMax );
	tData **data = io_image.data();
	for ( int aY=0; aY<height; aY++ )
	{
		tData *aL = data[aY];
		for ( int aX=0; aX<width; aX++ )
			aL[aX] = ElMin( i_dstMax, tBase(aL[aX])*aMul );
	}
}

/*
template <class tData, class tBase>
void initial_convolution( Im2D<tData,tBase> &o_dst )
{
	const cTypePyramide & aTP = mAppli.Params().TypePyramide();
	if ( aTP.PyramideGaussienne().IsInit() )
	{
		const double aSigmD = mImGlob.InitialDeltaSigma();
		if ( aSigmD!=0. )
		{
			Im1D<tBase,tBase> aIKerD = ImGaussianKernel(aSigmD);
			SetConvolSepXY( true, aSigmD, *this, aIKerD, mNbShift );
		}
	}
}
*/

template <class Type>
void cTplImInMem<Type>::LoadFile(Fonc_Num aFonc,const Box2di & aBox,GenIm::type_el aTypeFile)
{
	ResizeOctave(aBox.sz());
	ELISE_COPY(mIm.all_pts(), aFonc, mIm.out());

	if ( mAppli.Params().MaximDyn().ValWithDef(nbb_type_num(aTypeFile)<=8) &&
	     type_im_integral(mType) &&
	     (!signed_type_num(mType) ) )
	{
		const tBase theoricalMax = (tBase)(double(numeric_limits<tBase>::max()) / (double)(1 << mNbShift));
		const tBase typeMax = (tBase)numeric_limits<Type>::max();
		Type aMaxT = min<Type>(theoricalMax, typeMax);

		tBase aMul = 0;
		if ( mAppli.Params().ValMaxForDyn().IsInit() )
		{
			tBase aMaxTm1 = aMaxT-1;
			aMul = round_ni(aMaxTm1/mAppli.Params().ValMaxForDyn().Val());

			for (int aY=0 ; aY<mSz.y ; aY++)
			{
				Type * aL = mData[aY];
				for (int aX=0 ; aX<mSz.x ; aX++)
					aL[aX] = ElMin(aMaxTm1,tBase(aL[aX]*aMul));
			}
		}
		else
		{
			Type aMinV, aMaxV;
			mIm.getMinMax(aMinV, aMaxV);
			if (fabs(aMaxV - aMinV)>0.0001)
				aMul = (aMaxT - 1) / (aMaxV - aMinV);
			else
				aMul = 1;
			mIm.ramp(aMinV, aMul);
		}

		mImGlob.SetDyn(aMul);
		mImGlob.SetMaxValue( aMaxT-1 );
	}
	else{
		Type aMinV, aMaxV;
		mIm.getMinMax(aMinV, aMaxV);

		#ifdef __DEBUG_DIGEO_NORMALIZE_FLOAT_OCTAVE
			Type aMul;
			if (mAppli.Params().ValMaxForDyn().IsInit())
			{
				aMul = Type(1) / (Type)(mAppli.Params().ValMaxForDyn().Val());
				mIm.multiply(aMul);
			}
			else
			{
				aMul = Type(1) / (aMaxV - aMinV);
				mIm.ramp(aMinV, aMul);
			}
			mImGlob.SetDyn(aMul);
			mImGlob.SetMaxValue(1);
		#else
			mImGlob.SetDyn(1);
			mImGlob.SetMaxValue((REAL8)aMaxV);
		#endif
	}

	const cTypePyramide & aTP = mAppli.Params().TypePyramide();
	if ( aTP.PyramideGaussienne().IsInit() )
	{
		const double aSigmD = mImGlob.InitialDeltaSigma();
		if ( aSigmD!=0. )
		{
			//~ Im1D<tBase,tBase> aIKerD = ImGaussianKernel(aSigmD);
			//~ SetConvolSepXY( true, aSigmD, *this, aIKerD, mNbShift );
			mAppli.convolve(mIm,aSigmD,mIm);
		}
	}
}

template <class Type> Im2DGen cTplImInMem<Type>::Im(){ return TIm(); }

template <class Type> typename cTplImInMem<Type>::tBase * cTplImInMem<Type>::DoG(){ return ( mDoG.size()==0?NULL:mDoG.data() ); }

/*
template <class Type>  void  cTplImInMem<Type>::SetOrigOct(cTplImInMem<Type> * anOrig)
{
    mOrigOct = anOrig;
}
*/

template <class Type>  void  cTplImInMem<Type>::SetMereSameDZ(cTplImInMem<Type> * aTMere)
{
    
/*
    ELISE_ASSERT((mType==aMere->TypeEl()),"cImInMem::SetMere type mismatch");
    tTImMem * aTMere = static_cast<tTImMem *>(aMere);
*/

    ELISE_ASSERT((mTMere==0) && ((aTMere==0)||(aTMere->mTFille==0)),"cImInMem::SetMere");
    mTMere = aTMere;
    if (aTMere)
       aTMere->mTFille = this;
}



template <class Type>  double  cTplImInMem<Type>::CalcGrad2Moy()
{
    double aRes = 0;
    for (int anY = 0 ; anY< mSz.y-1 ; anY++)
    {
        Type * aL0 = mData[anY];
        Type * aL1 = mData[anY+1];
        for (int anX = 0 ; anX< mSz.x-1 ; anX++)
        {
            aRes += ElSquare(double(aL0[anX]-aL0[anX+1]))+ElSquare(double(aL0[anX]-aL1[anX]));
        }
    }

    return aRes/((mSz.y-1)*(mSz.x-1));
}


template <class Type>
void cTplImInMem<Type>::computeDoG( const cTplImInMem<Type> &i_nextScale )
{
    mDoG.resize( mSz.x*mSz.y );
    tBase *itDog = mDoG.data();
    Type  *itCurrentScale,
          *itNextScale;
    int x, y;
    for ( y=0 ; y<mSz.y; y++ )
    {
        itCurrentScale = mData[y];
        itNextScale = i_nextScale.mData[y];
        x = mSz.x;
        //while ( x-- ) ( *itDog++ )=(tBase)( *itCurrentScale++ )-(tBase)( *itNextScale++ );
        while ( x-- ) ( *itDog++ )=(tBase)( *itNextScale++ )-(tBase)( *itCurrentScale++ );
    }
}

template <class Type>
void cTplImInMem<Type>::saveGaussian()
{
	string filename = getValue_iTile_dz_iLevel( mAppli.tiledOutputGaussianExpression() );

	if ( mAppli.doRawTestOutput() )
	{
		MultiChannel<Type> channels;
		channels.link(mIm);
		channels.write_raw(filename);
	}
	else
	{
		Im2D<U_INT1,INT> outImg( mIm.tx(), mIm.ty() );
		ELISE_COPY( mIm.all_pts(), round_ni(mIm.in()/mImGlob.GetDyn()), outImg.out() );
		save_tiff( filename, outImg );
	}
}


/****************************************/
/*                                      */
/*           from cConvolSpec.cpp       */
/*                                      */
/****************************************/

string cImInMem::getValue_iTile_dz_iLevel( const Expression &e, int iLevelOffset ) const
{
	return mAppli.getValue_iTile_dz_iLevel( e, mAppli.currentBoxIndex(), mOct.Niv(), mKInOct+iLevelOffset );
}

string cImInMem::getValue_dz_iLevel( const Expression &e, int iLevelOffset ) const
{
	return mAppli.getValue_dz_iLevel( e, mOct.Niv(), mKInOct+iLevelOffset );
}

#include "Digeo_GaussFilter.cpp"
#include "Digeo_Detecteurs.cpp"
#include "Digeo_Pyram.cpp"

// Force instanciation

InstantiateClassTplDigeo(cTplImInMem)



/****************************************/
/*                                      */
/*             cImInMem                 */
/*                                      */
/****************************************/

cImInMem::cImInMem
(
       cImDigeo & aIGlob,
       const Pt2di & aSz,
       GenIm::type_el aType,
       cOctaveDigeo & anOct,
       double aResolOctaveBase,
       int    aKInOct,
       int    IndexSigma
) :
    mAppli            (aIGlob.Appli()),
    mImGlob           (aIGlob),
    mOct              (anOct),
    mSz               (aSz),
    mType             (aType),
    mResolGlob        (anOct.Niv()),
    mResolOctaveBase  (aResolOctaveBase),
    mKInOct           (aKInOct),
    mIndexSigma       (IndexSigma),
    mMere             (0),
    mFille            (0),
    mOrientateTime    (0.),
    mDescribeTime     (0.),
    mMergeTilesTime   (0.),
    mKernelTot        (1,1.0),
    mFirstSauv        (true),
    mFileTheoricalMaxValue( (1<</*aIGlob.TifF().bitpp()*/aIGlob.bitpp())-1 ),
    mUsed_points_map(NULL)
{
    if ( aIGlob.Appli().isVerbose() )
    {
        cout << "\tgaussian of index " << aKInOct << endl;
        cout << "\t\tresolution in octave = " << mResolOctaveBase << endl;
        cout << "\t\tglobal resolution    = " << mResolOctaveBase*anOct.Niv()*mImGlob.Resol() << endl;
    }
}

void  cImInMem::ResizeOctave(const Pt2di & aSz)
{
   mOct.ResizeAllImages(aSz);
}


double cImInMem::ScaleInOct() const
{
   return mResolOctaveBase;
}

double cImInMem::ScaleInit() const
{
    return ScaleInOct() * mOct.Niv();
}

int cImInMem::level() const { return mKInOct; }

void cImInMem::SetMere(cImInMem * aMere)
{
    ELISE_ASSERT((mMere==0) && (aMere->mFille==0),"cImInMem::SetMere");
    mMere = aMere;
    aMere->mFille = this;
}



void cImInMem::SauvIm(const std::string & aAdd)
{    
   if ( ! mAppli.Params().SauvPyram().IsInit()) return;

   const cTypePyramide & aTP = mAppli.Params().TypePyramide();
   cSauvPyram aSP = mAppli.Params().SauvPyram().Val();
   std::string aDir =  /*mAppli.DC() + */aSP.Dir().Val();
   ELISE_fp::MkDirSvp(aDir);

   std::string aNRes = ToString(mResolGlob);
   if (aTP.PyramideGaussienne().IsInit())
   {
       aNRes = aNRes + "_Sigma" + ToString(round_ni(100*mResolOctaveBase));
   }

   std::string aName =     aDir
                         + aAdd
                         + mAppli.ICNM()->Assoc1To2
                           (
                                aSP.Key().Val(),
                                mImGlob.Basename(),
                                aNRes,
                                true
                           );


   if (mFirstSauv)
   {
      L_Arg_Opt_Tiff aLArgTiff = Tiff_Im::Empty_ARG;
      int aStrip = aSP.StripTifFile().Val();
      if (aStrip>0)
          aLArgTiff = aLArgTiff +  Arg_Tiff(Tiff_Im::AStrip(aStrip));

      Pt2di aSz = mOct.BoxImCalc().sz();

      bool isCreate;

      Tiff_Im aTF = Tiff_Im::CreateIfNeeded
              (
                  isCreate,
                  aName.c_str(),
                  aSz,
                  (aSP.Force8B().Val() ? GenIm::u_int1 : Im().TypeEl()),
                  Tiff_Im::No_Compr,
                  Tiff_Im::BlackIsZero,
                  aLArgTiff
              );
       // std::cout << "CREATE " << aName << " " << isCreate << "\n";
   }
   Tiff_Im aTF(aName.c_str());

   const Box2di & aBoxOut = mOct.BoxCurOut();
   const Pt2dr  aP0In = mOct.BoxCurIn()._p0;
   Pt2di  aP0Glob = mOct.BoxImCalc()._p0;

   Fonc_Num aFonc = Im().in_proj()[Virgule(FX+aP0Glob.x-aP0In.x,FY+aP0Glob.y-aP0In.y)];
   aFonc = aSP.Force8B().Val() ? Min(255,aFonc*aSP.Dyn().Val()) : aFonc;

   ELISE_COPY
   (
       rectangle(aBoxOut._p0-aP0Glob,aBoxOut._p1-aP0Glob),
       aFonc,
       aTF.out()
   );
/*
      if (aSP.Force8B().Val())
      {
          Tiff_Im::Create8BFromFonc(aName,mSz,Min(255,Im().in()*aSP.Dyn().Val()));
      }
      else
      {
          Tiff_Im::CreateFromIm(Im(),aName,aLArgTiff);
      }
*/
   mFirstSauv = false;
}

    // ACCESSOR 

vector<cPtsCaracDigeo>       & cImInMem::featurePoints()       { return mFeaturePoints; }
const vector<cPtsCaracDigeo> & cImInMem::featurePoints() const { return mFeaturePoints; }

vector<DigeoPoint>       & cImInMem::orientedPoints()       { return mOrientedPoints; }
const vector<DigeoPoint> & cImInMem::orientedPoints() const { return mOrientedPoints; }

GenIm::type_el  cImInMem::TypeEl() const { return mType; }
Pt2di cImInMem::Sz() const {return mSz;}
int cImInMem::RGlob() const {return mResolGlob;}
double cImInMem::ROct() const {return mResolOctaveBase;}
cImInMem *  cImInMem::Mere() {return mMere;}
cOctaveDigeo &  cImInMem::Oct() {return mOct;}

template <class tData, class tComp>
void setBorder( Im2D<tData,tComp> &i_image, tData i_value )
{
	if ( i_image.tx()<1 || i_image.ty()<1 ) return;

	tData *data = i_image.data_lin();

	// set first line
	for ( int i=0; i<i_image.tx(); i++ )
		data[i] = i_value;
	// set last line
	memcpy( data+i_image.tx()*(i_image.ty()-1), data, i_image.tx()*sizeof(tData) );

	// set first and last columns
	const int offsetLast = i_image.tx()-1;
	for ( int i=0; i<i_image.ty(); i++ )
	{
		tData *line = i_image.data()[i];
		line[0] = line[offsetLast] = i_value;
	}
}

template <class tData, class tComp>
void gradient( const Im2D<tData,tComp> &i_image, REAL8 i_maxValue, Im2D<REAL4,REAL8> &o_gradient )
{
    o_gradient.Resize( Pt2di( i_image.tx()*2, i_image.ty() ) );

    const REAL8 coef = REAL8(0.5)/i_maxValue;
    const int c1 = -i_image.sz().x;
    int offset = i_image.sz().x+1;
    const tData *src = i_image.data_lin()+offset;
    REAL4 *dst = o_gradient.data_lin()+2*offset;
    REAL8 gx, gy, theta;
    int width_2 = i_image.sz().x-2,
        y = i_image.sz().y-2,
        x;
    while ( y-- )
    {
        x = width_2;
        while ( x-- )
        {
            gx = ( REAL8 )( coef*( REAL8(src[1])-REAL8(src[-1]) ) );
            gy = ( REAL8 )( coef*( REAL8(src[i_image.sz().x])-REAL8(src[c1]) ) );
            dst[0] = (REAL4)std::sqrt( gx*gx+gy*gy );

            theta = std::fmod( REAL8( std::atan2( gy, gx ) ), REAL8( 2*M_PI ) );
            if ( theta<0 ) theta+=2*M_PI;
            dst[1] = (REAL4)theta;

            src++; dst+=2;
        }
        src+=2; dst+=4;
    }
}

template void gradient<REAL4,REAL8>( const Im2D<REAL4,REAL8> &i_image, REAL8 i_maxValue, Im2D<REAL4,REAL8> &o_gradient );
template void gradient<U_INT2,INT>( const Im2D<U_INT2,INT> &i_image, REAL8 i_maxValue, Im2D<REAL4,REAL8> &o_gradient );
template void gradient<U_INT1,INT>( const Im2D<U_INT1,INT> &i_image, REAL8 i_maxValue, Im2D<REAL4,REAL8> &o_gradient );


template <class Type>
const Im2D<REAL4,REAL8> & cTplImInMem<Type>::getGradient() { return mImGlob.getGradient( TIm(), mOct.GetMaxValue() ); }

// return the number of possible orientations (at most DIGEO_NB_MAX_ANGLES)
int orientate( const Im2D<REAL4,REAL8> &i_gradient, const cPtsCaracDigeo &i_p, REAL8 o_angles[DIGEO_MAX_NB_ANGLES] )
{
	static REAL8 histo[DIGEO_ORIENTATION_NB_BINS];

	int xi = ((int) (i_p.mPt.x+0.5)) ;
	int yi = ((int) (i_p.mPt.y+0.5)) ;
	const REAL8 sigmaw = DIGEO_ORIENTATION_WINDOW_FACTOR*i_p.mLocalScale;
	const int W = (int)ceil( 3*sigmaw );

    //std::cout << "kp: scale " << i_p.mScale  << " localscale: " << i_p.mLocalScale <<  ", sigmaw " << sigmaw << " W " <<W<< "\n";


	// fill the SIFT histogram
	const INT width  = i_gradient.sz().x/2,
	          height = i_gradient.sz().y;
    REAL8 dx, dy, r2,
          wgt, mod, ang;
    int   offset;
    const REAL4 *p = i_gradient.data_lin()+( xi+yi*width )*2;

    std::fill( histo, histo+DIGEO_ORIENTATION_NB_BINS, 0 );
    for ( int ys=std::max( -W, 1-yi ); ys<=std::min( W, height-2-yi ); ys++ )
    {
        for ( int xs=std::max( -W, 1-xi ); xs<=std::min( W, width-2-xi ); xs++ )
        {
            dx = xi+xs-i_p.mPt.x;
            dy = yi+ys-i_p.mPt.y;
            r2 = dx*dx+dy*dy;

            // limit to a circular window
            if ( r2>=W*W+0.5 ) continue;

            wgt    = ::exp( -r2/( 2*sigmaw*sigmaw ) );
            offset = ( xs+ys*width )*2;
            mod    = p[offset];
            ang    = p[offset+1];

            int bin = (int)floor( DIGEO_ORIENTATION_NB_BINS*ang/( 2*M_PI ) );
            histo[bin] += mod*wgt;
        }
    }

    REAL8 prev;
    // smooth histogram
    // mean of a bin and its two neighbour values (x6)
    REAL8 *itHisto,
           first, mean;
    int iHisto,
        iIter = 6;
    while ( iIter-- )
    {
        itHisto = histo;
        iHisto  = DIGEO_ORIENTATION_NB_BINS-2;
        first = prev = *itHisto;
        *itHisto = ( histo[DIGEO_ORIENTATION_NB_BINS-1]+( *itHisto )+itHisto[1] )/3.; itHisto++;
        while ( iHisto-- )
        {
            mean = ( prev+(*itHisto)+itHisto[1] )/3.;
            prev = *itHisto;
            *itHisto++ = mean;
        }
        *itHisto = ( prev+( *itHisto )+first )/3.; itHisto++;
    }

    // find histogram's peaks
    // peaks are values > 80% of histoMax and > to both its neighbours
    REAL8 histoMax = 0.8*( *std::max_element( histo, histo+DIGEO_ORIENTATION_NB_BINS ) ),
          v, next, di;
    int nbAngles = 0;
    for ( int i=0; i<DIGEO_ORIENTATION_NB_BINS; i++ )
    {
        v = histo[i];
        prev = histo[ ( i==0 )?DIGEO_ORIENTATION_NB_BINS-1:i-1 ];
        next = histo[ ( i==( DIGEO_ORIENTATION_NB_BINS-1 ) )?0:i+1 ];
        if ( ( v>histoMax ) && ( v>prev ) && ( v>next ) )
        {
            // we found a peak
            // compute angle by quadratic interpolation
            di = -0.5*( next-prev )/( next+prev-2*v );
            o_angles[nbAngles++] = 2*M_PI*( i+di+0.5 )/DIGEO_ORIENTATION_NB_BINS;
            if ( nbAngles==DIGEO_MAX_NB_ANGLES ) return DIGEO_MAX_NB_ANGLES;
        }
    }
    return nbAngles;
}

void cImInMem::orientate()
{
	mOrientedPoints.clear();
	if ( !mAppli.doForceGradientComputation() && mFeaturePoints.size()==0 )
		return;

	const Im2D<REAL4,REAL8> &srcGradient = getGradient();

	mAppli.times()->start();

	mOrientedPoints.resize( mFeaturePoints.size() );
	REAL8 angles[DIGEO_MAX_NB_ANGLES];
	int nbAngles;
	size_t nbSkipped = 0;

	const cPtsCaracDigeo *itSrc = mFeaturePoints.data();
	DigeoPoint *itDst = mOrientedPoints.data();
	size_t iFeature = mFeaturePoints.size();
	while ( iFeature-- ){
		const cPtsCaracDigeo &srcPoint = *itSrc++;
		nbAngles = ::orientate( srcGradient, srcPoint, angles );
		if ( nbAngles!=0 )
		{
			DigeoPoint &dstPoint = *itDst++;
			dstPoint.x = srcPoint.mPt.x;
			dstPoint.y = srcPoint.mPt.y;
			dstPoint.scale = srcPoint.mScale;
			switch ( srcPoint.mType )
			{
			case eSiftMaxDog: dstPoint.type=DigeoPoint::DETECT_LOCAL_MAX; break;
			case eSiftMinDog: dstPoint.type=DigeoPoint::DETECT_LOCAL_MIN; break;
			default: dstPoint.type=DigeoPoint::DETECT_UNKNOWN; break;
			}

			dstPoint.entries.resize( (size_t)nbAngles );
			for ( int iAngle=0; iAngle<nbAngles; iAngle++ )
				dstPoint.entries[iAngle].angle = angles[iAngle];
		}
		else
			nbSkipped++;
	}
	mOrientedPoints.resize( mOrientedPoints.size()-nbSkipped );

	mAppli.times()->stop("orientate");
}

#define atd(dbinx,dbiny,dbint) *(dp + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

// o_descritpor must be of size DIGEO_DESCRIPTOR_SIZE
void describe( const Im2D<REAL4,REAL8> &i_gradient, REAL8 i_x, REAL8 i_y, REAL8 i_localScale, REAL8 i_angle, REAL8 *o_descriptor )
{
	REAL8 st0 = sinf( i_angle ),
	      ct0 = cosf( i_angle );

	int xi = int( i_x+0.5 );
	int yi = int( i_y+0.5 );

	const REAL8 SBP = DIGEO_DESCRIBE_MAGNIFY*i_localScale;
	const int  W   = (int)ceil( sqrt( 2.0 )*SBP*( DIGEO_DESCRIBE_NBP+1 )/2.0+0.5 );

	/* Offsets to move in the descriptor. */
	/* Use Lowe's convention. */
	const int binto = 1 ;
	const int binyo = DIGEO_DESCRIBE_NBO*DIGEO_DESCRIBE_NBP;
	const int binxo = DIGEO_DESCRIBE_NBO;

	std::fill( o_descriptor, o_descriptor+DIGEO_DESCRIPTOR_SIZE, 0 ) ;

    /* Center the scale space and the descriptor on the current keypoint.
    * Note that dpt is pointing to the bin of center (SBP/2,SBP/2,0).
    */
	const INT width  = i_gradient.sz().x/2,
	      height = i_gradient.sz().y;
	const REAL4 *p = i_gradient.data_lin()+( xi+yi*width )*2;
	REAL8 *dp = o_descriptor+( DIGEO_DESCRIBE_NBP/2 )*( binyo+binxo );

	#define atd(dbinx,dbiny,dbint) *(dp + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

	/*
	* Process pixels in the intersection of the image rectangle
	* (1,1)-(M-1,N-1) and the keypoint bounding box.
	*/
	const REAL8 wsigma = DIGEO_DESCRIBE_NBP/2 ;
	int  offset;
	REAL8 mod, angle, theta,
	      dx, dy,
	      nx, ny, nt;
    int  binx, biny, bint;
    REAL8 rbinx, rbiny, rbint;
    int dbinx, dbiny, dbint;
    REAL weight, win;
    for ( int dyi=std::max( -W, 1-yi ); dyi<=std::min( W, height-2-yi ); dyi++ )
    {
        for ( int dxi=std::max( -W, 1-xi ); dxi<=std::min( W, width-2-xi ); dxi++ )
        {
            // retrieve
            offset = ( dxi+dyi*width )*2;
            mod    = p[ offset ];
            angle  = p[ offset+1 ];

            theta  = -angle+i_angle;
            if ( theta>=0 )
                theta = std::fmod( theta, REAL8( 2*M_PI ) );
            else
                theta = 2*M_PI+std::fmod( theta, REAL8( 2*M_PI ) );

            // fractional displacement
            dx = xi+dxi-i_x;
            dy = yi+dyi-i_y;

            // get the displacement normalized w.r.t. the keypoint
            // orientation and extension.
            nx = ( ct0*dx + st0*dy )/SBP ;
            ny = ( -st0*dx + ct0*dy )/SBP ;
            nt = DIGEO_DESCRIBE_NBO*theta/( 2*M_PI ) ;

            // Get the gaussian weight of the sample. The gaussian window
            // has a standard deviation equal to NBP/2. Note that dx and dy
            // are in the normalized frame, so that -NBP/2 <= dx <= NBP/2.
             win = std::exp( -( nx*nx+ny*ny )/( 2.0*wsigma*wsigma ) );

            // The sample will be distributed in 8 adjacent bins.
            // We start from the ``lower-left'' bin.
            binx = std::floor( nx-0.5 );
            biny = std::floor( ny-0.5 );
            bint = std::floor( nt );
            rbinx = nx-( binx+0.5 );
            rbiny = ny-( biny+0.5 );
            rbint = nt-bint;

            // Distribute the current sample into the 8 adjacent bins
            for ( dbinx=0; dbinx<2; dbinx++ )
            {
                for ( dbiny=0; dbiny<2; dbiny++ )
                {
                    for ( dbint=0; dbint<2; dbint++ )
                    {
                        if ( ( ( binx+dbinx ) >= ( -(DIGEO_DESCRIBE_NBP/2)   ) ) &&
                             ( ( binx+dbinx ) <  ( DIGEO_DESCRIBE_NBP/2      ) ) &&
                             ( ( biny+dbiny ) >= ( -( DIGEO_DESCRIBE_NBP/2 ) ) ) &&
                             ( ( biny+dbiny ) <  ( DIGEO_DESCRIBE_NBP/2      ) ) )
                        {
                            weight = win*mod
                                    *std::fabs( 1-dbinx-rbinx )
                                    *std::fabs( 1-dbiny-rbiny )
                                    *std::fabs( 1-dbint-rbint );

                            atd( binx+dbinx, biny+dbiny, ( bint+dbint )%DIGEO_DESCRIBE_NBO ) += weight ;
                        }
                    }
                }
            }
        }
    }
}

void normalizeDescriptor( REAL8 *io_descriptor )
{
    REAL8 norm    = 0;
    int   i       = DIGEO_DESCRIPTOR_SIZE;
    REAL8 *itDesc = io_descriptor;
    while ( i-- ){
        norm += ( *itDesc )*( *itDesc );
        itDesc++;
    }
    
    norm = std::sqrt( norm )+std::numeric_limits<REAL8>::epsilon();

    i      = DIGEO_DESCRIPTOR_SIZE;
    itDesc = io_descriptor;
    while ( i-- ){
        *itDesc = ( *itDesc )/norm;
        itDesc++;
    }
}

void truncateDescriptor( REAL8 *io_descriptor )
{
    int    i      = DIGEO_DESCRIPTOR_SIZE;
    REAL8 *itDesc = io_descriptor;
    while ( i-- ){
        if ( ( *itDesc )>DIGEO_DESCRIBE_THRESHOLD )
            ( *itDesc )=DIGEO_DESCRIBE_THRESHOLD;
        itDesc++;
    }
}

void normalize_and_truncate( REAL8 *io_descriptor )
{
	normalizeDescriptor( io_descriptor );
	truncateDescriptor( io_descriptor );
	normalizeDescriptor( io_descriptor );
}

void cImInMem::describe()
{
	if ( !mAppli.doForceGradientComputation() && mOrientedPoints.size()==0 ) return;

	const Im2D<REAL4,REAL8> &srcGradient = getGradient();

	mAppli.times()->start();

	double octaveTrueSamplingPace = mOct.trueSamplingPace();
	DigeoPoint *itPoint = mOrientedPoints.data();
	size_t iPoint = mOrientedPoints.size();
	while ( iPoint-- )
	{
		DigeoPoint &p = *itPoint++;
		for ( size_t iAngle=0; iAngle<p.entries.size(); iAngle++ ){
			DigeoPoint::Entry &entry = p.entry(iAngle);
			::describe( srcGradient, p.x, p.y, p.scale/octaveTrueSamplingPace, entry.angle, entry.descriptor );
			normalize_and_truncate( entry.descriptor );
		}

		// prepare points for outputting
		p.x *= octaveTrueSamplingPace;
		p.y *= octaveTrueSamplingPace;
		p.scale = ScaleInit();
	}

	mAppli.times()->stop("describe");
}

static void __inconsistent_nb_channels_msg( const string &i_filename, int i_nbChannels, int i_expectedNbChannels )
{
	if ( i_nbChannels==i_expectedNbChannels ) return;
	stringstream ss;
	ss << "cImInMem::mergeTiles: inconsistent number of channels for file [" << i_filename << "] : " << i_nbChannels << " channels, expecting " << i_expectedNbChannels;
	ELISE_ASSERT( false, ss.str().c_str() );
}

void cImInMem::mergeTiles( const Expression &i_inputExpression, const cDecoupageInterv2D &i_tiles, const Expression &i_outputExpression, int i_iLevelOffset )
{
	ElTimer chrono;

	// grid has grid_nx columns and grid_ny lines of images, grid_n = grid_nx*grid_ny
	const int grid_n  = i_tiles.NbInterv(),
	          grid_nx = i_tiles.NbX(),
	          grid_ny = grid_n/grid_nx;

	if ( grid_n==0 ) return;

	// retrieve all filenames
	vector<string> filenames(grid_n);
	for ( int iTile=0; iTile<grid_n; iTile++ )
		filenames[iTile] = mAppli.getValue_iTile_dz_iLevel( i_inputExpression, iTile, mOct.Niv(), mKInOct+i_iLevelOffset );

	// retrieve full image size
	int fullWidth, fullHeight, nbChannels;
	{
		Tiff_Im tiff( filenames[0].c_str() );
		nbChannels = tiff.nb_chan();
		fullWidth  = tiff.sz().x;
		fullHeight = tiff.sz().y;
	}
	for ( int iTile=1; iTile<grid_nx; iTile++ )
	{
		Tiff_Im tiff( filenames[iTile].c_str() );
		__inconsistent_nb_channels_msg( filenames[iTile], tiff.nb_chan(), nbChannels );
		fullWidth += tiff.sz().x;
	}
	for ( int yTile=1; yTile<grid_ny; yTile++ )
	{
		const int iTile = yTile*grid_nx; 
		Tiff_Im tiff( filenames[iTile].c_str() );
		__inconsistent_nb_channels_msg( filenames[iTile], tiff.nb_chan(), nbChannels );
		fullHeight += tiff.sz().y;
	}

	Tiff_Im fullTiff(
				mAppli.getValue_iTile_dz_iLevel( i_outputExpression, 0, mOct.Niv(), mKInOct+i_iLevelOffset ).c_str(),
				Pt2di(fullWidth,fullHeight),
				GenIm::u_int1,
				Tiff_Im::No_Compr,
				nbChannels==1 ? Tiff_Im::BlackIsZero : Tiff_Im::RGB,
				Tiff_Im::Empty_ARG );
	int offsetY = 0;
	int h = 0;
	for ( int iTileY=0; iTileY<grid_ny; iTileY++ )
	{
		int offsetX = 0;
		for ( int iTileX=0; iTileX<grid_nx; iTileX++ )
		{
			const string &tileFilename = filenames[iTileX+iTileY*grid_nx];
			Tiff_Im tiff( tileFilename.c_str() );

			__inconsistent_nb_channels_msg( tileFilename, tiff.nb_chan(), nbChannels );

			ELISE_COPY
			(
				rectangle( Pt2di(offsetX,offsetY), Pt2di(offsetX,offsetY)+tiff.sz() ),
				trans( tiff.in(), Pt2di(-offsetX,-offsetY) ),
				fullTiff.out()
			);
			offsetX += tiff.sz().x;
			h = tiff.sz().y;

			if ( mAppli.doSuppressTiledOutputs() ) ELISE_fp::RmFile( tileFilename );
		}
		offsetY += h;
	}

	mMergeTilesTime += chrono.uval();
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
Footer-MicMac-eLiSe-25/06/2007*/
