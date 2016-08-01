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

#include "../../uti_phgrm/MICMAC/Jp2ImageLoader.h"

Video_Win * aW1Digeo;
// Video_Win * aW2Digeo;
// Video_Win * aW3Digeo;
// Video_Win * aW4Digeo;
Video_Win * aW5Digeo;

void  calc_norm_grad
      (
            double ** out,
            double *** in,
            const Simple_OPBuf_Gen & arg
      )
{
    Tjs_El_User.ElAssert
    (
          arg.dim_in() == 1,
          EEM0 << "calc_norm_grad requires dim out = 1 for func"
    );

   double * l0 = in[0][0];
   double * l1 = in[0][1];

   double * g2 = out[0];

   for (INT x=arg.x0() ;  x<arg.x1() ; x++)
   {
       g2[x] = ElSquare(l0[x+1]-l0[x]) + ElSquare(l1[x]-l0[x]);
   }
}

Fonc_Num norm_grad(Fonc_Num f)
{
     return create_op_buf_simple_tpl
            (
                0,  // Nouvelle syntaxe
                calc_norm_grad,
                f,
                1,
                Box2di(Pt2di(0,0),Pt2di(1,1))
            );
}

cInterfImageAbs* cInterfImageAbs::create( std::string const &aName, unsigned int aMaxLoadAll )
{
#if defined (__USE_JP2__)
	//on recupere l'extension
	int placePoint = -1;
	for(int l = (int)(aName.size() - 1);(l>=0)&&(placePoint==-1);--l)
	{
		if (aName[l]=='.')
		{
			placePoint = l;
		}
	}
	std::string ext = std::string("");
	if (placePoint!=-1)
	{
		ext.assign(aName.begin()+placePoint+1,aName.end());
	}
	//std::cout << "Extension : "<<ext<<std::endl;
	// on teste l'extension
	if ((ext==std::string("jp2"))|| (ext==std::string("JP2")) || (ext==std::string("Jp2")))
	{
		return new cInterfImageLoader( aName, aMaxLoadAll );
	}
#endif
	return new cInterfImageTiff( aName, aMaxLoadAll );
}

cInterfImageTiff::cInterfImageTiff( std::string const &aName, unsigned int aMaxSizeForFullLoad ):
	mTiff( new Tiff_Im( Tiff_Im::StdConvGen( aName, 1, true ) ) ) // 1 = nb channels, true = keep 16 bits values
{
	ELISE_ASSERT( mTiff.get()!=NULL, "cInterfImageTiff::cInterfImageTiff: cannot load TIFF image" );

	Tiff_Im &tiff = *mTiff.get();
	const Pt2di sz = tiff.sz();

	if ( (unsigned int)sz.x*(unsigned int)sz.y<aMaxSizeForFullLoad )
	{
		mFullImage.reset( Ptr_D2alloc_im2d( tiff.type_el(), sz.x, sz.y ) );
		ELISE_COPY
		(
			mFullImage->all_pts(),
			mTiff->in(),
			mFullImage->out()
		);
	}
}

double cInterfImageTiff::Som()const
{
	double aSom;
	ELISE_COPY
	(
		mTiff->all_pts(),
		norm_grad( mFullImage.get()==NULL ? mTiff->in_proj() : mFullImage->in_proj() ),
		sigma(aSom)
	);
	return aSom;
}

Im2DGen cInterfImageAbs::getWindow( Pt2di P0, const Pt2di &windowSize, int askedMargin, int &o_marginX, int &o_marginY )
{
	#ifdef __DEBUG_DIGEO
		Pt2di _p1 = P0+windowSize;
DoNothingButRemoveWarningUnused(_p1);
		ELISE_DEBUG_ERROR( P0.x<0, "cInterfImageAbs::getWindow", "P0.x = " << P0.x << " <0" );
		ELISE_DEBUG_ERROR( _p1.x>sz().x, "cInterfImageAbs::getWindow", "_p1.x = " << _p1.x << " > i_src.sz().x = " << sz() );
		ELISE_DEBUG_ERROR( P0.y<0, "cInterfImageAbs::getWindow", "P0.y = " << P0.y << " <0" );
		ELISE_DEBUG_ERROR( _p1.y>sz().y, "cInterfImageAbs::getWindow", "_p1.y = " << _p1.y << " > i_src.sz().y = " << sz() );
	#endif

	Pt2di p0( P0.x-askedMargin, P0.y-askedMargin );
	Pt2di p1( P0.x+windowSize.x+askedMargin, P0.y+windowSize.y+askedMargin );
	o_marginX = o_marginY = askedMargin;
	if ( p0.x<0 ){ o_marginX+=p0.x; p0.x=0; }
	if ( p0.y<0 ){ o_marginY+=p0.y; p0.y=0; }
	if ( p1.x>sz().x ) p1.x = sz().x;
	if ( p1.y>sz().y ) p1.y = sz().y;

	return getWindow( p0, p1-p0 );
}

Im2DGen cInterfImageTiff::getWindow( Pt2di const &P0, Pt2di windowSize )
{
	Fonc_Num fn = mTiff->in_proj();
	// if image is loaded in memory, use it rather than the Tiff_Im
	if ( mFullImage.get()!=NULL ) fn = mFullImage->in_proj();

	mWindow.reset( Ptr_D2alloc_im2d( mTiff->type_el(), windowSize.x, windowSize.y ) );
	ELISE_COPY( mWindow->all_pts(), trans( fn, P0 ), mWindow->out() );

	return *mWindow;
}

cInterfImageLoader::cInterfImageLoader( std::string const &aName, unsigned int aMaxSizeForFullLoad )
{
	#ifdef __USE_JP2__
		mLoader.reset( new JP2ImageLoader(aName) );

		cInterfModuleImageLoader &loader = *mLoader.get();

		ELISE_ASSERT( loader.PreferedTypeOfResol(1)==eUnsignedChar, "cInterfImageLoader::cInterfImageLoader: only one-channel uint8 jpeg2000 images are handled" );

		const JP2ImageLoader::tPInt &sz = loader.Sz(1);
		if ( (unsigned int)( sz.real()*sz.imag() )>aMaxSizeForFullLoad ) return;

		/*
		// load all image in memory
		mFullImage.reset( new Im2D<REAL4,REAL>( sz.real(), sz.imag() ) );
		loader.LoadCanalCorrel(
			sLowLevelIm<float>(
				mFullImage->data_lin(),
				mFullImage->data(),
				sz ),
			1, //deZoom
			cInterfModuleImageLoader::tPInt(0,0), //aP0Im
			cInterfModuleImageLoader::tPInt(0,0), //aP0File
			sz );
		*/
		mFullImage.reset( new Im2D<U_INT1,INT>( sz.real(), sz.imag() ) );
		vector<sLowLevelIm<U_INT1> > lowLevelIms;
		lowLevelIms.push_back( sLowLevelIm<U_INT1>( mFullImage->data_lin(), mFullImage->data(), sz ) );
		loader.LoadNCanaux(
			lowLevelIms,
			0, // flags
			1, //deZoom
			cInterfModuleImageLoader::tPInt(0,0), //aP0Im
			cInterfModuleImageLoader::tPInt(0,0), //aP0File
			sz );
	#endif
	ELISE_ASSERT( false, "cInterfImageLoader::cInterfImageLoader: no JPEG2000 image loader available" );
}

int cInterfImageLoader::bitpp()const
{
	switch (mLoader->PreferedTypeOfResol(1)) {
		case eUnsignedChar:
			return 8;
		case eSignedShort:
		case eUnsignedShort:
			return 16;
		case eFloat:
			return 32;
		default:
			break;
	}
	return 0;
}

GenIm::type_el cInterfImageLoader::type_el()const
{
	switch (mLoader->PreferedTypeOfResol(1)) {
		case eUnsignedChar:
			return GenIm::u_int1;
		case eSignedShort:
			return GenIm::int2;
		case eUnsignedShort:
			return GenIm::u_int2;
		case eFloat:
			return GenIm::real4;
		default:
			break;
	}
	return GenIm::no_type;
}

/*
double cInterfImageLoader::Som()const
{
	double aSom=0;
	int dl = 1000;
	Im2D<float,double> buffer( sz().x, dl+1 );
	for(int l=0;l<sz().y;l+=dl)
	{
		mLoader->LoadCanalCorrel(
			sLowLevelIm<float>(
				buffer.data_lin(),
				buffer.data(),
				Elise2Std(buffer.sz()) ),
			1, //deZoom
			cInterfModuleImageLoader::tPInt(0,0), //aP0Im
			cInterfModuleImageLoader::tPInt(0,l), //aP0File
			cInterfModuleImageLoader::tPInt(buffer.sz().x,std::min(sz().y-l,buffer.sz().y)) );
		double aSomLin;
		Pt2di aSz = buffer.sz() - Pt2di(1,1);
		ELISE_COPY
		(
			rectangle(Pt2di(0,0),aSz),
			norm_grad(buffer.in_proj()),
			sigma(aSomLin)
		);
		aSom += aSomLin;
	}
	return aSom;
}
*/

double cInterfImageLoader::Som()const
{
	double aSom=0;
	int dl = 1000;
	Im2D<U_INT1,INT> buffer( sz().x, dl+1 );
	vector<sLowLevelIm<U_INT1> > lowLevelIms;
	lowLevelIms.push_back( sLowLevelIm<U_INT1>( buffer.data_lin(), buffer.data(), Elise2Std(buffer.sz()) ) );
	for(int l=0;l<sz().y;l+=dl)
	{
		int bufferHeight = std::min( sz().y-l, buffer.sz().y );
		if ( buffer.ty()!=bufferHeight ) buffer.Resize( Pt2di(buffer.tx(),bufferHeight) );
		mLoader->LoadNCanaux(
			lowLevelIms,
			0, // flags
			1, //deZoom
			cInterfModuleImageLoader::tPInt(0,0), //aP0Im
			cInterfModuleImageLoader::tPInt(0,l), //aP0File
			cInterfModuleImageLoader::tPInt( buffer.tx(), buffer.ty() ) );
		double aSomLin;
		Pt2di aSz = buffer.sz() - Pt2di(1,1);
		ELISE_COPY
		(
			rectangle(Pt2di(0,0),aSz),
			norm_grad(buffer.in_proj()),
			sigma(aSomLin)
		);
		aSom += aSomLin;
	}
	return aSom;
}

Im2DGen cInterfImageLoader::getWindow( Pt2di const &P0, Pt2di windowSize )
{
	mWindow.Resize(windowSize);

	if ( mFullImage.get()!=NULL )
		ELISE_COPY( mWindow.all_pts(), trans( mFullImage->in(), P0 ), mWindow.out() );
	else
	{
		// no image is loaded in memory, get window from file
		const cInterfModuleImageLoader::tPInt sz = Elise2Std(windowSize);
		vector<sLowLevelIm<U_INT1> > lowLevelIms;
		lowLevelIms.push_back( sLowLevelIm<U_INT1>( mWindow.data_lin(), mWindow.data(), sz ) );
		mLoader->LoadNCanaux(
			lowLevelIms,
			0, // flags
			1, //deZoom
			cInterfModuleImageLoader::tPInt(0,0), //aP0Im
			Elise2Std(P0), //aP0File
			sz );
	}

	// image is fully loaded in memory, use this image as source for the Fonc_Num
	return mWindow;
}

TIm2D<float,double>* cInterfImageLoader::cropReal4(Pt2di const &P0, Pt2di const &SzCrop)const
{
	std_unique_ptr<TIm2D<float,double> > anTIm2D(new TIm2D<float, double>(SzCrop));
	mLoader->LoadCanalCorrel(
		sLowLevelIm<float>(
			anTIm2D->_the_im.data_lin(),
			anTIm2D->_the_im.data(),
			Elise2Std(anTIm2D->sz()) ),
		1, //deZoom
		cInterfModuleImageLoader::tPInt(0,0), //aP0Im
		cInterfModuleImageLoader::tPInt(P0.x,P0.y), //aP0File
		cInterfModuleImageLoader::tPInt(SzCrop.x,SzCrop.y) );
	return anTIm2D.release();
}

/*
TIm2D<U_INT1,INT>* cInterfImageLoader::cropUInt1(Pt2di const &P0, Pt2di const &SzCrop)const
{
	std::auto_ptr<TIm2D<U_INT1,INT> > anTIm2D(new TIm2D<U_INT1,INT>(SzCrop));
	mLoader->LoadCanalCorrel(
		sLowLevelIm<U_INT1>(
			anTIm2D->_the_im.data_lin(),
			anTIm2D->_the_im.data(),
			Elise2Std(anTIm2D->sz()) ),
		1,//deZoom
		cInterfModuleImageLoader::tPInt(0,0),//aP0Im
		cInterfModuleImageLoader::tPInt(P0.x,P0.y),//aP0File
		cInterfModuleImageLoader::tPInt(SzCrop.x,SzCrop.y) );
	return anTIm2D.release();
}
*/

TIm2D<U_INT1,INT>* cInterfImageLoader::cropUInt1(Pt2di const &P0, Pt2di const &SzCrop)const
{
	std_unique_ptr<TIm2D<U_INT1,INT> > anTIm2D(new TIm2D<U_INT1,INT>(SzCrop));
	vector<sLowLevelIm<U_INT1> > lowLevelIms;
	lowLevelIms.push_back( sLowLevelIm<U_INT1>( anTIm2D->_the_im.data_lin(), anTIm2D->_the_im.data(), Elise2Std(anTIm2D->sz()) ) );
	mLoader->LoadNCanaux(
		lowLevelIms,
		0, // flags
		1, // deZoom
		cInterfModuleImageLoader::tPInt(0,0), // aP0Im
		cInterfModuleImageLoader::tPInt(P0.x,P0.y), // aP0File
		cInterfModuleImageLoader::tPInt(SzCrop.x,SzCrop.y) );
	return anTIm2D.release();
}



/****************************************/
/*                                      */
/*             cImDigeo                 */
/*                                      */
/****************************************/

cImDigeo::cImDigeo
(
   int                 aNum,
   const cImageDigeo & aIMD,
   const std::string & aName,
   cAppliDigeo &       anAppli
) :
  mFullname    (aName),
  mAppli       (anAppli),
  mIMD         (aIMD),
  mNum         (aNum),
  mInterfImage ( cInterfImageAbs::create( mFullname, anAppli.loadAllImageLimit() ) ),
  mResol       (aIMD.ResolInit().Val()),
  mSzGlobR1    ( mInterfImage->sz() ),
  mBoxGlobR1   ( Pt2di(0,0), mSzGlobR1 ),
  mBoxImR1     ( mBoxGlobR1 ),
  //mBoxImCalc   ( round_ni(Pt2dr(mBoxImR1._p0)/mResol), round_ni(Pt2dr(mBoxImR1._p1)/mResol) ),
  mBoxImCalc   ( Pt2dr(0.,0.), Pt2dr() ),
  mSzMax       (0,0),
  mG2MoyIsCalc (false),
  mDyn         (1.0),
  mFileInMem   (NULL),
  mSigma0      ( anAppli.Params().Sigma0().Val() ),
  mSigmaN      ( anAppli.Params().SigmaN().Val() ),
  mGradientSource(NULL)
{
	mBoxImCalc = Box2di( Pt2di(0,0), Pt2di( mInterfImage->sz().x, mInterfImage->sz().y ) );

	SplitDirAndFile( mDirectory, mBasename, mFullname );

    if ( Appli().isVerbose() )
    {
        cout << "resol0 : " << mResol << endl;
        cout << "sigmaN : " << SigmaN() << endl;
        cout << "sigma0 : " << Sigma0() << endl;
    }

   mInterfImage->bitpp();
   mFileInMem = mInterfImage->fullImage();

	if ( mFileInMem!=NULL )
	{
		mG2MoyIsCalc= true;
		mGradMoy = sqrt(mFileInMem->MoyG2());
	}
	else
	{
		Pt2di aSz = mInterfImage->sz() - Pt2di(1,1);
		double aSom = mInterfImage->Som();
		aSom /= aSz.x * double(aSz.y);
		mG2MoyIsCalc= true;
		mGradMoy = sqrt(aSom);
   }

   // compute gaussians' standard-deviation
	double sigmaN = SigmaN()*( 1./Resol() );
   mInitialDeltaSigma = sqrt( ElSquare(mSigma0)-ElSquare(sigmaN) );
   if ( mAppli.isVerbose() ) cout << "initial convolution sigma : " << mInitialDeltaSigma << ( mInitialDeltaSigma==0.?"(no convolution)":"" ) << endl;
}

double cImDigeo::Resol() const
{
   return mResol;
}


const Box2di & cImDigeo::BoxImCalc() const
{
   return mBoxImCalc;
}

const std::vector<cOctaveDigeo *> &   cImDigeo::Octaves() const
{
   return mOctaves;
}

double cImDigeo::Sigma0() const { return mSigma0; }

double cImDigeo::SigmaN() const { return mSigmaN; }

double cImDigeo::InitialDeltaSigma() const { return mInitialDeltaSigma; }

void cImDigeo::NotifUseBox(const Box2di & aBox) { mSzMax.SetSup( aBox.sz() ); }

void cImDigeo::AllocImages()
{
   Pt2di aSz = mSzMax;
   int aNivDZ = 0;

	const int lastPace = mAppli.lastDz();
   cOctaveDigeo * aLastOct = 0;
   int iOctave = 0;
   for ( int aDz = 1 ; aDz <=lastPace; aDz*=2 )
   {
       cOctaveDigeo * anOct =   aLastOct                                                   ?
                                aLastOct->AllocDown( mAppli.octaveType(iOctave),*this,aDz,aSz)       :
                                cOctaveDigeo::AllocTop( mAppli.octaveType(iOctave),*this,aDz,aSz)       ;
       mOctaves.push_back(anOct);
       const cTypePyramide & aTP = mAppli.Params().TypePyramide();
       if (aTP.NivPyramBasique().IsInit())
       {
            // mVIms.push_back(cImInMem::Alloc (*this,aSz,TypeOfDeZoom(aDz), *anOct, 1.0));
            // C'est l'image Bas qui servira
            //         mVIms.push_back(anOct->AllocIm(1.0,0));
             ELISE_ASSERT( false, "cImDigeo::AllocImages: PyramBasique not implemented" );
       }
       else if ( aTP.PyramideGaussienne().IsInit() )
       {
            const cPyramideGaussienne &  aPG = aTP.PyramideGaussienne().Val();
            int aNbIm = aPG.NbByOctave().Val();
            if (aPG.NbInLastOctave().IsInit() && (aDz*2>lastPace)) aNbIm = aPG.NbInLastOctave().Val();
            int aK0 = 0;
            if (aDz==1) aK0 = aPG.IndexFreqInFirstOctave().Val();
            anOct->SetNbImOri(aNbIm);

            if ( mAppli.isVerbose() )
            {
                cout << "octave " << mOctaves.size()-1 << " (" << eToString( mAppli.octaveType(iOctave) ) << ")" << endl;
                cout << "\tsampling pace    = " << aDz << endl;
                cout << "\tnumber of levels = " << aNbIm << endl;
            }

            for (int aK=aK0 ; aK< aNbIm+3 ; aK++)
            {
                double aSigma =  mSigma0*pow(2.0,aK/double(aNbIm));
                //mVIms.push_back(cImInMem::Alloc (*this,aSz,TypeOfDeZoom(aDz), *anOct,aSigma));
                mVIms.push_back((anOct->AllocIm(aSigma,aK,aNivDZ*aNbIm+(aK-aK0))));
            }
       }
       aSz = ( aSz+Pt2di(1,1) )/2;
       aNivDZ++;

       aLastOct = anOct;
       iOctave++;
   }

   for (int aK=1 ; aK<int(mVIms.size()) ; aK++)
      mVIms[aK]->SetMere(mVIms[aK-1]);
}

bool cImDigeo::PtResolCalcSauv(const Pt2dr & aP)
{
   return    (aP.x>=mBoxCurOut._p0.x)
          && (aP.x <mBoxCurOut._p1.x)
          && (aP.y>=mBoxCurOut._p0.y)
          && (aP.y <mBoxCurOut._p1.y) ;
}

void save_tiff( const string &i_filename, Im2DGen i_img, bool i_rgb )
{
	//if ( ELISE_fp::exist_file( i_filename ) ) ELISE_fp::RmFile( i_filename );
	ELISE_COPY
	(
		i_img.all_pts(),
		i_rgb ? Virgule(i_img.in(),i_img.in(),i_img.in()) : i_img.in(),
		Tiff_Im(
			i_filename.c_str(),
			i_img.sz(),
			GenIm::u_int1,
			Tiff_Im::No_Compr,
			i_rgb ? Tiff_Im::RGB : Tiff_Im::BlackIsZero,
			Tiff_Im::Empty_ARG ).out()
	);
}

void cImDigeo::LoadImageAndPyram(const Box2di & aBoxIn,const Box2di & aBoxOut)
{
	ELISE_DEBUG_ERROR( mInterfImage==NULL, "cImDigeo::LoadImageAndPyram", "mInterfImage==NULL" );
	ELISE_DEBUG_ERROR( mOctaves.size()==0, "cImDigeo::LoadImageAndPyram", "mOctaves.size()==0" );

    const cTypePyramide & aTP = mAppli.Params().TypePyramide();

    mBoxCurIn = aBoxIn;
    mBoxCurOut = Box2di( aBoxIn._p0/mResol, aBoxIn._p1/mResol );

    mAppli.times()->start();

    mSzCur = aBoxIn.sz();
    mP0Cur = aBoxIn._p0;

    Pt2di p0( mP0Cur.x, mP0Cur.y ),
          p1( p0.x+mSzCur.x, p0.y+mSzCur.y );

    int fullW = mInterfImage->sz().x, fullH = mInterfImage->sz().y;
    if ( p0.x<0 ) p0.x=0;
    if ( p0.y<0 ) p0.y=0;
    if ( p1.x>fullW ) p1.x=fullW;
    if ( p1.y>fullH ) p1.y=fullH;

    for ( size_t aK=0 ; aK<mOctaves.size(); aK++ )
       mOctaves[aK]->SetBoxInOut(aBoxIn,aBoxOut);

	Fonc_Num aF;
	if ( mResol==1. )
		aF = mInterfImage->getWindow( mP0Cur, mSzCur ).in_proj();
	else
	{
		ELISE_ASSERT( mResol<=1, "cImDigeo::LoadImageAndPyram: starting with an octave >1 is not handled yet" );

		int marginX = 0, marginY = 0;
		//Im2DGen window = mInterfImage->getWindow( mP0Cur, mSzCur, 1, marginX, marginY ); // 1 = askedMargin

		Im2DGen window = mInterfImage->getWindow( mP0Cur, mSzCur );
		aF = StdFoncChScale_Bilin( window.in_proj(), Pt2dr( (REAL)marginX,(REAL)marginY ), Pt2dr(mResol,mResol) );
	}

	//mOctaves[0]->FirstImage()->LoadFile(aF,aBoxIn,GenIm::u_int1/*mInterfImage->type_el()*/);
	mOctaves[0]->FirstImage()->LoadFile(aF,mBoxCurOut,GenIm::real4/*mInterfImage->type_el()*/);

	mAppli.times()->stop("tile loading");

	if ( mAppli.doSaveTiles() )
	{
		mAppli.times()->start();
		Im2DGen window = mInterfImage->getWindow( mP0Cur, mSzCur );
		string filename = mAppli.getValue_iTile( mAppli.tiledOutputExpression(), mAppli.currentBoxIndex() );

		if ( mAppli.doRawTestOutput() )
		{
			/*
			MultiChannel channels;
			channels.link(window);
			channels.duplicateLastChannel(2);
			channels.write_raw(filename);
			*/
		}
		else
			save_tiff( filename, window, true ); // true = rgb
		mAppli.times()->stop(DIGEO_TIME_OUTPUTS);
	}

	if ( !mAppli.doComputeCarac() ) return;

	mAppli.times()->start();

    for ( int aK=0 ; aK< int(mVIms.size()) ; aK++ )
    {
       if ( aK>0 )
       {
          if (aTP.NivPyramBasique().IsInit())
             mVIms[aK]->VMakeReduce_121( *(mVIms[aK-1]) );
          else if ( aTP.PyramideGaussienne().IsInit() )
             mVIms[aK]->ReduceGaussienne();
       }
       mVIms[aK]->SauvIm();
    }

    for (int aKOct=0 ; aKOct<int(mOctaves.size()) ; aKOct++)
        mOctaves[aKOct]->PostPyram();

    mAppli.times()->stop("pyramid computation");

	if ( mAppli.doSaveGaussians() ) saveGaussians();
}

void cImDigeo::DoCalcGradMoy(int aDZ)
{
   if (mG2MoyIsCalc)
      return;

   mG2MoyIsCalc = true;

   if (mAppli.MultiBloc())
   {
      ELISE_ASSERT(false,"DoCalcGradMoy : Multi Bloc a gerer");
   }

   ElTimer aChrono;
   mGradMoy = sqrt(GetOctOfDZ(aDZ).FirstImage()->CalcGrad2Moy());

   std::cout << "Grad = " << GradMoyCorrecDyn() <<  " Time =" << aChrono.uval() << "\n";
}


void cImDigeo::DoSiftExtract()
{
   ELISE_ASSERT(false,"cImDigeo::DoSiftExtract deprecated");    
}

cOctaveDigeo * cImDigeo::SVPGetOctOfDZ(int aDZ)
{
   for (int aK=0 ; aK<int(mOctaves.size()) ; aK++)
   {
      if (mOctaves[aK]->Niv() == aDZ)
      {
          return mOctaves[aK];
      }
   }
   return 0;
}

cOctaveDigeo & cImDigeo::GetOctOfDZ(int aDZ)
{
   cOctaveDigeo * aRes = SVPGetOctOfDZ(aDZ);

   ELISE_ASSERT(aRes!=0,"cAppliDigeo::GetOctOfDZ");

   return *aRes;
}


double cImDigeo::GetDyn() const
{
    return mDyn;
}

void cImDigeo::SetDyn(double aDyn)
{
    mDyn = aDyn;
}

REAL8 cImDigeo::GetMaxValue() const
{
    return mMaxValue;
}

void cImDigeo::SetMaxValue(REAL8 i_maxValue)
{
    mMaxValue = i_maxValue;
}

const Pt2di& cImDigeo::SzCur() const { return mSzCur; }
const Pt2di& cImDigeo::P0Cur() const { return mP0Cur; }

const std::string  & cImDigeo::Fullname() const { return mFullname; }
const std::string  & cImDigeo::Directory() const { return mDirectory; }
const std::string  & cImDigeo::Basename() const { return mBasename; }

cAppliDigeo &  cImDigeo::Appli() {return mAppli;}
const cImageDigeo &  cImDigeo::IMD() {return mIMD;}

double cImDigeo::GradMoyCorrecDyn() const 
{
   ELISE_ASSERT(mG2MoyIsCalc,"cImDigeo::G2Moy");
   return mGradMoy * mDyn;
}

void cImDigeo::detect()
{
	mAppli.times()->start();
	for ( size_t iOctave=0; iOctave<mOctaves.size(); iOctave++ )
		mOctaves[iOctave]->DoAllExtract();
	mAppli.times()->stop("point detection");
}

template <class DataType,class ComputeType>
const Im2D<REAL4,REAL8> & cImDigeo::getGradient( const Im2D<DataType,ComputeType> &i_src, REAL8 i_srcMaxValue )
{
	if ( mGradientSource!=(void*)i_src.data_lin() || mGradientMaxValue!=i_srcMaxValue  )
	{
		mAppli.times()->start();

		mGradientSource = (void*)i_src.data_lin();
		mGradientMaxValue = i_srcMaxValue;
		ElTimer chrono;
		gradient<DataType,ComputeType>( i_src, i_srcMaxValue, mGradient );
		mAppli.upNbComputedGradients();

		mAppli.times()->stop("gradient");
	}
	return mGradient;
}

template const Im2D<REAL4,REAL8> & cImDigeo::getGradient<REAL4,REAL8>( const Im2D<REAL4,REAL8> &i_image, REAL8 i_maxValue );
template const Im2D<REAL4,REAL8> & cImDigeo::getGradient<U_INT2,INT>( const Im2D<U_INT2,INT> &i_image, REAL8 i_maxValue );

void get_channel_min_max( const REAL4 *i_channel, int i_width, int i_height, int i_nbChannels, const int i_iChannel, REAL4 &o_min, REAL4 &o_max )
{
	unsigned int iPix = i_width*i_height;

	if ( iPix==0 ) return;

	const REAL4 *itPix = i_channel+i_iChannel;
	o_min = o_max = *itPix;
	while ( iPix-- )
	{
		REAL4 v = *itPix;
		if ( v<o_min ) o_min=v;
		if ( v>o_max ) o_max=v;
		itPix += i_nbChannels;
	}
}

void channel_to_Im2D( const REAL4 *i_channel, int i_width, int i_height, int i_nbChannels, const int i_iChannel, Im2D<REAL4,REAL> &o_im2d )
{
	o_im2d.Resize( Pt2di(i_width,i_height) );
	unsigned int iPix = i_width*i_height;

	if ( iPix<=0 ) return;

	const REAL4 *itSrc = i_channel+i_iChannel;
	REAL4 *itDst = o_im2d.data_lin();
	while ( iPix-- )
	{
		*itDst++ = *itSrc++;
		itSrc += i_nbChannels;
	}
}

void scale_min_max( REAL4 *i_channel, int i_width, int i_height, REAL4 i_min, REAL4 i_max )
{
	unsigned int iPix = i_width*i_height;

	if ( iPix<=0 ) return;

	REAL4 *it = i_channel;
	REAL4 scale = 255./(i_max-i_min);
	while ( iPix-- )
	{
		*it = scale*( (*it)-i_min );
		it++;
	}
}

void save_gradient_component_raw( const Im2D<REAL4, REAL8> &i_gradient, const int i_iComponent, const string &i_filename )
{
	Im2D<REAL4, REAL> imgToSave;
	channel_to_Im2D( i_gradient.data_lin(), i_gradient.tx()/2, i_gradient.ty(), 2, i_iComponent, imgToSave );
	// __MULTI_CHANNEL
	//imgToSave.write_raw(i_filename);
}

void save_gradient_component_tiff( const Im2D<REAL4, REAL8> &i_gradient, const int i_iComponent, const string &i_filename )
{
	REAL4 minv = 0., maxv = 0.;
	get_channel_min_max( i_gradient.data_lin(), i_gradient.tx()/2, i_gradient.ty(), 2, i_iComponent, minv, maxv );
	Im2D<REAL4, REAL> imgToSave;
	channel_to_Im2D( i_gradient.data_lin(), i_gradient.tx()/2, i_gradient.ty(), 2, i_iComponent, imgToSave );
	scale_min_max(  imgToSave.data_lin(), i_gradient.tx()/2, i_gradient.ty(), minv, maxv );

	ELISE_COPY
	(
		imgToSave.all_pts(),
		round_ni( imgToSave.in() ),
		Tiff_Im(
			i_filename.c_str(),
			imgToSave.sz(),
			GenIm::u_int1,
			Tiff_Im::No_Compr,
			Tiff_Im::BlackIsZero,
			Tiff_Im::Empty_ARG ).out()
	);
}

void cImDigeo::orientateAndDescribe()
{
	for ( size_t iOctave=0; iOctave<mOctaves.size(); iOctave++ )
	{
		vector<cImInMem*> &images = mOctaves[iOctave]->VIms();

		for ( size_t iLevel=1; iLevel<images.size()-2; iLevel++ )
		{
			cImInMem &image = *images[iLevel];

			#ifdef DIGEO_NO_ANGLE
				const cPtsCaracDigeo *itSrc = image.featurePoints().data();
				image.orientedPoints().resize(image.featurePoints().size());
				DigeoPoint *itDst = image.orientedPoints().data();
				size_t iFeature = image.featurePoints().size();
				while ( iFeature-- )
				{
					const cPtsCaracDigeo &srcPoint = *itSrc++;
					DigeoPoint &dstPoint = *itDst++;

					dstPoint.x = srcPoint.mPt.x;
					dstPoint.y = srcPoint.mPt.y;
					dstPoint.scale = srcPoint.mScale;

					dstPoint.entries.resize(1);
					dstPoint.entries[0].angle = 0.;
					switch (srcPoint.mType)
					{
					case eSiftMaxDog: dstPoint.type = DigeoPoint::DETECT_LOCAL_MAX; break;
					case eSiftMinDog: dstPoint.type = DigeoPoint::DETECT_LOCAL_MIN; break;
					default: dstPoint.type=DigeoPoint::DETECT_UNKNOWN; break;
					}
				}
			#else
				image.orientate();
			#endif
			image.describe();

			if ( mAppli.doSaveGradients() )
			{
				mAppli.times()->start();

				const Im2D<REAL4,REAL8> &gradient = image.getGradient();
				string filename_norm = image.getValue_iTile_dz_iLevel( mAppli.tiledOutputGradientNormExpression(), -1 ),
				       filename_angle = image.getValue_iTile_dz_iLevel( mAppli.tiledOutputGradientAngleExpression(), -1 );
				if ( mAppli.doRawTestOutput() )
				{
					save_gradient_component_raw( gradient, 0, filename_norm );
					save_gradient_component_raw( gradient, 1, filename_angle );
				}
				else
				{
					save_gradient_component_tiff( gradient, 0, filename_norm );
					save_gradient_component_tiff( gradient, 1, filename_angle );
				}

				mAppli.times()->stop(DIGEO_TIME_OUTPUTS);
			}
		}
	}
}

void cImDigeo::saveGaussians() const
{
	mAppli.times()->start();
	for ( size_t i=0; i<mVIms.size(); i++ )
		mVIms[i]->saveGaussian();
	mAppli.times()->stop(DIGEO_TIME_OUTPUTS);
}

size_t cImDigeo::addAllPoints( list<DigeoPoint> &o_allPoints ) const
{
	// copy all points from all cImInMem
	size_t nbAddedPoints = 0;
	for ( size_t i=0; i<mVIms.size(); i++ )
	{
		const vector<DigeoPoint> &points = mVIms[i]->orientedPoints();
		const DigeoPoint *itSrc = points.data();
		size_t iSrc = points.size();
		nbAddedPoints += iSrc;
		while ( iSrc-- ) o_allPoints.push_back( *itSrc++ );
	}
	return nbAddedPoints;
}

void skip_pgm_comments( ifstream &io_stream )
{
	int c;
	while ( !io_stream.eof() )
	{
		c = io_stream.peek();
		if ( c=='#' ) // a comment, ending with a '\n'
		{
			while ( (c=io_stream.get())!='\n' ) cout << c;
			cout << c;
		}
		else if ( c==' ' || c=='\t' || c=='\r' || c=='\n' ) // space characters
			io_stream.get();
		else
			return;
	}
}

bool read_pgm_header( const string &i_filename, unsigned int &o_width, unsigned int &o_height, unsigned int &o_maxValue, string &o_format )
{
	ifstream f( i_filename.c_str() );
	if ( !f ) return false;
	skip_pgm_comments(f);
	f >> o_format;
	skip_pgm_comments(f);
	f >> o_width;
	skip_pgm_comments(f);
	f >> o_height;
	skip_pgm_comments(f);
	f >> o_maxValue;
	return true;
}

void drawWindow( unsigned char *io_dst, unsigned int i_dstW, unsigned int i_dstH, unsigned int i_nbChannels,
                 unsigned int i_offsetX, unsigned int i_offsetY, const unsigned char *i_src, unsigned int i_srcW, unsigned int i_srcH )
{
	ELISE_ASSERT( i_nbChannels!=0, "drawWindow: nbChannels = 0" );
	ELISE_ASSERT( ( i_offsetX+i_srcW-1<i_dstW ) &&
	              ( i_offsetY+i_srcH-1<i_dstH ),
	              "drawWindow: at least part of the source window is out of destination image"  );

	io_dst += ( i_offsetX+i_offsetY*i_dstW )*i_nbChannels;
	const unsigned int srcLineSize = i_srcW*i_nbChannels,
	                   dstLineSize = i_dstW*i_nbChannels;
	while ( i_srcH-- ){
		memcpy( io_dst, i_src, srcLineSize );
		i_src += srcLineSize;
		io_dst += dstLineSize;
	}
}

unsigned int cImDigeo::getNbFeaturePoints() const
{
	unsigned int nbTotalFeaturePoints = 0;
	for ( size_t iImage=0; iImage<mVIms.size(); iImage++ )
	{
		// process first image
		const cImInMem &image = *mVIms[iImage];
		nbTotalFeaturePoints += (unsigned int)image.featurePoints().size();
	}
	return nbTotalFeaturePoints;
}

void cImDigeo::plotPoints() const
{
	mAppli.times()->start();

	const string tileFilename = mAppli.getValue_iTile( mAppli.tiledOutputExpression(), mAppli.currentBoxIndex() );
	Tiff_Im tiff( tileFilename.c_str() );

	ELISE_ASSERT( tiff.nb_chan()==3, (string("cImDigeo::plotPoints: file [")+tileFilename+"] should be in RGB colorspace").c_str() );
	ELISE_ASSERT( tiff.type_el()==GenIm::u_int1, (string("cImDigeo::plotPoints: file [")+tileFilename+"] should of type uint8").c_str() );

	vector<Im2DGen*> channels = tiff.ReadVecOfIm();
	U_INT1 *green = ( (Im2D<U_INT1,INT>*)channels[1] )->data_lin();
	int width = tiff.sz().x, height = tiff.sz().y;
	for ( size_t iImage=0; iImage<mVIms.size(); iImage++ )
	{
		const vector<DigeoPoint> &points = mVIms[iImage]->orientedPoints();
		const DigeoPoint *itPoint = points.data();
		size_t iPoint = points.size();
		while ( iPoint-- )
		{
			int x = round_ni( itPoint->x );
			int y = round_ni( ( itPoint++ )->y );
			if ( x>=0 && x<width && y>=0 && y<height )
				green[ x+y*width ] = 255;
			else
				cerr << ELISE_RED_WARNING << "cImDigeo::plotPoints: a point is outside the tile : " << x << ',' << y << endl;
		}
	}

	ELISE_COPY
	(
		channels[0]->all_pts(),
		Virgule( channels[0]->in(), channels[1]->in(), channels[2]->in() ),
		Tiff_Im(
			tileFilename.c_str(),
			channels[0]->sz(),
			GenIm::u_int1,
			Tiff_Im::No_Compr,
			Tiff_Im::RGB,
			Tiff_Im::Empty_ARG ).out()
	);

	for ( int iChannel=0; iChannel<tiff.nb_chan(); iChannel++ )
		delete channels[iChannel];

	mAppli.times()->stop("points plotting");
}


bool cImDigeo::mergeTiles( const Expression &i_inputExpression, int i_minLevel, int i_maxLevel, const cDecoupageInterv2D &i_tiles,
                           const Expression &i_outputExpression, int i_iLevelOffset ) const
{
	// retrieve all input filenames
	for ( size_t iImage=0; iImage<mVIms.size(); iImage++ )
	{
		cImInMem &image= *mVIms[iImage];
		if ( image.level()<i_minLevel || image.level()>i_maxLevel ) continue;
		image.mergeTiles( i_inputExpression, i_tiles, i_outputExpression, i_iLevelOffset );
	}
	return true;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe �  
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
