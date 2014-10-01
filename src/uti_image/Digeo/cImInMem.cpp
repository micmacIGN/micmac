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

bool save_pgm( const string &i_filename, unsigned char *i_image, unsigned int i_width, unsigned int i_height )
{
	ofstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	f << "P5" << endl;
	f << i_width << ' ' << i_height << endl;
	f << 255 << endl;
	f.write( (const char *)i_image, i_width*i_height );
	return true;
}

template <class T>
bool save_pgm( const string &i_filename, const T *i_image, unsigned int i_width, unsigned int i_height, T i_min, T i_max )
{
	unsigned int iPix = i_width*i_height;
	unsigned char *img = new unsigned char[iPix];
	unsigned char *itDst = img;
	const double scale = 255./( (double)i_max-(double)i_min );
	const double mind = (double)i_min;
	while ( iPix-- ){
		int v = (int)( ( (double)i_image[0]-mind )*scale+0.5 );
		if ( v>255 )
			v = 255;
		else if ( v<0 )
			v = 0;
		itDst[0] = (unsigned char)v;
		i_image++; itDst++;
	}
	bool res = save_pgm( i_filename, img, i_width, i_height );
	delete [] img;
	return res;
}

template <class T>
bool save_pgm( const string &i_filename, const T *i_image, unsigned int i_width, unsigned int i_height, double i_scale )
{
	unsigned int iPix = i_width*i_height;
	unsigned char *img = new unsigned char[iPix];
	unsigned char *itDst = img;
	while ( iPix-- ){
		int v = round_ni( (double)i_image[0]*i_scale );
		if ( v>255 )
			v = 255;
		else if ( v<0 )
			v = 0;
		itDst[0] = (unsigned char)v;
		i_image++; itDst++;
	}
	bool res = save_pgm( i_filename, img, i_width, i_height );
	delete [] img;
	return res;
}

bool save_ppm( const string &i_filename, unsigned char *i_image, unsigned int i_width, unsigned int i_height )
{
	ofstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	f << "P6" << endl;
	f << i_width << ' ' << i_height << endl;
	f << 255 << endl;
	f.write( (const char *)i_image, 3*i_width*i_height );

	#ifdef __DEBUG_DIGEO
		if ( f.bad() ) cerr << "save_ppm: cannot read " << 3*i_width*i_height << " bytes" << endl;
	#endif

	return true;
}

bool load_pgm( const string &i_filename, unsigned char *&o_image, unsigned int &o_width, unsigned int &o_height )
{
	o_image = NULL;
	o_width = o_height = 0;
	ifstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	string str;
	f >> str;
	if ( str!="P5" ) return false;
	f >> o_width;
	f >> o_height;
	int maxValue;
	f >> maxValue;
	f.get();
	unsigned int nbBytes = o_width*o_height;
	o_image = new unsigned char[nbBytes];
	f.read( (char *)o_image, nbBytes );
	return true;
}

bool load_ppm( const string &i_filename, unsigned char *&o_image, unsigned int &o_width, unsigned int &o_height )
{
	o_image = NULL;
	o_width = o_height = 0;
	ifstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	string str;
	f >> str;
	if ( str!="P6" ) return false;
	f >> o_width;
	f >> o_height;
	int maxValue;
	f >> maxValue;
	f.get();
	unsigned int nbBytes = 3*o_width*o_height;
	o_image = new unsigned char[nbBytes];
	f.read( (char *)o_image, nbBytes );
	return true;
}

template <class T>
bool save_ppm( const string &i_filename, const T *i_image, unsigned int i_width, unsigned int i_height, T i_min, T i_max )
{
	unsigned int iPix = i_width*i_height;
	unsigned char *img = new unsigned char[3*iPix];
	unsigned char *itDst = img;
	const double scale = 255./( (double)i_max-(double)i_min );
	const double mind = (double)i_min;
	while ( iPix-- ){
		int v = (int)( ( (double)i_image[0]-mind )*scale+0.5 );
		if ( v>255 )
			v = 255;
		else if ( v<0 )
			v = 0;
		itDst[0] = itDst[1] = itDst[2] = (unsigned char)v;
		i_image++; itDst+=3;
	}
	bool res = save_ppm( i_filename, img, i_width, i_height );
	delete [] img;
	return res;
}


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
 
template <class Type> void cTplImInMem<Type>::LoadFile(Fonc_Num aFonc,const Box2di & aBox,GenIm::type_el aTypeFile)
{
	ResizeOctave(aBox.sz());
	ELISE_COPY(mIm.all_pts(), aFonc, mIm.out());

	if ( mAppli.doSaveTiles() ) save_ppm<Type>( mAppli.outputTilesDirectory()+getTiledOutputBasename()+".ppm", mData[0], mSz.x, mSz.y, (Type)0, 255 );

	if ( mAppli.Params().MaximDyn().ValWithDef(nbb_type_num(aTypeFile)<=8) &&
	     type_im_integral(mType) &&
	     (!signed_type_num(mType) ) )
	{
		int aMinT,aMaxT;
		min_max_type_num(mType,aMinT,aMaxT);
		aMaxT = ElMin(aMaxT-1,1<<19);  // !!! LIES A NbShift ds PyramideGaussienne
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
			std::cout << "\tMultiplieur in : " << aMul  << "\n";
		}
		else
		{
			Type aMaxV = aMinT;
			for (int aY=0 ; aY<mSz.y ; aY++)
			{
				Type * aL = mData[aY];
				for (int aX=0 ; aX<mSz.x ; aX++)
					ElSetMax(aMaxV,aL[aX]);
			}
			aMul = (aMaxT-1) / aMaxV;
			if ( mAppli.Params().ShowTimes().Val() )
				std::cout << "\tMultiplieur in : " << aMul << " MaxVal " << tBase(aMaxV ) << " MaxType " << aMaxT << "\n";
			if (aMul > 1)
			{
				for (int aY=0 ; aY<mSz.y ; aY++)
				{
					Type * aL = mData[aY];
					for (int aX=0 ; aX<mSz.x ; aX++)
						aL[aX] *= aMul;
				}
				SauvIm("ReDyn_");
			}
		}

		mImGlob.SetDyn(aMul);
		mImGlob.SetMaxValue( aMaxT-1 );
	}
	else{
		Type aMaxV = numeric_limits<Type>::min();
		for (int aY=0 ; aY<mSz.y ; aY++)
		{
			Type * aL = mData[aY];
			for (int aX=0 ; aX<mSz.x ; aX++)
				ElSetMax(aMaxV,aL[aX]);
		}

		#ifdef __DEBUG_DIGEO_NORMALIZE_FLOAT_OCTAVE
			const Type  mul = (Type)1/(Type)( mAppli.Params().ValMaxForDyn().IsInit()?mAppli.Params().ValMaxForDyn().Val():mFileTheoricalMaxValue );
			for (int aY=0 ; aY<mSz.y ; aY++)
			{
				Type * aL = mData[aY];
				for (int aX=0 ; aX<mSz.x ; aX++)
					aL[aX] *= mul;
			}
			mImGlob.SetDyn(mul);
			mImGlob.SetMaxValue( 1 );
		#else
			mImGlob.SetDyn(1);
			mImGlob.SetMaxValue( (REAL8)aMaxV );
		#endif
	}

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
    
    #ifdef __DEBUG_DIGEO_DOG_OUTPUT
        saveDoG( "dog_digeo" );
    #endif

    #ifdef __DEBUG_DIGEO_DOG_INPUT
        loadDoG( "dog_sift" );
    #endif
}

template <class Type>
bool load_raw( const string &i_filename, Type *o_image, unsigned int i_width, unsigned int i_height )
{
	ELISE_ASSERT( false, (string("save_raw ")+El_CTypeTraits<Type>::Name()).c_str() );
	return false;
}

template <>
bool load_raw( const string &i_filename, float *o_image, unsigned int i_width, unsigned int i_height )
{
	ifstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	U_INT4 sz[2];
	f.read( (char*)sz, 8 );
	if ( sz[0]!=i_width || sz[1]!=i_height ) return false;
	f.read( (char*)o_image, i_width*i_height*sizeof(float) );
	return true;
}

template <class Type>
bool cTplImInMem<Type>::load_raw( const string &i_filename )
{
	return ::load_raw( i_filename, mIm.data_lin(), (unsigned int)mIm.tx(), (unsigned int)mIm.ty() );
}

template <>
bool cTplImInMem<U_INT2>::saveGaussian_pgm() const
{
	const string filename = mAppli.outputGaussiansDirectory()+getTiledOutputBasename()+".gaussian.pgm";
	return ::save_pgm<U_INT2>( filename, mIm.data_lin(), (unsigned int)mIm.tx(), (unsigned int)mIm.ty(), 1./mImGlob.GetDyn() );
}

template <>
bool cTplImInMem<REAL4>::saveGaussian_pgm() const
{
	const string filename = mAppli.outputGaussiansDirectory()+getTiledOutputBasename()+".gaussian.pgm";
	return ::save_pgm<REAL4>( filename, mIm.data_lin(), (unsigned int)mIm.tx(), (unsigned int)mIm.ty(), 1./mImGlob.GetDyn() );
}

template <>
bool cTplImInMem<U_INT1>::saveGaussian_pgm() const { return false; }

template <>
bool cTplImInMem<INT>::saveGaussian_pgm() const { return false; }

template <>
bool cTplImInMem<REAL>::saveGaussian_pgm() const { return false; }

template <class Type>
bool cTplImInMem<Type>::saveGaussian() const { return saveGaussian_pgm(); }

template <class Type>
bool cTplImInMem<Type>::loadGaussian_raw( const std::string &in_dir )
{
    if ( !ELISE_fp::IsDirectory(in_dir) ) return false;

    RealImage1 img;
    stringstream ss;
    ss << in_dir << "/gaussian_" << setfill('0') << setw(2) << mOct.Niv() << '_' << mKInOct << ".raw";
    if ( !img.loadRaw( ss.str() ) )
        cerr << "cTplImInMem::loadGaussians: failed to load \"" << ss.str() << "\"" << endl;
    img.toArray( TIm().data_lin() );
    return true;
}

template <class Type>
void cTplImInMem<Type>::saveDoG( const string &out_dir ) const
{
    if ( !ELISE_fp::IsDirectory(out_dir) )
    {
       cerr << "cTplImInMem::saveDoG: creating output directory \"" << out_dir << "\"" << endl;
       ELISE_fp::MkDir( out_dir );
    }

    RealImage1 img( mSz.x, mSz.y, mDoG );
    stringstream ss;
    ss << out_dir << "/dog_" << setfill('0') << setw(2) << mOct.Niv() << '_' << mKInOct;
    img.saveRaw( ss.str()+".raw" );
    img.savePGM( ss.str()+".pgm", true );
}

template <class Type>
void cTplImInMem<Type>::loadDoG( const string &in_dir )
{


    if ( !ELISE_fp::IsDirectory(in_dir) )
        cerr << "cTplImInMem::loadDoG: input directory \"" << in_dir << "\" does not exist" << endl;

    RealImage1 img;
    stringstream ss;
    ss << in_dir << "/dog_" << setfill('0') << setw(2) << mOct.Niv() << '_' << mKInOct << ".raw";

    if ( !img.loadRaw( ss.str() ) )
        cerr << "cTplImInMem::loadDoG: failed to load \"" << ss.str() << "\"" << endl;
    img.toVector( mDoG );
}

/****************************************/
/*                                      */
/*           from cConvolSpec.cpp       */
/*                                      */
/****************************************/

std::string ToNCC_Str(int aV)
{
	return aV>=0 ? ToString(aV) : ("M"+ToString(-aV));
}

std::string ToNCC_Str(double aV)
{
	return aV>=0 ? ToString(round_ni(aV*1000)) : ("M"+ToString(round_ni(-aV*1000)));
}


template <class Type> std::string NameClassConvSpec(Type *)
{
    static int  aCpt=0;
	
    std::string aRes =  std::string("cConvolSpec_") 
	+ El_CTypeTraits<Type>::Name()
	+ std::string("_Num") + ToString(aCpt++) ;
	
    return aRes;
}

static void LineSym(FILE * aFile,int aVal,int aK)
{
	fprintf(aFile,"                              +   %d*(In[%d]+In[%d])\n",aVal,aK,-aK);
}
static void LineSym(FILE * aFile,double aVal,int aK)
{
	fprintf(aFile,"                              +   %lf*(In[%d]+In[%d])\n",aVal,aK,-aK);
}
static void LineStd(FILE * aFile,int aVal,int aK)
{
	fprintf(aFile,"                              +   %d*(In[%d])\n",aVal,aK);
}
static void LineStd(FILE * aFile,double aVal,int aK)
{
	fprintf(aFile,"                              +   %lf*(In[%d])\n",aVal,aK);
}


static void  PutVal(FILE * aFile,int aVal)
{
	fprintf(aFile,"%d",aVal);
}
static void  PutVal(FILE * aFile,double aVal)
{
	fprintf(aFile,"%lf",aVal);
}


template <class Type> 
void cTplImInMem <Type>::MakeClassConvolSpec
(
 bool Increm,
 double aSigma,
 FILE * aFileH,
 FILE * aFileCpp,
 tBase* aFilter,
 int aDeb,
 int aFin,
 int aNbShit
 )
{
    if (!aFileH) 
		return;
	
    cConvolSpec<Type> * aRes = cConvolSpec<Type>::GetExisting(aFilter,aDeb,aFin,aNbShit,true);
    if (aRes)
    {
        return;
    }
    // std::cout << "xxxxxx--- NEW  "  << aFilter[aDeb] << " " << aFilter[0] <<  " " << aFilter[aFin] << "\n";
    aRes = cConvolSpec<Type>::GetOrCreate(aFilter,aDeb,aFin,aNbShit,true);
	
	
    // std::string aNClass = NameClassConvSpec(aFilter,aDeb,aFin);
    std::string aNClass = NameClassConvSpec((Type *)0);
    std::string aNType = El_CTypeTraits<Type>::Name();
    std::string aNTBase = El_CTypeTraits<tBase>::Name();
	
    fprintf(aFileH,"/* Sigma %lf  ModeIncrem %d */\n",aSigma,int(Increm));
    fprintf(aFileH,"class %s : public cConvolSpec<%s>\n",aNClass.c_str(),aNType.c_str());
    fprintf(aFileH,"{\n");
    fprintf(aFileH,"   public :\n");
    fprintf(aFileH,"      bool IsCompiled() const {return true;}\n");
    fprintf(aFileH,"      void Convol(%s * Out,%s * In,int aK0,int aK1)\n",aNType.c_str(),aNType.c_str());
    fprintf(aFileH,"      {\n");
    fprintf(aFileH,"          In+=aK0;\n");
    fprintf(aFileH,"          Out+=aK0;\n");
    fprintf(aFileH,"          for (int aK=aK0; aK<aK1 ; aK++)\n");
    fprintf(aFileH,"          {\n");
    fprintf(aFileH,"               *(Out++) =  (\n");
    if (El_CTypeTraits<Type>::IsIntType())
		fprintf(aFileH,"                                %d\n",(1<<aNbShit)/2);
    else
		fprintf(aFileH,"                                 0\n");
    for (int aK=aDeb ; aK <=aFin ; aK++)
    {
        if ((-aK>=aDeb) && (-aK<=aFin) && (aK) && (aFilter[aK]==aFilter[-aK]))
        {
            if (aK<0)
				LineSym(aFileH,aFilter[aK],aK);
        }
        else
        {
			LineStd(aFileH,aFilter[aK],aK);
        }
    }
    if (El_CTypeTraits<Type>::IsIntType())
		fprintf(aFileH,"                           )>>%d;\n",aNbShit);
    else
		fprintf(aFileH,"                           );\n");
    fprintf(aFileH,"               In++;\n");
    fprintf(aFileH,"          }\n");
    fprintf(aFileH,"      }\n\n");
    fprintf(aFileH,"      %s(%s * aFilter):\n",aNClass.c_str(),aNTBase.c_str());
    fprintf(aFileH,"           cConvolSpec<%s>(aFilter-(%d),%d,%d,%d,false) ",aNType.c_str(),aDeb,aDeb,aFin,aNbShit);
    fprintf(aFileH,"      {\n");
    fprintf(aFileH,"      }\n");
    fprintf(aFileH,"};\n\n");
	
	
    fprintf(aFileCpp,"   {\n");
    fprintf(aFileCpp,"      %s theCoeff[%d] ={",aNTBase.c_str(),aFin-aDeb+1);
	
    for (int aK=aDeb ; aK <=aFin ; aK++)
    {
		if (aK!=aDeb) fprintf(aFileCpp,",");
		PutVal(aFileCpp,aFilter[aK]);
    }
    fprintf(aFileCpp,"};\n");
    fprintf(aFileCpp,"         new %s(theCoeff);\n",aNClass.c_str());
    fprintf(aFileCpp,"   }\n");
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



void cImInMem::SetMere(cImInMem * aMere)
{
    ELISE_ASSERT((mMere==0) && (aMere->mFille==0),"cImInMem::SetMere");
    mMere = aMere;
    aMere->mFille = this;
}



void cImInMem::SauvIm(const std::string & aAdd)
{
    #if defined(__DEBUG_DIGEO_GAUSSIANS_OUTPUT_RAW) || defined(__DEBUG_DIGEO_GAUSSIANS_OUTPUT_PGM)
        saveGaussians( "gaussians_digeo" );
    #endif

    #ifdef __DEBUG_DIGEO_GAUSSIANS_INPUT
        loadGaussians( "gaussians_sift" );
    #endif
    
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

bool save_gradient( const Im2D<REAL4, REAL8> &i_gradient, const int i_offset, const string &i_filename )
{
	unsigned int iPix = i_gradient.tx()*i_gradient.ty()/2;
	REAL4 *channel = new REAL4[iPix];

	// retrieve min/max
	REAL4 minv = numeric_limits<REAL4>::max(),
	      maxv = -numeric_limits<REAL4>::max();
	const REAL4 *itPix = i_gradient.data_lin();
	REAL4 *itChan = channel;
	while ( iPix-- ){
		REAL4 v = itPix[i_offset];
		if ( v<minv ) minv=v;
		if ( v>maxv ) maxv=v;

		*itChan++ = itPix[i_offset];
		itPix += 2;
	}

	if ( !save_pgm( i_filename, channel, (unsigned int)i_gradient.tx()/2, (unsigned int)i_gradient.ty(), minv, maxv ) ) return false;

	delete [] channel;
	return true;
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

string cImInMem::getTiledOutputBasename( int i_tile ) const
{
	stringstream ss;
	ss << mOct.getTiledOutputBasename(i_tile) << '_' << setfill('0') << setw(2) << mKInOct;
	return ss.str();
}

string cImInMem::getTiledOutputBasename() const { return getTiledOutputBasename(mAppli.currentBoxIndex()); }

string cImInMem::getReconstructedOutputBasename() const
{
	stringstream ss;
	ss << mOct.getReconstructedOutputBasename() << '_' << setfill('0') << setw(2) << mKInOct;
	return ss.str();
}

string cImInMem::getTiledGradientNormOutputName( int i_tile ) const { return mAppli.outputGradientsNormDirectory()+getTiledOutputBasename(i_tile)+".gradient.norm.pgm"; }
string cImInMem::getTiledGradientNormOutputName() const { return getTiledGradientNormOutputName( mAppli.currentBoxIndex() ); }

string cImInMem::getTiledGradientAngleOutputName( int i_tile ) const { return mAppli.outputGradientsAngleDirectory()+getTiledOutputBasename(i_tile)+".gradient.angle.pgm"; }
string cImInMem::getTiledGradientAngleOutputName() const { return getTiledGradientAngleOutputName( mAppli.currentBoxIndex() ); }

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
	if ( mKInOct<1 || mKInOct>mAppli.nbLevels() || ( !mAppli.doForceGradientComputation() && mFeaturePoints.size()==0 ) )
	{
		mOrientateTime = 0.;
		return;
	}

	const Im2D<REAL4,REAL8> &srcGradient = getGradient();

	ElTimer chrono;
	mOrientedPoints.resize( mFeaturePoints.size() );
	double octaveTrueSamplingPace = mOct.trueSamplingPace();
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
			dstPoint.x = srcPoint.mPt.x*octaveTrueSamplingPace;
			dstPoint.y = srcPoint.mPt.y*octaveTrueSamplingPace;
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
	mOrientateTime = chrono.uval();
}

//double cImDigeo::detectTime() const { return mDetectTime; }

double cImInMem::orientateTime() const { return mOrientateTime; }

double cImInMem::describeTime() const { return mDescribeTime; }




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
	if ( mKInOct<1 || mKInOct>mAppli.nbLevels() || ( !mAppli.doForceGradientComputation() && mOrientedPoints.size()==0 ) ) return;

	const Im2D<REAL4,REAL8> &srcGradient = getGradient();
	ElTimer chrono;
	double octaveTrueSamplingPace = mOct.trueSamplingPace();

	DigeoPoint *itPoint = mOrientedPoints.data();
	size_t iPoint = mOrientedPoints.size();
	while ( iPoint-- ){
		DigeoPoint &p = *itPoint++;
		for ( size_t iAngle=0; iAngle<p.entries.size(); iAngle++ ){
			DigeoPoint::Entry &entry = p.entry(iAngle);
			::describe( srcGradient, p.x, p.y, p.scale/octaveTrueSamplingPace, entry.angle, entry.descriptor );
			normalize_and_truncate( entry.descriptor );
		}
	}
	mDescribeTime = chrono.uval();
}

bool cImInMem::reconstructFromTiles( const string &i_directory, const string &i_postfix, int i_gridWidth ) const
{
	int nbTiles = mAppli.NbInterv();
	int gridHeight = nbTiles/i_gridWidth;

	ELISE_ASSERT( (nbTiles%i_gridWidth)==0, "cImDigeo::reconstructFromTiles incorrect grid width" );

	unsigned int fullW = 0, fullH = 0;
	unsigned int w, h, maxv;
	unsigned int nbChannels = 1;
	string format;

	// retrive full image size
	if ( !read_pgm_header( i_directory+getTiledOutputBasename(0)+i_postfix, fullW, fullH, maxv, format ) ){
		cout << "cannot read [" << i_directory+getTiledOutputBasename(0)+i_postfix << "]" << endl;
		return false;
	}
	if ( format=="P6" ) nbChannels=3;

	for ( int iTile=1; iTile<i_gridWidth; iTile++ )
	{
		string tileFilename = i_directory+getTiledOutputBasename(iTile)+i_postfix;
		ELISE_ASSERT( read_pgm_header( tileFilename, w, h, maxv, format ), (string("cImDigeo::reconstructFromTiles cannot read pgm header from [")+tileFilename+"]").c_str() );
		fullW += w;
	}
	for ( int iTile=1; iTile<gridHeight; iTile++ )
	{
		string tileFilename = i_directory+getTiledOutputBasename(iTile)+i_postfix;
		ELISE_ASSERT( read_pgm_header( tileFilename, w, h, maxv, format ), (string("cImDigeo::reconstructFromTiles cannot read pgm header from [")+tileFilename+"]").c_str() );
		fullH += h;
	}

	unsigned char *data = new unsigned char[fullW*fullH*nbChannels];
	unsigned int offsetY = 0;
	for ( int iTileY=0; iTileY<gridHeight; iTileY++ )
	{
		unsigned int offsetX = 0;
		for ( int iTileX=0; iTileX<i_gridWidth; iTileX++ )
		{
			const string &windowFilename = i_directory+getTiledOutputBasename(iTileX+iTileY*i_gridWidth)+i_postfix;
			unsigned char *window = NULL;
			if ( nbChannels==1 )
				load_pgm( windowFilename, window, w, h );
			else
				load_ppm( windowFilename, window, w, h );

			drawWindow( data, fullW, fullH, nbChannels, offsetX, offsetY, window, w, h );
			delete [] window;
			offsetX += w;
			if ( mAppli.doSuppressTiledOutputs() ) ELISE_fp::RmFile( windowFilename );
		}
		offsetY += h;
	}

	const string outputName = i_directory+getReconstructedOutputBasename()+i_postfix;
	if ( nbChannels==1 )
	{
		if ( !save_pgm( outputName, data, fullW, fullH  ) ) return false;
	}
	else
	{
		if ( !save_ppm( outputName, data, fullW, fullH ) ) return false;
	}
	delete [] data;

	return true;
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
