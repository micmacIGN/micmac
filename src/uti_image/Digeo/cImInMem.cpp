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

bool save_ppm( const string &i_filename, unsigned char *i_image, unsigned int i_width, unsigned int i_height )
{
	ofstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	f << "P6" << endl;
	f << i_width << ' ' << i_height << endl;
	f << 255 << endl;
	f.write( (const char *)i_image, 3*i_width*i_height );
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
/*             cTplImInMem               */
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
     mData   (0)
{
    ResizeImage(aSz);
    mNbShift =  mAppli.TypePyramide().PyramideGaussienne().Val().NbShift().Val();
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

template <class Type> bool cTplImInMem<Type>::InitRandom()
{
   if (! mAppli.SectionTest().IsInit())
      return false;
   const cSectionTest & aST = mAppli.SectionTest().Val();
   if (! aST.GenereAllRandom().IsInit())
      return false;


    const cGenereAllRandom & aGAR = aST.GenereAllRandom().Val();

   ELISE_COPY
   (
      mIm.all_pts(),
      255*(gauss_noise_1(aGAR.SzFilter())>0),
      mIm.out()
   );

   return true;
}
 
template <class Type> void cTplImInMem<Type>::LoadFile(Fonc_Num aFonc,const Box2di & aBox,GenIm::type_el aTypeFile)
{
	ResizeOctave(aBox.sz());
	ELISE_COPY(mIm.all_pts(), aFonc, mIm.out());
	if ( mAppli.MaximDyn().ValWithDef(nbb_type_num(aTypeFile)<=8) &&
	     type_im_integral(mType) &&
	     (!signed_type_num(mType) ) ){
		int aMinT,aMaxT;
		min_max_type_num(mType,aMinT,aMaxT);
		aMaxT = ElMin(aMaxT-1,1<<19);  // !!! LIES A NbShift ds PyramideGaussienne
		tBase aMul = 0;

		if ( mAppli.ValMaxForDyn().IsInit() ){
			tBase aMaxTm1 = aMaxT-1;
			aMul = round_ni(aMaxTm1/mAppli.ValMaxForDyn().Val()) ;

			for (int aY=0 ; aY<mSz.y ; aY++){
				Type * aL = mData[aY];
				for (int aX=0 ; aX<mSz.x ; aX++)
				aL[aX] = ElMin(aMaxTm1,tBase(aL[aX]*aMul));
			}
			std::cout << " Multiplieur in : " << aMul  << "\n";
		}
		else{
			Type aMaxV = aMinT;
			for (int aY=0 ; aY<mSz.y ; aY++){
				Type * aL = mData[aY];
				for (int aX=0 ; aX<mSz.x ; aX++)
				ElSetMax(aMaxV,aL[aX]);
			}
			aMul = (aMaxT-1) / aMaxV;
			if ( mAppli.ShowTimes().Val() )
				std::cout << " Multiplieur in : " << aMul << " MaxVal " << tBase(aMaxV ) << " MaxType " << aMaxT << "\n";
			if (aMul > 1){
				for (int aY=0 ; aY<mSz.y ; aY++){
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
		for (int aY=0 ; aY<mSz.y ; aY++){
			Type * aL = mData[aY];
			for (int aX=0 ; aX<mSz.x ; aX++)
				ElSetMax(aMaxV,aL[aX]);
		}

		#ifdef __DEBUG_DIGEO_NORMALIZE_FLOAT_OCTAVE
			const Type  mul = (Type)1/(Type)( mAppli.ValMaxForDyn().IsInit()?mAppli.ValMaxForDyn().Val():mFileTheoricalMaxValue );
			for (int aY=0 ; aY<mSz.y ; aY++){
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

	if ( mAppli.doSaveTiles() ) save_ppm<Type>( mAppli.currentTileFullname()+".ppm", mData[0], mSz.x, mSz.y, (Type)0, (Type)mImGlob.GetMaxValue() );

	const cTypePyramide & aTP = mAppli.TypePyramide();
	if ( aTP.PyramideGaussienne().IsInit() ){
		const double aSigmD = mImGlob.InitialDeltaSigma();
		if ( aSigmD!=0. ){
			Im1D<tBase,tBase> aIKerD = ImGaussianKernel(aSigmD);
			SetConvolSepXY( true, aSigmD, *this, aIKerD, mNbShift );
		}
	}

   if (mAppli.SectionTest().IsInit())
   {
      const cSectionTest & aST = mAppli.SectionTest().Val();
      if (aST.GenereRandomRect().IsInit())
      {
         cGenereRandomRect aGRR = aST.GenereRandomRect().Val();
         ELISE_COPY(mIm.all_pts(),0, mIm.out());
         for (int aK= 0 ; aK < aGRR.NbRect() ; aK++)
         {
             Pt2di aP = round_ni(aBox.RandomlyGenereInside()) - aBox._p0;
             int aL = 1 +NRrandom3(aGRR.SzRect());
             int aH = 1 +NRrandom3(aGRR.SzRect());
             ELISE_COPY
             (
                   rectangle(aP-Pt2di(aL,aH),aP+Pt2di(aL,aH)),
                   255*NRrandom3(),
                   mIm.out()
             );
         }
      }
      if (aST.GenereCarroyage().IsInit())
      {
         cGenereCarroyage aGC = aST.GenereCarroyage().Val();
         ELISE_COPY(mIm.all_pts(),255*((FX/aGC.PerX()+FY/aGC.PerY())%2), mIm.out());
      }

      InitRandom();
   }
}

template <class Type> Im2DGen cTplImInMem<Type>::Im(){ return TIm(); }

template <class Type> typename cTplImInMem<Type>::tBase * cTplImInMem<Type>::DoG(){ return ( mDoG.size()==0?NULL:&mDoG[0] ); }

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
    if ( mDoG.size()==0 ) return;
    tBase *itDog = &mDoG[0];
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

#ifdef __WITH_GAUSS_SEP_FILTER
	bool load_raw( const string &i_filename, REAL *i_image, unsigned int i_width, unsigned int i_height )
	{
		ELISE_ASSERT( false, "save_raw(REAL*)");
		return false;
	}

	bool load_raw( const string &i_filename, U_INT1 *i_image, unsigned int i_width, unsigned int i_height )
	{
		ELISE_ASSERT( false, "save_raw(REAL8*)");
		return false;
	}

	bool load_raw( const string &i_filename, INT *i_image, unsigned int i_width, unsigned int i_height )
	{
		ELISE_ASSERT( false, "save_raw(REAL8*)");
		return false;
	}
#endif

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
bool cTplImInMem<U_INT2>::saveGaussian_pgm( const std::string &i_filename ) const
{
	stringstream ss;
	ss << i_filename << '_' << setfill('0') << setw(2) << mKInOct << ".pgm";
	return ::save_pgm<U_INT2>( ss.str(), mIm.data_lin(), (unsigned int)mIm.tx(), (unsigned int)mIm.ty(), 0, 65535 );
}

template <>
bool cTplImInMem<REAL4>::saveGaussian_pgm( const std::string &i_filename ) const
{
	stringstream ss;
	ss << i_filename << '_' << setfill('0') << setw(2) << mKInOct << ".pgm";
	return ::save_pgm<REAL4>( ss.str(), mIm.data_lin(), (unsigned int)mIm.tx(), (unsigned int)mIm.ty(), 0.f, 1.f );
}

template <>
bool cTplImInMem<U_INT1>::saveGaussian_pgm( const std::string &i_filename ) const { return false; }

template <>
bool cTplImInMem<INT>::saveGaussian_pgm( const std::string &i_filename ) const { return false; }

template <>
bool cTplImInMem<REAL>::saveGaussian_pgm( const std::string &i_filename ) const { return false; }

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
    mKernelTot        (1,1.0),
    mFirstSauv        (true),
    mFileTheoricalMaxValue( (1<</*aIGlob.TifF().bitpp()*/aIGlob.bitpp())-1 ),
    mUsed_points_map(NULL)
   #ifdef __DEBUG_DIGEO_STATS
      ,mCount_eTES_Uncalc(0),
       mCount_eTES_instable_unsolvable(0),
       mCount_eTES_instable_tooDeepRecurrency(0),
       mCount_eTES_instable_outOfImageBound(0),
       mCount_eTES_GradFaible(0),
       mCount_eTES_TropAllonge(0),
       mCount_eTES_Ok(0)
   #endif
{
    if ( aIGlob.Appli().mVerbose )
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
    
   if ( ! mAppli.SauvPyram().IsInit()) return;

   const cTypePyramide & aTP = mAppli.TypePyramide();
   cSauvPyram aSP = mAppli.SauvPyram().Val();
   std::string aDir =  mAppli.DC() + aSP.Dir().Val();
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
                                mImGlob.Name(),
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

std::vector<cPtsCaracDigeo> &  cImInMem::VPtsCarac() {return mVPtsCarac;}
const std::vector<cPtsCaracDigeo> &  cImInMem::VPtsCarac() const {return mVPtsCarac;}

GenIm::type_el  cImInMem::TypeEl() const { return mType; }
Pt2di cImInMem::Sz() const {return mSz;}
int cImInMem::RGlob() const {return mResolGlob;}
double cImInMem::ROct() const {return mResolOctaveBase;}
cImInMem *  cImInMem::Mere() {return mMere;}
cOctaveDigeo &  cImInMem::Oct() {return mOct;}



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
