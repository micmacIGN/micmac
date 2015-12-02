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

/****************************************/
/*                                      */
/*             cAppliDigeo              */
/*                                      */
/****************************************/

extern const char * theNameVar_ParamDigeo[];

void cAppliDigeo::loadParametersFromFile( const string &i_templateFilename, const string &i_parametersFilename )
{
	AddEntryStringifie( "include/XML_GEN/ParamDigeo.xml", theNameVar_ParamDigeo, true );

	// construct parameters structure from template and parameters files
	cResultSubstAndStdGetFile<cParamDigeo> aP2 
	                                       (
	                                           0, NULL,                      // int argc,char **argv
	                                           i_parametersFilename,         // const std::string & aNameFileObj
	                                           i_templateFilename,           // const std::string & aNameFileSpecif
	                                           "ParamDigeo",                 // const std::string & aNameTagObj
	                                           "ParamDigeo",                 // const std::string & aNameTagType
	                                           "DirectoryChantier",          // const std::string & aNameTagDirectory
	                                           "FileChantierNameDescripteur" // const std::string & aNameTagFDC
	                                       );

	mParamDigeo = aP2.mObj;
	mICNM = aP2.mICNM;

	return ;
}

template <class tData>
void cAppliDigeo::allocateConvolutionHandler( ConvolutionHandler<tData> *&o_convolutionHandler )
{
	o_convolutionHandler = new ConvolutionHandler<tData>;

	if (isVerbose())
	{
		const size_t nbCompiledConvolutions = o_convolutionHandler->nbConvolutions();
		cout << "--- " << nbCompiledConvolutions << " compiled convolution" << (nbCompiledConvolutions>1?'s':'\0')
		     << " of type " << El_CTypeTraits<tData>::Name() << endl;
	}
}

string cAppliDigeo::defaultParameterFile(){ return Basic_XML_MM_File("Digeo-Parameters.xml"); }

cAppliDigeo::cAppliDigeo( const string &i_parametersFilename ):
	mParamDigeo  (NULL),
	mImage       (NULL),
	mICNM        (NULL),
	mDecoupInt   (cDecoupageInterv2D::SimpleDec(Pt2di(10,10),10,0)),
	mDoIncrementalConvolution( true ),
	mSiftCarac   (NULL),
	mExpressionIntegerDictionnary( new map<string,int>() ),
	mNbSlowConvolutionsUsed_uint2(0),
	mNbSlowConvolutionsUsed_real4(0),
	mRefinementMethod(eRefine3D),
	mShowTimes(false),
	mNbComputedGradients(0),
	mNbLevels(1),
	mDoForceGradientComputation(false),
	mDoPlotPoints(false),
	mDoGenerateConvolutionCode(true),
	mDoRawTestOutput(false),
	mTimes( NULL ),
	mUseSampledConvolutionKernels(false),
	mConvolutionHandler_uint2(NULL),
	mConvolutionHandler_real4(NULL)
{
	MapTimes *times = new MapTimes;
	times->start();

	loadParametersFromFile( StdGetFileXMLSpec( "ParamDigeo.xml" ), i_parametersFilename );

	mSiftCarac = Params().SiftCarac().PtrVal();
	mVerbose = Params().Verbose().Val();

	if ( isVerbose() )
	{
		cout << "--- using ";
		if ( i_parametersFilename==Basic_XML_MM_File("Digeo-Parameters.xml") ) cout << " default ";
		cout << "parameters file [" << i_parametersFilename << ']' << endl;
	}

	if ( Params().ConvolIncrem().IsInit() )
		mDoIncrementalConvolution = Params().ConvolIncrem().Val();
	if ( Params().SiftCarac().IsInit() && Params().SiftCarac().Val().RefinementMethod().IsInit() )
		mRefinementMethod = Params().SiftCarac().Val().RefinementMethod().Val();
	if ( Params().ShowTimes().IsInit() && Params().ShowTimes().Val() )
		mShowTimes = true;
	if ( Params().DigeoSectionImages().ImageDigeo().NbOctetLimitLoadImageOnce().IsInit() && Params().ShowTimes().Val() )
		mLoadAllImageLimit = Params().DigeoSectionImages().ImageDigeo().NbOctetLimitLoadImageOnce().Val();
	if ( Params().TypePyramide().PyramideGaussienne().IsInit() )
	{
		const cPyramideGaussienne aPyramideGaussienne = Params().TypePyramide().PyramideGaussienne().Val();
		mNbLevels = aPyramideGaussienne.NbByOctave().Val();
		mGaussianNbShift = aPyramideGaussienne.NbShift().Val();
		mGaussianEpsilon = aPyramideGaussienne.EpsilonGauss().Val();
		mGaussianSurEch  = aPyramideGaussienne.SurEchIntegralGauss().Val();
		if (aPyramideGaussienne.SampledConvolutionKernels().IsInit() && aPyramideGaussienne.SampledConvolutionKernels().Val() ) mUseSampledConvolutionKernels = true;
	}
	mDoComputeCarac = Params().ComputeCarac();
	if ( Params().GenereCodeConvol().IsInit() ) mDoGenerateConvolutionCode = true;

	processTestSection();

	if ( Params().GenereCodeConvol().IsInit() )
	{
		const cGenereCodeConvol &genereCodeConvol = Params().GenereCodeConvol().Val();
		mConvolutionCodeFileBase = MMDir() + genereCodeConvol.DirectoryCodeConvol().Val() + genereCodeConvol.FileBaseCodeConvol().Val() + "_";
	}

	mTimes = ( doShowTimes() ? times : (Times*)new NoTimes );

	if ( isVerbose() )
	{
		cout << "saving tiles                      : " << (doSaveTiles()?"yes":"no") << endl;
		cout << "saving gaussians                  : " << (doSaveGaussians()?"yes":"no") << endl;
		cout << "saving gradients                  : " << (doSaveGradients()?"yes":"no") << endl;
		cout << "force gradient computation        : " << (doForceGradientComputation()?"yes":"no") << endl;
		cout << "refinement                        : " << eToString(mRefinementMethod) << endl;
		cout << "downsampling                      : " << eToString(Params().ReducDemiImage().Val()) << endl;
		cout << "gaussian kernels                  : " << ( useSampledConvolutionKernels()?"sampled":"integral" ) << endl;
		if ( !useSampledConvolutionKernels() )
		{
			cout << "\tnb shift : " << mGaussianNbShift << endl;
			cout << "\tepsilon  : " << mGaussianEpsilon << endl;
			cout << "\tsurEch   : " << mGaussianSurEch << endl;
		}
	}

	const cTypePyramide & aTP = Params().TypePyramide();
	if ( aTP.NivPyramBasique().IsInit() )
		mDzLastOctave = aTP.NivPyramBasique().Val();
	else if ( aTP.PyramideGaussienne().IsInit() )
	{
		mDzLastOctave = aTP.PyramideGaussienne().Val().NivOctaveMax();
		for ( int dz=1; dz<=mDzLastOctave; dz<<=1 )
		{
			GenIm::type_el type = TypeOfDeZoom(dz);
			if ( type==GenIm::u_int2 && mConvolutionHandler_uint2==NULL ) allocateConvolutionHandler(mConvolutionHandler_uint2);
			if ( type==GenIm::real4 && mConvolutionHandler_real4==NULL ) allocateConvolutionHandler(mConvolutionHandler_real4);
			mOctaveTypes.push_back(type);
		}
	}
	else
		ELISE_ASSERT( false, "cImDigeo::AllocImages PyramideImage" );

	times->stop("appli construction");
	if ( !doShowTimes() ) delete times;
}

GenIm::type_el cAppliDigeo::octaveType( int iOctave ) const
{
	ELISE_DEBUG_ERROR( iOctave<0 || iOctave>=(int)mOctaveTypes.size(), "cAppliDigeo::octaveType", "iOctave out of range: " << iOctave << "(max " << mOctaveTypes.size() );
	return mOctaveTypes[iOctave];
}

cAppliDigeo::~cAppliDigeo()
{
	delete mParamDigeo;
	delete mICNM;
	delete mExpressionIntegerDictionnary;
	delete mTimes;
	delete mConvolutionHandler_uint2;
	delete mConvolutionHandler_real4;
}

int cAppliDigeo::nbLevels() const { return mNbLevels; }

void cAppliDigeo::upNbComputedGradients() { mNbComputedGradients++; }

int cAppliDigeo::nbComputedGradients() const { return mNbComputedGradients; }

const cParamDigeo & cAppliDigeo::Params() const { return *mParamDigeo; }

cParamDigeo & cAppliDigeo::Params() { return *mParamDigeo; }

cSiftCarac * cAppliDigeo::SiftCarac()
{
   return mSiftCarac;
}

cSiftCarac * cAppliDigeo::RequireSiftCarac()
{
   ELISE_ASSERT(mSiftCarac!=0,"cAppliDigeo::RequireSiftCarac");
   return mSiftCarac;
}

void cAppliDigeo::loadImage( const string &i_filename )
{
	processImageName( i_filename );
	AllocImages();
	InitAllImage();
}

void cAppliDigeo::AllocImages()
{
	ELISE_ASSERT( mImage==NULL, "cAppliDigeo::AllocImages: image already allocated" );
	mImage = new cImDigeo( 0, Params().DigeoSectionImages().ImageDigeo(), imageFullname(), *this );
}

bool cAppliDigeo::MultiBloc() const
{
  return Params().DigeoDecoupageCarac().IsInit();
}

void cAppliDigeo::InitAllImage()
{
  cImDigeo &image = getImage();
  Box2di aBox = image.BoxImCalc();
  Pt2di aSzGlob = aBox.sz();
  int    aBrd=0;
  int    aSzMax = aSzGlob.x + aSzGlob.y;
  if ( Params().DigeoDecoupageCarac().IsInit() )
  {
     aBrd = Params().DigeoDecoupageCarac().Val().Bord();
     aSzMax = Params().DigeoDecoupageCarac().Val().SzDalle();
  }

  mDecoupInt = cDecoupageInterv2D (aBox,Pt2di(aSzMax,aSzMax),Box2di(aBrd));

  // Les images s'itialisent en fonction de la Box
	for (int aKB=0; aKB<mDecoupInt.NbInterv() ; aKB++)
		image.NotifUseBox(mDecoupInt.KthIntervIn(aKB));
	image.AllocImages();
}

int cAppliDigeo::NbInterv() const
{
   return mDecoupInt.NbInterv();
}

void cAppliDigeo::DoOneInterv( int aKB )
{
	mBoxIn = mDecoupInt.KthIntervIn(aKB);
	mBoxOut = mDecoupInt.KthIntervOut(aKB);
	getImage().LoadImageAndPyram(mBoxIn,mBoxOut);
}

void cAppliDigeo::LoadOneInterv(int aKB)
{
    mCurrentBoxIndex = aKB;
    DoOneInterv(aKB);
}

Box2di cAppliDigeo::getInterv( int aKB ) const { return mDecoupInt.KthIntervIn(aKB); }

        //   ACCESSEURS BASIQUES

cInterfChantierNameManipulateur * cAppliDigeo::ICNM() {return mICNM;}

string cAppliDigeo::imageFullname() const { return mImageFullname; }

string cAppliDigeo::outputGaussiansDirectory() const { return mOutputGaussiansDirectory; }

string cAppliDigeo::outputTilesDirectory() const { return mOutputTilesDirectory; }

string cAppliDigeo::outputGradientsNormDirectory() const { return mOutputGradientsNormDirectory; }

string cAppliDigeo::outputGradientsAngleDirectory() const { return mOutputGradientsAngleDirectory; }

bool cAppliDigeo::doSaveGaussians() const { return mDoSaveGaussians; }

bool cAppliDigeo::doSaveTiles() const { return mDoSaveTiles; }

bool cAppliDigeo::doSaveGradients() const { return mDoSaveGradients; }

bool cAppliDigeo::doMergeOutputs() const { return mMergeOutputs; }

bool cAppliDigeo::doSuppressTiledOutputs() const { return mSuppressTiledOutputs; }

bool cAppliDigeo::doForceGradientComputation() const { return mDoForceGradientComputation; }

bool cAppliDigeo::doPlotPoints() const { return mDoPlotPoints; }

bool cAppliDigeo::doGenerateConvolutionCode() const { return mDoGenerateConvolutionCode; }

int cAppliDigeo::currentBoxIndex() const { return mCurrentBoxIndex; }

bool cAppliDigeo::doIncrementalConvolution() const { return mDoIncrementalConvolution; }

bool cAppliDigeo::isVerbose() const { return mVerbose; }

ePointRefinement cAppliDigeo::refinementMethod() const { return mRefinementMethod; }

bool cAppliDigeo::doShowTimes() const { return mShowTimes; }

bool cAppliDigeo::doComputeCarac() const { return mDoComputeCarac; }

bool cAppliDigeo::doRawTestOutput() const { return mDoRawTestOutput; }

double cAppliDigeo::loadAllImageLimit() const { return mLoadAllImageLimit; }

Times * const cAppliDigeo::times() const { return mTimes; }

void cAppliDigeo::processImageName( const string &i_imageFullname )
{
	ELISE_ASSERT( ELISE_fp::exist_file(i_imageFullname), ( string("processImageName: image to process do not exist [")+i_imageFullname+"]" ).c_str() );

	mImageFullname = i_imageFullname;
	SplitDirAndFile( mImagePath, mImageBasename, mImageFullname );

	expressions_partial_completion();
}

cImDigeo & cAppliDigeo::getImage()
{
	ELISE_ASSERT( mImage!=NULL, "cAppliDigeo::getImage: image is not allocated" );
	return *mImage;
}

const cImDigeo & cAppliDigeo::getImage() const
{
	ELISE_ASSERT( mImage!=NULL, "cAppliDigeo::getImage: image is not allocated" );
	return *mImage;
}

static void removeEndingSlash( string &i_str )
{
	if ( i_str.length()==0 ) return;
	const char c = *i_str.rbegin();
	if ( c=='/' || c=='\\' ) i_str.resize(i_str.length()-1);
}

static void normalizeOutputPath( string &i_path, bool i_doCreateDirectory )
{
	removeEndingSlash( i_path );
	if ( i_path.length()==0 )
		i_path="./";
	else
	{
		i_path.append("/");
		if ( i_doCreateDirectory ) ELISE_fp::MkDirRec( i_path );
	}
}

static void check_expression_has_variables( const Expression &i_e, const list<string> &i_variables, const Expression *i_motherExpression=NULL )
{
	if ( i_e.isValid() && i_e.hasVariables(i_variables) ) return;

	if ( !i_e.isValid() )
	{
		stringstream ss;
		ss << "expression [" << i_e.toString() << "] is invalid";
		ELISE_ASSERT( false, ss.str().c_str() );
	}

	stringstream ss;
	ss << "expression [" << i_e.toString() << "]";
	if ( i_motherExpression!=NULL ) ss << ", build from [" << i_motherExpression->toString() << "],";
	ss << " is missing variable" << (i_variables.size()<2?'\0':'s');
	list<string>::const_iterator it = i_variables.begin();
	while ( it!=i_variables.end() )
	{
		if ( !i_e.hasVariable(*it) ) ss << " [" << (*it) << ']';
		it++;
	}

	ELISE_ASSERT( false, ss.str().c_str() );
}

void cAppliDigeo::expressions_partial_completion()
{
	map<string,string> dico;
	dico["outputTilesDirectory"] = mOutputTilesDirectory;
	dico["outputGaussiansDirectory"] = mOutputGaussiansDirectory;
	dico["outputGradientsAngleDirectory"] = mOutputGradientsAngleDirectory;
	dico["outputGradientsNormDirectory"] = mOutputGradientsNormDirectory;
	dico["imageBasename"] = mImageBasename;

	if ( !doSaveTiles() && !doMergeOutputs() && !doSaveGaussians() && !doSaveGradients() ) return;

	list<string> neededVariables_merge;
	neededVariables_merge.push_back("dz");
	neededVariables_merge.push_back("iLevel");
	list<string> neededVariables_tiled;
	neededVariables_tiled.push_back("iTile");

	if ( doSaveTiles() )
	{
		mTiledOutput_expr = mTiledOutput_base_expr;
		mTiledOutput_expr.replace(dico);

		check_expression_has_variables( mTiledOutput_expr, neededVariables_tiled, &mTiledOutput_base_expr );

		if ( doMergeOutputs() )
		{
			mMergedOutput_expr = mMergedOutput_base_expr;
			mMergedOutput_expr.replace(dico);
		}
	}

	neededVariables_tiled.insert( neededVariables_tiled.end(), neededVariables_merge.begin(), neededVariables_merge.end() );

	if ( doSaveGaussians() )
	{
		mTiledOutputGaussian_expr = mTiledOutputGaussian_base_expr;
		mTiledOutputGaussian_expr.replace(dico);

		// check expression has the mandatory variables (see processTestSection() for more variable checking)
		check_expression_has_variables( mTiledOutputGaussian_expr, neededVariables_tiled, &mTiledOutputGaussian_base_expr );

		if ( doMergeOutputs() )
		{
			mMergedOutputGaussian_expr = mMergedOutputGaussian_base_expr;
			mMergedOutputGaussian_expr.replace(dico);

			check_expression_has_variables( mMergedOutputGaussian_expr, neededVariables_merge, &mMergedOutputGaussian_base_expr );
		}
	}

	if ( doSaveGradients() )
	{
		mTiledOutputGradientNorm_expr = mTiledOutputGradientNorm_base_expr;
		mTiledOutputGradientNorm_expr.replace(dico);
		mTiledOutputGradientAngle_expr = mTiledOutputGradientAngle_base_expr;
		mTiledOutputGradientAngle_expr.replace(dico);

		check_expression_has_variables( mTiledOutputGradientNorm_expr, neededVariables_tiled, &mTiledOutputGradientNorm_base_expr );
		check_expression_has_variables( mTiledOutputGradientAngle_expr, neededVariables_tiled, &mTiledOutputGradientAngle_base_expr );

		if ( doMergeOutputs() )
		{
			mMergedOutputGradientNorm_expr = mMergedOutputGradientNorm_base_expr;
			mMergedOutputGradientNorm_expr.replace(dico);
			mMergedOutputGradientAngle_expr = mMergedOutputGradientAngle_base_expr;
			mMergedOutputGradientAngle_expr.replace(dico);

			check_expression_has_variables( mMergedOutputGradientNorm_expr, neededVariables_merge, &mMergedOutputGradientNorm_base_expr );
			check_expression_has_variables( mMergedOutputGradientAngle_expr, neededVariables_merge, &mMergedOutputGradientAngle_base_expr );
		}
	}
}

void cAppliDigeo::processTestSection()
{
	const cSectionTest *mSectionTest = ( Params().SectionTest().IsInit()?&Params().SectionTest().Val():NULL );
	mDoSaveGaussians = mDoSaveTiles = mDoSaveGradients = mMergeOutputs = mSuppressTiledOutputs = false;

	// process DigeoTestOutput
	if ( mSectionTest!=NULL && mSectionTest->DigeoTestOutput().IsInit() )
	{
		const cDigeoTestOutput &testOutput = mSectionTest->DigeoTestOutput().Val();

		string testOutputSuffix = ".tif";
		if ( testOutput.RawOutput().IsInit() && testOutput.RawOutput().Val() )
		{
			mDoRawTestOutput = true;
			testOutputSuffix = ".raw";
		}

		// process output of original tiles images
		if ( testOutput.OutputTiles().IsInit() && mSectionTest->OutputTiles().Val() )
		{
			mDoSaveTiles = true;
			mOutputTilesDirectory = testOutput.OutputTilesDirectory().Val();
			normalizeOutputPath( mOutputTilesDirectory, true );

			// TODO: use a XML parameter for these expressions
			mTiledOutput_base_expr = string("${outputTilesDirectory}${imageBasename}_tile${iTile:3}")+testOutputSuffix;
			mMergedOutput_base_expr = string("${outputTilesDirectory}${imageBasename}_merged")+testOutputSuffix;

			// check expression has the mandatory variables (see expressions_partial_completion() for more variable checking)
			list<string> neededVariables;
			neededVariables.push_back("outputTilesDirectory");
			neededVariables.push_back("imageBasename");
			check_expression_has_variables( mTiledOutput_base_expr, neededVariables, NULL );
			check_expression_has_variables( mMergedOutput_base_expr, neededVariables, NULL );
		}
		if ( doSaveTiles() && testOutput.PlotPointsOnTiles().IsInit() && testOutput.PlotPointsOnTiles().Val() )
			mDoPlotPoints = true;

		// process output of gaussian images
		if ( testOutput.OutputGaussians().IsInit() && testOutput.OutputGaussians().Val() )
		{
			mDoSaveGaussians = true;
			mOutputGaussiansDirectory = testOutput.OutputGaussiansDirectory().Val();
			normalizeOutputPath( mOutputGaussiansDirectory, true ); 

			// TODO: use a XML parameter for these expressions
			mTiledOutputGaussian_base_expr = string("${outputGaussiansDirectory}${imageBasename}_tile${iTile:3}_dz${dz:3}_lvl${iLevel:3}.gaussian")+testOutputSuffix;
			mMergedOutputGaussian_base_expr = string("${outputGaussiansDirectory}${imageBasename}_merged_dz${dz:3}_lvl${iLevel:3}.gaussian")+testOutputSuffix;

			// check expression has the mandatory variables (see expressions_partial_completion() for more variable checking)
			list<string> neededVariables;
			neededVariables.push_back("outputGaussiansDirectory");
			neededVariables.push_back("imageBasename");
			check_expression_has_variables( mTiledOutputGaussian_base_expr, neededVariables, NULL );
			check_expression_has_variables( mMergedOutputGaussian_base_expr, neededVariables, NULL );
		}

		// process output of gradient images
		if ( testOutput.OutputGradients().IsInit() && testOutput.OutputGradients().Val() )
		{
			mDoSaveGradients = true;
			mOutputGradientsAngleDirectory = testOutput.OutputGradientsAngleDirectory().Val();
			normalizeOutputPath( mOutputGradientsAngleDirectory, true );
			mOutputGradientsNormDirectory = testOutput.OutputGradientsNormDirectory().Val();
			normalizeOutputPath( mOutputGradientsNormDirectory, true );

			// TODO: use a XML parameter for these expressions
			mTiledOutputGradientNorm_base_expr = string("${outputGradientsNormDirectory}${imageBasename}_tile${iTile:3}_dz${dz:3}_lvl${iLevel:3}.gradient.norm")+testOutputSuffix;
			mMergedOutputGradientNorm_base_expr = string("${outputGradientsNormDirectory}${imageBasename}_merged_dz${dz:3}_lvl${iLevel:3}.gradient.norm")+testOutputSuffix;
			mTiledOutputGradientAngle_base_expr = string("${outputGradientsAngleDirectory}${imageBasename}_tile${iTile:3}_dz${dz:3}_lvl${iLevel:3}.gradient.angle")+testOutputSuffix;
			mMergedOutputGradientAngle_base_expr = string("${outputGradientsAngleDirectory}${imageBasename}_merged_dz${dz:3}_lvl${iLevel:3}.gradient.angle")+testOutputSuffix;

			// check expression has the mandatory variables (see expressions_partial_completion() for more variable checking)
			list<string> neededVariables;
			neededVariables.push_back("outputGradientsNormDirectory");
			neededVariables.push_back("imageBasename");
			check_expression_has_variables( mTiledOutputGradientNorm_base_expr, neededVariables, NULL );
			check_expression_has_variables( mMergedOutputGradientNorm_base_expr, neededVariables, NULL );
			neededVariables.pop_front();
			neededVariables.push_back("outputGradientsAngleDirectory");
			check_expression_has_variables( mTiledOutputGradientAngle_base_expr, neededVariables, NULL );
			check_expression_has_variables( mMergedOutputGradientAngle_base_expr, neededVariables, NULL );
		}

		if ( testOutput.MergeTiles().IsInit() && testOutput.MergeTiles().Val() &&
		     ( mDoSaveTiles || mDoSaveGaussians || mDoSaveGradients ) )
			mMergeOutputs = true;

		if ( testOutput.SuppressTiles().IsInit() && testOutput.SuppressTiles().Val() )
			mSuppressTiledOutputs = true;

		if ( testOutput.ForceGradientComputation().IsInit() && testOutput.ForceGradientComputation().Val() )
			mDoForceGradientComputation = true;
	}
}

string cAppliDigeo::getConvolutionClassesFilename( string i_type )
{
	return mConvolutionCodeFileBase+i_type+".classes.h";
}

string cAppliDigeo::getConvolutionInstantiationsFilename( string i_type )
{
	return mConvolutionCodeFileBase+i_type+".instantiations.h";
}

void cAppliDigeo::mergeOutputs() const
{
	if ( !doMergeOutputs() ) return;

	times()->start();

	if ( doSaveTiles() )
	{
		if ( isVerbose() ) cout << "merging raw tiles" << endl;
		mImage->Octaves()[0]->FirstImage()->mergeTiles( mTiledOutput_expr, mDecoupInt, mMergedOutput_expr );
	}

	if ( doSaveGaussians() )
	{
		if ( isVerbose() ) cout << "merging gaussian tiles" << endl;
		mImage->mergeTiles( mTiledOutputGaussian_expr, 0, nbLevels()+2, mDecoupInt, mMergedOutputGaussian_expr );
	}

	if ( doSaveGradients() )
	{
		if ( isVerbose() ) cout << "merging gradient tiles" << endl;
		mImage->mergeTiles( mTiledOutputGradientAngle_expr, 0, nbLevels()-1, mDecoupInt, mMergedOutputGradientAngle_expr );
		mImage->mergeTiles( mTiledOutputGradientNorm_expr, 0, nbLevels()-1, mDecoupInt, mMergedOutputGradientNorm_expr );
	}

	times()->stop("tiles merging");
}

const map<string,int> & cAppliDigeo::dictionnary_tile_dz_level( int i_tile, int i_dz, int i_level ) const
{
	(*mExpressionIntegerDictionnary)["iTile"] = i_tile;
	(*mExpressionIntegerDictionnary)["dz"] = i_dz;
	(*mExpressionIntegerDictionnary)["iLevel"] = i_level;
	return *mExpressionIntegerDictionnary;
}
       
string cAppliDigeo::getValue_iTile_dz_iLevel( const Expression &e, int iTile, int dz, int iLevel ) const { return e.value( dictionnary_tile_dz_level( iTile, dz, iLevel ) ); }

string cAppliDigeo::getValue_dz_iLevel( const Expression &e, int dz, int iLevel ) const { return e.value( dictionnary_tile_dz_level( 0, dz, iLevel ) ); }

string cAppliDigeo::getValue_iTile( const Expression &e, int iTile ) const { return e.value( dictionnary_tile_dz_level( iTile, 0, 0 ) ); }

const Expression & cAppliDigeo::tiledOutputExpression() const { return mTiledOutput_expr; }
const Expression & cAppliDigeo::mergedOutputExpression() const { return mMergedOutput_expr; }

const Expression & cAppliDigeo::tiledOutputGaussianExpression() const { return mTiledOutputGaussian_expr; }
const Expression & cAppliDigeo::mergedOutputGaussianExpression() const { return mMergedOutputGaussian_expr; }

const Expression & cAppliDigeo::tiledOutputGradientNormExpression() const { return mTiledOutputGradientNorm_expr; }
const Expression & cAppliDigeo::mergedOutputGradientNormExpression() const { return mMergedOutputGradientNorm_expr; }

const Expression & cAppliDigeo::tiledOutputGradientAngleExpression() const { return mTiledOutputGradientAngle_expr; }
const Expression & cAppliDigeo::mergedOutputGradientAngleExpression() const { return mMergedOutputGradientAngle_expr; }

int    cAppliDigeo::gaussianNbShift() const { return mGaussianNbShift; }
double cAppliDigeo::gaussianEpsilon() const { return mGaussianEpsilon; }
int    cAppliDigeo::gaussianSurEch() const { return mGaussianSurEch; }
bool   cAppliDigeo::useSampledConvolutionKernels() const { return mUseSampledConvolutionKernels; }

GenIm::type_el cAppliDigeo::TypeOfDeZoom(int aDZ) const
{
	GenIm::type_el aRes = GenIm::no_type;
	int aDZMax = -10000000;
	std::list<cTypeNumeriqueOfNiv>::const_iterator itP=Params().TypeNumeriqueOfNiv().begin();
	while ( itP!=Params().TypeNumeriqueOfNiv().end() )
	{
		int aNiv = itP->Niv();
		if  ( (aNiv>=aDZMax) && (aNiv<=aDZ) )
		{
			aRes = Xml2EL(itP->Type());
			aDZMax = aNiv;
		}
		itP++;
	}
	return aRes;
}

bool cAppliDigeo::generateConvolutionCode() const
{
	if ( mConvolutionHandler_uint2!=NULL && !generateConvolutionCode(*mConvolutionHandler_uint2) ) return false;
	if ( mConvolutionHandler_real4!=NULL && !generateConvolutionCode(*mConvolutionHandler_real4) ) return false;
	return true;
}

template <class T>
void cAppliDigeo::createGaussianKernel( double aSigma, ConvolutionKernel1D<T> &oKernel ) const
{
	if ( useSampledConvolutionKernels() )
		sampledGaussianKernel<T>( aSigma, gaussianNbShift(), oKernel );
	else
		integralGaussianKernel<T>( aSigma, gaussianNbShift(), gaussianEpsilon(), gaussianSurEch(), oKernel );
}

template <class tData>
void cAppliDigeo::convolve( const Im2D<tData,TBASE> &aSrc, double aSigma, const Im2D<tData,TBASE> &oDst )
{
	ELISE_DEBUG_ERROR( aSrc.tx()!=oDst.tx() || aSrc.ty()!=oDst.ty(), "cAppliDigeo::convolve<" << El_CTypeTraits<tData>::Name() << ">", "aSrc.sz() = " << aSrc.sz() << " != oDst.sz() = " << oDst.sz() );

	ConvolutionKernel1D<TBASE> kernel;
	createGaussianKernel(aSigma,kernel);

	ELISE_DEBUG_ERROR( convolutionHandler<tData>()==NULL, "cAppliDigeo::convolve<" << El_CTypeTraits<tData>::Name() << ">", "convolutionHandler<tData>()==NULL" );

	cConvolSpec<tData> *convolution1d = convolutionHandler<tData>()->getConvolution(kernel);

	if ( !convolution1d->IsCompiled() ) upNbSlowConvolutionsUsed<tData>();

	convolution<tData>( (const tData **)aSrc.data(), aSrc.tx(), aSrc.ty(), *convolution1d, oDst.data() );
}

template <class tData>
bool cAppliDigeo::generateConvolutionCode( const ConvolutionHandler<tData> &aConvolutionHandler ) const
{
	if ( aConvolutionHandler.nbConvolutionsNotCompiled()==0 ) return true;

	const string codeFilename = MMDir()+"src/uti_image/Digeo/"+ConvolutionHandler<tData>::defaultCodeBasename();
	if ( aConvolutionHandler.generateCode(codeFilename) )
	{
		if ( isVerbose() ) cout << "--- convolution code generated in " << codeFilename << endl;
		return true;
	}

	ELISE_WARNING( "an error occured while generating convolution code for type " << El_CTypeTraits<tData>::Name() );

	return false;
}

template <class T>
ConvolutionHandler<T> * cAppliDigeo::convolutionHandler()
{
	ELISE_DEBUG_ERROR(true, "cAppliDigeo::convolutionHandler<" << El_CTypeTraits<T>::Name() << ">", "unhandled type " );
	return NULL;
}

template <> ConvolutionHandler<U_INT2> * cAppliDigeo::convolutionHandler(){ return mConvolutionHandler_uint2; }
template <> ConvolutionHandler<REAL4> * cAppliDigeo::convolutionHandler(){ return mConvolutionHandler_real4; }

// cAppliDigeo template methods
template void cAppliDigeo::createGaussianKernel<INT>( double aSigma, ConvolutionKernel1D<INT> &oKernel ) const;
template void cAppliDigeo::convolve<U_INT2>( const Im2D<U_INT2,INT> &aSrc, double aSigma, const Im2D<U_INT2,INT> &oDst );
template bool cAppliDigeo::generateConvolutionCode<U_INT2>( const ConvolutionHandler<U_INT2> &aConvolutionHandler ) const;

template void cAppliDigeo::createGaussianKernel<REAL>( double aSigma, ConvolutionKernel1D<REAL> &oKernel ) const;
template void cAppliDigeo::convolve<REAL4>( const Im2D<REAL4,REAL> &aSrc, double aSigma, const Im2D<REAL4,REAL> &oDst );
template bool cAppliDigeo::generateConvolutionCode<REAL4>( const ConvolutionHandler<REAL4> &aConvolutionHandler ) const;

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
