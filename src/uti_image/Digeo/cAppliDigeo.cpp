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
 
cAppliDigeo::cAppliDigeo():
	mParamDigeo  (NULL),
	mImage       (NULL),
	mICNM        (NULL),
	mDecoupInt   (cDecoupageInterv2D::SimpleDec(Pt2di(10,10),10,0)),
	mDoIncrementalConvolution( true ),
	mSiftCarac   (NULL),
	mNbSlowConvolutionsUsed_uint2(0),
	mNbSlowConvolutionsUsed_real4(0),
	mRefinementMethod(eRefine3D),
	mShowTimes(false),
	mNbComputedGradients(0),
	mNbLevels(1),
	mDoForceGradientComputation(false)
{
	loadParametersFromFile( StdGetFileXMLSpec( "ParamDigeo.xml" ), Basic_XML_MM_File( "Digeo-Parameters.xml" ) );

	mSiftCarac = Params().SiftCarac().PtrVal();
	mVerbose = Params().Verbose().Val();
	if ( Params().ConvolIncrem().IsInit() ) mDoIncrementalConvolution = Params().ConvolIncrem().Val();
	if ( Params().SiftCarac().IsInit() && Params().SiftCarac().Val().RefinementMethod().IsInit() ) mRefinementMethod = Params().SiftCarac().Val().RefinementMethod().Val();
	if ( Params().ShowTimes().IsInit() && Params().ShowTimes().Val() ) mShowTimes = true;
	if ( Params().DigeoSectionImages().ImageDigeo().NbOctetLimitLoadImageOnce().IsInit() && Params().ShowTimes().Val() ) mLoadAllImageLimit = Params().DigeoSectionImages().ImageDigeo().NbOctetLimitLoadImageOnce().Val();
	if ( Params().TypePyramide().PyramideGaussienne().IsInit() ) mNbLevels = Params().TypePyramide().PyramideGaussienne().Val().NbByOctave().Val();

	processTestSection();
	InitConvolSpec();

	// __DEL
	cout << "nb levels : " << mNbLevels << endl;
	cout << "force gradient output : " << (doForceGradientComputation()?"true":"false") << endl;

	if ( Params().GenereCodeConvol().IsInit() )
	{
		const cGenereCodeConvol &genereCodeConvol = Params().GenereCodeConvol().Val();
		mConvolutionCodeFileBase = MMDir() + genereCodeConvol.DirectoryCodeConvol().Val() + genereCodeConvol.FileBaseCodeConvol().Val() + "_";
	}

	if ( isVerbose() ) cout << "refinement: " << eToString( mRefinementMethod ) << endl;
}

cAppliDigeo::~cAppliDigeo()
{
	if ( mParamDigeo!=NULL ) delete mParamDigeo;
	if ( mICNM!=NULL ) delete mICNM;
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
	if ( !Params().ComputeCarac() ) return;

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

     // cOctaveDigeo & GetOctOfDZ(int aDZ);


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

bool cAppliDigeo::doReconstructOutputs() const { return mReconstructOutputs; }

bool cAppliDigeo::doSuppressTiledOutputs() const { return mSuppressTiledOutputs; }

bool cAppliDigeo::doForceGradientComputation() const { return mDoForceGradientComputation; }

int cAppliDigeo::currentBoxIndex() const { return mCurrentBoxIndex; }

bool cAppliDigeo::doIncrementalConvolution() const { return mDoIncrementalConvolution; }

bool cAppliDigeo::isVerbose() const { return mVerbose; }

ePointRefinement cAppliDigeo::refinementMethod() const { return mRefinementMethod; }

bool cAppliDigeo::showTimes() const { return mShowTimes; }

double cAppliDigeo::loadAllImageLimit() const { return mLoadAllImageLimit; }

void cAppliDigeo::processImageName( const string &i_imageFullname )
{
	ELISE_ASSERT( ELISE_fp::exist_file(i_imageFullname), ( string("processImageName: image to process do not exist [")+i_imageFullname+"]" ).c_str() );

	mImageFullname = i_imageFullname;
	SplitDirAndFile( mImagePath, mImageBasename, mImageFullname );
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

string cAppliDigeo::outputTiledBasename( int i_iTile ) const
{
	stringstream ss;
	ss << mImageBasename << "_tile" << setfill('0') << setw(3) << i_iTile;
	return ss.str();
}

string cAppliDigeo::currentTiledBasename() const
{
	return outputTiledBasename( currentBoxIndex() );
}

string cAppliDigeo::currentTiledFullname() const
{
	return outputTilesDirectory()+"/"+currentTiledBasename();
}

static void removeEndingSlash( string &i_str )
{
	if ( i_str.length()==0 ) return;
	const char c = *i_str.rbegin();
	if ( c=='/' || c=='\\' ) i_str.resize(i_str.length()-1);
}

static void processOutputPath( string &i_path )
{
	removeEndingSlash( i_path );
	if ( i_path.length()==0 )
		i_path="./";
	else{
		i_path.append("/");
		ELISE_fp::MkDirRec( i_path );
	}
}

void cAppliDigeo::processTestSection()
{
	const cSectionTest *mSectionTest = ( Params().SectionTest().IsInit()?&Params().SectionTest().Val():NULL );
	mDoSaveGaussians = mDoSaveTiles = mDoSaveGradients = mReconstructOutputs = mSuppressTiledOutputs = false;
	if ( mSectionTest!=NULL && mSectionTest->DigeoTestOutput().IsInit() ){
		const cDigeoTestOutput &testOutput = mSectionTest->DigeoTestOutput().Val();
		if ( testOutput.OutputGaussians().IsInit() && testOutput.OutputGaussians().Val() ){
			mDoSaveGaussians = true;
			mOutputGaussiansDirectory = testOutput.OutputGaussiansDirectory().ValWithDef("");
			processOutputPath( mOutputGaussiansDirectory );
		}
		if ( testOutput.OutputTiles().IsInit() && mSectionTest->OutputTiles().Val() ){
			mDoSaveTiles = true;
			mOutputTilesDirectory = testOutput.OutputTilesDirectory().ValWithDef("");
			processOutputPath( mOutputTilesDirectory );
		}
		if ( testOutput.OutputGradients().IsInit() && mSectionTest->OutputGradients().Val() ){
			mDoSaveGradients = true;
			mOutputGradientsAngleDirectory = testOutput.OutputGradientsAngleDirectory().ValWithDef("");
			processOutputPath( mOutputGradientsAngleDirectory );
			mOutputGradientsNormDirectory = testOutput.OutputGradientsNormDirectory().ValWithDef("");
			processOutputPath( mOutputGradientsNormDirectory );
		}
		if ( testOutput.MergeTiles().IsInit() && testOutput.MergeTiles().Val() )
			mReconstructOutputs = true;
		if ( testOutput.SuppressTiles().IsInit() && testOutput.SuppressTiles().Val() )
			mSuppressTiledOutputs = true;
		if ( testOutput.ForceGradientComputation().IsInit() && testOutput.ForceGradientComputation().Val() )
			mDoForceGradientComputation = true;
	}
}

void cAppliDigeo::InitConvolSpec()
{
	__InitConvolSpec<U_INT2>();
	__InitConvolSpec<REAL4>();
}

string cAppliDigeo::getConvolutionClassesFilename( string i_type )
{
	return mConvolutionCodeFileBase+i_type+".classes.h";
}

string cAppliDigeo::getConvolutionInstantiationsFilename( string i_type )
{
	return mConvolutionCodeFileBase+i_type+".instantiations.h";
}

void cAppliDigeo::reconstructFullOutputImages() const
{
	if ( doSaveGaussians() )
	{
		ELISE_ASSERT( getImage().reconstructFromTiles( outputGaussiansDirectory(), ".gaussian.pgm", mDecoupInt.NbX() ),
		              "cAppliDigeo::reconstructFullOutputImages: gaussian image failed" );
	}

	if ( doSaveTiles() )
	{
		ELISE_ASSERT( getImage().Octaves()[0]->FirstImage()->reconstructFromTiles( outputTilesDirectory(), ".ppm", mDecoupInt.NbX() ),
		              "cAppliDigeo::reconstructFullOutputImages: tiled image failed" );
	}

	if ( doSaveGradients() )
	{
		ELISE_ASSERT( getImage().reconstructFromTiles( outputGradientsNormDirectory(), ".gradient.norm.pgm", mDecoupInt.NbX() ),
		              "cAppliDigeo::reconstructFullOutputImages: gradient norm image failed" );
		ELISE_ASSERT( getImage().reconstructFromTiles( outputGradientsAngleDirectory(), ".gradient.angle.pgm", mDecoupInt.NbX() ),
		              "cAppliDigeo::reconstructFullOutputImages: gradient angle image failed" );
	}
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
