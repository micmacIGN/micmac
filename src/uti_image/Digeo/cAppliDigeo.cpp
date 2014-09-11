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

void cAppliDigeo::loadParametersFromFile( const string &i_templateFilename, const string &i_parametersFilename, const string &i_imageFullname )
{
	string imageDirectory, imageBasename;
	SplitDirAndFile( imageDirectory, imageBasename, i_imageFullname );

	AddEntryStringifie( "include/XML_GEN/ParamDigeo.xml", theNameVar_ParamDigeo, true );

	// construct parameters structure from template and parameters files
	vector<char *> args;
	args.push_back( strdup( ( string( "+Im1=" )+imageBasename ).c_str() ) );
	args.push_back( strdup( ( string( "DirectoryChantier=" )+imageDirectory ).c_str() ) );

	cResultSubstAndStdGetFile<cParamDigeo> aP2 
	                                       (
	                                           args.size(), args.data(),     // int argc,char **argv
	                                           i_parametersFilename,         // const std::string & aNameFileObj
	                                           i_templateFilename,           // const std::string & aNameFileSpecif
	                                           "ParamDigeo",                 // const std::string & aNameTagObj
	                                           "ParamDigeo",                 // const std::string & aNameTagType
	                                           "DirectoryChantier",          // const std::string & aNameTagDirectory
	                                           "FileChantierNameDescripteur" // const std::string & aNameTagFDC
	                                       );

	// free strings duplicates
	for ( size_t i=0; i<args.size(); i++ ) 
		free( args[i] );

	mParamDigeo = aP2.mObj;
	mICNM = aP2.mICNM;

	return ;
}
 
cAppliDigeo::cAppliDigeo( const string &i_imageFilename ):
	mParamDigeo  (NULL),
	mICNM        (NULL),
	mDecoupInt   (cDecoupageInterv2D::SimpleDec(Pt2di(10,10),10,0)),
	mDoIncrementalConvolution( true ),
	mSiftCarac   (NULL),
	mNbSlowConvolutionsUsed_uint2(0),
	mNbSlowConvolutionsUsed_real4(0)
{
	loadParametersFromFile( StdGetFileXMLSpec( "ParamDigeo.xml" ), Basic_XML_MM_File( "Digeo-Parameters.xml" ), i_imageFilename );
	//mParamDigeo = new cParamDigeo( *(aParam.mObj) );
	mSiftCarac = Params().SiftCarac().PtrVal();
	mVerbose = Params().Verbose().Val();
	if ( Params().ConvolIncrem().IsInit() ) mDoIncrementalConvolution = Params().ConvolIncrem().Val();

	AllocImages();
	InitAllImage();

	processImageName();
	processTestSection();
	InitConvolSpec();

	if ( Params().GenereCodeConvol().IsInit() ){
		const cGenereCodeConvol &genereCodeConvol = Params().GenereCodeConvol().Val();
		mConvolutionCodeFileBase = MMDir() + genereCodeConvol.DirectoryCodeConvol().Val() + genereCodeConvol.FileBaseCodeConvol().Val() + "_";
	}
}

cAppliDigeo::cAppliDigeo
( 
   cResultSubstAndStdGetFile<cParamDigeo> aParam,
   cAppliDigeo *                          aMaterAppli,
   cModifGCC *                            aModif,
   bool                                   IsLastGCC

) :
    mParamDigeo  (NULL),
    mICNM        (aParam.mICNM),/*
    mDC          (aParam.mDC),
    mMaster      (aMaterAppli),
    mModifGCC    (aModif),
    mLastGCC     (IsLastGCC),
    mFileGGC_H   (aMaterAppli ? aMaterAppli->mFileGGC_H : 0),
    mFileGGC_Cpp (aMaterAppli ? aMaterAppli->mFileGGC_Cpp : 0),*/
    mDecoupInt   (cDecoupageInterv2D::SimpleDec(Pt2di(10,10),10,0)),
    mDoIncrementalConvolution( true ),
    mSiftCarac   (NULL),
    mNbSlowConvolutionsUsed_uint2(0),
    mNbSlowConvolutionsUsed_real4(0)
{
	mParamDigeo = new cParamDigeo( *(aParam.mObj) );
	mSiftCarac = Params().SiftCarac().PtrVal();
	mVerbose = Params().Verbose().Val();
	if ( Params().ConvolIncrem().IsInit() ) mDoIncrementalConvolution = Params().ConvolIncrem().Val();

   processImageName();
   processTestSection();
   InitConvolSpec();

	if ( Params().GenereCodeConvol().IsInit() ){
		const cGenereCodeConvol &genereCodeConvol = Params().GenereCodeConvol().Val();
		mConvolutionCodeFileBase = MMDir() + genereCodeConvol.DirectoryCodeConvol().Val() + genereCodeConvol.FileBaseCodeConvol().Val() + "_";
	}
}

cAppliDigeo::~cAppliDigeo()
{
	if ( mParamDigeo!=NULL ) delete mParamDigeo;
	if ( mICNM!=NULL ) delete mICNM;
}

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

void cAppliDigeo::AllocImages()
{
   for 
   (
       std::list<cImageDigeo>::const_iterator itI=Params().ImageDigeo().begin();
       itI!=Params().ImageDigeo().end();
       itI++
   )
   {
       std::list<std::string> aLS = mICNM->StdGetListOfFile(itI->KeyOrPat());
       for 
       (
            std::list<std::string>::const_iterator itS=aLS.begin();
            itS!=aLS.end();
            itS++
       )
       {
           mVIms.push_back(new cImDigeo(mVIms.size(),*itI,mICNM->Dir()+(*itS),*this));
       }
   }
}

cImDigeo & cAppliDigeo::SingleImage()
{
    ELISE_ASSERT(mVIms.size()==1,"cAppliDigeo::SingleImage");
    return *(mVIms[0]);
}

bool cAppliDigeo::MultiBloc() const
{
  return Params().DigeoDecoupageCarac().IsInit();
}



void cAppliDigeo::InitAllImage()
{
	if ( !Params().ComputeCarac() ) return;

	/*
  if (GenereCodeConvol().IsInit() && (mFileGGC_H ==0))
  {
      cGenereCodeConvol & aGCC = GenereCodeConvol().Val();
      std::string aName = aGCC.File().Val();
      std::string aDir = aGCC.Dir().Val();
      mFileGGC_H = FopenNN(MMDir()+aDir+aName+".h","w","cAppliDigeo::DoCarac");
      mFileGGC_Cpp = FopenNN(MMDir()+aDir+aName+".cpp","w","cAppliDigeo::DoCarac");

      fprintf(mFileGGC_H,"#include \"Digeo.h\"\n\n");
      // fprintf(mFileGGC_H,"namespace NS_ParamDigeo {\n");


      fprintf(mFileGGC_Cpp,"#include \"%s.h\"\n\n",aName.c_str());
      // fprintf(mFileGGC_Cpp,"namespace NS_ParamDigeo {\n");
      fprintf(mFileGGC_Cpp,"void cAppliDigeo::InitConvolSpec()\n");
      fprintf(mFileGGC_Cpp,"{\n");
      fprintf(mFileGGC_Cpp,"    static bool theFirst = true;\n");
      fprintf(mFileGGC_Cpp,"    if (! theFirst) return;\n");
      fprintf(mFileGGC_Cpp,"    theFirst = false;\n");
      fprintf(mFileGGC_Cpp,"\n");

  }
	*/

  Box2di aBox = mVIms[0]->BoxImCalc();
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
  for (int aKI=0 ; aKI<int(mVIms.size()) ; aKI++)
  {
      for (int aKB=0; aKB<mDecoupInt.NbInterv() ; aKB++)
      {
          mVIms[aKI]->NotifUseBox(mDecoupInt.KthIntervIn(aKB));
      }
      mVIms[aKI]->AllocImages();
  }


	/*
  if (GenereCodeConvol().IsInit() &&  mLastGCC)
  {
       DoAllInterv();
      // fprintf(mFileGGC_H,"}\n");
      ElFclose(mFileGGC_H);

      fprintf(mFileGGC_Cpp,"}\n");
      // fprintf(mFileGGC_Cpp,"}\n");
      ElFclose(mFileGGC_Cpp);
  }
  */
}

void cAppliDigeo::DoAllInterv()
{
   for (int aKB=0; aKB<mDecoupInt.NbInterv() ; aKB++)
   {
         std::cout << "Boxes to do " << mDecoupInt.NbInterv()  - aKB << "\n";
         DoOneInterv(aKB,true);
   }
}


int cAppliDigeo::NbInterv() const
{
   return mDecoupInt.NbInterv();
}

void cAppliDigeo::DoOneInterv(int aKB,bool DoExtract)
{
   mBoxIn = mDecoupInt.KthIntervIn(aKB);
   mBoxOut = mDecoupInt.KthIntervOut(aKB);
   for (size_t aKI=0 ; aKI<mVIms.size() ; aKI++)
   {
          mVIms[aKI]->LoadImageAndPyram(mBoxIn,mBoxOut);
          if ( DoExtract ) mVIms[aKI]->DoExtract();
   }
}

void cAppliDigeo::LoadOneInterv(int aKB)
{
    mCurrentBoxIndex = aKB;
    if ( doSaveTiles() ) ELISE_fp::MkDir( outputTilesDirectory() );
    DoOneInterv(aKB,false);
}

Box2di cAppliDigeo::getInterv( int aKB ) const { return mDecoupInt.KthIntervIn(aKB); }

void cAppliDigeo::DoAll()
{
     AllocImages();
     ELISE_ASSERT(mVIms.size(),"NoImage selected !!");
     InitAllImage();
     DoAllInterv();
}

     // cOctaveDigeo & GetOctOfDZ(int aDZ);


        //   ACCESSEURS BASIQUES



cInterfChantierNameManipulateur * cAppliDigeo::ICNM() {return mICNM;}

/*
const std::string & cAppliDigeo::DC() const { return mDC; }
* 
FILE *  cAppliDigeo::FileGGC_H()
{
   return mFileGGC_H;
}

FILE *  cAppliDigeo::FileGGC_Cpp()
{
   return mFileGGC_Cpp;
}

cModifGCC * cAppliDigeo::ModifGCC() const 
{ 
   return mModifGCC; 
}
*/


string cAppliDigeo::imageFullname() const { return mImageFullname; }

string cAppliDigeo::outputGaussiansDirectory() const { return mOutputGaussiansDirectory; }

string cAppliDigeo::outputTilesDirectory() const { return mOutputTilesDirectory; }

bool cAppliDigeo::doSaveGaussians() const { return mDoSaveGaussians; }

bool cAppliDigeo::doSaveTiles() const { return mDoSaveGaussians; }

int cAppliDigeo::currentBoxIndex() const { return mCurrentBoxIndex; }

bool cAppliDigeo::doIncrementalConvolution() const { return mDoIncrementalConvolution; }

void cAppliDigeo::processImageName()
{
	#ifdef __DEBUG_DIGEO
		ELISE_ASSERT( ImageDigeo().size()==1, ( string("setImageFullname: only one file can be processed at a time (nb ImageDigeo = ")+ToString((int)ImageDigeo().size()) ).c_str() );
	#endif
	for ( std::list<cImageDigeo>::const_iterator itI=Params().ImageDigeo().begin(); itI!=Params().ImageDigeo().end(); itI++ ) 
		mImageFullname = mICNM->Dir()+itI->KeyOrPat();
	SplitDirAndFile( mImagePath, mImageBasename, mImageFullname );
}

string cAppliDigeo::outputTileBasename( int i_iTile ) const
{
	stringstream ss;
	ss << mImageBasename << "_tile" << setfill('0') << setw(3) << i_iTile;
	return ss.str();
}

string cAppliDigeo::currentTileBasename() const
{
	return outputTileBasename( currentBoxIndex() );
}

string cAppliDigeo::currentTileFullname() const
{
	return outputTilesDirectory()+"/"+currentTileBasename();
}

void removeEndingSlash( string &i_str )
{
	if ( i_str.length()==0 ) return;
	const char c = *i_str.rbegin();
	if ( c=='/' || c=='\\' ) i_str.resize(i_str.length()-1);
}

void cAppliDigeo::processTestSection()
{
	const cSectionTest *mSectionTest = ( Params().SectionTest().IsInit()?&Params().SectionTest().Val():NULL );
	mDoSaveGaussians = mDoSaveTiles = false;
	if ( mSectionTest!=NULL && mSectionTest->DigeoTestOutput().IsInit() ){
		const cDigeoTestOutput &testOutput = mSectionTest->DigeoTestOutput().Val();
		if ( testOutput.OutputGaussians().IsInit() && testOutput.OutputGaussians().Val() ){
			mDoSaveGaussians = true;
			mOutputGaussiansDirectory = testOutput.OutputGaussiansDirectory().ValWithDef("");
			removeEndingSlash( mOutputGaussiansDirectory );
		}
		if ( testOutput.OutputTiles().IsInit() && mSectionTest->OutputTiles().Val() ){
			mDoSaveTiles = true;
			mOutputTilesDirectory = testOutput.OutputTilesDirectory().ValWithDef("");
			removeEndingSlash( mOutputTilesDirectory );
		}
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
