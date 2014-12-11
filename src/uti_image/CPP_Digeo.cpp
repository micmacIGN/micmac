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

#include "Digeo/Digeo.h"

// fait le boulot pour une octave quand ses points d'intéret ont été calculés
// ajoute les points à o_list (on passe la même liste à toutes les octaves)
template <class Type,class tBase> void orientate_and_describe_all(cTplOctDig<Type> * anOct, list<DigeoPoint> &o_list);

// calcul le descripteur d'un point
void describe( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracDigeo &i_p, REAL8 i_angle, REAL8 *o_descriptor );

#define atd(dbinx,dbiny,dbint) *(dp + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

// o_descritpor must be of size DIGEO_DESCRIPTOR_SIZE
void describe( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracDigeo &i_p, REAL8 i_angle, REAL8 *o_descriptor )
{
    REAL8 st0 = sinf( i_angle ),
		  ct0 = cosf( i_angle );

	int xi = int( i_p.mPt.x+0.5 );
    int yi = int( i_p.mPt.y+0.5 );

    const REAL8 SBP = DIGEO_DESCRIBE_MAGNIFY*i_p.mLocalScale;
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
            dx = xi+dxi-i_p.mPt.x;
            dy = yi+dyi-i_p.mPt.y;

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

template <class Type,class tBase> void orientate_and_describe_all(cTplOctDig<Type> * anOct, list<DigeoPoint> &o_list)
{
  Im2D<REAL4,REAL8> imgGradient;
  DigeoPoint p;
  const std::vector<cTplImInMem<Type> *> &  aVIm = anOct->cTplOctDig<Type>::VTplIms();
  double trueSamplingPace = anOct->Niv()*anOct->ImDigeo().Resol();
  REAL8 angles[DIGEO_MAX_NB_ANGLES];
  int nbAngles;

  for (int aKIm=0 ; aKIm<int(aVIm.size()) ; aKIm++)
  {
       cTplImInMem<Type> & anIm = *(aVIm[aKIm]);
       Im2D<Type,tBase> aTIm = anIm.TIm();

		p.scale = anIm.ScaleInit();
		std::vector<cPtsCaracDigeo> &  aVPC = anIm.featurePoints();
		if ( aVPC.size()!=0 )
		{
			gradient( aTIm, anOct->GetMaxValue(), imgGradient );

			for ( unsigned int i=0; i<aVPC.size(); i++ ){
				p.x = aVPC[i].mPt.x*trueSamplingPace;
				p.y = aVPC[i].mPt.y*trueSamplingPace;
				switch ( aVPC[i].mType ){
				case eSiftMaxDog: p.type=DigeoPoint::DETECT_LOCAL_MAX; break;
				case eSiftMinDog: p.type=DigeoPoint::DETECT_LOCAL_MIN; break;
				default: p.type=DigeoPoint::DETECT_UNKNOWN; break;
				}
				aVPC[i].mLocalScale = aVPC[i].mScale/trueSamplingPace;
				nbAngles = orientate( imgGradient, aVPC[i], angles );
				if ( nbAngles!=0 ){
					p.entries.resize(nbAngles);
					for ( int iAngle=0; iAngle<nbAngles; iAngle++ ){
						DigeoPoint::Entry &entry = p.entry(iAngle);
						describe( imgGradient, aVPC[i], entry.angle, entry.descriptor );
						normalize_and_truncate( entry.descriptor );
					}
					o_list.push_back( p );
			   }
		   }
	   }
  }
}

// add i_v to the coordinates of the i_nbPoints last points of io_points
void translate_points( list<DigeoPoint> &io_points, size_t i_nbPoints, const Pt2di &i_v )
{
	#ifdef __DEBUG_DIGEO
		if ( i_nbPoints>io_points.size() ){ cerr << "translate_points: trying to translate " << i_nbPoints << " out of " << io_points.size() << endl; exit(EXIT_FAILURE); }
	#endif
	const double tx = (double)i_v.x,
	             ty = (double)i_v.y;
	list<DigeoPoint>::reverse_iterator itPoint = io_points.rbegin();
	while ( i_nbPoints-- ){
		itPoint->x = itPoint->x+tx;
		itPoint->y = itPoint->y+ty;
		itPoint++;
	}
}

template <class T>
bool generate_convolution_code( cAppliDigeo &i_appli )
{
	if ( i_appli.nbSlowConvolutionsUsed<T>()==0 ) return true;

	const string typeName = El_CTypeTraits<T>::Name();
	if ( i_appli.isVerbose() ) cout << "WARNING: " << i_appli.nbSlowConvolutionsUsed<T>() << " slow convolutions of type " << typeName << " have been used" << endl;

	string lowerTypeName = El_CTypeTraits<T>::Name();
	for ( size_t i=0; i<lowerTypeName.length(); i++ ) lowerTypeName[i] = ::tolower(lowerTypeName[i]);

	string classFilename = i_appli.getConvolutionClassesFilename( lowerTypeName );
	string instantiationsFilename = i_appli.getConvolutionInstantiationsFilename( lowerTypeName );
	if ( !ELISE_fp::exist_file( classFilename ) || !ELISE_fp::exist_file( instantiationsFilename ) ){
		cout << "WARNING: source code do not seem to be available, no convolution code generated for type " << typeName << endl;
		return false;
	}
	
	if ( !cConvolSpec<T>::generate_classes( classFilename ) ){
		cout << "WARNING: generated convolution couldn't be saved to " << classFilename << endl;
		return false;
	}

	if ( !cConvolSpec<T>::generate_instantiations( instantiationsFilename ) ){
		cout << "WARNING: generated convolution couldn't be saved to " << instantiationsFilename << endl;
		return false;
	}
	if ( i_appli.isVerbose() ) cout << "convolution code has been generated for type " << typeName << ", compile again to improve speed with the same parameters" << endl;
	return true;
}

int Digeo_main( int argc, char **argv )
{
	ElTimer chrono;

	if ( argc!=4 )
	{
		cerr << "Digeo: usage : mm3d Digeo input_filename -o output_filename" << endl;
		return EXIT_FAILURE;
	}

	std::string inputName  = argv[1];
	std::string outputName = argv[3];

	cAppliDigeo appli;
	appli.times()->start();
	appli.loadImage( inputName );
	appli.times()->stop("pyramid structure");

	cImDigeo &image = appli.getImage();

    if ( appli.isVerbose() )
    {
       cout << "number of tiles : " << appli.NbInterv() << endl;
       cout << "tile size : " << appli.Params().DigeoDecoupageCarac().Val().SzDalle() << endl;
       cout << "margin : " << appli.Params().DigeoDecoupageCarac().Val().Bord() << endl;
    }

    list<DigeoPoint> total_list;
    for (int aKBox = 0 ; aKBox<appli.NbInterv() ; aKBox++)
    {
        appli.LoadOneInterv(aKBox);  // Calcul et memorise la pyramide gaussienne

        if ( !appli.doComputeCarac() ) continue;

        Box2di box = appli.getInterv( aKBox );
        if ( appli.isVerbose() ) cout << "processing tile " << aKBox << " of origin " << box._p0 << " and size " << box.sz() << endl;

        image.detect();

        if ( appli.isVerbose() ) cout << "\t" << image.getNbFeaturePoints() << " feature points" << endl;

        image.orientateAndDescribe();

        if ( appli.doPlotPoints() ) image.plotPoints();

        size_t nbTilePoints = image.addAllPoints( total_list );
        translate_points( total_list, nbTilePoints, box._p0 );

        if ( appli.isVerbose() ) cout << "\t" << nbTilePoints << " points" << endl;
    }

	appli.mergeOutputs();

	if ( appli.doGenerateConvolutionCode() )
	{
		generate_convolution_code<U_INT2>( appli );
		generate_convolution_code<REAL4>( appli );
	}
	else if ( appli.isVerbose() && ( appli.nbSlowConvolutionsUsed<U_INT2>()!=0 || appli.nbSlowConvolutionsUsed<REAL4>()!=0 ) )
		cout << "skipping convolution code generation" << endl;

	ElTimer chronoSave;
	cout << total_list.size() << " points" << endl;
	if ( !DigeoPoint::writeDigeoFile( outputName, total_list ) ) cerr << "Digeo: ERROR: unable to save points to file " << outputName << endl;
	double saveTime = chronoSave.uval();

	if ( appli.doShowTimes() )
	{
		MapTimes *times = (MapTimes*)appli.times();
		double totalMonitoredTime = saveTime+times->totalTime();

		cout << "Total time = " << chrono.uval() << " (" << totalMonitoredTime << ")" << endl;
		times->printTimes( "\t" );
		cout << "\tsave :" << saveTime << endl;
	}

	cout << "nb computed gradient = " << appli.nbComputedGradients() << endl;

	return EXIT_SUCCESS;
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
