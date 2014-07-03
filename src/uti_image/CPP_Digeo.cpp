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

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#define DIGEO_ORIENTATION_NB_BINS 36
#define DIGEO_ORIENTATION_NB_MAX_ANGLES 4
#define DIGEO_ORIENTATION_WINDOW_FACTOR 1.5
#define DIGEO_DESCRIBE_NBO 8
#define DIGEO_DESCRIBE_NBP 4
#define DIGEO_DESCRIBE_MAGNIFY 3.
#define DIGEO_DESCRIBE_THRESHOLD 0.2

// fait le boulot pour une octave quand ses points d'intéret ont été calculés
// ajoute les points à o_list (on passe la même liste à toutes les octaves)
template <class Type,class tBase> void orientate_and_describe_all(cTplOctDig<Type> * anOct, list<DigeoPoint> &o_list);

// calcul le gradient d'une image, le résultat est forcément en REAL4, peut-être faut-il la templatiser
// le gradient fait le double de la largeur de l'image source car il contient deux valeurs pour un pixel (la norme puis l'orientation du gradient)
template <class tData, class tComp>
void gradient( const Im2D<tData,tComp> &i_image, Im2D<REAL4,REAL8> &o_gradient );

// calcul les angles d'orientation pour un point (au plus DIGEO_NB_MAX_ANGLES angles)
int orientate( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracDigeo &i_p, REAL8 o_angles[DIGEO_ORIENTATION_NB_MAX_ANGLES] );

// calcul le descripteur d'un point
void describe( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracDigeo &i_p, REAL8 i_angle, REAL8 *o_descriptor );

// = normalizeDescriptor + truncateDescriptor + normalizeDescriptor
void normalize_and_truncate( REAL8 *io_descriptor );

void normalizeDescriptor( REAL8 *io_descriptor );
// tronque à DIGEO_DESCRIBE_THRESHOLD
void truncateDescriptor( REAL8 *io_descriptor );

// lit/écrit une liste de point au format siftpp_tgi
inline void write_DigeoPoint_binary_legacy( std::ostream &output, const DigeoPoint &p );
inline void read_DigeoPoint_binary_legacy( std::istream &output, DigeoPoint &p );
// lit/écrit un point au format siftpp_tgi
bool write_digeo_points( const string &i_filename, const list<DigeoPoint> &i_list );
bool read_digeo_points( const string &i_filename, vector<DigeoPoint> &o_list );

//----------

template <class tData, class tComp>
void gradient( const Im2D<tData,tComp> &i_image, REAL8 i_maxValue, Im2D<REAL4,REAL8> &o_gradient )
{
    o_gradient.Resize( Pt2di( i_image.sz().x*2, i_image.sz().y ) );

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

// return the number of possible orientations (cannot be greater than DIGEO_NB_MAX_ANGLES)
int orientate( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracDigeo &i_p, REAL8 o_angles[DIGEO_ORIENTATION_NB_MAX_ANGLES] )
{
	static REAL8 histo[DIGEO_ORIENTATION_NB_BINS];

	int xi = ((int) (i_p.mPt.x+0.5)) ;
    int yi = ((int) (i_p.mPt.y+0.5)) ;

    const REAL8 sigmaw = DIGEO_ORIENTATION_WINDOW_FACTOR*i_p.mScale;
	const int W = (int)floor( 3*sigmaw );

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
            r2 = dx*dx+dy*dy ;

            // limit to a circular window
            if ( r2>=W*W+0.5 ) continue;
    
            wgt    = ::exp( -r2/( 2*sigmaw*sigmaw ) );
            offset = ( xs+ys*width )*2;
            mod    = p[offset];
            ang    = p[offset+1];

            int bin = (int) floor( DIGEO_ORIENTATION_NB_BINS*ang/( 2*M_PI ) ) ;
            histo[bin] += mod*wgt ;

        }
    }

    REAL8 prev;
    // smooth histogram  (Vedaldi style)
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
        while ( iHisto-- ){
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
        v    = histo[i];
        prev = histo[ ( i==0 )?DIGEO_ORIENTATION_NB_BINS-1:i-1 ];
        next = histo[ ( i==( DIGEO_ORIENTATION_NB_BINS-1 ) )?0:i+1 ];
        if ( ( v>histoMax ) && ( v>prev ) && ( v>next ) )
        {
            // we found a peak
            // compute angle by quadratic interpolation
            di = -0.5*( next-prev )/( next+prev-2*v ) ;
            o_angles[nbAngles++] = 2*M_PI*( i+di+0.5 )/DIGEO_ORIENTATION_NB_BINS;
            if ( nbAngles==DIGEO_ORIENTATION_NB_MAX_ANGLES ) return DIGEO_ORIENTATION_NB_MAX_ANGLES;
        }
    }
    return nbAngles;
}

#define atd(dbinx,dbiny,dbint) *(dp + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

// o_descritpor must be of size DIGEO_DESCRIPTOR_SIZE
void describe( const Im2D<REAL4,REAL8> &i_gradient, cPtsCaracDigeo &i_p, REAL8 i_angle, REAL8 *o_descriptor )
{
    REAL8 st0 = sinf( i_angle ),
		  ct0 = cosf( i_angle );

	int xi = int( i_p.mPt.x+0.5 );
    int yi = int( i_p.mPt.y+0.5 );

    const REAL8 SBP = DIGEO_DESCRIBE_MAGNIFY*i_p.mScale;
    const int  W   = (int)floor( sqrt( 2.0 )*SBP*( DIGEO_DESCRIBE_NBP+1 )/2.0+0.5 );

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

// same format as siftpp_tgi (not endian-wise)
// Real_ values are cast to float
// descriptors are cast from float to unsigned char values d[i]->(unsigned char)(512*d[i])
inline void write_DigeoPoint_binary_legacy( std::ostream &output, const DigeoPoint &p )
{
	float float_value = (float)p.x; output.write( (char*)&float_value, sizeof( float ) );
	float_value = (float)p.y; output.write( (char*)&float_value, sizeof( float ) );
	float_value = (float)p.scale; output.write( (char*)&float_value, sizeof( float ) );
	float_value = (float)p.angle; output.write( (char*)&float_value, sizeof( float ) );
	static unsigned char uchar_desc[DIGEO_DESCRIPTOR_SIZE];
	int i=DIGEO_DESCRIPTOR_SIZE; const REAL8 *itReal=p.descriptor; unsigned char *it_uchar=uchar_desc;
	while (i--)	(*it_uchar++)=(unsigned char)( 512*(*itReal++) );
    output.write( (char*)uchar_desc, DIGEO_DESCRIPTOR_SIZE );
}

// same format as siftpp_tgi (not endian-wise)
// Real_ values are cast to float
// descriptors are cast from unsigned char to float values d[i]->d[i]/512
inline void read_DigeoPoint_binary_legacy( std::istream &output, DigeoPoint &p )
{
	float float_values[4];
	output.read( (char*)float_values, 4*sizeof( float ) );
	p.x 	= (REAL8)float_values[0];
	p.y 	= (REAL8)float_values[1];
	p.scale = (REAL8)float_values[2];
	p.angle = (REAL8)float_values[3];
	static unsigned char uchar_desc[DIGEO_DESCRIPTOR_SIZE];
	const REAL8 k = REAL8(1)/REAL8(512);
    output.read( (char*)uchar_desc, DIGEO_DESCRIPTOR_SIZE );
	int i=DIGEO_DESCRIPTOR_SIZE; REAL8 *itReal=p.descriptor; unsigned char *it_uchar=uchar_desc;
	while (i--) (*itReal++)=(REAL8)(*it_uchar++)*k;
}

bool write_digeo_points( const string &i_filename, const list<DigeoPoint> &i_list )
{
	// write in siftpp_tgi format
	
    ofstream f( i_filename.c_str(), ios::binary );

    if ( !f ) return false;

    U_INT4 nbPoints  = i_list.size(),
		   dimension = DIGEO_DESCRIPTOR_SIZE;
    f.write( (char*)&nbPoints, 4 );
    f.write( (char*)&dimension, 4 );
    list<DigeoPoint>::const_iterator it = i_list.begin();
    while ( nbPoints-- )
        write_DigeoPoint_binary_legacy( f, *it++ );
    f.close();
    return true;
}

bool read_digeo_points( const string &i_filename, vector<DigeoPoint> &o_list )
{
	// read in siftpp_tgi format
	
    ifstream f( i_filename.c_str(), ios::binary );

    if ( !f ) return false;

    U_INT4 nbPoints, dimension;
    f.read( (char*)&nbPoints, 4 );
    f.read( (char*)&dimension, 4 );
    
    o_list.resize( nbPoints );
    if ( dimension!=DIGEO_DESCRIPTOR_SIZE ){
		cerr << "ERROR: read_siftPoint_list " << i_filename << ": descriptor's dimension is " << dimension << " and should be " << DIGEO_DESCRIPTOR_SIZE << endl;
		return false;
	}
	if ( nbPoints==0 ) return true;
	DigeoPoint *itPoint = &o_list[0];
    while ( nbPoints-- )
        read_DigeoPoint_binary_legacy( f, *itPoint++ );
    f.close();

    return true;
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

template <class Type,class tBase> void orientate_and_describe_all(cTplOctDig<Type> * anOct, list<DigeoPoint> &o_list)
{
  Im2D<REAL4,REAL8> imgGradient;
  REAL8 angles[DIGEO_ORIENTATION_NB_MAX_ANGLES];
  DigeoPoint p;
  int nbAngles;
  const std::vector<cTplImInMem<Type> *> &  aVIm = anOct->cTplOctDig<Type>::VTplIms();
  
  for (int aKIm=0 ; aKIm<int(aVIm.size()) ; aKIm++)
  {
       cTplImInMem<Type> & anIm = *(aVIm[aKIm]);
       Im2D<Type,tBase> aTIm = anIm.TIm();
       //std::cout << "   #  Sz " << aTIm.sz() << " SInit:" <<  anIm.ScaleInit() << " SOct:" << anIm.ScaleInOct() << endl;

	   p.scale = anIm.ScaleInit();
       std::vector<cPtsCaracDigeo> &  aVPC = anIm.VPtsCarac();
       //std::cout << "NB PTS " <<aVPC.size()<<endl;
	   if ( aVPC.size()!=0 )
	   {
		   gradient( aTIm, anOct->GetMaxValue(), imgGradient );
		   for ( unsigned int i=0; i<aVPC.size(); i++ )
		   {
			   p.x=aVPC[i].mPt.x*anOct->Niv(); p.y=aVPC[i].mPt.y*anOct->Niv(); p.type=(DigeoPoint::DetectType)aVPC[i].mType;
			   nbAngles = orientate( imgGradient, aVPC[i], angles );
			   if ( nbAngles!=0 )
			   {
				   for ( int iAngle=0; iAngle<nbAngles; iAngle++ ){
						describe( imgGradient, aVPC[i], angles[iAngle], p.descriptor );
						normalize_and_truncate( p.descriptor );
						p.angle = angles[iAngle];
						o_list.push_back( p );
				   }
			   }
		   }
	   }
  }
}

int Digeo_main( int argc, char **argv )
{
	if ( argc!=4 ){
		cerr << "Digeo: usage : mm3d Digeo input_filename -o output_filename" << endl;
		return EXIT_FAILURE;
	}
    std::string inputName  = argv[1];
	std::string outputName = argv[3];

	cParamAppliDigeo aParam;
    aParam.mSauvPyram = false;
    aParam.mSigma0 = 1.0;

    aParam.mResolInit = 1.0;

    cAppliDigeo * anAD = DigeoCPP(inputName,aParam);
    cImDigeo &  anImD = anAD->SingleImage(); // Ici on ne mape qu'une seule image à la fois

    list<DigeoPoint> total_list;

    for (int aKBox = 0 ; aKBox<anAD->NbInterv() ; aKBox++)
    {
        anAD->LoadOneInterv(aKBox);  // Calcul et memorise la pyramide gaussienne
        const std::vector<cOctaveDigeo *> & aVOct = anImD.Octaves();
        
        for (int aKo=0 ; aKo<int(aVOct.size()) ; aKo++)
        {
            cOctaveDigeo & anOct = *(aVOct[aKo]);
            cTplOctDig<U_INT2> * aUI2_Oct = anOct.U_Int2_This();  // entre aUI2_OctaUI2_Oct et  aR4_Oct
            cTplOctDig<REAL4> * aR4_Oct = anOct.REAL4_This();     // un et un seul doit etre != 0
            
			if ( anAD->mVerbose )
			{
				cout << "\tOctave " << aKo << " of type ";
				if (aUI2_Oct !=0) cout << "uint2" << endl;
				else if (aR4_Oct !=0) cout << "real4" << endl;
				else cout << "unknown" << endl;
			}
			
			anOct.DoAllExtract();
			
			#ifdef __DEBUG_DIGEO_STATS
			   if ( anAD->mVerbose && ( anOct.VIms().size()!=0 ) )
			   {
				   size_t iImg = anOct.VIms().size(),
					      countRefined     = 0,
					      countUncalc      = 0,
					      countInstable    = 0,
					      countInstable2   = 0,
					      countInstable3   = 0,
					      countGradFaible  = 0,
					      countTropAllonge = 0,
					      countOk          = 0,
					      countExtrema;
				   cImInMem *const *itImg = &( anOct.VIms()[0] );
				   while ( iImg-- )
				   {
					   countRefined     += ( *itImg )->VPtsCarac().size();
					   countUncalc      += ( *itImg )->mCount_eTES_Uncalc;
					   countInstable    += ( *itImg )->mCount_eTES_instable_unsolvable;
					   countInstable2   += ( *itImg )->mCount_eTES_instable_tooDeepRecurrency;
					   countInstable3   += ( *itImg )->mCount_eTES_instable_outOfImageBound;
					   countGradFaible  += ( *itImg )->mCount_eTES_GradFaible;
					   countTropAllonge += ( *itImg )->mCount_eTES_TropAllonge;
					   countOk          += ( *itImg )->mCount_eTES_Ok;
					   itImg++;
				   }
				   countExtrema = countInstable+countInstable2+countInstable3+countGradFaible+countTropAllonge+countOk;
				   cout << "\t\textrema detected                    \t" << countExtrema << endl;
				   cout << "\t\tafter refinement and on-edge removal\t" << countRefined << endl;
				   cout << "\t\t------------------------------------" << endl;
				   cout << "\t\teTES_Uncalc                       \t" << countUncalc << endl;
				   cout << "\t\teTES_instable_unsolvable          \t" << countInstable << endl;
				   cout << "\t\teTES_instable_tooDeepRecurrency   \t" << countInstable2 << endl;
				   cout << "\t\teTES_instable_outOfImageBound     \t" << countInstable3 << endl;
				   cout << "\t\teTES_GradFaible                   \t" << countGradFaible << endl;
				   cout << "\t\teTES_TropAllonge                  \t" << countTropAllonge << endl;
				   cout << "\t\teTES_Ok                           \t" << countOk << endl;
			   }
			#endif
			
			if (aUI2_Oct !=0)
				orientate_and_describe_all<U_INT2,INT>(aUI2_Oct, total_list);
			if (aR4_Oct !=0) 
			orientate_and_describe_all<REAL4,REAL8>(aR4_Oct, total_list);
        }
    }
	cout << total_list.size() << " points" << endl;
	if ( !write_digeo_points( outputName, total_list ) ){
		cerr << "Digeo: ERROR: unable to save points to file " << outputName << endl;
	}
	return EXIT_SUCCESS;
}





/*
        bool     mExigeCodeCompile;
        int      mNivFloatIm;        // Ne depend pas de la resolution
*/


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
