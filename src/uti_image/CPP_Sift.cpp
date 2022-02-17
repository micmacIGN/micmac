#include <iostream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <limits>

#include "Sift.h"

using namespace std;

class sift_parameters_t {
public:
	bool   verbose;
	bool   print_help;
	string output_filename;
	string prefix;
	string keypoints_filename;
	bool   save_gaussians;
	int    nb_octaves;
	int    nb_levels;
	int    first_octave;
	Real_  strength_threshold;
	int    onedge_threshold;
	bool   no_descriptors;
	bool   no_orientations;
	int    chunk_size;
	string basename;    // current image filename without extension (used to output gaussians)
    
	sift_parameters_t():
		verbose( false ),
		print_help( false ),
		save_gaussians( false ),
		nb_octaves( 7 ),
		nb_levels( 3 ),
		first_octave( -1 ),
		strength_threshold( default_strength_threshold ),
		onedge_threshold( default_onedge_threshold ),
		no_descriptors( false ),
		no_orientations( false ),
		chunk_size( 1024 ){}
};

void print_usage();

inline void verbose_func        ( const string &val, sift_parameters_t &p ){ p.verbose=true; }
inline void help_func           ( const string &val, sift_parameters_t &p ){ print_usage(); exit(EXIT_SUCCESS); }
inline void prefix_func         ( const string &val, sift_parameters_t &p ){ p.prefix=val; }
inline void output_func         ( const string &val, sift_parameters_t &p ){ p.output_filename=val; }
inline void keypoints_func      ( const string &val, sift_parameters_t &p ){ /*p.keypoints_filename=true;*/ cerr << "keypoints argument is not implemented yet" << endl; exit(EXIT_FAILURE); }
inline void save_gss_func       ( const string &val, sift_parameters_t &p ){ p.save_gaussians=true; }
inline void octaves_func        ( const string &val, sift_parameters_t &p ){ p.nb_octaves=atoi( val.c_str() ); }
inline void levels_func         ( const string &val, sift_parameters_t &p ){ p.nb_levels=atoi( val.c_str() ); }
inline void first_octave_func   ( const string &val, sift_parameters_t &p ){ p.first_octave=atoi( val.c_str() ); }
inline void threshold_func      ( const string &val, sift_parameters_t &p ){ p.strength_threshold=atof( val.c_str() ); }
inline void edge_threshold_func ( const string &val, sift_parameters_t &p ){ p.nb_levels=atoi( val.c_str() ); }
inline void chunk_size_func     ( const string &val, sift_parameters_t &p ){ p.chunk_size=atoi( val.c_str() ); }
inline void no_orientations_func( const string &val, sift_parameters_t &p ){ p.no_orientations = (atoi(val.c_str()) != 0); }
inline void no_descriptors_func ( const string &val, sift_parameters_t &p ){ p.no_descriptors = true; }

typedef struct {
    const string long_name;
    char         short_name;
    bool         needValue;
    const string description;
    void         (*func)( const string &, sift_parameters_t & );
} parameter_t;

parameter_t g_parameters_list[] = {
    { "verbose",         'v',  false, "verbose mode", verbose_func },
    { "help",            'h',  false, "print help", help_func },
    { "prefix",          'p',  true,  "prefix output filename", prefix_func },
    { "output",          'o',  true,  "set output filename", output_func },
    { "keypoints",       'k',  true,  "load a key-points set and compute their descriptors", keypoints_func },
    { "save-gss",        's',  false, "save gausians", save_gss_func },
    { "octaves",         'O',  true,  "set number of octaves", octaves_func },
    { "levels",          'S',  true,  "set number of level per octave", levels_func },
    { "first-octave",    'f',  true,  "set first octave index", first_octave_func },
    { "threshold",       't',  true,  "set detection threshold", threshold_func },
    { "edge-threshold",  'e',  true,  "set on-edge threshold", edge_threshold_func },
    { "chunk-size",      'c',  true,  "set tile size", chunk_size_func },
    { "no-orientations", '\0', false, "only detect key-points without angles and descriptors", no_orientations_func },
    { "no-descriptors",  '\0', false, "detect key-points and compute their angles", no_descriptors_func },
    { "",                '\0', false, "", NULL },
};

void print_usage()
{
	cout << "usage: mm3d Sift input_image" << endl;
	parameter_t *itParam = g_parameters_list;
	while ( itParam->func!=NULL )
	{
		cout << "--" << itParam->long_name;
		if ( itParam->needValue ) cout << " value";
		if ( itParam->short_name!='\0' ) cout << "(-" << itParam->short_name << ')';
		cout << ": " << itParam->description << endl;
		itParam++;
	}
}

void process_image( const RealImage1 &i_image, const sift_parameters_t &i_params, const string &i_imageBasename, Siftator &i_gaussPyramid, list<SiftPoint> &o_siftPoints )
{
    i_gaussPyramid.compute_gaussians( i_image );

    if ( i_params.save_gaussians )
        i_gaussPyramid.save_gaussians( i_imageBasename, i_params.verbose );

    // detect extremum
    list<Extremum> extrema;
    list<RefinedPoint> refinedPoints;
    SiftPoint siftPoint;
    list<RefinedPoint>::iterator itRefinedPoint;
    int nbAngles, iAngle;
    Real_ angles[m_maxNbAngles];
    Real_ descriptor[SIFT_DESCRIPTOR_SIZE];
    for ( int o=0; o<i_params.nb_octaves; o++  )
    {
		if ( i_params.verbose ) cout << "\toctave " << o+i_params.first_octave << endl;
		
        i_gaussPyramid.setCurrentOctave( o );
        i_gaussPyramid.compute_differences_of_gaussians();

        extrema.clear();
        refinedPoints.clear();

        i_gaussPyramid.getExtrema( extrema );
        
		if ( i_params.verbose ) cout << "\t\textrema detected\t\t\t" << extrema.size() << endl;
        
        i_gaussPyramid.refinePoints( extrema, refinedPoints );

		if ( i_params.verbose ) cout << "\t\tafter refinement and on-edge removal\t" << refinedPoints.size() << endl;
        
        int iRefinedPoint = (int)refinedPoints.size();
        if ( iRefinedPoint!=0 )
        {
            // compute orientation
            i_gaussPyramid.compute_gradients();

            list<RefinedPoint>::iterator itRefinedPoint = refinedPoints.begin();
            while ( iRefinedPoint-- )
            {
                nbAngles = i_gaussPyramid.orientations( *itRefinedPoint, angles );
                for ( iAngle=0; iAngle<nbAngles; iAngle++ )
                {
                    i_gaussPyramid.descriptor( *itRefinedPoint, angles[iAngle], descriptor );

                    Siftator::normalizeDescriptor( descriptor );
                    Siftator::truncateDescriptor( descriptor );
                    Siftator::normalizeDescriptor( descriptor );

                    siftPoint.x = itRefinedPoint->rx;
                    siftPoint.y = itRefinedPoint->ry;
                    siftPoint.scale = itRefinedPoint->rs;
                    siftPoint.angle = angles[iAngle];
                    memcpy( siftPoint.descriptor, descriptor, SIFT_DESCRIPTOR_SIZE*sizeof( Real_ ) );

                    o_siftPoints.push_back( siftPoint );
                }
                itRefinedPoint++;
            }
        }
    }
}

// long names alternate assignation is for exemple with --prefix argument --prefix=VALUE
inline bool getEqualValue( string &io_str, string &o_value ){
    size_t pos = io_str.find( '=' );
    if ( pos==string::npos || pos==io_str.length()-1 ){
        io_str = io_str.substr( 2 );
        return false;
    }
    o_value = io_str.substr( pos+1 );
    io_str = io_str.substr( 2, pos-2 );
    return true;
}

// short names alternate assignation is for exemple with -p argument : -pVALUE
inline bool getJointValue( const string &i_str, string &o_value ){
    if ( i_str.length()<3 ) return false;
    o_value = i_str.substr( 2 );
    return true;
}

#define USE_NEXT_ARG_AS_VALUE( count, args, value )\
{\
	if ( (count)==0 )\
	{\
		cerr << "ERROR: argument " << (arg) << " needs a value but none is specified" << endl;\
		return false;\
	}\
	(value) = *(args)++;\
	(count)--;\
}

// returns if all arguments could be processed or not
bool process_arguments( int i_argc, char **i_argv, sift_parameters_t &o_parameters, list<string> &o_image_list )
{
	string arg, value;
	parameter_t *itParam = NULL;
	bool matchFound;
	while ( i_argc-- )
	{
		arg = *i_argv++;

		if ( arg.length()==0 ) continue; // should not be necessary

		if ( arg[0]!='-' )
		{
		// this is not an argument, then its a filename
		o_image_list.push_back(arg);
		continue;
		}

		if ( arg.length()<2 ){ cerr << "missing name parameter after '-'" << endl; return false; }

		matchFound = false;
		value.clear();
		if ( arg[1]=='-' )
		{
			// this is a long-form argument
			if ( arg.length()<3 ){ cerr << "missing parameter name after \"--\"" << endl; return false; }
			bool foundEqualValue = getEqualValue( arg, value );
			itParam = g_parameters_list;
			while ( itParam->func!=NULL )
			{
				if ( arg==itParam->long_name )
				{
					if ( itParam->needValue && !foundEqualValue ) USE_NEXT_ARG_AS_VALUE( i_argc, i_argv, value );
					matchFound = true;
					break;
				}
				itParam++;
			}
		}
		else
		{
			// this is a short-form argument
			itParam = g_parameters_list;
			char arg1 = arg[1];
			while ( itParam->func!=NULL )
			{
				if ( arg1==itParam->short_name )
				{
					if ( itParam->needValue && !getJointValue( arg, value ) ) USE_NEXT_ARG_AS_VALUE( i_argc, i_argv, value );
					matchFound = true;
					break;
				}
				itParam++;
			}
		}
		if ( !matchFound ){ cerr << "ERROR: parameter " << i_argv[-1] << " does not exit" << endl; return false; }

		itParam->func( value, o_parameters );
	}

	return true;
}

inline void toupper( const string &i_str, string &o_str )
{
    size_t i = i_str.length();
    o_str.resize( i_str.length() );
    string::const_iterator src = i_str.begin();
    string::iterator dst = o_str.begin();
    while ( i-- ) ( *dst++ )=toupper(*src++);
}

// o_ext is set to the upper-case of the shortest extension (after the last point)
// o_base is the rest of string
// return if it's degenerated case of not (ie. there's no extension or the whole string is the extension)
// in this case, o_base is set to i_str and o_ext is set to an empty string
inline bool getExtension( const string &i_str, string &o_base, string &o_ext )
{
    size_t pos = i_str.find_last_of( '.' );
    if ( pos==string::npos ||
         pos==0 ||
         pos==i_str.length()-1 ){
        o_base=i_str;
        o_ext.clear();
        return false;
    }
    toupper( i_str.substr( pos+1 ), o_ext );
    o_base = i_str.substr( 0, pos );
    return true;
}

// replace the longest prefix of string ending by a '/' or a '\' with i_prefix
// return false if it's a degenerated case (ie the whole string is the prefix)
inline bool replace_prefix( string &io_str, const string &i_prefix )
{
    size_t pos = io_str.find_last_of( "/\\" );
    if ( pos==io_str.length()-1 ) return false;
    if ( pos!=string::npos ) io_str=io_str.substr( pos+1 );
    io_str = i_prefix+io_str;
    return true;
}

int Sift_main( int argc, char**argv )
{
    sift_parameters_t parameters;
    list<string>      image_list;

    if ( !process_arguments( argc-1, argv+1, parameters, image_list ) )
        return EXIT_FAILURE;

    if ( image_list.empty()){ cerr << "ERROR: no intput image file" << endl; exit(-1); }
    if ( parameters.output_filename.length()!=0 )
    {
        if ( image_list.size()>1 ){ cerr << "ERROR: an output filename is specified but there is more than one image file" << endl; exit(-1); }
        if ( parameters.prefix.length()!=0 ){ cerr << "ERROR: --prefix and --output are incompatible parameters" << endl; exit(-1); }
    }

    /*
    // compute a value for nbOctaves depending of image's size
    if ( parameters.nb_octaves<0 )
        #ifdef __ORIGINAL__
            parameters.nb_octaves = std::max( int( std::floor( log2( std::min( i_image.width(), i_image.height() ) ) )-parameters.first_octave-3 ), 1 ) ;
        #else
            parameters.nb_octaves = std::max( int( std::floor( log2( std::min( i_image.width(), i_image.height() ) ) )-parameters.first_octave-4 ), 1 ) ;
        #endif
    */

    Siftator gaussPyramid( parameters.nb_octaves, parameters.nb_levels, parameters.first_octave );
    gaussPyramid.setStrengthThreshold( parameters.strength_threshold );
    gaussPyramid.setOnEdgeThreshold( parameters.onedge_threshold );

    if ( parameters.verbose ){
        cout << "-> extra parameters" << endl;
        cout << "chunk size                                  : " << parameters.chunk_size << endl;
        cout << "compute orientations                        : " << ( parameters.no_orientations?"no":"yes" ) << endl;
        cout << "compute descriptors                         : " << ( parameters.no_descriptors?"no":"yes" ) << endl;
        cout << "saving gaussians                            : " << ( parameters.save_gaussians?"yes":"no" ) << endl;
        cout << "-> sift parameters" << endl;
        gaussPyramid.print_parameters( cout );
    }

    string outname, outputBasename, extension;
    RealImage1 image;
    list<SiftPoint> siftPoints;
    for ( list<string>::iterator itFilename=image_list.begin(); itFilename!=image_list.end(); itFilename++ )
    {
        siftPoints.clear();
        // extract extension (to be compare with known file formats)
        getExtension( *itFilename, outputBasename, extension );
        // replace prefix if needed
        if ( parameters.prefix.length()!=0 )
            if ( !replace_prefix( outputBasename, parameters.prefix ) ){ cerr << "WARN: skipping image with invalid filename " << ( *itFilename ) << endl; continue; }

        if ( parameters.output_filename.length()==0 )
            outname = outputBasename+".key";
        else{
            outname = parameters.output_filename;
            // remove extension from output_filename
            size_t pos  = parameters.output_filename.find_last_of( '.' );
            if ( ( pos==string::npos ) || ( pos==0 ) )
                outputBasename.clear();
            else
                outputBasename = parameters.output_filename.substr( 0, pos-1 );
        }

        image.load( *itFilename );

        if ( parameters.verbose )
            cout << "processing image : " << *itFilename << ' ' << image.width() << 'x' << image.height() << endl;

        if ( parameters.chunk_size>0 )
        {
            list<SiftPoint> siftPoints_sub;

            // split image into sub area
            vector<RoiWindow_2d> grid;
            clusterize_2d( ImageSize( image.width(), image.height() ),
                           ImageSize( parameters.chunk_size, parameters.chunk_size ),
                           ImageSize( 0, 0 ), // no overlap
                           grid );

            // process all sub-images
            list<SiftPoint>::iterator itPoint;
            int iWin = (int)grid.size();
            vector<RoiWindow_2d>::iterator itWindow = grid.begin();
            while ( iWin-- )
            {
                RealImage1 subimage;
                image.getWindow( *itWindow, subimage );
                siftPoints_sub.clear();

                if ( parameters.verbose ) cout << "processing chunk " << itWindow->m_x0 << ',' << itWindow->m_y0 << ' ' << subimage.width() << 'x' << subimage.height() << "..." << endl;
                process_image( subimage, parameters, outputBasename, gaussPyramid, siftPoints_sub );

                int iPoint = (int)siftPoints_sub.size();
                if ( parameters.verbose ) cout << "chunk " << itWindow->m_x0 << ',' << itWindow->m_y0 << ' ' << subimage.width() << 'x' << subimage.height() << " : " << iPoint << " points" << endl;

                // shift points into orignal image's coordinate system
                // and add points to the main list
                itPoint = siftPoints_sub.begin();
                while ( iPoint-- )
                {
                    itPoint->x += (Real_)itWindow->m_x0;
                    itPoint->y += (Real_)itWindow->m_y0;
                    siftPoints.push_back( *itPoint++ );
                }

                itWindow++;
            }
        }
        else{
            // process full length
            process_image( image, parameters, outputBasename, gaussPyramid, siftPoints ); // process one image per file
        }
        cout << siftPoints.size() << " sift points" << endl;

        if ( parameters.verbose )
            cout << "saving sift points to " << outname << endl;

        if ( !write_siftPoint_list( outname, siftPoints ) ){
            cerr << "ERROR: unable open/write in file [" << outname << ']' << endl;
            return EXIT_FAILURE;
        }
    }

    return 0;
}
