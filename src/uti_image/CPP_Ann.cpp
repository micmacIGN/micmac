#include <iomanip>

#include "AnnSearcher.h"

using namespace std;

static string clipShortestExtension( const string &i_filename )
{
	size_t pos = i_filename.find_last_of( "." );
	if ( pos>=i_filename.length()-1 ) return string();
	return i_filename.substr( 0, pos );
}

static string getBasename( const string &i_filename )
{
	size_t pos = i_filename.find_last_of( "/\\" );
	if ( pos>=i_filename.length()-1 ) return string();
	return i_filename.substr( pos+1, i_filename.length()-pos+1 );
}

static inline bool ann_read_digeo_file( const string &i_filename, vector<DigeoPoint> &i_array, bool i_removeMin, bool i_removeMax, bool i_removeUnknown )
{
	if ( !DigeoPoint::readDigeoFile( i_filename, false/*do no use multiple angles*/, i_array ) ){
		cerr << "ERROR: Ann: cannot read file [" << i_filename << ']' << endl;
		return false;
	}

	// __DEL
	for ( size_t iPoint=0; iPoint<i_array.size(); iPoint++ ){
		if ( i_array[iPoint].descriptors.capacity()!=i_array[iPoint].descriptors.size() ){
			cerr << "-----------------------> " << i_array[iPoint].descriptors.capacity() << " != " << i_array[iPoint].descriptors.size() << endl;
		}
	}

	if ( i_removeMin ) DigeoPoint::removePointsOfType( DigeoPoint::DETECT_LOCAL_MIN, i_array );
	if ( i_removeMax ) DigeoPoint::removePointsOfType( DigeoPoint::DETECT_LOCAL_MAX, i_array );
	if ( i_removeUnknown ) DigeoPoint::removePointsOfType( DigeoPoint::DETECT_UNKNOWN, i_array );

	return true;
}

void ann_usage()
{
	cerr << "usage: Ann [-removeMin] [-removeMax] [-removeUnknown] point_file1 point_file2 [output_file]" << endl;
	exit(EXIT_FAILURE);
}

int Ann_main( int argc, char **argv )
{
	if ( argc<3 ) ann_usage();

	bool removeMin = false,
	     removeMax = false,
	     removeUnknown = false;
	string name0, name1, output_name;

	// process arguments
	for ( int i=1; i<argc; i++ ){
		if ( strcmp( argv[i], "-ignoreMin" )==0 ) removeMin=true;
		else if ( strcmp( argv[i], "-ignoreMax" )==0 ) removeMax=true;
		else if ( strcmp( argv[i], "-ignoreUnknown" )==0 ) removeUnknown=true;
		else if ( name0.length()==0 ) name0 = argv[i];
		else if ( name1.length()==0 ) name1 = argv[i];
		else if ( output_name.length()==0 ) output_name = argv[i];
		else ann_usage();
	}

	if ( output_name.length()==0 ){
		// construct output name from input names
		// if in0 = dir0/name0.ext0 and in1 = dir1/name1.ext1
		// with ext0 and ext1 the shortest extensions of in0 and in1
		// then out = dir0/name0.-.name1.result
		string name0_clipped = clipShortestExtension( name0 );
		string name1_clipped_basename = clipShortestExtension( getBasename( name1 ) );
		output_name = name0_clipped+".-."+name1_clipped_basename+".result";
	}

	list<V2I> matchedCoupleIndices;
	vector<DigeoPoint> array0, array1;
	if ( !ann_read_digeo_file( name0, array0, removeMin, removeMax, removeUnknown ) ) return EXIT_FAILURE;
	if ( !ann_read_digeo_file( name1, array1, removeMin, removeMax, removeUnknown ) ) return EXIT_FAILURE;
	const size_t nbPoints0 = array0.size(),
	             nbPoints1 = array1.size();

	match_lebris( array0, array1, matchedCoupleIndices );

	if ( !matchedCoupleIndices.empty() )
	{
		unfoldMatchingCouples( array0, array1, matchedCoupleIndices );
		neighbourFilter( array0, array1, matchedCoupleIndices );

		if ( !matchedCoupleIndices.empty() )
		{
			unfoldMatchingCouples( array0, array1, matchedCoupleIndices );
			neighbourFilter( array0, array1, matchedCoupleIndices );
		}
	}
	write_matches_ascii( output_name, array0, array1, matchedCoupleIndices );

	cout << name0 << " : " << nbPoints0 << " points " << name1 << " : " << nbPoints1 << " points => " << matchedCoupleIndices.size() << " matches" << endl;

	return EXIT_SUCCESS;
}
