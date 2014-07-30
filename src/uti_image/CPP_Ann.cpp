#include <iomanip>

#include "AnnSearcher.h"

using namespace std;

inline bool ann_read_sift_file( const string &i_filename, vector<DigeoPoint> &array )
{
	if ( !DigeoPoint::readDigeoFile( i_filename, array ) ){
		cerr << "ERROR: Ann: unable to read file " << i_filename << endl;
		return false;
	}

	return true;
}

int Ann_main( int argc, char **argv )
{
    if ( argc<3 ){
		cerr << "ERROR: ANN search needs two arguments: the names of the two files listing SIFT points to match" << endl;
		return EXIT_FAILURE;
	}

	string output_name;
	if ( argc==4 )
		output_name = argv[3];
	else
	{
		// construct output name from input names
		// if in0 = dir0/name0.ext0 and in1 = dir1/name1.ext1
		// with ext0 and ext1 the shortest extensions of in0 and in1
		// then out = dir0/name0.-.name1.result
		string name0 = argv[1],
		       name1 = argv[2];
		name0 = name0.substr( 0, name0.find_last_of( "." ) );
		
		size_t pos0 = name1.find_last_of( "/\\" ),
			   pos1 = name1.find_last_of( "." );
		if ( pos0==string::npos ) pos0=0;
		else if ( pos1==string::npos ) pos1=name1.length();
		else if ( pos0==name1.length()-1 ){
			cerr << "Ann: ERROR: invalid filename " << name1 << endl;
			return EXIT_FAILURE;
		}
		else pos0++;
		name1 = name1.substr( pos0, pos1-pos0 );
		
		output_name = name0+".-."+name1+".result";
	}

	list<V2I> matchedCoupleIndices;
	vector<DigeoPoint> array0, array1;
	if ( !ann_read_sift_file( argv[1], array0 ) ) return EXIT_FAILURE;
	if ( !ann_read_sift_file( argv[2], array1 ) ) return EXIT_FAILURE;

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

	cout << matchedCoupleIndices.size() << " matches" << endl;

    return EXIT_SUCCESS;
}
