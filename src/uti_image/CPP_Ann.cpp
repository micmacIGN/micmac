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

void match_and_filter( vector<DigeoPoint> &io_array0, vector<DigeoPoint> &io_array1, list<V2I> &io_matchedCoupleIndices )
{
	match_lebris( io_array0, io_array1, io_matchedCoupleIndices );

	if ( !io_matchedCoupleIndices.empty() )
	{
		unfoldMatchingCouples( io_array0, io_array1, io_matchedCoupleIndices );
		neighbourFilter( io_array0, io_array1, io_matchedCoupleIndices );

		if ( !io_matchedCoupleIndices.empty() )
		{
			unfoldMatchingCouples( io_array0, io_array1, io_matchedCoupleIndices );
			neighbourFilter( io_array0, io_array1, io_matchedCoupleIndices );
		}
	}
}

bool process_couple( const string &i_inFilename0, const string &i_inFilename1, string i_outFilename,
                     bool i_removeMin, bool i_removeMax, bool i_removeUnknown, bool i_noSplitTypes )
{
	if ( i_outFilename.length()==0 ){
		// construct output name from input names
		// if in0 = dir0/name0.ext0 and in1 = dir1/name1.ext1
		// with ext0 and ext1 the shortest extensions of in0 and in1
		// then out = dir0/name0.-.name1.result
		string name0_clipped = clipShortestExtension( i_inFilename0 );
		string name1_clipped_basename = clipShortestExtension( getBasename( i_inFilename1 ) );
		i_outFilename = name0_clipped+".-."+name1_clipped_basename+".result";
	}
	
	vector<DigeoPoint> allPointsArray0, allPointsArray1;
	if ( !ann_read_digeo_file( i_inFilename0, allPointsArray0, i_removeMin, i_removeMax, i_removeUnknown ) ) return false;
	if ( !ann_read_digeo_file( i_inFilename1, allPointsArray1, i_removeMin, i_removeMax, i_removeUnknown ) ) return false;
	const size_t nbPoints0 = allPointsArray0.size(),
	             nbPoints1 = allPointsArray1.size();
	size_t totalMatches = 0;
	if ( i_noSplitTypes ){
		list<V2I> matchedCoupleIndices;
		match_and_filter( allPointsArray0, allPointsArray1, matchedCoupleIndices );
		write_matches_ascii( i_outFilename, allPointsArray0, allPointsArray1, matchedCoupleIndices );
		totalMatches = matchedCoupleIndices.size();
	}
	else{
		DigeoTypedVectors typedVectors0( allPointsArray0 ),
		                  typedVectors1( allPointsArray1 );
		vector<list<V2I> > matchedCoupleIndices(DigeoPoint::nbDetectTypes);
		for ( size_t iType=0; iType<DigeoPoint::nbDetectTypes; iType++ ){
			vector<DigeoPoint> &array0 = typedVectors0.m_points[iType],
			                   &array1 = typedVectors1.m_points[iType];
			if ( array0.size()!=0 && array1.size()!=0 ) match_and_filter( array0, array1, matchedCoupleIndices[iType] );
			cout << DetectType_to_string( (DigeoPoint::DetectType)iType ) << " : " << matchedCoupleIndices[iType].size() << " matches" << endl;
			write_matches_ascii( i_outFilename, array0, array1, matchedCoupleIndices[iType], iType!=0 );
			totalMatches += matchedCoupleIndices[iType].size();
		}
	}
	cout << i_inFilename0 << " : " << nbPoints0 << " points " << i_inFilename1 << " : " << nbPoints1 << " points => " << totalMatches << " matches" << endl;
	return true;
}

extern void SplitDirAndFile( string &i_directory, string &i_basename, const string &i_fullname );
extern list<string> RegexListFileMatch( const string &i_directory, const string &i_pattern, int i_maxDepth, bool i_outDirectory );

#ifdef __ANN_SEARCHER_TIME
	extern double g_construct_tree_time;
	extern double g_query_time;
#endif

int Ann_main( int argc, char **argv )
{
	bool removeMin = false,
	     removeMax = false,
	     removeUnknown = false,
	     noSplitTypes = false;
	string name0, name1, output_name;

	// process arguments
	for ( int i=1; i<argc; i++ ){
		if ( strcmp( argv[i], "-ignoreMin" )==0 ) removeMin=true;
		else if ( strcmp( argv[i], "-ignoreMax" )==0 ) removeMax=true;
		else if ( strcmp( argv[i], "-ignoreUnknown" )==0 ) removeUnknown=true;
		else if ( strcmp( argv[i], "-noSplitTypes" )==0 ) noSplitTypes=true;
		else if ( name0.length()==0 ) name0 = argv[i];
		else if ( name1.length()==0 ) name1 = argv[i];
		else if ( output_name.length()==0 ) output_name = argv[i];
		else ann_usage();
	}

	if ( name1.length()==0 ){
		// pattern mode, process all couple of files possible for that pattern

		string directory, pattern;
		SplitDirAndFile( directory, pattern, name0 );

		list<string> filenames = RegexListFileMatch( directory, pattern, 1, false );
		cout << filenames.size() << " files" << endl;
		list<string>::const_iterator itFilename0 = filenames.begin();
		while ( itFilename0!=filenames.end() ){
			list<string>::const_iterator itFilename1 = itFilename0;
			itFilename1++;
			while ( itFilename1!=filenames.end() ){
				cout << "processing " << *itFilename0 << ' ' << *itFilename1 << endl;

				if ( !process_couple( directory+*itFilename0, directory+*itFilename1, output_name, removeMin, removeMax, removeUnknown, noSplitTypes ) ) return EXIT_FAILURE;

				itFilename1++;
			}
			itFilename0++;
		}

		#ifdef __ANN_SEARCHER_TIME
			cout << "tree construction time = " << g_construct_tree_time << " ms " << endl;
			cout << "query time = " << g_query_time << " ms " << endl;
		#endif

		return EXIT_SUCCESS;
	}

	return ( process_couple( name0, name1, output_name, removeMin, removeMax, removeUnknown, true/*noSplitTypes*/ )?EXIT_SUCCESS:EXIT_FAILURE );
}
