#include "ann_utils.h"

using namespace std;

string clipShortestExtension( const string &i_filename )
{
	size_t pos = i_filename.find_last_of( "." );
	if ( pos>=i_filename.length()-1 ) return string();
	return i_filename.substr( 0, pos );
}

string getBasename( const string &i_filename )
{
	size_t pos = i_filename.find_last_of( "/\\" );
	if ( pos>=i_filename.length()-1 ) return string();
	return i_filename.substr( pos+1, i_filename.length()-pos+1 );
}

string ann_create_output_filename( const string &i_inFilename0, const string &i_inFilename1 )
{
	string name0_clipped = clipShortestExtension( i_inFilename0 );
	string name1_clipped_basename = clipShortestExtension( getBasename( i_inFilename1 ) );
	return name0_clipped+".-."+name1_clipped_basename+".result";
}
