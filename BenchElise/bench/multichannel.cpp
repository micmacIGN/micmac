#define __DEBUG

#include "StdAfx.h"
#include "../src/uti_image/Digeo/MultiChannel.h"

using namespace std;

const int width = 256;
const int height = 256;
const string temporaryTiffOutputFilename = "multichannel.tif";
const string temporaryRawOutputFilename = "multichannel.raw";
const string temporaryPnmOutputFilename = "multichannel.pnm";
string error_prefix;


//----------------------------------------------------------------------
// MultiChannel generation and comparison functions
//----------------------------------------------------------------------

template<class tData>
void generate_shade( tData **io_image, int i_width, int i_height )
{
	for ( int y=0; y<i_height; y++ )
		for ( int x=0; x<i_width; x++ )
			io_image[y][x] = (tData)x;
}

template<class tData>
void generate_shade( MultiChannel<tData> &io_channels )
{
	const size_t nbBytes = (size_t)io_channels.width()*(size_t)io_channels.height()*sizeof(tData);
	generate_shade( io_channels.data(0), io_channels.width(), io_channels.height() );
	for ( size_t i=1; i<io_channels.nbChannels(); i++ )
		memcpy( io_channels.data_lin(i), io_channels.data_lin(0), nbBytes );
}

template<class T>
T random_value()
{
	T res;
	unsigned char *it = (unsigned char*)&res;
	size_t i = sizeof(T);
	while ( i-- ) *it++ = (unsigned char)rand()%256;
	return res;
}

template<class tData>
void generate_random_content( tData *io_image, int i_width, int i_height )
{
	size_t i = (size_t)i_width*(size_t)i_height;
	while ( i-- ) *io_image++ = random_value<tData>();
}

template<class tData>
void generate_random_content( MultiChannel<tData> &io_channels )
{
	for ( size_t i=0; i<io_channels.nbChannels(); i++ )
		generate_random_content( io_channels.data_lin(i), io_channels.width(), io_channels.height() );
}

template <class tData>
void compare( const MultiChannel<tData> &i_a, const MultiChannel<tData> &i_b )
{
	if ( !i_a.hasSameDimensions(i_b) )
		__elise_error( error_prefix << "dimensions are different " << i_a.width() << 'x' << i_a.height() << 'x' << i_a.nbChannels() << " != "
			<< i_b.width() << 'x' << i_b.height() << 'x' << i_b.nbChannels() );
	size_t firstDifferentChannel;
	if ( !i_a.hasSameData( i_b, firstDifferentChannel ) )
		__elise_error( error_prefix << "data are different in channel " << firstDifferentChannel );
}


//----------------------------------------------------------------------
// tiff I/O test functions
//----------------------------------------------------------------------

template <class tData>
void test_multi_channel_tiff_write_read( MultiChannel<tData> &io_channels )
{
}

template <class tData>
void test_io_tiff( int i_nbChannels )
{
	MultiChannel<tData> channels( width, height, i_nbChannels );
	//generate_random_content(channels);
	generate_shade(channels);
	test_multi_channel_tiff_write_read(channels);

	if ( !channels.write_tiff(temporaryTiffOutputFilename) ) __elise_error( error_prefix << "cannot write tiff" );

	MultiChannel<tData> readChannels;
	if ( !readChannels.read_tiff(temporaryTiffOutputFilename) ) __elise_error( error_prefix << "cannot read tiff" );

	compare( channels, readChannels );
}

template <class tData>
void test_multi_channel_tiff()
{
	error_prefix = string("test_multi_channel_tiff: ") + "MultiChannel<" + El_CTypeTraits<tData>::Name() + ">, [" + temporaryTiffOutputFilename + "]: ";
	test_io_tiff<tData>(1);
	test_io_tiff<tData>(3);
	cout << error_prefix << "done" << endl;
}


//----------------------------------------------------------------------
// raw I/O test functions
//----------------------------------------------------------------------

template <class tData>
void test_multi_channel_raw()
{
	error_prefix = string("test_multi_channel_raw: ") + "MultiChannel<" + El_CTypeTraits<tData>::Name() + ">, [" + temporaryRawOutputFilename + "]: ";

	MultiChannel<tData> channels( width, height, 1 );
	generate_random_content(channels);
	//generate_shade(channels);

	if ( !channels.write_raw(temporaryRawOutputFilename) ) __elise_error( error_prefix << "cannot write raw" );

	MultiChannel<tData> readChannels;
	if ( !readChannels.read_raw(temporaryRawOutputFilename) ) __elise_error( error_prefix << "cannot read raw" );

	compare( channels, readChannels );

	cout << error_prefix << "done" << endl;
}


//----------------------------------------------------------------------
// pgm/ppm I/O test functions
//----------------------------------------------------------------------

template <class tData>
void test_io_pnm( int i_nbChannels )
{
	MultiChannel<tData> channels( width, height, i_nbChannels );
	channels.resize( width, height, i_nbChannels );
	//generate_random_content(channels);
	generate_shade(channels);

	stringstream ss;
	ss << "multichannels." << eToString(channels.typeEl()) << '.' << i_nbChannels << ( i_nbChannels==1?".pgm":".ppm" );
	string temporaryPnmOutputFilename = ss.str();

	if ( !channels.write_pnm(temporaryPnmOutputFilename) ) __elise_error( error_prefix << "cannot write raw" );

	MultiChannel<tData> readChannels;
	if ( !readChannels.read_pnm(temporaryPnmOutputFilename) ) __elise_error( error_prefix << "cannot read raw" );

	compare( channels, readChannels );
}

template <class tData>
void test_multi_channel_pnm()
{
	error_prefix = string("test_multi_channel_pnm: ") + "MultiChannel<" + El_CTypeTraits<tData>::Name() + ">: ";
	test_io_pnm<tData>(1);
	test_io_pnm<tData>(3);
	cout << error_prefix << "done" << endl;
}

template <class tData>
void test_tuple_conversion()
{
	error_prefix = string("test_tuple_conversion: ") + "MultiChannel<" + El_CTypeTraits<tData>::Name() + ">: ";

	MultiChannel<tData> channels( width, height, 3 );
	generate_shade(channels);
	tData *tupleData = channels.newTupleArray();
	MultiChannel<tData> channelsFromTuple( width, height, 3 );
	channelsFromTuple.setFromTuple(tupleData);
	delete [] tupleData;

	compare( channels, channelsFromTuple );

	cout << error_prefix << "done" << endl;
}


int main( int argc, char **argv )
{
	srand( time(NULL) );

/*
	// __DEL
	string filename = "/home/jbelvaux/data/test.pgm";
	list<string> comments;
	int width, height;
	size_t nbChannels;
	GenIm::type_el type;
	if ( !multi_channels_read_pnm_header( filename, width, height, nbChannels, type ) )
		__elise_error( "cannot read header of [" << filename << ']' );
	cout << "[" << filename << "] " << width << 'x' << height << 'x' << nbChannels << ' ' << eToString(type) << endl;
	list<string>::const_iterator it= comments.begin();
	cout << "comments:" << endl;
	while ( it!=comments.end() )
		cout << "\t[" << (*it++) << ']' << endl;
*/

	test_tuple_conversion<U_INT2>();

	// test MutliChannel tiff I/O
	test_multi_channel_tiff<U_INT1>();
	test_multi_channel_tiff<U_INT2>();

	// test MutliChannel raw I/O
	test_multi_channel_raw<U_INT1>();
	test_multi_channel_raw<INT1>();
	test_multi_channel_raw<U_INT2>();
	test_multi_channel_raw<INT2>();
	test_multi_channel_raw<INT4>();
	test_multi_channel_raw<REAL4>();
	test_multi_channel_raw<REAL8>();
	test_multi_channel_raw<REAL16>();
	
	// test MutliChannel pnm I/O
	test_multi_channel_pnm<U_INT1>();
	test_multi_channel_pnm<U_INT2>();
}
