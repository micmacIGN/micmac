#include "StdAfx.h"
#include "../src/uti_image/Digeo/MultiChannel.h"

using namespace std;

const int width = 256;
const int height = 256;
const string temporaryTiffOutputFilename = "multichannel.tif";
const string temporaryRawOutputFilename = "multichannel.raw";
string error_prefix;


//----------------------------------------------------------------------
// tiff I/O test functions
//----------------------------------------------------------------------

template<class tData>
void generate_shade( tData **io_image, int i_width, int i_height )
{
	for ( int y=0; y<i_height; y++ )
		for ( int x=0; x<i_width; x++ )
			io_image[y][x] = (tData)x;
}

template <class tData>
void test_multi_channel_tiff_write_read( MultiChannel<tData> &io_channels )
{
	if ( !io_channels.write_tiff(temporaryTiffOutputFilename) ) __elise_error( error_prefix << "cannot write tiff" );

	MultiChannel<tData> readChannels;
	if ( !readChannels.read_tiff(temporaryTiffOutputFilename) ) __elise_error( error_prefix << "cannot read tiff" );

	if ( io_channels.width()!=readChannels.width() ||
	     io_channels.height()!=readChannels.height() ||
	     io_channels.nbChannels()!=readChannels.nbChannels() )
		__elise_error( error_prefix << "read MultiChannel " << readChannels.width() << 'x' << readChannels.height() << 'x' << readChannels.nbChannels() <<
			"!= src MultiChannel " << io_channels.width() << 'x' << io_channels.height() << 'x' << io_channels.nbChannels() );

	const size_t nbBytes = (size_t)io_channels.width()*(size_t)io_channels.height()*sizeof(tData);
	for ( size_t i=0; i<io_channels.nbChannels(); i++ )
		if ( memcmp( io_channels[i].data_lin(), readChannels[i].data_lin(), nbBytes ) ) __elise_error( error_prefix << "read data != written data for channel " << i );
}

template <class tData>
void test_multi_channel_tiff()
{
	error_prefix = string("test_multi_channel_tiff: ") + "MultiChannel<" + El_CTypeTraits<tData>::Name() + ">, [" + temporaryTiffOutputFilename + "]: ";

	MultiChannel<tData> channels( width, height, 1 );
	generate_shade( channels.data(0), channels.width(), channels.height() );
	test_multi_channel_tiff_write_read(channels);

	channels.duplicateLastChannel(2);
	test_multi_channel_tiff_write_read(channels);

	cout << error_prefix << "done" << endl;
}


//----------------------------------------------------------------------
// raw I/O test functions
//----------------------------------------------------------------------

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
void test_multi_channel_raw()
{
	error_prefix = string("test_multi_channel_raw: ") + "MultiChannel<" + El_CTypeTraits<tData>::Name() + ">, [" + temporaryRawOutputFilename + "]: ";

	MultiChannel<tData> channels( width, height, 1 );
	generate_random_content(channels);

	if ( !channels.write_raw(temporaryRawOutputFilename) ) __elise_error( error_prefix << "cannot write raw" );

	MultiChannel<tData> readChannels;
	if ( !readChannels.read_raw(temporaryRawOutputFilename) ) __elise_error( error_prefix << "cannot read raw" );

	cout << error_prefix << "done" << endl;
}

int main( int argc, char **argv )
{
	srand( time(NULL) );

	// test MutliChannel tiff I/O
	test_multi_channel_tiff<U_INT2>();
	test_multi_channel_tiff<U_INT1>();

	// test MutliChannel raw I/O
	test_multi_channel_raw<U_INT1>();
	test_multi_channel_raw<INT1>();
	test_multi_channel_raw<U_INT2>();
	test_multi_channel_raw<INT2>();
	test_multi_channel_raw<INT4>();
	test_multi_channel_raw<REAL4>();
	test_multi_channel_raw<REAL8>();
	test_multi_channel_raw<REAL16>();
}
