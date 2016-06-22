#ifndef __MULTI_CHANNEL__
#define __MULTI_CHANNEL__

#ifdef NO_ELISE
	#include "TypeTraits.h"
#else
	#include "StdAfx.h"
#endif

#include "debug.h"

typedef enum{
	FF_Raw,
	FF_Tiff,
	FF_Jpeg2000,
	FF_Pnm,
	FF_Unknown
} FileFormat_t;

template <class tData>
class MultiChannel
{
public:
	typedef typename El_CTypeTraits<tData>::tBase tBase;

	class Iterator
	{
	public:
		vector<tData *> mIterators;

		Iterator();

		size_t size() const;

		void set( const vector<Im2D<tData,tBase>*> &i_v );

		void operator ++(int);
		void operator --(int);
		void operator +=( int i_offset);
		void operator -=( int i_offset);

		bool operator ==( const Iterator &i_it ) const;
		bool operator !=( const Iterator &i_it ) const;

		tData & operator [] ( size_t i_index );
		const tData & operator [] ( size_t i_index ) const;
	};

	MultiChannel();
	MultiChannel( const MultiChannel<tData> &i_b );
	MultiChannel( const Im2D<tData,tBase> &i_im2d );
	MultiChannel( int i_width, int i_height, int i_nbChannels );

	~MultiChannel();

	void resize( int i_width, int i_height, int i_nbChannels );

	void set( const MultiChannel<tData> &i_b );
	void set( size_t i_iChannel, const Im2D<tData,tBase> &i_im2d );
	void link( Im2D<tData,tBase> &i_im2d );
	void link( vector<Im2D<tData,tBase> *> &i_im2ds );

	MultiChannel<tData> & operator =( const MultiChannel<tData> &i_b );

	void clear();

	GenIm::type_el typeEl() const;
	int width() const;
	int height() const;
	size_t nbChannels() const;

	//Im2D<tData,tBase> & operator []( size_t i_iChannel );
	const Im2D<tData,tBase> & operator []( size_t i_iChannel ) const;
	tData ** data( size_t i_iChannel ) const;
	tData * data_lin( size_t i_iChannel ) const;

	const Iterator & begin() const { return mBegin; }
	Iterator & begin() { return mBegin; }

	const Iterator & end() const { return mEnd; }
	Iterator & end() { return mEnd; }

	void write_raw_v1( ostream &io_ostream, bool i_reverseByteOrder ) const;
	bool write_raw( const string &i_filename, U_INT4 i_version, bool i_writeBigEndian ) const;
	bool write_raw( const string &i_filename ) const; // use last file version and processor's endianness
	bool read_raw_v1( istream &io_stream, bool i_reverseByteOrder );
	bool read_raw( const string &i_filename, VersionedFileHeader *o_header=NULL );

	bool write_tiff( const string &i_filename ) const;
	bool read_tiff( const string &i_filename );
	bool read_tiff( Tiff_Im &i_tiff );

	bool read_pnm( const string &i_filename, list<string> *o_comments=NULL );

	bool write_pnm( const string &i_filename ) const;

	void duplicateLastChannel( size_t i_nbDuplicates );
	void suppressLastChannels(size_t i_nbToSuppress);
	void toGrayScale(Im2D<tData, tBase> &oImage);

	bool hasSameDimensions( const MultiChannel<tData> &i_b ) const;
	bool hasSameData( const MultiChannel<tData> &i_b, size_t &i_firstDifferentChannel ) const;
	bool hasSameData( const MultiChannel<tData> &i_b ) const;

	size_t nbPixels() const;
	size_t nbValues() const;
	size_t nbChannelBytes() const;
	size_t nbBytes() const;

	// create a copy of the MultiChannel organized as an array of tuples
	// user have the responsability to delete the non-NULL returned array
	tData * newTupleArray() const;
	// o_dst should have at least the same number of elements as the MultiChannels (ie width*height*nbChannels)
	void toTupleArray( tData *i_dst ) const;
	// i_src should have at least the same number of elements as the MultiChannels (ie width*height*nbChannels)
	void setFromTuple( const tData *i_src ) const;

private:
	void set_begin_end();

	vector<Im2D<tData,tBase> *> mChannels;
	int                         mWidth;
	int                         mHeight;
	Iterator                    mBegin;
	Iterator                    mEnd;
};

template <class T> void __clear_vector( vector<T*> &v );

void multi_channels_read_raw_header_v1( istream &io_istream, bool i_reverseByteOrder, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type );
bool multi_channels_read_raw_header( const string &i_filename, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type, VersionedFileHeader *o_header=NULL );
bool multi_channels_read_pnm_header( istream &io_istream, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type, list<string> *o_comments = NULL );
bool multi_channels_read_pnm_header( const string &i_filename, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type, list<string> *o_comments = NULL );
bool multi_channels_read_header( const string &i_filename, FileFormat_t &o_format, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type );

#ifdef __DEBUG_MULTI_CHANNEL
	void __check_multi_channel_consistency( vector<Im2DGen *> &i_channels );
#endif

#include "MultiChannel.inline.h"

#endif // __MULTI_CHANNEL__
