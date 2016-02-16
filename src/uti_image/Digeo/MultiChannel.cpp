#include "MultiChannel.h"

#include "StdAfx.h"

using namespace std;

//----------------------------------------------------------------------
// MultiChannel::Iterator methods
//----------------------------------------------------------------------

template <class tData>
void MultiChannel<tData>::Iterator::set( const vector<Im2D<tData,tBase>*> &i_v )
{
	#ifdef __DEBUG_MULTI_CHANNEL
		__check_multi_channel_consistency(i_v);
	#endif
	mIterators.resize( i_v.size() );
	for ( size_t i=0; i<mIterators.size(); i++ )
		mIterators[i] = i_v[i]->data_lin();
}


//----------------------------------------------------------------------
// MultiChannel methods
//----------------------------------------------------------------------

template <class tData>
void MultiChannel<tData>::set( const MultiChannel<tData> &i_b )
{
	resize( i_b.mWidth, i_b.mHeight, (int)i_b.mChannels.size() );
	const size_t nbPix = (size_t)mWidth*(size_t)mHeight;
	for ( size_t i=0; i<mChannels.size(); i++ )
		memcpy( mChannels[i]->data_lin(), i_b.mChannels[i]->data_lin(), nbPix );
}

template <class tData>
void MultiChannel<tData>::set( size_t i_iChannel, const Im2D<tData,tBase> &i_im2d )
{
	ELISE_DEBUG_ERROR( i_iChannel>=mChannels.size(), "MultiChannel::set(Im2D)", "i_iChannel>=mChannels.size()" );
	ELISE_DEBUG_ERROR( i_im2d.tx()!=mWidth, "MultiChannel::set(Im2D)", "i_im2d.tx()!=mWidth" );
	ELISE_DEBUG_ERROR( i_im2d.ty()!=mHeight, "MultiChannel::set(Im2D)", "i_im2d.ty()!=mHeight" );

	memcpy( mChannels[i_iChannel]->data_lin(), i_im2d.data_lin(), (size_t)mWidth*(size_t)mHeight*sizeof(tData) );
}

template <class tData>
void MultiChannel<tData>::link( Im2D<tData,tBase> &i_im2d )
{
	resize( i_im2d.tx(), i_im2d.ty(), 1 );
	*mChannels[0] = i_im2d;
}


template <class tData>
void MultiChannel<tData>::resize( int i_width, int i_height, int i_nbChannels )
{
	ELISE_DEBUG_ERROR( i_width<0, "MultiChannel::resize(int,int,int)", "invalid width : " << i_width );
	ELISE_DEBUG_ERROR( i_height<0, "MultiChannel::resize(int,int,int)", "invalid height : " << i_height );
	ELISE_DEBUG_ERROR( i_nbChannels<0, "MultiChannel::resize(int,int,int)", "invalid number of channels : " << i_nbChannels );

	if ( mWidth==i_width && mHeight==i_height && mChannels.size()==(size_t)i_nbChannels ) return;

	vector<Im2D<tData,tBase> *> newChannels( (size_t)i_nbChannels );

	// resize already allocated channels
	size_t i;
	const size_t nbToResize = min<size_t>( mChannels.size(), newChannels.size() );
	for ( i=0; i<nbToResize; i++ )
	{
		newChannels[i] = mChannels[i];
		newChannels[i]->Resize( Pt2di(i_width,i_height) );
	}

	// delete supernumerary channels
	while ( i<mChannels.size() ) delete mChannels[i++];

	// allocate new channels
	while ( i<newChannels.size() )
	{
		newChannels[i++] = new Im2D<tData,tBase>(i_width,i_height);
		ELISE_DEBUG_ERROR( newChannels[i-1]==NULL, "MultiChannel::resize(int,int,int)", "allocation failed" );
	}

	mChannels.swap(newChannels);
	mWidth = i_width;
	mHeight = i_height;
	set_begin_end();
}

template <class tData>
void MultiChannel<tData>::set_begin_end()
{
	mBegin.set(mChannels);
	mEnd = mBegin;
	mEnd += mWidth*mHeight;
}


//----------------------------------------------------------------------
// raw I/O
//----------------------------------------------------------------------

template <class tData>
void MultiChannel<tData>::write_raw_v1( ostream &io_ostream, bool i_reverseByteOrder ) const
{
	// write header
	U_INT4 ui4[4] = { (U_INT4)mWidth, (U_INT4)mHeight, (U_INT4)mChannels.size(), (U_INT4)typeEl() }; // 1 = nb channels
	if ( i_reverseByteOrder )
	{
		byte_inv_4( ui4 );
		byte_inv_4( ui4+1 );
		byte_inv_4( ui4+2 );
		byte_inv_4( ui4+3 );
		if ( sizeof(tData)==1 ) i_reverseByteOrder = false; // no need to reverse 1 byte data
	}
	io_ostream.write( (const char *)ui4, 16 );

	if ( mChannels.size()==0 ) return;

	// write channels' data
	tData *tmpData = NULL;
	size_t nbElements = (size_t)mWidth*(size_t)mHeight, nbBytes = nbElements*sizeof(tData);

	// allocate temporary memory for data inversion
	if ( i_reverseByteOrder ) tmpData = new tData[nbElements]; 

	for ( size_t i=0; i<mChannels.size(); i++ )
	{
		const tData *data = mChannels[i]->data_lin();

		if ( i_reverseByteOrder  )
		{
			memcpy( tmpData, data, nbBytes );

			tData *it = tmpData;
			size_t i = nbElements;
			while ( i-- ) inverseByteOrder(it++);

			data = tmpData;
		}

		io_ostream.write( (const char*)data, nbBytes );
	}

	if ( tmpData!=NULL ) delete [] tmpData;
}

template <class tData>
bool MultiChannel<tData>::write_raw( const string &i_filename, U_INT4 i_version, bool i_writeBigEndian ) const
{
	ofstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;

	VersionedFileHeader header( VFH_RawIm2D, i_version, i_writeBigEndian );
	header.write(f);

	const bool reverseByteOrder = ( i_writeBigEndian!=MSBF_PROCESSOR() );
	switch( i_version ){
	case 1: write_raw_v1( f, reverseByteOrder ); return true;
	default:
		ELISE_DEBUG_ERROR( true, "MultiChannel::write_raw", "unkown version number " << i_version);
		return false;
	}
}

void multi_channels_read_raw_header_v1( istream &io_istream, bool i_reverseByteOrder, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type )
{
	U_INT4 ui4[4]; // 0 = width, 1 = height, 2 = nb channels, 3 = type
	io_istream.read( (char*)ui4, 16 );

	if ( i_reverseByteOrder )
	{
		byte_inv_4(ui4);
		byte_inv_4(ui4+1);
		byte_inv_4(ui4+2);
		byte_inv_4(ui4+3);
	}
	
	o_width = (int)ui4[0];
	o_height = (int)ui4[1];
	o_nbChannels = (size_t)ui4[2];
	o_type = (GenIm::type_el)ui4[3];
}

bool multi_channels_read_raw_header( const string &i_filename, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type, VersionedFileHeader *o_header )
{
	ifstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;

	VersionedFileHeader header;
	if ( !header.read_known( VFH_RawIm2D, f ) ) return false;
	if ( o_header!=NULL ) *o_header=header;
	bool reverseByteOrder = ( header.isMSBF()!=MSBF_PROCESSOR() );

	switch ( header.version() ){
	case 1:
		multi_channels_read_raw_header_v1( f, reverseByteOrder, o_width, o_height, o_nbChannels, o_type );
		break;
	default:
		ELISE_DEBUG_WARNING( true, "multi_channels_read_raw_header", "unknown version number " << header.version() );
		return false;
	}
	return true;
}

template <class tData>
bool MultiChannel<tData>::read_raw_v1( istream &io_istream, bool i_reverseByteOrder )
{
	int readWidth, readHeight;
	size_t readNbChannels;
	GenIm::type_el readType;
	multi_channels_read_raw_header_v1( io_istream, i_reverseByteOrder, readWidth, readHeight, readNbChannels, readType );
	if ( sizeof(tData)==1 ) i_reverseByteOrder = false; // no need to reverse 1 byte data

	if ( readType!=typeEl() ) return false;

	resize(readWidth, readHeight, (int)readNbChannels);
	const size_t nbElements = (size_t)mWidth*(size_t)mHeight;
	for ( size_t iChannel=0; iChannel<readNbChannels; iChannel++ )
	{
		io_istream.read( (char*)mChannels[iChannel]->data_lin(), nbElements*sizeof(tData) );

		if ( i_reverseByteOrder )
		{
			size_t i = nbElements;
			tData *it = mChannels[iChannel]->data_lin();
			while ( i-- ) inverseByteOrder(it++);
		}
	}

	return true;
}

template <class tData>
bool MultiChannel<tData>::read_raw( const string &i_filename, VersionedFileHeader *o_header )
{
	ifstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;

	VersionedFileHeader header;
	if ( !header.read_known( VFH_RawIm2D, f ) )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<tData>::read_raw", "cannot read versioned file header" );
		return false;
	}
	if ( o_header!=NULL ) *o_header=header;
	bool reverseByteOrder = ( header.isMSBF()!=MSBF_PROCESSOR() );

	switch ( header.version() ){
	case 1:
		return read_raw_v1( f, reverseByteOrder );
	default:
		ELISE_DEBUG_ERROR( true, "im2d_read_raw(iostream &, vector<Im2DGen*> &, VersionedFileHeader *)", "nknown version number " << header.version() );
		return false;
	}
}


//----------------------------------------------------------------------
// tiff I/O
//----------------------------------------------------------------------

template <class tData>
bool MultiChannel<tData>::read_tiff( Tiff_Im &i_tiff ) // Tiff_Im is not const because of ReadVecOfIm
{
	if ( i_tiff.type_el()!=typeEl() )
	{
		ELISE_DEBUG_ERROR(true, "MultiChannel<" << El_CTypeTraits<tData>::Name() << ">::read_tiff", "cannot read image of type " << eToString(i_tiff.type_el()));
		return false;
	}

	vector<Im2DGen *> tiffChannels = i_tiff.ReadVecOfIm();

	#if 0
		resize( i_tiff.sz().x, i_tiff.sz().y, i_tiff.nb_chan() );

		for ( size_t i=0; i<mChannels.size(); i++ )
			set( i, *(Im2D<tData,tBase>*)tiffChannels[i] );

		for (size_t i = 0; i < tiffChannels.size(); i++)
			delete tiffChannels[i];
	#else
		clear();
		mChannels.resize(tiffChannels.size());
		for (size_t i = 0; i < mChannels.size(); i++)
			mChannels[i] = (Im2D<tData,tBase>*)tiffChannels[i];
		mWidth = mChannels[0]->tx();
		mHeight = mChannels[0]->ty();
	#endif

	return true;
}

template <class tData>
bool MultiChannel<tData>::write_tiff( const string &i_filename ) const
{
	ELISE_DEBUG_ERROR( typeEl()!=GenIm::u_int2 && typeEl()!=GenIm::u_int1, "MultiChannel: write_tiff", "unhandled type " << eToString( typeEl() ) );

	GenIm::type_el tiffOutputType = ( sizeof(tData)>1 ? GenIm::u_int2 : GenIm::u_int1 );
	Fonc_Num inFonc;
	Tiff_Im::PH_INTER_TYPE colorSpace;
	if ( mChannels.size()==1 )
	{
		inFonc = mChannels[0]->in();
		colorSpace = Tiff_Im::BlackIsZero;
	}
	else if ( mChannels.size()==3 )
	{
		inFonc = Virgule( mChannels[0]->in(), mChannels[1]->in(), mChannels[2]->in() );
		colorSpace = Tiff_Im::RGB;
	}
	else
		return false;

	ELISE_COPY
	(
		mChannels[0]->all_pts(),
		inFonc,
		Tiff_Im(
			i_filename.c_str(),
			Pt2di( mWidth, mHeight ),
			tiffOutputType,
			Tiff_Im::No_Compr,
			colorSpace,
			Tiff_Im::Empty_ARG ).out()
	);

	return true;
}


//----------------------------------------------------------------------
// PGM/PPM I/O
//----------------------------------------------------------------------

template <class tData>
bool MultiChannel<tData>::write_pnm( const string &i_filename ) const
{
	ELISE_DEBUG_ERROR( false, "MultiChannel::write_pnm", "unhandled type : " << eToString( typeEl() ) );
	return false;
}

void write_pnm_header( ostream &io_ostream, int i_width, int i_height, size_t i_nbChannels, int i_maxValue )
{
	ELISE_DEBUG_ERROR( i_nbChannels!=1 && i_nbChannels!=3, "MultiChannel::write_pnm_header", "invalid number of channels : " << i_nbChannels );
	ELISE_DEBUG_ERROR( i_maxValue!=255 && i_maxValue!=65535, "MultiChannel::write_pnm_header", "invalid max value : " << i_maxValue );

	io_ostream << (i_nbChannels==1?"P5":"P6") << endl;
	io_ostream << i_width << ' ' << i_height << endl;
	io_ostream << i_maxValue << endl;
}

inline bool __pnm_is_whitespace( char c ){ return c==' ' || c=='\t' || c=='\r' || c=='\n'; }

static void __pnm_read_values( istream &io_istream, list<string> &o_values, list<string> *o_comments )
{
	char c;
	string currentValue(25,' '), comment;
	currentValue.clear();
	do
	{
		if ( io_istream.peek()=='#' )
		{
			if ( currentValue.length()!=0 )
			{
				o_values.push_back(currentValue);
				currentValue.clear();
			}
			getline( io_istream, comment );
			if ( o_comments!=NULL && comment.length()>1 ) o_comments->push_back( comment.substr(1) );
		}
		io_istream.get(c);
		if ( __pnm_is_whitespace(c) )
		{
			if ( currentValue.length()!=0 )
			{
				o_values.push_back(currentValue);
				currentValue.clear();
			}
		}
		else
			currentValue.append( 1, c );
	}
	while ( !io_istream.eof() && o_values.size()<4 );
}

static void __array_byte_inv2( U_INT2 *io_array, size_t i_nbElements )
{
	while ( i_nbElements-- ) byte_inv_2( io_array++ );
}

bool multi_channels_read_pnm_header( istream &io_istream, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type, list<string> *o_comments )
{
	list<string> values;
	__pnm_read_values( io_istream, values, o_comments );
	if ( values.size()!=4 )
	{
		ELISE_DEBUG_ERROR( true, "multi_channels_read_pnm_header", "invalid number of values : " << values.size() );
		return false;
	}

	list<string>::iterator itValue = values.begin();
	string id = *itValue++; 
	if ( id=="P5" ) o_nbChannels = 1;
	else if ( id=="P6" ) o_nbChannels = 3;
	else
	{
		ELISE_DEBUG_ERROR( true, "multi_channels_read_pnm_header", "unknown id : " << id );
		return false;
	}

	o_width  = atoi( (*itValue++).c_str() );
	o_height = atoi( (*itValue++).c_str() );
	int maxValue = atoi( (*itValue++).c_str() );

	if ( maxValue==255 ) o_type = GenIm::u_int1;
	else if ( maxValue==65535 ) o_type = GenIm::u_int2;
	else
	{
		ELISE_DEBUG_ERROR( true, "multi_channels_read_pnm_header", "invalid max value : " << maxValue );
		return false;
	}

	return true;
}

bool multi_channels_read_pnm_header( const string &i_filename, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type, list<string> *o_comments )
{
	ifstream f( i_filename.c_str(), ios::binary );
	if ( !f )
	{
		ELISE_DEBUG_ERROR( true, "multi_channels_read_pnm_header", "cannot open file [" << i_filename << ']' );
		return false;
	}
	return multi_channels_read_pnm_header( f, o_width, o_height, o_nbChannels, o_type, o_comments );
}

template <>
bool MultiChannel<U_INT1>::write_pnm( const string &i_filename ) const
{
	ELISE_DEBUG_ERROR( mChannels.size()!=1 && mChannels.size()!=3 , "MultiChannel<U_INT1>::write_pnm", "nbChannels() = " << nbChannels() << " (should be 1 or 3)" );

	ofstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	write_pnm_header( f, mWidth, mHeight, mChannels.size(), 255 );

	if ( mChannels.size()==1 )
	{
		f.write( (const char *)mChannels[0]->data_lin(), ( (size_t)mWidth )*( (size_t)mHeight ) );
		return true;
	}

	const size_t nbElements = nbPixels()*nbChannels();
	U_INT1 *tupleData = newTupleArray();
	f.write( (const char *)tupleData, nbElements*2 );
	delete [] tupleData;

	return true;
}

template <>
bool MultiChannel<U_INT2>::write_pnm( const string &i_filename ) const
{
	ELISE_DEBUG_ERROR( mChannels.size()!=1 && mChannels.size()!=3, "MultiChannel<U_INT2>::write_pnm", "nbChannels() = " << nbChannels() << " (should be 1 or 3)" );

	ofstream f( i_filename.c_str(), ios::binary );
	if ( !f ) return false;
	write_pnm_header( f, mWidth, mHeight, mChannels.size(), 65535 );

	if ( MSBF_PROCESSOR() && mChannels.size()==1 )
	{
		f.write( (const char *)mChannels[0]->data_lin(), nbPixels()*2 );
		return true;
	}

	const size_t nbVal = nbValues();
	U_INT2 *buffer = new U_INT2[nbVal];
	toTupleArray(buffer);
	if ( !MSBF_PROCESSOR() ) __array_byte_inv2( buffer, nbVal );

	f.write( (const char *)buffer, nbVal*2 );
	delete [] buffer;

	return true;
}

template <class tData>
bool MultiChannel<tData>::read_pnm( const string &i_filename, list<string> *o_comments )
{
	ELISE_DEBUG_ERROR( true, "MultiChannel<" << El_CTypeTraits<tData>::Name() << ">::read_pnm", "type inconsistent with pnm formats, only U_INT1 and U_INT2 are allowed" );
	return false;
}

template <>
bool MultiChannel<U_INT1>::read_pnm( const string &i_filename, list<string> *o_comments )
{
	int width, height;
	size_t nbChannels;
	GenIm::type_el type;
	ifstream f( i_filename.c_str(), ios::binary );

	if ( !f )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<U_INT1>::read_pnm", "cannot open file [" << i_filename << "] for reading" );
		return false;
	}

	if ( !multi_channels_read_pnm_header( f, width, height, nbChannels, type, o_comments ) )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<U_INT1>::read_pnm", "cannot read pnm header in [" << i_filename << ']' );
		return false;
	}

	if ( type!=GenIm::u_int1 )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<U_INT1>::read_pnm", "incompatible image type " << eToString(type) );
		return false;
	}

	resize(width, height, (int)nbChannels);
	if ( nbChannels==1 )
		f.read( (char*)mChannels[0]->data_lin(), nbChannelBytes() );
	else
	{
		U_INT1 *buffer = new U_INT1[nbValues()];
		f.read( (char*)buffer, nbBytes() );
		setFromTuple(buffer);
		delete [] buffer;
	}

	return true;
}

template <>
bool MultiChannel<U_INT2>::read_pnm( const string &i_filename, list<string> *o_comments )
{
	int width, height;
	size_t nbChannels;
	GenIm::type_el type;
	ifstream f( i_filename.c_str(), ios::binary );

	if ( !f )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<U_INT2>::read_pnm", "cannot open file [" << i_filename << "] for reading" );
		return false;
	}

	if ( !multi_channels_read_pnm_header( f, width, height, nbChannels, type, o_comments ) )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<U_INT2>::read_pnm", "cannot read pnm header in [" << i_filename << ']' );
		return false;
	}

	if ( type!=GenIm::u_int2 )
	{
		ELISE_DEBUG_ERROR( true, "MultiChannel<U_INT2>::read_pnm", "incompatible image type " << eToString(type) );
		return false;
	}

	resize(width, height, (int)nbChannels);

	if ( !MSBF_PROCESSOR() && nbChannels==1 )
	{
		f.read( (char*)mChannels[0]->data_lin(), nbBytes() );
		return true;
	} 

	const size_t nbVal = nbValues();
	U_INT2 *buffer = new U_INT2[nbVal];
	f.read( (char*)buffer, nbBytes() );
	if ( MSBF_PROCESSOR() ) __array_byte_inv2( buffer, nbVal );
	setFromTuple(buffer);
	delete [] buffer;

	return true;
}


//----------------------------------------------------------------------
// all formats functions
//----------------------------------------------------------------------

bool multi_channels_read_header( const string &i_filename, FileFormat_t &o_format, int &o_width, int &o_height, size_t &o_nbChannels, GenIm::type_el &o_type )
{
	if ( multi_channels_read_raw_header( i_filename, o_width, o_height, o_nbChannels, o_type ) )
	{
		o_format = FF_Raw;
		return true;
	}

	if ( Tiff_Im::IsTiff( i_filename.c_str() ) )
	{
		Tiff_Im tiff( i_filename.c_str() );
		o_format = FF_Tiff;
		o_width = tiff.sz().x;
		o_height = tiff.sz().x;
		o_nbChannels = tiff.nb_chan();
		o_type = tiff.type_el();
		return true;
	}

	if ( multi_channels_read_pnm_header( i_filename, o_width, o_height, o_nbChannels, o_type ) )
	{
		o_format = FF_Pnm;
		return true;
	}

	o_format = FF_Unknown;
	return false;
}


template <class tData>
void MultiChannel<tData>::duplicateLastChannel( size_t i_nbDuplicates )
{
	ELISE_DEBUG_ERROR( mChannels.size()==0, "MultiChannel<tData>::duplicateLastChannel", "mChannels.size()==0" );

	const tData *src = mChannels.back()->data_lin();
	size_t i = mChannels.size();
	resize( mWidth, mHeight, (int)(mChannels.size() + i_nbDuplicates));

	ELISE_DEBUG_ERROR( src!=mChannels[i-1]->data_lin(), "MultiChannel<tData>::duplicateLastChannel", "resize has reallocated image memory" );

	const size_t nbBytes = (size_t)mWidth*(size_t)mHeight*sizeof(tData);
	while ( i<mChannels.size() )
		memcpy( mChannels[i++]->data_lin(), src, nbBytes );
}

template <class tData>
void MultiChannel<tData>::suppressLastChannels(size_t i_nbToSuppress)
{
	if (i_nbToSuppress > mChannels.size())
	{
		ELISE_DEBUG_ERROR(true, "MultiChannel<tData>::suppressLastChannels", "i_nbToSuppress = " << i_nbToSuppress << " > " << mChannels.size() << " = mChannels.size()" );
		clear();

		return;
	}

	const size_t finalNbChannels = mChannels.size() - i_nbToSuppress;
	for (size_t i = finalNbChannels; i < mChannels.size(); i++)
		delete mChannels[i];

	mChannels.resize(finalNbChannels);
}

template <class tData>
void MultiChannel<tData>::toGrayScale(Im2D<tData, TBASE> &oImage)
{
	const size_t nbChannels = mChannels.size();
	if (nbChannels == 0 || nbChannels == 2)
	{
		ELISE_DEBUG_ERROR(true, "MultiChannel<tData>::toGrayScale", "invalid nbChannels = " << nbChannels);
		return;
	}

	oImage.Resize(mChannels[0]->sz());

	if (nbChannels == 1)
	{
		memcpy(oImage.data_lin(), mChannels[0]->data_lin(), size_t(oImage.tx()) * size_t(oImage.ty()));
		return;
	}

	if (nbChannels >= 3)
	{
		const tData *itRed = data_lin(0), *itGreen = data_lin(1), *itBlue = data_lin(2);
		tData *itDst = oImage.data_lin();
		size_t iPix = size_t(oImage.tx()) * size_t(oImage.ty());
		while (iPix--)
			*itDst++ = round_ni(0.299 * float(*itRed++) + 0.587 * float(*itGreen++) + 0.114 * float(*itBlue++));
		return;
	}
}

template <class tData>
bool MultiChannel<tData>::hasSameData( const MultiChannel<tData> &i_b, size_t &i_firstDifferentChannel ) const
{
	const size_t nbBytes = width()*height()*sizeof(tData);
	for ( size_t i=0; i<nbChannels(); i++ )
		if ( memcmp( mChannels[i]->data_lin(), i_b.mChannels[i]->data_lin(), nbBytes )!=0 )
		{
			i_firstDifferentChannel = i;
			return false;
		}
	return true;
}

template <class tData>
tData * MultiChannel<tData>::newTupleArray() const
{
	tData *res = new tData[nbPixels()*nbChannels()];
	toTupleArray(res);
	return res;
}

template <class tData>
void MultiChannel<tData>::toTupleArray( tData *o_dst ) const
{
	if (nbChannels() == 1)
	{
		memcpy(o_dst, mChannels[0]->data_lin(), nbPixels() * sizeof(tData));
		return;
	}

	const size_t nbChan = nbChannels();
	const size_t nbPix = nbPixels();
	for ( size_t iChannel=0; iChannel<nbChan; iChannel++ )
	{
		tData *itDst = o_dst+iChannel;
		const tData *itSrc = mChannels[iChannel]->data_lin();
		size_t i = nbPix;
		while ( i-- )
		{
			*itDst = *itSrc++;
			itDst += nbChan;
		}
	}
}

template <class tData>
void MultiChannel<tData>::setFromTuple( const tData *i_src ) const
{
	const size_t nbChan = mChannels.size();
	const size_t nbPix = nbPixels();
	for ( size_t iChannel=0; iChannel<nbChan; iChannel++ )
	{
		const tData *itSrc = i_src+iChannel;
		tData *itDst = mChannels[iChannel]->data_lin();
		size_t i = nbPix;
		while ( i-- )
		{
			*itDst++ = *itSrc;
			itSrc += nbChan;
		}
	}
}


//----------------------------------------------------------------------
// related functions
//----------------------------------------------------------------------

#ifdef __DEBUG_MULTI_CHANNEL
	template <class tData, class tBase>
	void __check_multi_channel_consistency( const vector<Im2D<tData,tBase> *> &i_channels )
	{
		if ( i_channels.size()==0 ) return;

		const GenIm::type_el type = i_channels[0]->TypeEl();
		const Pt2di size = i_channels[0]->sz();

		for ( size_t i=1; i<i_channels.size(); i++ )
		{
			if ( i_channels[i]==NULL ) __elise_error( "__check_multi_channel_consistency: channel " << i << " = NULL " );
			if ( i_channels[i]->sz()!=size || i_channels[i]->TypeEl()!=type )
				__elise_error( "__check_multi_channel_consistency: channel "<<i<<": " << i_channels[i]->sz() << ' ' << eToString(i_channels[i]->TypeEl())
					<< " channel 0: " << size << ' ' << eToString(type) );
		}
	}
#endif


//----------------------------------------------------------------------
// instantiation
//----------------------------------------------------------------------

template class MultiChannel<U_INT1>;
template class MultiChannel<INT1>;
template class MultiChannel<U_INT2>;
template class MultiChannel<INT2>;
template class MultiChannel<INT4>;
template class MultiChannel<REAL4>;
template class MultiChannel<REAL8>;
template class MultiChannel<REAL16>;
