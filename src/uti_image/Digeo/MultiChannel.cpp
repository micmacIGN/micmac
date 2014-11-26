#include "StdAfx.h"
#include "MultiChannel.h"

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
void MultiChannel<tData>::set( size_t i_iChannel, const Im2D<tData,MultiChannel<tData>::tBase > &i_im2d )
{
	__elise_debug_error( i_iChannel>=mChannels.size(), "MultiChannel::set(Im2D): i_iChannel>=mChannels.size()" );
	__elise_debug_error( i_im2d.tx()!=mWidth, "MultiChannel::set(Im2D): i_im2d.tx()!=mWidth" );
	__elise_debug_error( i_im2d.ty()!=mHeight, "MultiChannel::set(Im2D): i_im2d.ty()!=mHeight" );

	memcpy( mChannels[i_iChannel]->data_lin(), i_im2d.data_lin(), (size_t)mWidth*(size_t)mHeight*sizeof(tData) );
}

template <class tData>
void MultiChannel<tData>::link( Im2D<tData,tBase> &i_im2d )
{
	clear();
	mWidth = i_im2d.tx();
	mHeight = i_im2d.ty();
	mChannels.resize(1);
	mChannels[0] = &i_im2d;
}


template <class tData>
void MultiChannel<tData>::resize( int i_width, int i_height, int i_nbChannels )
{
	__elise_debug_error( i_width<0, "MultiChannel: resize(int,int,int): invalid width : " << i_width );
	__elise_debug_error( i_height<0, "MultiChannel: resize(int,int,int): invalid height : " << i_height );
	__elise_debug_error( i_nbChannels<0, "MultiChannel: resize(int,int,int): invalid number of channels : " << i_nbChannels );

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
	while ( i<mChannels.size() )
		delete mChannels[i++];

	// allocate new channels
	while ( i<newChannels.size() )
	{
		newChannels[i++] = new Im2D<tData,tBase>(i_width,i_height);
		__elise_debug_error( newChannels[i-1]==NULL, "MultiChannel: resize(int,int,int): allocation failed" );
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
	}
	io_ostream.write( (const char *)ui4, 16 );

	if ( mChannels.size()==0 ) return;

	// write channels' data
	tData *tmpData = NULL;
	size_t nbElements = 0, nbBytes = 0;
	for ( size_t i=0; i<mChannels.size(); i++ )
	{
		const tData *data = mChannels[i]->data_lin();
		if ( i_reverseByteOrder && sizeof(tData)>1 )
		{
			if ( tmpData==NULL )
			{
				nbElements = (size_t)mWidth*(size_t)mHeight;
				nbBytes = nbElements*sizeof(tData);
				tmpData = new tData[nbBytes];
			}

			memcpy( tmpData, data, nbBytes );

			tData *it = tmpData;
			size_t i = nbElements;
			while ( i-- ) inverseByteOrder(it++);

			data = tmpData;
		}

		io_ostream.write( (const char*)data, nbElements*sizeof(tData) );
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
		__elise_debug_warning( true, "MultiChannel::write_raw: unkown version number " << i_version);
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
		__elise_debug_warning( true, "multi_channels_read_raw_header: unknown version number " << header.version() );
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

	if ( readType!=typeEl() ) return false;

	resize( readWidth, readHeight, readNbChannels );
	const size_t nbElements = (size_t)mWidth*(size_t)mHeight;
	for ( size_t iChannel=0; iChannel<readNbChannels; iChannel++ )
	{
		io_istream.read( (char*)mChannels[iChannel]->data_lin(), nbElements*sizeof(tData) );

		if ( i_reverseByteOrder && sizeof(tData)>1 )
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
		__elise_debug_warning( true, "MultiChannel<tData>::read_raw: cannot read versioned file header" );
		return false;
	}
	if ( o_header!=NULL ) *o_header=header;
	bool reverseByteOrder = ( header.isMSBF()!=MSBF_PROCESSOR() );

	switch ( header.version() ){
	case 1:
		return read_raw_v1( f, reverseByteOrder );
	default:
		__elise_debug_warning( true, "im2d_read_raw(iostream &, vector<Im2DGen*> &, VersionedFileHeader *): unknown version number "<<header.version() );
		return false;
	}
}


//----------------------------------------------------------------------
// tiff I/O
//----------------------------------------------------------------------

template <class tData>
bool MultiChannel<tData>::read_tiff( Tiff_Im &i_tiff ) // Tiff_Im is not const because of ReadVecOfIm
{
	if ( i_tiff.type_el()!=typeEl() ) return false;

	vector<Im2DGen *> tiffChannels = i_tiff.ReadVecOfIm();
	resize( i_tiff.sz().x, i_tiff.sz().y, i_tiff.nb_chan() );

	for ( size_t i=0; i<mChannels.size(); i++ )
		set( i, *(Im2D<tData,tBase>*)tiffChannels[i] );

	return true;
}

template <class tData>
bool MultiChannel<tData>::write_tiff( const string &i_filename ) const
{
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
	}

	o_format = FF_Unknown;

	return false;
}


template <class tData>
void MultiChannel<tData>::duplicateLastChannel( size_t i_nbDuplicates )
{
	__elise_debug_error( mChannels.size()==0, "duplicateLastChannel: mChannels.size()==0" );

	const tData *src = mChannels.back()->data_lin();
	size_t i = mChannels.size();
	resize( mWidth, mHeight, mChannels.size()+i_nbDuplicates );

	__elise_debug_error( src!=mChannels[i-1]->data_lin(), "duplicateLastChannel: resize has reallocated image memory" );

	const size_t nbBytes = (size_t)mWidth*(size_t)mHeight*sizeof(tData);
	while ( i<mChannels.size() )
		memcpy( mChannels[i++]->data_lin(), src, nbBytes );
}

/*
template <class tSrc, class tDst>
void MultiChannel<tSrc>::convert( ValueConverter<tSrc,tDst> &i_converter, MultiChannel<tDst> &o_dst ) const
{
	o_dst.resize( mWidth, mHeight, mChannels.size() );
	const Iterator itSrc = begin();
	Iterator itDst = o_dst.begin();
	while ( itSrc!=end() )
	{
		for ( size_t i=0; i<it.size(); i++ )
		{
			itDst[i] = i_converter( itSrc[i] );
			itDst++; itSrc++;
		}
	}
}
*/


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
