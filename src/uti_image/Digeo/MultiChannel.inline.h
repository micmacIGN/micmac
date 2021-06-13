// this file should be included by MultiChannel.h solely

//----------------------------------------------------------------------
// MultiChannel::Iterator inline methods
//----------------------------------------------------------------------

/*
template <class tSrc, class tDst, class tCompute>
inline LinearConverter<tSrc,tDst,tCompute>::LinearConverter( tSrc i_minSrc, tSrc i_maxSrc, tDst i_minDst, tDst i_maxDst ):
	mMinSrc( (tCompute)i_minSrc ),
	mMinDst( (tCompute)i_minDst),
	mScale( ( (tCompute)i_maxDst-(tCompute)i_minDst )/( (tCompute)i_maxSrc-(tCompute)i_minSrc ) )
{
}

template <class tSrc, class tDst, class tCompute>
inline tDst LinearConverter<tSrc,tDst,tCompute>::operator()( const tSrc &i_src ){ return (tDst)( ( ( (tCompute)-mMinSrc )*mScale )+mMinDst ); }
*/


//----------------------------------------------------------------------
// MultiChannel::Iterator inline methods
//----------------------------------------------------------------------

template <class tData>
inline MultiChannel<tData>::Iterator::Iterator(){}

template <class tData>
inline size_t MultiChannel<tData>::Iterator::size() const { return mIterators.size(); }

template <class tData>
inline void MultiChannel<tData>::Iterator::operator ++(int)
{
	for ( size_t i=0; i<mIterators.size(); i++ )
		mIterators[i]++;
}

template <class tData>
inline void MultiChannel<tData>::Iterator::operator --(int)
{
	for ( size_t i=0; i<mIterators.size(); i++ )
		mIterators[i]++;
}

template <class tData>
inline void MultiChannel<tData>::Iterator::operator +=( int i_offset)
{
	for ( size_t i=0; i<mIterators.size(); i++ )
		mIterators[i] += i_offset;
}

template <class tData>
inline bool MultiChannel<tData>::Iterator::operator ==( const Iterator &i_it ) const
{
	ELISE_DEBUG_ERROR(
		( mIterators.size()==0 || i_it.mIterators.size()==0 ),
		"MultiChannel::Iterator::operator ==(const Iterator &)",
		"mIterator.size()==0 || i_it.mIterators.size()==0" );
	return mIterators[0]==i_it.mIterators[0];
}

template <class tData>
inline bool MultiChannel<tData>::Iterator::operator !=( const Iterator &i_it ) const { return !( *this==i_it ); }

template <class tData>
inline void MultiChannel<tData>::Iterator::operator -=( int i_offset){ *this += -i_offset; }

template <class tData>
inline tData & MultiChannel<tData>::Iterator::operator [] ( size_t i_index )
{
	ELISE_DEBUG_ERROR(
		i_index>=mIterators.size(),
		"MultiChannel<tData>::Iterator::operator []",
		"MultiChannel::Iterator::operator[](size_t): index "<<i_index<<" out of range [0,"<<mIterators.size()-1<<"]" );
	return *mIterators[i_index];
}

template <class tData>
inline const tData & MultiChannel<tData>::Iterator::operator [] ( size_t i_index ) const
{
	ELISE_DEBUG_ERROR(
		i_index>=mIterators.size(),
		"MultiChannel<tData>::Iterator::operator []",
		"MultiChannel::Iterator::operator[](size_t) const: index "<<i_index<<" out of range [0,"<<mIterators.size()-1<<"]" );
	return *mIterators[i_index];
}


//----------------------------------------------------------------------
// MultiChannel inline methods
//----------------------------------------------------------------------

template <class tData>
inline MultiChannel<tData>::MultiChannel():mWidth(0),mHeight(0){}

template <class tData>
inline MultiChannel<tData>::MultiChannel( const MultiChannel<tData> &i_b ):mWidth(0),mHeight(0){ set(i_b); }

template <class tData>
inline MultiChannel<tData>::MultiChannel( const Im2D<tData,tBase> &i_im2d ):mWidth(0),mHeight(0)
{
	resize( i_im2d.tx(), i_im2d.ty(), 1 );
	set( 0, i_im2d );
}

template <class tData>
inline MultiChannel<tData>::MultiChannel( int i_width, int i_height, int i_nbChannels ):mWidth(0),mHeight(0){ resize( i_width, i_height, i_nbChannels ); }

template <class tData>
inline void MultiChannel<tData>::clear(){ resize(0,0,0); }

template <class tData>
inline MultiChannel<tData>::~MultiChannel(){ clear(); }

template <class tData>
inline MultiChannel<tData> & MultiChannel<tData>::operator =( const MultiChannel<tData> &i_b )
{
	set(i_b);
	return *this;
}

template <class tData>
inline GenIm::type_el MultiChannel<tData>::typeEl() const { return type_of_ptr( (tData*)NULL ); }

template <class tData>
inline int MultiChannel<tData>::width() const { return mWidth; }

template <class tData>
inline int MultiChannel<tData>::height() const { return mHeight; }

template <class tData>
inline size_t MultiChannel<tData>::nbChannels() const { return mChannels.size(); }

template <class tData>
inline const Im2D<tData,typename MultiChannel<tData>::tBase> & MultiChannel<tData>::operator []( size_t i_iChannel ) const
{
	ELISE_DEBUG_ERROR( i_iChannel>=mChannels.size(), "const Im2D<tData,tBase> MultiChannel::operator [](size_t) const", "i_iChannel>=mChannels.size()" );
	return *mChannels[i_iChannel];
}

template <class tData>
tData ** MultiChannel<tData>::data( size_t i_iChannel ) const
{
	ELISE_DEBUG_ERROR( i_iChannel>=mChannels.size(), "tData ** MultiChannel::data(size_t) const", "i_iChannel>=mChannels.size()" );
	return mChannels[i_iChannel]->data();
}

template <class tData>
tData * MultiChannel<tData>::data_lin( size_t i_iChannel ) const
{
	ELISE_DEBUG_ERROR( i_iChannel>=mChannels.size(), "tData * MultiChannel::data_lin(size_t) const", "i_iChannel>=mChannels.size()" );
	return mChannels[i_iChannel]->data_lin();
}

template <class tData>
inline bool MultiChannel<tData>::write_raw( const string &i_filename ) const
{
	return write_raw( i_filename, g_versioned_headers_list[VFH_RawIm2D].last_handled_version, MSBF_PROCESSOR() );
}

template <class tData>
inline bool MultiChannel<tData>::read_tiff( const string &i_filename )
{
	Tiff_Im tiff( i_filename.c_str() );
	return read_tiff( tiff );
}

template <class tData>
inline bool MultiChannel<tData>::hasSameDimensions( const MultiChannel<tData> &i_b ) const
{
	return ( width()==i_b.width() && height()==i_b.height() && nbChannels()==i_b.nbChannels() );
}

template <class tData>
inline bool MultiChannel<tData>::hasSameData( const MultiChannel<tData> &i_b ) const
{
	size_t firstDifferentChannel;
	return hasSameData( i_b, firstDifferentChannel );
}

template <class tData>
inline size_t MultiChannel<tData>::nbPixels() const { return ( (size_t)mWidth )*( (size_t)mHeight ); }

template <class tData>
inline size_t MultiChannel<tData>::nbChannelBytes() const { return nbPixels()*sizeof(tData); }

template <class tData>
inline size_t MultiChannel<tData>::nbBytes() const { return nbChannelBytes()*mChannels.size(); }

template <class tData>
inline size_t MultiChannel<tData>::nbValues() const { return nbPixels()*mChannels.size(); }


//----------------------------------------------------------------------
// related functions
//----------------------------------------------------------------------

template <class T>
inline void __clear_vector( vector<T*> &v )
{
	for ( size_t i=0; i<v.size(); i++ )
		delete v[i];
	v.clear();
}
