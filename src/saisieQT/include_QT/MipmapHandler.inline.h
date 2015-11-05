// this file should be included by MipmapHandler.h solely

// ---------------------------------------------------------------------
// MipmapHandler::Mipmap
// ---------------------------------------------------------------------

inline MipmapHandler::Mipmap::Mipmap( const std::string &aFilename, unsigned int aSubScale, bool aForceGray ):
	mFilename(aFilename),
	mSubScale(aSubScale),
	mForceGray(aForceGray),
	mData(NULL),
	mWidth(0), mHeight(0)
#ifdef __DEBUG
	, mIsInit(false)
#endif
{
}

inline bool MipmapHandler::Mipmap::is( const std::string &aFilename, unsigned int aSubScale, bool aForceGray ) const
{
	return mFilename == aFilename && mSubScale == aSubScale && mForceGray == aForceGray;
}

//~ inline MipmapHandler::Mipmap::~Mipmap()
//~ {
	//~ release();
//~ }

inline bool MipmapHandler::Mipmap::writeTiff( std::string *oErrorMessage )
{
	return writeTiff(mCacheFilename, oErrorMessage);
}

inline bool MipmapHandler::Mipmap::hasData() const
{
	return mData != NULL;
}


// ---------------------------------------------------------------------
// MipmapHandler
// ---------------------------------------------------------------------

inline MipmapHandler::MipmapHandler( size_t aMaxLoaded ):
	mMaxLoaded(aMaxLoaded)
{
}

inline MipmapHandler::~MipmapHandler()
{
	clear();
}


// ---------------------------------------------------------------------
// related functions
// ---------------------------------------------------------------------

template <class tData>
inline void mixRGB( const tData *aRedData, const tData *aGreenData, const tData *aBlueData, size_t aNbPixels, tData *oData )
{
	while (aNbPixels--)
	{
		oData[0] = *aRedData++;
		oData[1] = *aGreenData++;
		oData[2] = *aBlueData++;
		oData += 3;
	}
}

template <class tData>
inline void scanlineCopy( const tData *aSrc, size_t aSrcWidth, size_t aHeight, tData *aDst, size_t aDstWidth )
{
	while (aHeight--)
	{
		memcpy(aDst, aSrc, aSrcWidth * sizeof(tData));
		aSrc += aSrcWidth;
		aDst += aDstWidth;
	}
}
