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
{
}

inline bool MipmapHandler::Mipmap::is( const std::string &aFilename, unsigned int aSubScale, bool aForceGray ) const
{
	return mFilename == aFilename && mSubScale == aSubScale && mForceGray == aForceGray;
}

inline MipmapHandler::Mipmap::~Mipmap()
{
	release();
}
