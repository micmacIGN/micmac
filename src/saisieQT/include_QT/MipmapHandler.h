#ifndef __SCENE_VIEW__
#define __SCENE_VIEW__

#include <string>
#include <vector>

class MipmapHandler
{
private:
	class Mipmap
	{
	public:
		std::string mFilename;
		unsigned int mSubScale;
		bool mForceGray;

		std::string mCacheFilename;
		unsigned char *mData;
		unsigned int mWidth, mHeight;
		size_t mNbPixels;


		Mipmap( const std::string &aFilename, unsigned int aSubScale, bool aForceGray );
		~Mipmap();

		bool is( const std::string &aFilename, unsigned int aSubcale, bool aForceGray ) const;

		bool init();
		void release();
		bool read();
	};

	Mipmap * getExisting( const std::string &aFilename, unsigned int aSubScale, bool aForceGray );
	bool add( const std::string &aFilename, unsigned int aSubScale, bool aForceGray, Mipmap *&oMipmap );
	Mipmap * ask( const std::string &aFilename, unsigned int aSubScale, bool aForceGray );

public:
		bool ask( const std::string &aFilename, unsigned int aSubScale, bool aForceGray, unsigned int &oWidth, unsigned int &oHeight );
		bool getData( const std::string &aFilename, unsigned int aSubScale, bool aForceGray, unsigned char *&oData, unsigned int &oWidth, unsigned int &oHeight );

		std::vector<Mipmap> mMipmaps;
};

void grayToRGB( const unsigned char *aGrayData, size_t aNbPixel, unsigned char *oData );

void mixRGB( const unsigned char *aRedData, const unsigned char *aGreenData, const unsigned char *aBlueData, size_t aNbPixel, unsigned char *oData );

void scanlineCopy( const unsigned char *aSrc, size_t aSrcWidth, size_t aHeight, unsigned char *aDst, size_t aDstWidth );

#include "MipmapHandler.inline.h"

#endif
