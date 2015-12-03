#ifndef __SCENE_VIEW__
#define __SCENE_VIEW__

#include <string>
#include <vector>
#include <list>

class MipmapHandler
{
public:
	class Mipmap
	{
	friend class MipmapHandler;

	public:
		std::string mFilename;
		unsigned int mSubScale;
		bool mForceGray;

		std::string mCacheFilename;
		unsigned char *mData;
		unsigned int mWidth, mHeight;
		size_t mNbPixels;
		unsigned int mNbChannels, mNbBitsPerChannel;
		size_t mNbBytes;

		Mipmap( const std::string &aFilename, unsigned int aSubScale, bool aForceGray );
		~Mipmap();

		bool is( const std::string &aFilename, unsigned int aSubcale, bool aForceGray ) const;

		bool writeTiff( const std::string &aOutputFilename, std::string *oErrorMessage = NULL );
		bool writeTiff( std::string *oErrorMessage = NULL );

		bool init();

		bool hasData() const;
	private:
		void release();
		bool read();

		#ifdef __DEBUG
			bool mIsInit;
		#endif
	};

public:
		MipmapHandler( size_t aMaxLoaded = 1 );
		~MipmapHandler();
		void setMaxLoaded( size_t aMaxLoaded );
		Mipmap * releaseOneMipmap( const Mipmap *aMipmapToKeep = NULL );

		Mipmap * getExisting( const std::string &aFilename, unsigned int aSubScale, bool aForceGray );
		bool add( const std::string &aFilename, unsigned int aSubScale, bool aForceGray, Mipmap *&oMipmap );
		Mipmap * ask( const std::string &aFilename, unsigned int aSubScale, bool aForceGray );
		void release();

		bool isLoaded( const Mipmap &aMipmap ) const;
		void release( Mipmap &aMipmap );
		bool read( Mipmap &aMipmap );

		void clear();

		#ifdef __DEBUG
			size_t nbLoadedMipmaps() const;
		#endif

		std::vector<Mipmap *> mMipmaps;
		size_t mMaxLoaded;
		std::list<Mipmap *> mLoadedMipmaps;
};

void rgb888_to_red8( const unsigned char *aRgbData, size_t aWidth, size_t aHeight, unsigned int aPadding, unsigned char *oData );

void gray8_to_rgb888( const unsigned char *aGrayData, size_t aWidth, size_t aHeight, unsigned char *oData, unsigned int aPadding );

template <class tData>
void mixRGB( const tData *aRedData, const tData *aGreenData, const tData *aBlueData, size_t aNbPixel, tData *oData );

template <class tData>
void scanlineCopy( const unsigned char *aSrc, size_t aSrcWidth, size_t aHeight, unsigned char *aDst, size_t aDstWidth );

#include "MipmapHandler.inline.h"

#endif
