#include "StdAfx.h"
#include "MipmapHandler.h"
#include "../src/uti_image/Digeo/MultiChannel.h"

//~ #define WRITE_READ_MIPMAP
//~ #define VERBOSE_MIPMAP_HANDLER

#ifdef VERBOSE_MIPMAP_HANDLER
	int gNbDeletetedMipmaps = 0;
#endif

using namespace std;

// ---------------------------------------------------------------------
// MipmapHandler::Mipmap
// ---------------------------------------------------------------------

#ifdef WRITE_READ_MIPMAP
	int gNbReadMipmap = 0;
#endif

bool MipmapHandler::Mipmap::init()
{
	#ifdef __DEBUG
		ELISE_DEBUG_ERROR(mIsInit == true, "MipmapHandler::Mipmap::init", "mIsInit == true");
		mIsInit = true;
	#endif

	if ( !ELISE_fp::exist_file(mFilename)) return false;

	mCacheFilename = NameFileStd(mFilename, 3, false, true, false, true); // 3 = aNbChan, false = cons16B, true = ExigNoCompr, false = Create, true = ExigB8
	if (mForceGray || !ELISE_fp::exist_file(mCacheFilename))
		mCacheFilename = NameFileStd(mFilename, 1, false, true, false, true); // 1 = aNbChan, false = cons16B, true = ExigNoCompr, false = Create, true = ExigB8

	if ( !ELISE_fp::exist_file(mCacheFilename)) cout << "creating cache file [" << mCacheFilename << ']' << endl;

	Tiff_Im cacheTiff(mCacheFilename.c_str());
	ELISE_DEBUG_ERROR(cacheTiff.nb_chan() < 1, "MipmapHandler::Mipmap::init", "cacheTiff.nb_chan() = " << cacheTiff.nb_chan() << " < 1");
	ELISE_DEBUG_ERROR(cacheTiff.bitpp() < 1, "MipmapHandler::Mipmap::init", "cacheTiff.bitpp() = " << cacheTiff.bitpp() << " < 1");

	if (cacheTiff.type_el() != GenIm::u_int1)
	{
		cerr << "invalid type: " << eToString(cacheTiff.type_el()) << endl;
		return false;
	}

	if (cacheTiff.nb_chan() != 1 && cacheTiff.nb_chan() != 3)
	{
		cerr << "invalid number of channels: " << cacheTiff.nb_chan() << endl;
		return false;
	}

	mWidth = cacheTiff.sz().x;
	mHeight = cacheTiff.sz().y;
	mNbPixels = size_t(mWidth) * size_t(mHeight);
	mNbChannels = (unsigned int)cacheTiff.nb_chan();
	mNbBitsPerChannel = (unsigned int)cacheTiff.bitpp();
	mNbBytes = mNbPixels * size_t(mNbChannels) * size_t(mNbBitsPerChannel >> 3);

	return true;
}

void MipmapHandler::Mipmap::release()
{
	if (mData != NULL)
	{
		#ifdef VERBOSE_MIPMAP_HANDLER
			cout << "VERBOSE_MIPMAP_HANDLER: release mData = {" << (void *)mData << "} of mipmap {" << this << "}" << endl;
		#endif
		delete [] mData;
	}
	mData = NULL;
}

MipmapHandler::Mipmap::~Mipmap()
{
	release();

	#ifdef VERBOSE_MIPMAP_HANDLER
		cout << "VERBOSE_MIPMAP_HANDLER: " << gNbDeletetedMipmaps++ << ": deleting Mipmap {" << this << "} of name [" << mFilename << "]" << endl;
		//~ if (gNbDeletetedMipmaps == 2) cin.get();
	#endif
}

bool MipmapHandler::Mipmap::read()
{
	#ifdef __DEBUG
		ELISE_DEBUG_ERROR(mIsInit == false, "MipmapHandler::Mipmap::read", "mIsInit == false");
	#endif

	release();

	MultiChannel<U_INT1> channels;
	if ( !channels.read_tiff(mCacheFilename)) return false;

	mData = new unsigned char[mNbBytes];
	#ifdef VERBOSE_MIPMAP_HANDLER
		cout << "VERBOSE_MIPMAP_HANDLER: allocate mData = {" << (void *)mData << "} of mipmap {" << this << "}" << endl;
	#endif

	channels.toTupleArray(mData);

	#ifdef WRITE_READ_MIPMAP
		stringstream ss;
		ss << "read_mipmap_" << setw(3) << setfill('0') << gNbReadMipmap++ << ".tif";
		writeTiff(ss.str());
		cout << "WRITE_READ_MIPMAP: read mipmap written to [" << ss.str() << ']' << endl;
	#endif

	return true;
}

bool MipmapHandler::Mipmap::writeTiff( const string &aOutputFilename, string *oErrorMessage )
{
	if (mData == NULL)
	{
		if (oErrorMessage != NULL) *oErrorMessage = "no data loaded";
		return false;
	}

	if ((mNbChannels != 1 && mNbChannels != 3) || (mNbBitsPerChannel != 8 && mNbBitsPerChannel != 16))
	{
		if (oErrorMessage != NULL)
		{
			stringstream ss;
			ss << "invalid format: " << mNbChannels << " channels, " << mNbBitsPerChannel << " bits per channel";
			*oErrorMessage = ss.str();
		}
		return false;
	}

	if (mNbBitsPerChannel == 8)
	{
		MultiChannel<U_INT1> channels(mWidth, mHeight, mNbChannels);
		channels.setFromTuple(mData);
		channels.write_tiff(aOutputFilename);
	}
	else
	{
		MultiChannel<U_INT2> channels(mWidth, mHeight, mNbChannels);
		channels.setFromTuple((U_INT2 *)mData);
		channels.write_tiff(aOutputFilename);
	}

	return ELISE_fp::exist_file(mFilename);
}


// ---------------------------------------------------------------------
// MipmapHandler
// ---------------------------------------------------------------------

MipmapHandler::Mipmap * MipmapHandler::getExisting( const std::string &aFilename, unsigned int aSubscale, bool aForceGray )
{
	Mipmap **it = mMipmaps.data();
	size_t i = mMipmaps.size();
	while (i--)
	{
		if ((**it).is(aFilename, aSubscale, aForceGray)) return *it;
		it++;
	}
	return NULL;
}

bool MipmapHandler::add( const std::string &aFilename, unsigned int aSubscale, bool aForceGray, MipmapHandler::Mipmap *&oMipmap )
{
	oMipmap = getExisting(aFilename, aSubscale, aForceGray);
	if (oMipmap != NULL) return false;

	mMipmaps.push_back(new Mipmap(aFilename, aSubscale, aForceGray));
	oMipmap = mMipmaps.back();

	#ifdef VERBOSE_MIPMAP_HANDLER
		cout << "VERBOSE_MIPMAP_HANDLER: adding mipmap {" << oMipmap << "} of name [" << oMipmap->mFilename << "]" << endl;
	#endif

	return true;
}

MipmapHandler::Mipmap * MipmapHandler::ask( const std::string &aFilename, unsigned int aSubScale, bool aForceGray )
{
	Mipmap *mipmap;
	if (add(aFilename, aSubScale, aForceGray, mipmap) && !mipmap->init())
	{
		mMipmaps.pop_back();
		return NULL;
	}
	return mipmap;
}

MipmapHandler::Mipmap * MipmapHandler::releaseOneMipmap( const MipmapHandler::Mipmap *aMipmapToKeep )
{
	list<Mipmap *>::iterator itMipmap = mLoadedMipmaps.begin();
	while (itMipmap != mLoadedMipmaps.end())
	{
		if (*itMipmap != aMipmapToKeep)
		{
			Mipmap *released = *itMipmap;
			released->release();
			mLoadedMipmaps.erase(itMipmap);
			return released;
		}
		itMipmap++;
	}

	ELISE_DEBUG_ERROR(true, "MipmapHandler::releaseOneMipmap", "failed to release a Mipmap");

	return NULL;
}

void MipmapHandler::setMaxLoaded( size_t aMaxLoaded )
{
	cout << "setMaxLoaded = " << aMaxLoaded << endl;

	mMaxLoaded = aMaxLoaded;
	if (mLoadedMipmaps.size() <= aMaxLoaded) return;
	size_t i = mLoadedMipmaps.size() - aMaxLoaded;
	list<Mipmap *>::iterator it = mLoadedMipmaps.begin();
	while (i--)
	{
		(**it).release();
		it = mLoadedMipmaps.erase(it);
	}
}

bool MipmapHandler::read( Mipmap &aMipmap )
{
	if (mLoadedMipmaps.size() >= mMaxLoaded) releaseOneMipmap(&aMipmap);

	bool result = aMipmap.read();
	mLoadedMipmaps.push_back(&aMipmap);
	return result;
}

#ifdef __DEBUG
	size_t MipmapHandler::nbLoadedMipmaps() const
	{
		size_t result = 0;
		Mipmap * const *it = mMipmaps.data();
		size_t i = mMipmaps.size();
		while (i--) if ((**it++).mData != NULL) result++;
		return result;
	}
#endif

void MipmapHandler::release()
{
	list<Mipmap *>::iterator it = mLoadedMipmaps.begin();
	while (it != mLoadedMipmaps.end())
	{
		(**it).release();
		it = mLoadedMipmaps.erase(it);
	}

	ELISE_DEBUG_ERROR(nbLoadedMipmaps() != 0, "MipmapHandler::release", "nbLoadedMipmaps() = " << nbLoadedMipmaps() << " != 0");
}

void MipmapHandler::clear()
{
	Mipmap * const *it = mMipmaps.data();
	size_t i = mMipmaps.size();
	while (i--) delete (*it++);

	mMipmaps.clear();
	mLoadedMipmaps.clear();
}

bool MipmapHandler::isLoaded( const Mipmap &aMipmap ) const
{
	bool result = false;
	list<Mipmap *>::const_iterator it = mLoadedMipmaps.begin();
	while (it != mLoadedMipmaps.end())
	{
		if (*it == &aMipmap)
		{
			result = true;
			break;
		}
	}

	ELISE_DEBUG_ERROR(result != (aMipmap.mData != NULL), "MipmapHandler::isLoaded", "result = " << result << " != (aMipmap.mData != NULL) = " << (aMipmap.mData != NULL));
	return result;
}

void MipmapHandler::release( Mipmap &aMipmap )
{
	ELISE_DEBUG_ERROR( !isLoaded(aMipmap), "MipmapHandler::release", "!isLoaded(aMipmap)");

	list<Mipmap *>::iterator it = mLoadedMipmaps.begin();
	while (it != mLoadedMipmaps.end())
	{
		if (*it == &aMipmap)
		{
			aMipmap.release();
			mLoadedMipmaps.erase(it);
			return;
		}
	}

	ELISE_DEBUG_ERROR(true, "MipmapHandler::release", "try to release a Mipmap that is not loaded");
}


// ---------------------------------------------------------------------
// related functions
// ---------------------------------------------------------------------

void gray8_to_rgb888( const unsigned char *aGrayData, size_t aWidth, size_t aHeight, unsigned char *oData, unsigned int aPadding )
{
	while (aHeight--)
	{
		size_t x = aWidth;
		while (x--)
		{
			oData[0] = oData[1] = oData[2] = *aGrayData++;
			oData += 3;
		}
		oData += aPadding;
	}
}

void rgb888_to_red8( const unsigned char *aRgbData, size_t aWidth, size_t aHeight, unsigned int aPadding, unsigned char *oData )
{
	while (aHeight--)
	{
		size_t x = aWidth;
		while (x--)
		{
			*oData++ = *aRgbData;
			aRgbData += 3;
		}
		aRgbData += aPadding;
	}
}
