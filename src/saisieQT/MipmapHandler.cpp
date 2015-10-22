#include "StdAfx.h"
#include "MipmapHandler.h"

using namespace std;

// ---------------------------------------------------------------------
// MipmapHandler::Mipmap
// ---------------------------------------------------------------------

bool MipmapHandler::Mipmap::init()
{
	if ( !ELISE_fp::exist_file(mFilename))
	{
		cerr << "input file [" << mFilename << "] does not exist" << endl;
		return false;
	}

	mCacheFilename = NameFileStd(mFilename, 3, false, true, false, true); // 3 = aNbChan, false = cons16B, true = ExigNoCompr, false = Create, true = ExigB8
	if (mForceGray || !ELISE_fp::exist_file(mCacheFilename))
		mCacheFilename = NameFileStd(mFilename, 1, false, true, false, true); // 1 = aNbChan, false = cons16B, true = ExigNoCompr, false = Create, true = ExigB8

	cout << "mCacheFilename = [" << mCacheFilename << ']' << endl;

	//~ cElDate cacheFileDate = cElDate::NoDate;
	//~ if (  || cacheFileDate < imageFileDate)
		//~ cout << "cache is outdated" << endl;
	//~ else

	if ( !ELISE_fp::exist_file(mCacheFilename))
		cout << "creating cache file [" << mCacheFilename << ']' << endl;

	Tiff_Im cacheTiff(mCacheFilename.c_str());

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

	return true;
}

void MipmapHandler::Mipmap::release()
{
	delete [] mData;
	mData = NULL;
}

bool MipmapHandler::Mipmap::read()
{
	release();

	mData = new unsigned char[mNbPixels * 3];

	Tiff_Im tiff(mCacheFilename.c_str());

	if (tiff.nb_chan() == 1)
	{
		Im2DGen channel = tiff.ReadIm();
		const U_INT1 *gray = ((Im2D_U_INT1 *)&channel)->data_lin();
		grayToRGB(gray, mNbPixels, mData);

		return true;
	}

	if (tiff.nb_chan() == 3)
	{
		std::vector<Im2DGen *> channels = tiff.ReadVecOfIm();
		const U_INT1 *red = ((Im2D_U_INT1 *)channels[0])->data_lin(),
						 *green = ((Im2D_U_INT1 *)channels[1])->data_lin(),
						 *blue = ((Im2D_U_INT1 *)channels[2])->data_lin();
		mixRGB(red, green, blue, mNbPixels, mData);

		delete channels[0];
		delete channels[1];
		delete channels[2];

		return true;
	}

	return false;
}


// ---------------------------------------------------------------------
// MipmapHandler
// ---------------------------------------------------------------------

MipmapHandler::Mipmap * MipmapHandler::getExisting( const std::string &aFilename, unsigned int aSubscale, bool aForceGray )
{
	Mipmap *it = mMipmaps.data();
	size_t i = mMipmaps.size();
	while (i--)
	{
		if (it->is(aFilename, aSubscale, aForceGray)) return &*it;
		it++;
	}
	return NULL;
}

bool MipmapHandler::add( const std::string &aFilename, unsigned int aSubscale, bool aForceGray, MipmapHandler::Mipmap *&oMipmap )
{
	oMipmap = getExisting(aFilename, aSubscale, aForceGray);
	if (oMipmap != NULL) return false;

	mMipmaps.push_back(Mipmap(aFilename, aSubscale, aForceGray));
	oMipmap = &mMipmaps.back();
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

bool MipmapHandler::ask( const std::string &aFilename, unsigned int aSubScale, bool aForceGray, unsigned int &oWidth, unsigned int &oHeight )
{
	Mipmap *mipmap = ask(aFilename, aSubScale, aForceGray);
	if (mipmap == NULL) return false;
	oWidth = mipmap->mWidth;
	oHeight = mipmap->mHeight;
	return true;
}

bool MipmapHandler::getData( const std::string &aFilename, unsigned int aSubScale, bool aForceGray, unsigned char *&oData, unsigned int &oWidth, unsigned int &oHeight )
{
	Mipmap *mipmap = ask(aFilename, aSubScale, aForceGray);
	if (mipmap == NULL || !mipmap->read()) return false;

	oData = mipmap->mData;
	oWidth = mipmap->mWidth;
	oHeight = mipmap->mHeight;
	return true;
}


// ---------------------------------------------------------------------
// related functions
// ---------------------------------------------------------------------

void grayToRGB( const U_INT1 *aGrayData, size_t aNbPixels, U_INT1 *oData )
{
	while (aNbPixels--)
	{
		oData[0] = oData[1] = oData[2] = *aGrayData++;
		oData += 3;
	}
}

void mixRGB( const U_INT1 *aRedData, const U_INT1 *aGreenData, const U_INT1 *aBlueData, size_t aNbPixels, U_INT1 *oData )
{
	while (aNbPixels--)
	{
		oData[0] = *aRedData++;
		oData[1] = *aGreenData++;
		oData[2] = *aBlueData++;
		oData += 3;
	}
}

void scanlineCopy( const unsigned char *aSrc, size_t aSrcWidth, size_t aHeight, unsigned char *aDst, size_t aDstWidth )
{
	while (aHeight--)
	{
		memcpy(aDst, aSrc, aSrcWidth);
		aSrc += aSrcWidth;
		aDst += aDstWidth;
	}
}
