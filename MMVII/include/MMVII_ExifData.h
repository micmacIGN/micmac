#ifndef MMVII_EXIFDATA_H
#define MMVII_EXIFDATA_H

#include <vector>
#include <string>
#include <optional>
#include <cstdint>

namespace MMVII {


// FIXME CM: interface that with cDataFileIm2D (cache lazy read ?)

class cExifData {
public:
    cExifData() {}
    void reset();

    std::optional<std::string> mExifVersion;
    std::optional<unsigned> mPixelXDimension;
    std::optional<unsigned> mPixelYDimension;

    std::optional<double> mFocalLength_mm;
    std::optional<double> mFocalLengthIn35mmFilm_mm;

    std::optional<double> mFNumber;
    std::optional<double> mExposureTime_s;
    std::optional<double> mOrientation;

    std::optional<std::string> mMake;
    std::optional<std::string> mModel;
    std::optional<std::string> mLensMake;
    std::optional<std::string> mLensModel;

    std::optional<double> mXResolution;
    std::optional<double> mYResolution;
    std::optional<unsigned> mResolutionUnit;
    std::optional<double> mFocalPlaneXResolution;
    std::optional<double> mFocalPlaneYResolution;
    std::optional<unsigned> mFocalPlaneResolutionUnit;

    std::optional<std::string> mDateTime;
    std::optional<std::string> mDateTimeDigitized;
    std::optional<std::string> mDateTimeOriginal;
    std::optional<std::string> mSubSecTime;
    std::optional<std::string> mSubSecTimeDigitized;
    std::optional<std::string> mSubSecTimeOriginal;

    std::optional<double> mGPSAltitude_m;
    std::optional<double> mGPSLongitude_deg;
    std::optional<double> mGPSLatitude_deg;
    std::optional<std::string> mGPSTimeStamp;
    std::optional<std::string> mGPSDateStamp;
    std::optional<uint64_t> mGPSTimeUTC_s;
    std::optional<uint64_t> mGPSTimeUTC_ns;

    std::optional<double> mDateTimeNumber_s;                // Not Unix Epoch, but related to. Can be used to sort images.
    std::optional<double> mDateTimeDigitizedNumber_s;
    std::optional<double> mDateTimeOriginalNumber_s;

    bool FromFile(const std::string &aName, bool SVP=true);
    bool FromFileMainOnly(const std::string &aName, bool SVP=true);

    static std::vector<std::string> StringListFromFile(const std::string &aName, bool SVP=true);
    static bool FromFile(const std::string &aName, cExifData &anExif, bool SVP=true);
    static cExifData CreateFromFile(const std::string &aName, bool SVP=true);
    static bool FromFileMainOnly(const std::string &aName, cExifData &anExif, bool SVP=true);
    static cExifData CreateFromFileMainOnly(const std::string &aName, bool SVP=true);
};


} // namespace MMVII

#endif // MMVII_EXIFDATA_H
