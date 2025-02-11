#ifndef MMVII_EXIFDATA_H
#define MMVII_EXIFDATA_H

#include <vector>
#include <string>
#include <optional>
#include <cstdint>
#include <map>

namespace MMVII {


class cExifData {
public:
    cExifData() {}
    void reset();   // Reset all tags to nullopt

// Main Tags
    std::optional<unsigned> mPixelXDimension;
    std::optional<unsigned> mPixelYDimension;

    std::optional<double> mFocalLength_mm;
    std::optional<double> mFocalLengthIn35mmFilm_mm;

    std::optional<double> mFNumber;
    std::optional<double> mExposureTime_s;

    std::optional<std::string> mMake;
    std::optional<std::string> mModel;
    std::optional<std::string> mLensMake;
    std::optional<std::string> mLensModel;

// Other Tags
    std::optional<std::string> mExifVersion;
    std::optional<double> mOrientation;

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
    std::optional<uint64_t> mGPSTimeUTC_s;                  // Unix UTC time from GPS receiver
    std::optional<uint64_t> mGPSTimeUTC_ns;

    std::optional<double> mDateTimeNumber_s;                // Not Unix Epoch, but related to. Can be used to sort images.
    std::optional<double> mDateTimeDigitizedNumber_s;
    std::optional<double> mDateTimeOriginalNumber_s;

// Fill this structure from file aFileName (SVP: true: return false on error, false: halt program with error message)
    bool FromFile(const std::string &aFileName, bool SVP=true);
// Fill only main tags
    bool FromFileMainOnly(const std::string &aFileName, bool SVP=true);

/*
* static methods
*/

// Fill argument anExif
    static bool FromFile(const std::string &aFileName, cExifData &anExif, bool SVP=true);
// Return a struct
    static cExifData CreateFromFile(const std::string &aFileName, bool SVP=true);

// Idem for main tags only
    static bool FromFileMainOnly(const std::string &aFileName, cExifData &anExif, bool SVP=true);
    static cExifData CreateFromFileMainOnly(const std::string &aFileName, bool SVP=true);

// Return a list of ALL exif tags found from file
    static std::vector<std::string> StringListFromFile(const std::string &aName, bool SVP=true);

// Return a dict of all metadata strings stored by domain (contains exif if present but other metadata too)
    static std::map<std::string, std::vector<std::string>> AllMetadataFromFile(const std::string &aFileName, bool SVP=true);

};


} // namespace MMVII

#endif // MMVII_EXIFDATA_H
