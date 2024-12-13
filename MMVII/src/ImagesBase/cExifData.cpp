#include "MMVII_ExifData.h"
#include "MMVII_util.h"
#include "cGdalApi.h"
#include <gdal_priv.h>

#include <ctime>
#include <iomanip>

namespace MMVII
{

namespace {   // private namespace

// To store HMS or DMS data
struct cDH_MS {
    double unit,min,sec;
};

enum class eExtracType{Main,All};

// String to String
void ExifConvert(const std::string& aVal, std::optional<std::string>& aField)
{
    aField = aVal;
}

// String to Double  "2.3" or "(2.3)" or "23/10" or "(23/10)"
void ExifConvert(std::string aVal, std::optional<double>& aField)
{
    char *end;

    if (aVal.front() == '(' && aVal.back()==')')
    {
        aVal = aVal.substr(1,aVal.length()-2);
    }

    auto val = strtod(aVal.c_str(),&end);
    if (*end == 0 || *end == ' ')
    {
        aField = val;
    }
    else if (*end == '/')    // Some gdal exif (ex: .png) use original EXIF fractionnal representation : X/Y
    {
        char *sDen = end+1;
        auto dDen = strtod(sDen,&end);
        if (*end == 0 || *end == ' ') {
            aField = val / dDen;
        }
    }
}

// String to unsigned
void ExifConvert(const std::string& aVal, std::optional<unsigned>& aField)
{
    char *end;
    auto val = strtoul(aVal.c_str(),&end,0);
    if (*end == 0 || *end == ' ')
        aField = val;
}


// String to HMS or DMS : "(H) (M) (S)" or "Hp/Hq,Mp/Mq,Sp/Sq"
void ExifConvert(const std::string& aVal, std::optional<cDH_MS>& aField)
{
    cDH_MS anAngle;

    if (aVal.front() == '(')
    { // Format (deg) (min) (sec.fsec)
        if (3 == sscanf(aVal.c_str(),"(%lf) (%lf) (%lf)",&anAngle.unit,&anAngle.min,&anAngle.sec))
        {
            aField = anAngle;
        }
        return;
    }
    else
    { // Format deg_p/deg_q,min_p/min_q,sec_p/sec_q . Warning, can be: 35/1,2345/1000,0/1
        unsigned d_p,d_q,m_p,m_q,s_p,s_q;
        if (6 == sscanf(aVal.c_str(),"%u/%u,%u/%u,%u/%u",&d_p,&d_q,&m_p,&m_q,&s_p,&s_q)) {
            anAngle.unit = (double)d_p / d_q;
            anAngle.min = (double)m_p / m_q;
            anAngle.sec = (double)s_p / s_q;
            auto min_int = static_cast<long>(anAngle.min);
            if (min_int != anAngle.min) {
                // Fractional minute ...
                anAngle.sec += (anAngle.min - min_int) * 60;
                anAngle.min = min_int;
            }
            aField = anAngle;
        }
    }
}

// DMS to Decimal Degree with sign (South and West are negative)
void ExifDMStoDeg(const std::optional<cDH_MS>& aDMS, const std::optional<std::string>& aRef, std::optional<double> &aDeg)
{
    if (! aDMS)
        return;
    aDeg = aDMS->unit + aDMS->min / 60 + aDMS->sec / 3600;
    if (aRef && (*aRef == "S" || *aRef == "W"))
    {
        aDeg = - *aDeg;
    }
}

// Windows defines _mkgmtime instead of timegm
#ifdef _WIN32
# define timegm _mkgmtime
#endif


// Convert a date "YYYY:MM:DD HH:MM:SS" and a subsecond value "XXX" to an arbitrary double that keep ordering
void ExifDateTimeToNumber(const std::optional<std::string>& aDate, const std::optional<std::string>& aSubSec, std::optional<double>& aTime)
{
    if (! aDate)
        return;
    std::tm tm{};

    std::istringstream ss(*aDate);
    ss >> std::get_time(&tm, "%Y:%m:%d %H:%M:%S");
    if (ss.fail())
        return;
    tm.tm_isdst = 0;
    aTime = timegm(&tm) - 1073741824;    // Arbitrary offset (2^30) to add 1 bit of precision in the final double value at this time
    if (aSubSec)
    {
        char *end;
        double val = strtoul(aSubSec->c_str(),&end,0);
        if (*end == 0 || *end == ' ') {
            for (size_t i=0; i < aSubSec->length(); i++) {
                val = val / 10.0;
            }
            aTime = (*aTime) + val;
        }
    }
}

// Convert a GPS UTC Time to a Unix Epoch (second and nanosecond)
// Date is "YYYY:MM:SS", time is "(H) (M) (S)" or "Hp/Hq,Mp/Mq,Sp/Sq"  (second may be fractionnal)
void ExifGPSDateTimeToEpoch(const std::optional<std::string>& aDate,
                            const std::optional<std::string>& aTime,
                            std::optional<uint64_t>& aSec,
                            std::optional<uint64_t>& aNSec)
{
    if (!aDate || !aTime)
        return;
    std::tm tm{};
    if (3 != sscanf(aDate->c_str(),"%u:%u:%u)",&tm.tm_year,&tm.tm_mon,&tm.tm_mday))
        return;
    std::optional<cDH_MS> aHMS;
    ExifConvert(*aTime,aHMS);
    if (!aHMS)
        return;
    tm.tm_isdst = 0;
    tm.tm_year -=1900;
    tm.tm_mon -= 1;
    tm.tm_hour = aHMS->unit;
    tm.tm_min = aHMS->min;
    tm.tm_sec= static_cast<unsigned>(aHMS->sec);
    aSec = timegm(&tm);
    aNSec = (aHMS->sec - tm.tm_sec) / 1e9;
}

// Find key "aName" in aStrings and convert value to aField
template <typename T>
void ExifParse(const CPLStringList& aStrings, const char* aName, std::optional<T>& aField)
{
    std::string aKey("EXIF_");
    aKey += aName;
    auto pChar = aStrings.FetchNameValue(aKey.c_str());
    if (pChar == nullptr)
        return;

    std::string aVal(pChar);
    ExifConvert(aVal,aField);
}

// Fill struct aExif with all known tags (or only main known tags if requested)
void FillExif(const CPLStringList& aStrings, cExifData& aExif, eExtracType aType)
{
    aExif.reset();
#define EXIF_PARSE(name) ExifParse(aStrings,#name, aExif.m##name)
#define EXIF_UNIT_PARSE(name,unit) ExifParse(aStrings,#name, aExif.m##name##_##unit)

    EXIF_PARSE(PixelXDimension);
    EXIF_PARSE(PixelYDimension);

    EXIF_UNIT_PARSE(FocalLength,mm);
    EXIF_UNIT_PARSE(FocalLengthIn35mmFilm,mm);

    EXIF_PARSE(FNumber);
    EXIF_UNIT_PARSE(ExposureTime,s);

    EXIF_PARSE(Make);
    EXIF_PARSE(Model);
    EXIF_PARSE(LensMake);
    EXIF_PARSE(LensModel);

    if (aType == eExtracType::Main)
        return;

    EXIF_PARSE(ExifVersion);

    EXIF_PARSE(Orientation);

    EXIF_PARSE(XResolution);
    EXIF_PARSE(YResolution);
    EXIF_PARSE(ResolutionUnit);
    EXIF_PARSE(FocalPlaneXResolution);
    EXIF_PARSE(FocalPlaneYResolution);
    EXIF_PARSE(FocalPlaneResolutionUnit);

    EXIF_PARSE(DateTime);
    EXIF_PARSE(SubSecTime);
    EXIF_PARSE(DateTimeOriginal);
    EXIF_PARSE(SubSecTimeOriginal);
    EXIF_PARSE(DateTimeDigitized);
    EXIF_PARSE(SubSecTimeDigitized);

    EXIF_PARSE(GPSTimeStamp);
    EXIF_PARSE(GPSDateStamp);
    EXIF_PARSE(SubSecTimeDigitized);

    EXIF_UNIT_PARSE(GPSAltitude,m);
#undef EXIF_PARSE
#undef EXIF_UNIT_PARSE

// Convert DateTime info to a number to simplify ordering if needed
    ExifDateTimeToNumber(aExif.mDateTime,aExif.mSubSecTime,aExif.mDateTimeNumber_s);
    ExifDateTimeToNumber(aExif.mDateTimeOriginal,aExif.mSubSecTimeOriginal,aExif.mDateTimeOriginalNumber_s);
    ExifDateTimeToNumber(aExif.mDateTimeDigitized,aExif.mSubSecTimeDigitized,aExif.mDateTimeDigitizedNumber_s);

// Adjust sign of GPSAltitude
    if (aExif.mGPSAltitude_m) {
        // if EXIT_GPSAltitudeRef == 1 => Altitude is negative
        std::optional<unsigned> anAltitudeRef;
        ExifParse(aStrings,"GPSAltitudeRef",anAltitudeRef);
        if (anAltitudeRef && *anAltitudeRef == 1)
            aExif.mGPSAltitude_m = - *aExif.mGPSAltitude_m;
    }

// Extract GPS Latitude and Longitude and store them as signed decimal degrees
    std::optional<cDH_MS> aLongitude,aLatitude;
    std::optional<std::string> aLongitudeRef,aLatitudeRef;
    ExifParse(aStrings,"GPSLongitude",aLongitude);
    ExifParse(aStrings,"GPSLatitude",aLatitude);
    ExifParse(aStrings,"GPSLongitudeRef",aLongitudeRef);
    ExifParse(aStrings,"GPSLatidudeRef",aLatitudeRef);

    ExifDMStoDeg(aLongitude,aLongitudeRef,aExif.mGPSLongitude_deg);
    ExifDMStoDeg(aLatitude,aLatitudeRef,aExif.mGPSLatitude_deg);

// Convert GPS Date and Time to Unix Epoch (second and nanosec)
    ExifGPSDateTimeToEpoch(aExif.mGPSDateStamp, aExif.mGPSTimeStamp, aExif.mGPSTimeUTC_s, aExif.mGPSTimeUTC_ns);
}


CPLStringList GetExifMetadataRaw(const std::string &aName, bool SVP)
{

    auto onError = SVP ? cGdalApi::eOnError::ReturnNullptr : cGdalApi::eOnError::RaiseError;
    auto aDataSet = cGdalApi::OpenDataset(aName,GA_ReadOnly,onError);
    if (aDataSet == nullptr)
    {
        return CPLStringList();
    }
    auto aMetadataDomainList = CPLStringList(aDataSet->GetMetadataDomainList(),TRUE);

    if (aMetadataDomainList.FindString("") < 0) // Add empty domain if not already present
        aMetadataDomainList.AddString("");
    for (const auto& aMetadataDomain: aMetadataDomainList)
    {
        auto aMetadataList = CPLStringList((CSLConstList)aDataSet->GetMetadata(aMetadataDomain));
        if (aMetadataList.FindName("EXIF_ExifVersion") >= 0) {
            aMetadataList.Sort();
            cGdalApi::CloseDataset(aDataSet);
            return aMetadataList;
        }
    }
    cGdalApi::CloseDataset(aDataSet);
    return CPLStringList();
}

} // private namespace



// cExifData method
void cExifData::reset()
{
    *this = cExifData();
}


bool cExifData::FromFile(const std::string &aName, bool SVP)
{
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);
    FillExif(aMetadataList, *this, eExtracType::All);
    return ! aMetadataList.empty();
}

bool cExifData::FromFileMainOnly(const std::string &aName, bool SVP)
{
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);
    FillExif(aMetadataList, *this, eExtracType::Main);
    return ! aMetadataList.empty();
}



// cExifData static methods
std::vector<std::string> cExifData::StringListFromFile(const std::string &aName, bool SVP)
{
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);

    std::vector<std::string> anExifList;
    for (const auto& aMetadata: aMetadataList)
    {
        if (UCaseBegin("EXIF_",aMetadata))
        {
            anExifList.push_back(aMetadata);
        }
    }
    return anExifList;
}

bool cExifData::FromFile(const std::string &aName, cExifData &anExif, bool SVP)
{
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);
    FillExif(aMetadataList, anExif,eExtracType::All);
    return ! aMetadataList.empty();
}

bool cExifData::FromFileMainOnly(const std::string &aName, cExifData &anExif, bool SVP)
{
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);
    FillExif(aMetadataList, anExif,eExtracType::Main);
    return ! aMetadataList.empty();
}


cExifData cExifData::CreateFromFile(const std::string &aName, bool SVP)
{
    cExifData anExif;
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);
    FillExif(aMetadataList, anExif, eExtracType::All);
    return anExif;
}

cExifData cExifData::CreateFromFileMainOnly(const std::string &aName, bool SVP)
{
    cExifData anExif;
    auto aMetadataList = GetExifMetadataRaw(aName,SVP);
    FillExif(aMetadataList, anExif, eExtracType::Main);
    return anExif;
}

} // namespace MMVII
