#include "MMVII_ExifData.h"
#include "cMMVII_Appli.h"

namespace MMVII
{

class cAppli_ExifData : public cMMVII_Appli
{
public :
    cAppli_ExifData(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec,bool isBasic);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

private :
    std::string mNameIn;  ///< Input image name
    int mDisp;
};

cCollecSpecArg2007 & cAppli_ExifData::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return
        anArgObl
        <<   Arg2007(mNameIn,"Name of input file",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
        ;
}

cCollecSpecArg2007 & cAppli_ExifData::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return
        anArgOpt
        <<   AOpt2007(mDisp,"Disp","0->All known tags , 1->Main tags, 2->Raw strings",{eTA2007::HDV,{eTA2007::Range,"[0,2]"}})
        ;
}



template<typename T>
std::ostream& operator<<(std::ostream& os, std::optional<T> const& opt)
{
    return opt ? os << opt.value() : os << "<NULL>";
}


int cAppli_ExifData::Exe()
{
    for (const auto & aName : VectMainSet(0))
    {
        std::cout << "####### " << aName <<":" << std::endl;
        if (mDisp == 2) {
            auto anExifList = cExifData::StringListFromFile(aName);
            for (const auto &s : anExifList)
                std::cout << s << std::endl;
        } else {
            cExifData anExif;
            if (mDisp == 0)
                anExif = cExifData::CreateFromFile(aName);
            else
                anExif = cExifData::CreateFromFileMainOnly(aName);
            std::cout << std::setprecision(17);

#define DISP_EXIF(key) std::cout << #key << ": " << anExif.m##key << std::endl;

            DISP_EXIF(ExifVersion);

            DISP_EXIF(PixelXDimension);
            DISP_EXIF(PixelYDimension);

            DISP_EXIF(FocalLength_mm);
            DISP_EXIF(FocalLengthIn35mmFilm_mm);
            DISP_EXIF(FNumber);
            DISP_EXIF(ExposureTime_s);
            DISP_EXIF(Orientation);
            DISP_EXIF(Make);
            DISP_EXIF(Model);
            DISP_EXIF(LensMake);
            DISP_EXIF(LensModel);
            if (mDisp == 0) {
                DISP_EXIF(XResolution);
                DISP_EXIF(YResolution);
                DISP_EXIF(ResolutionUnit);
                DISP_EXIF(FocalPlaneXResolution);
                DISP_EXIF(FocalPlaneYResolution);
                DISP_EXIF(FocalPlaneResolutionUnit);

                DISP_EXIF(DateTime);
                DISP_EXIF(SubSecTime);
                DISP_EXIF(DateTimeOriginal);
                DISP_EXIF(SubSecTimeOriginal);
                DISP_EXIF(DateTimeDigitized);
                DISP_EXIF(SubSecTimeDigitized);

                DISP_EXIF(DateTimeNumber_s);
                DISP_EXIF(DateTimeOriginalNumber_s);
                DISP_EXIF(DateTimeDigitizedNumber_s);

                DISP_EXIF(GPSLongitude_deg);
                DISP_EXIF(GPSLatitude_deg);
                DISP_EXIF(GPSAltitude_m);

                DISP_EXIF(GPSDateStamp);
                DISP_EXIF(GPSTimeStamp);
                DISP_EXIF(GPSTimeUTC_s);
                DISP_EXIF(GPSTimeUTC_ns);

                DISP_EXIF(ExifVersion);
            }
       }
        std::cout << std::endl;
    }
    return EXIT_SUCCESS;
}


cAppli_ExifData:: cAppli_ExifData(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec,bool isBasic) :
    cMMVII_Appli (aVArgs,aSpec),
    mDisp(0)
{
}


tMMVII_UnikPApli Alloc_ExifData(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_ExifData(aVArgs,aSpec,true));
}

cSpecMMVII_Appli  TheSpec_ExifData
    (
        "ExifData",
        Alloc_ExifData,
        "Display Exif metadata from image file",
        {eApF::ImProc},
        {eApDT::Image},
        {eApDT::Console},
        __FILE__
        );


} // namespace MMVII
