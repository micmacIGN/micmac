#include "cGdalApi.h"

namespace MMVII {


// Should be called at least once bedore any use of GDAL API
void cGdalApi::InitGDAL()
{
    static bool isGdalInitialized = false;
    if (isGdalInitialized)
        return;
    GDALAllRegister();
    CPLSetErrorHandler(GdalErrorHandler);
    isGdalInitialized = true;
}


// To read General Metadata of an existing image file
void cGdalApi::GetFileInfo(const std::string& aName, eTyNums& aType, cPt2di& aSz, int& aNbChannel)
{
    auto aDataset = OpenDataset(aName, GA_ReadOnly, eOnError::RaiseError);
    aSz = cPt2di(aDataset->GetRasterXSize(), aDataset->GetRasterYSize());
    aNbChannel = aDataset->GetRasterCount();
    aType = ToMMVII( aDataset->GetRasterBand( 1 )->GetRasterDataType());
    CloseDataset(aDataset);
}


GDALDataset* cGdalApi::CreateDataset(const cDataFileIm2D& aDataFileIm2D, bool *createdInMemory)
{
    auto aName = aDataFileIm2D.Name();
    remove(aName.c_str());

    auto aGDALType = cGdalApi::FromMMVII(aDataFileIm2D.Type());
    auto aSz = aDataFileIm2D.Sz();
    auto aNbChan = aDataFileIm2D.NbChannel();

    GDALDataset* aGdalDataset;
    auto aGdalDriver = cGdalApi::GetDriver(aName);

    // Determinine if we can create the file and then write to it
    //  or if we must use an intermiediate in memory image in GDAL format.
    bool createFileNow = true;
    if (aDataFileIm2D.IsCreateAtFirstWrite() || aDataFileIm2D.IsCreatedNoUpdate()) {
        createFileNow = GDalDriverCanCreate(aGdalDriver);
    }

    if (createFileNow) {
        // Create the file now
        auto aGdalOptions = GetCreateOptions(aGdalDriver, aDataFileIm2D.CreateOptions());
        remove(aName.c_str());
        aGdalDataset = aGdalDriver->Create( aName.c_str(), aSz.x(), aSz.y(), aNbChan, aGDALType, aGdalOptions.List());
    } else {
        // Create image in memory, will be copied to file at end of GdalIO::operator() (WriteOnce() => called by GdalIO::operator())
        GDALDriver *pDriverMEM = GetGDALDriverManager()->GetDriverByName("MEM");
        aGdalDataset = pDriverMEM->Create( "", aSz.x(), aSz.y(), aNbChan, aGDALType, nullptr );
        if (createdInMemory)
            *createdInMemory = true;
    }
    if (!aGdalDataset) {
        MMVII_INTERNAL_ERROR(std::string("Can't create image file: ") + CPLGetLastErrorMsg() + " '" + aName +"'");
        return nullptr; // Never happens
    }
    return aGdalDataset;
}


// Create an image file if file not exits or has different characteristics: size, type or nb channels
void cGdalApi::CreateFileIfNeeded(const cDataFileIm2D& aDataFileIm2D)
{
    MMVII_INTERNAL_ASSERT_always(aDataFileIm2D.IsCreateAtFirstWrite(),"GDAL: Invalid use of GDAL API for image file creation");
    auto aName = aDataFileIm2D.Name();
    auto aSz = aDataFileIm2D.Sz();
    auto aDataset = OpenDataset(aName, GA_ReadOnly, eOnError::ReturnNullptr);
    if (aDataset != nullptr) {
        auto aType = aDataFileIm2D.Type();
        auto aNbChannel = aDataFileIm2D.NbChannel();
        cPt2di fileSz = cPt2di(aDataset->GetRasterXSize(), aDataset->GetRasterYSize());
        auto fileNbChan = aDataset->GetRasterCount();
        auto fileType = ToMMVII( aDataset->GetRasterBand( 1 )->GetRasterDataType());
        auto aGdalDriver = aDataset->GetDriver();
        CloseDataset(aDataset);
        if (fileSz == aSz && fileNbChan == aNbChannel && fileType == aType) {
            if (GDalDriverCanCreate(aGdalDriver)) {
                aDataFileIm2D.SetCreated();
            } else {
                aDataFileIm2D.SetCreatedNoUpdate();
            }
            return;     // No need to create
        }
    }
    // aDataset is either null here either already closed, no need to close it

    // Create and Init the image file with a blank image
    cIm2D<tU_INT1> anEmptyImg(aSz,nullptr,eModeInitImage::eMIA_Null);
    cGdalApi::ReadWrite(IoMode::Write,anEmptyImg.DIm(),aDataFileIm2D,cPt2di(0,0),1,cRect2::TheEmptyBox);
}



// cGdalApi : private methods

eTyNums cGdalApi::ToMMVII( GDALDataType aType )
{
    switch (aType)
    {
    case GDT_Byte    : return eTyNums::eTN_U_INT1 ;
    case GDT_UInt16  : return eTyNums::eTN_U_INT2 ;
    case GDT_Int16   : return eTyNums::eTN_INT2 ;
    case GDT_UInt32  : return eTyNums::eTN_U_INT4 ;
    case GDT_Int32   : return eTyNums::eTN_INT4 ;
    case GDT_Float32 : return eTyNums::eTN_REAL4 ;
    case GDT_Float64 : return eTyNums::eTN_REAL8 ;

    case GDT_CInt16   : MMVII_INTERNAL_ERROR("TyGdalToMMVII: GDAL Image type GDT_CInt16 not supported");
    case GDT_CInt32   : MMVII_INTERNAL_ERROR("TyGdalToMMVII: GDAL Image type GDT_CInt32 not supported");
    case GDT_CFloat32 : MMVII_INTERNAL_ERROR("TyGdalToMMVII: GDAL Image type GDT_CFloat32 not supported");
    case GDT_CFloat64 : MMVII_INTERNAL_ERROR("TyGdalToMMVII: GDAL Image type GDT_CFloat64 not supported");
    case GDT_Unknown  : MMVII_INTERNAL_ERROR("TyGdalToMMVII: GDAL Image type GDT_Unknown not supported");
    default: MMVII_INTERNAL_ERROR("TyGdalToMMVII: GDAL Image type #" + std::to_string(aType) + " not supported");
    }
    return eTyNums::eTN_UnKnown ;
}


GDALDataType cGdalApi::FromMMVII( eTyNums aType )
{
    switch (aType)
    {
    case eTyNums::eTN_U_INT1  : return GDT_Byte ;
    case eTyNums::eTN_INT2    : return GDT_Int16 ;
    case eTyNums::eTN_U_INT2  : return GDT_UInt16 ;
    case eTyNums::eTN_INT4    : return GDT_Int32 ;
    case eTyNums::eTN_U_INT4  : return GDT_UInt32 ;
    case eTyNums::eTN_REAL4   : return GDT_Float32 ;
    case eTyNums::eTN_REAL8   : return GDT_Float64 ;
    case eTyNums::eTN_INT1    :
    case eTyNums::eTN_INT8    :
    case eTyNums::eTN_REAL16  :
    default: MMVII_INTERNAL_ERROR("TyMMVIIToGdal: eTyNums::eTN_" + ToStr(aType) + " not supported by GDAL");
    }
    return GDT_Unknown ;
}

void cGdalApi::GdalErrorHandler(CPLErr aErrorCat, CPLErrorNum aErrorNum, const char *aMesg)
{
    if (aErrorCat == CE_Fatal) {    // Internal error in GDAL, cannot be handled at higher level
        MMVII_INTERNAL_ERROR("GDal fatal Error #" + std::to_string(aErrorNum) + ": " +aMesg);
    }
}


CPLStringList cGdalApi::GetCreateOptions(GDALDriver* aGdalDriver, const cDataFileIm2D::tOptions& aOptions)
{
    CPLStringList aGdalOptions;
    for (const auto& anOpt : aOptions)
    {
        aGdalOptions.AddString(anOpt.c_str());
    }
    if (! GDALValidateCreationOptions(aGdalDriver,aGdalOptions))
    {
        MMVII_INTERNAL_ERROR(std::string("GDal : Invalid options for driver ") + aGdalDriver->GetDescription() + "': " + ToStr(aOptions)) ;
        return aGdalOptions; // never happens
    }
    return aGdalOptions;
}


GDALDataset * cGdalApi::OpenDataset(const std::string& aName, GDALAccess aAccess, eOnError onError)
{
    InitGDAL();
    auto aHandle = GDALOpen( aName.c_str(), aAccess);
    auto aDataSet = GDALDataset::FromHandle(aHandle);
    if (aDataSet == nullptr && onError == eOnError::RaiseError)
    {
        MMVII_UserError(eTyUEr::eOpenFile,std::string("Can't open image file: ") + CPLGetLastErrorMsg());
        return nullptr; // never happens
    }
    return aDataSet;
}

void cGdalApi::CloseDataset(GDALDataset *aGdalDataset)
{
    if (aGdalDataset)
        GDALClose(GDALDataset::ToHandle(aGdalDataset));
}

std::map<std::string, std::vector<std::string>> cGdalApi::GetMetadata(const std::string &aName, eOnError onError)
{
    std::map<std::string, std::vector<std::string>> aMetadataMap;

    auto aDataSet = cGdalApi::OpenDataset(aName,GA_ReadOnly,onError);
    if (aDataSet == nullptr)
    {
        return aMetadataMap;
    }
    auto aMetadataDomainList = CPLStringList(aDataSet->GetMetadataDomainList(),TRUE);

    if (aMetadataDomainList.FindString("") < 0) // Add empty domain if not already present
        aMetadataDomainList.AddString("");
    for (int i=0; i<aMetadataDomainList.Count(); i++)
    {
        auto [it,ok] = aMetadataMap.emplace(aMetadataDomainList[i],std::vector<std::string>());
        auto aMetadataList = CPLStringList((CSLConstList)aDataSet->GetMetadata(aMetadataDomainList[i]));
        for (int i=0; i<aMetadataList.Count(); i++)
        {
            it->second.push_back(aMetadataList[i]);
        }
    }
    cGdalApi::CloseDataset(aDataSet);
    return aMetadataMap;
}


CPLStringList cGdalApi::GetExifMetadata(const std::string &aName, eOnError onError)
{
    auto aDataSet = cGdalApi::OpenDataset(aName,GA_ReadOnly,onError);
    if (aDataSet == nullptr)
    {
        return CPLStringList();
    }
    auto aMetadataDomainList = CPLStringList(aDataSet->GetMetadataDomainList(),TRUE);

    if (aMetadataDomainList.FindString("") < 0) // Add empty domain if not already present
        aMetadataDomainList.AddString("");
    for (int i=0; i<aMetadataDomainList.Count(); i++)
    {
        auto aMetadataList = CPLStringList((CSLConstList)aDataSet->GetMetadata(aMetadataDomainList[i]));
        if (aMetadataList.FindName("EXIF_ExifVersion") >= 0) {
            aMetadataList.Sort();
            cGdalApi::CloseDataset(aDataSet);
            return aMetadataList;
        }
    }
    cGdalApi::CloseDataset(aDataSet);
    return CPLStringList();
}


const std::map<std::string, std::string> &cGdalApi::SupportedDrivers()
{
    static std::map<std::string, std::string> cSupportedDrivers= {
        {"tif", "GTiff"},
        {"tiff", "GTiff"},
        {"dng", "GTiff"},
        {"jpg","JPEG"},
        {"jpeg","JPEG"},
        {"jp2","JP2OpenJPEG"},
        {"png","PNG"},
        {"bmp","BMP"},
        {"pnm","PNM"},
        {"gif","GIF"},
    };
    return cSupportedDrivers;
}

bool cGdalApi::IsPostFixNameImage(const std::string &aPost)
{
    const auto aDriverList = SupportedDrivers();
    auto aDriverIt = aDriverList.find(ToLower(aPost));
    return aDriverIt != aDriverList.end();
}


GDALDriver* cGdalApi::GetDriver(const std::string& aName)
{
    const auto aDriverList = SupportedDrivers();
    auto aDriverIt = aDriverList.find(ToLower(LastPostfix(aName,'.')));
    if (aDriverIt == aDriverList.end())
    {
        MMVII_INTERNAL_ERROR("MMVIITOGDal: Unsupported image format for " + aName);
        return nullptr; // never happens
    }
    auto aDriverName = aDriverIt->second;
    auto aGdalDriver =  GetGDALDriverManager()->GetDriverByName(aDriverName.c_str());
    if (!aGdalDriver) {
        MMVII_INTERNAL_ERROR(std::string("GDAL can't handle file format : ") + aDriverName);
        return nullptr; // never happens
    }
    return aGdalDriver;
}


bool  cGdalApi::GDalDriverCanCreate(GDALDriver* aGdalDriver)
{
    auto capabilityCreate = aGdalDriver->GetMetadataItem(GDAL_DCAP_CREATE);
    return (capabilityCreate != nullptr) && (strcmp(capabilityCreate,"YES") == 0);
}



} // namespace MMVII


