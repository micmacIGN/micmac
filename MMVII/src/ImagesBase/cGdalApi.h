#ifndef CGDALAPI_H
#define CGDALAPI_H

#include <MMVII_enums.h>
#include <MMVII_Error.h>
#include <MMVII_Ptxd.h>
#include <MMVII_Image2D.h>

#include <string>
#include <gdal_priv.h>

/*
 * GDAL API interface to MMVII to read/write image file
 *
 * To extend the list of file format supported, see cGdalApi::GetDriver() at end of file
 *
*/



namespace MMVII {




// Public API

class cGdalApi {
public:
    enum class IoMode {Read, Write};

    static void InitGDAL();

    static bool IsPostFixNameImage(const std::string& aPost);

    static void GetFileInfo(const std::string& aName, eTyNums& aType, cPt2di& aSz, int& aNbChannel);
    static void CreateFileIfNeeded(const cDataFileIm2D& aDataFileIm2D);

    template <class TypeIm>
    static void ReadWrite(IoMode aMode,
                          const cDataIm2D<TypeIm>& aImV2,
                          const cDataFileIm2D &aDF,
                          const cPt2di & aP0File,
                          double aDyn,
                          const cRect2& aR2Init
                          );

    template <class TypeIm>
    static void ReadWrite(IoMode aMode,
                          const cDataIm2D<TypeIm>& aImV2R,
                          const cDataIm2D<TypeIm>& aImV2G,
                          const cDataIm2D<TypeIm>& aImV2B,
                          const cDataFileIm2D &aDF,
                          const cPt2di & aP0File,
                          double aDyn,
                          const cRect2& aR2Init
                          );

    template <class TypeIm>
    static void ReadWrite(IoMode aMode,
                          const std::vector<const cDataIm2D<TypeIm>*>& aVecImV2,
                          const cDataFileIm2D &aDF,
                          const cPt2di & aP0File,
                          double aDyn,
                          const cRect2& aR2Init
                          );

private:
    template <typename, typename> friend class GdalIO;

    enum class eOnError {ReturnNullptr, RaiseError};

    static eTyNums ToMMVII( GDALDataType aType );
    static GDALDataType FromMMVII( eTyNums aType );

    // Global Gdal Error Handler. Not multithread safe !
    // Avoid printing of error message => each API call must test and handle error case.
    static void GdalErrorHandler(CPLErr aErrorCat, CPLErrorNum aErrorNum, const char *aMesg);
    static GDALDriver* GetDriver(const std::string& aName);
    static const std::map<std::string, std::string> &SupportedDrivers();
    static bool GDalDriverCanCreate(GDALDriver *aGdalDriver);

    static CPLStringList GetCreateOptions(GDALDriver* aGdalDriver, const cDataFileIm2D::tOptions& aOptions);

    // Return nullptr if error
    static GDALDataset * OpenDataset(const std::string &aName, GDALAccess aAccess, eOnError onError);
    static void CloseDataset(GDALDataset *aGdalDataset);

    static GDALDataset* CreateDataset(const cDataFileIm2D& aDataFileIm2D, bool *createdInMemory);

    static void SetCreated(const cDataFileIm2D &aDF) { aDF.SetCreated(); }
    static void SetCreatedNoUpdate(const cDataFileIm2D &aDF) { aDF.SetCreatedNoUpdate(); }
};


/****************************************************
 * PRIVATE
*****************************************************/


/*******************************************
 * GDalIO
 * Helper class that does the real IO
 * (cannot succeed to make it really private)
 ******************************************/
template <typename TypeIm, typename TypeFile>
class GdalIO {
public:
    typedef cDataIm2D<TypeIm> tIm;
    typedef std::vector<const tIm*> tvIm;
    typedef typename cGdalApi::IoMode IoMode;

    GdalIO()
        : mDataFile(nullptr)
        , mGdalDataset(nullptr)
        , mGdalNbChan(0)
        , mNbImg(0)
    {}
    void operator()(IoMode aMode, const cDataFileIm2D &aDF, const tvIm& aVecImV2, const cPt2di & aP0File,double aDyn,const cRect2& aR2)
    {
        mDataFile = &aDF;
        auto aName = mDataFile->Name();
        mGdalNbChan = mDataFile->NbChannel();
        mNbImg = static_cast<int>(aVecImV2.size());

        auto aIm2D = aVecImV2[0];
        cRect2 aRectFullIm(aIm2D->P0(), aIm2D->P1());
        cRect2 aRectIm =  (aR2 == cRect2::TheEmptyBox) ? aRectFullIm : aR2;
        cRect2 aRectFile (aRectIm.Translate(aP0File)) ;

        MMVII_INTERNAL_ASSERT_strong(aRectFile.IncludedIn(aDF), "Read/write out of image file (" + aName + ")");
        MMVII_INTERNAL_ASSERT_strong(aRectIm.IncludedIn(aRectFullIm), "Read/write out of Image buffer (" + aName + ")");

        auto notUpdatable = false;
        if (mDataFile->IsCreateAtFirstWrite())
        {
            if (aMode != IoMode::Write)
            {
                MMVII_INTERNAL_ERROR("GDAL read: image file created with CreateOnWrite() must be written before trying to read it (" + aName + ")");
            }
            if (aRectFile.Sz() != aDF.Sz() || aRectFile.P0() != cPt2di(0,0))
            {
                MMVII_INTERNAL_ERROR("GDAL write: image file created with CreateOnWrite() must be fully written on first write (" + aName + ")");
            }
            // Delayed file creation, Dataset can be created in memory depending of file format driver
            mGdalDataset = cGdalApi::CreateDataset(aDF, &notUpdatable);
            if (notUpdatable) {
                cGdalApi::SetCreatedNoUpdate(*mDataFile);
            } else {
                cGdalApi::SetCreated(*mDataFile);
            }
        }
        else if (mDataFile->IsCreatedNoUpdate() && aMode == IoMode::Write)
        {
            if (aRectFile.Sz() != aDF.Sz() || aRectFile.P0() != cPt2di(0,0))
            {
                MMVII_INTERNAL_ERROR("GDAL write: this image file format must be fully written on each write (" + aName + ")");
            }
            mGdalDataset = cGdalApi::CreateDataset(aDF, &notUpdatable);
        }
        else
        {
            mGdalDataset = cGdalApi::OpenDataset(aDF.Name(), aMode==IoMode::Read ? GA_ReadOnly : GA_Update, cGdalApi::eOnError::RaiseError);
        }

        if (aMode == IoMode::Read) {
            if (mGdalNbChan == mNbImg && mNbImg != 0) {
                GdalReadNtoN(aVecImV2,aRectIm,aRectFile,aDyn);     // file N -> N img channels
            } else if (mGdalNbChan == 1 && mNbImg != 0) {
                GdalRead1toN(aVecImV2,aRectIm,aRectFile,aDyn);     // file 1 -> N img channels
            } else if (mGdalNbChan != 0 && mNbImg == 1) {
                GdalReadNto1(aVecImV2,aRectIm,aRectFile,aDyn);     // file N -> 1 img channels
            } else {
                MMVII_INTERNAL_ERROR("Gdal read: Images vector size: " + std::to_string(mNbImg) + ", file channels: " + std::to_string(mGdalNbChan) + " (" + aName + ")");
            }
        } else {
            if (mGdalNbChan == mNbImg && mNbImg != 0) {
                GdalWriteNtoN(aVecImV2,aRectIm,aRectFile,aDyn);     // img N -> N file channels
            } else if (mGdalNbChan != 0 && mNbImg == 1) {
                GdalWrite1toN(aVecImV2,aRectIm,aRectFile,aDyn);     // img 1 -> N file channels
            } else {
                MMVII_INTERNAL_ERROR("Gdal write: Images vector size: " + std::to_string(mNbImg) + ", file channels: " + std::to_string(mGdalNbChan) + " (" + aName + ")");
            }
        }

        if (notUpdatable) {
            // Copy image from memory to file if image file driver needs it
            remove(aName.c_str());
            auto aGdalDriver = cGdalApi::GetDriver(aName);
            auto aGdalOptions = cGdalApi::GetCreateOptions(aGdalDriver, mDataFile->CreateOptions());
            remove(aName.c_str());
            auto mFinalDataset = aGdalDriver->CreateCopy(aName.c_str(), mGdalDataset, FALSE, aGdalOptions.List(), NULL, NULL);
            cGdalApi::CloseDataset(mFinalDataset);
        }
        cGdalApi::CloseDataset(mGdalDataset);
    }

private:
    // Helper class: manage N (1 by default) image buffers read from/write to file with GDAL
    // "Inherits" from outer class (GdalIO) the template parameter "TypeFile"
    class GDalBuffer
    {
    public:
        GDalBuffer(const cRect2& aRectIm, int nbChan=1)
            : mBuffer(nbChan)
            , mRectIm(aRectIm)
        {
            auto aSize = sizeof(TypeFile)*aRectIm.Sz().x()*aRectIm.Sz().y();
            for (auto& aBuf : mBuffer)
            {
                aBuf = (TypeFile*) cMemManager::Calloc(1,aSize);
            }
        }
        ~GDalBuffer()
        {
            for (auto& aBuf : mBuffer)
            {
                cMemManager::Free(aBuf);
            }
        }
        TypeFile operator()(const cPt2di& aPt, int chan=0) const
        {
            auto pt0 = aPt-mRectIm.P0();
            return mBuffer[chan][pt0.y() * mRectIm.Sz().x() + pt0.x()];
        }
        TypeFile& operator()(const cPt2di& aPt, int chan=0)
        {
            auto pt0 = aPt-mRectIm.P0();
            return mBuffer[chan][pt0.y() * mRectIm.Sz().x() + pt0.x()];
        }
        void *addr(int chan=0)
        {
            return mBuffer[chan];
        }

    private:
        std::vector<TypeFile*> mBuffer;
        cRect2 mRectIm;
    };


    // Read: Normal case: map each channel from image file to a corresponding cDataIm2D
    void GdalReadNtoN(const tvIm& aVecImV2, const cRect2& aRectIm, const cRect2& aRectFile, double aDyn)
    {
        GDalBuffer aBuffer(aRectIm);
        for (int aChan = 0; aChan < mGdalNbChan; aChan++)
        {
            auto aDataIm2D = const_cast<tIm*>(aVecImV2[aChan]); // If IoMode::Read the original parameter was not const, so we can de-const it.
            GDALRasterBand* aBand = mGdalDataset->GetRasterBand(aChan+1); // GDal channel begins at 1 (not 0)
            ReadWrite(GF_Read, aBand, aBuffer.addr(), aRectFile);
            for (const auto &aPt : aRectIm)
            {
                aDataIm2D->SetVTrunc(aPt, aBuffer(aPt) * aDyn);
            }
        }
    }

    // Read: Duplicate unique channel in image file to each cDataIm2D
    void GdalRead1toN(const tvIm& aVecImV2, const cRect2& aRectIm, const cRect2& aRectFile, double aDyn)
    {
        GDalBuffer aBuffer(aRectIm);
        GDALRasterBand* aBand = mGdalDataset->GetRasterBand(1); // GDal channel begins at 1 (not 0)
        ReadWrite(GF_Read, aBand, aBuffer.addr(), aRectFile);
        for (const auto &aPt : aRectIm)
        {
            auto aVal = aBuffer(aPt) * aDyn;
            for (int aImgNum = 0; aImgNum < mNbImg; aImgNum++)
            {
                auto aDataIm2D = const_cast<tIm*>(aVecImV2[aImgNum]); // If IoMode::Read the original parameter was not const, so we can de-const it.
                aDataIm2D->SetVTrunc(aPt, aVal);
            }
        }
    }

    // Read:  Average all channels from image file to the only cDataIm2D
    void GdalReadNto1(const tvIm& aVecImV2, const cRect2& aRectIm, const cRect2& aRectFile, double aDyn)
    {
        int nbChan = mGdalNbChan;
        if (nbChan == 4) // Probably RGBA, use only RGB
            nbChan = 3;
        GDalBuffer aBuffer(aRectIm,nbChan);
        auto aDataIm2D = const_cast<tIm*>(aVecImV2[0]); // If IoMode::Read the original parameter was not const, so we can de-const it.
        for (int i=0; i<nbChan; i++) {
            GDALRasterBand* aBand = mGdalDataset->GetRasterBand(i+1); // GDal channel begins at 1 (not 0)
            ReadWrite(GF_Read, aBand, aBuffer.addr(i), aRectFile);
        }

        for (const auto &aPt : aRectIm)
        {
            double sum = 0;
            for (int i=0; i<nbChan; i++)
            {
                sum += aBuffer(aPt,i);
            }
            aDataIm2D->SetVTrunc(aPt, (sum * aDyn) / nbChan);
        }
    }

    // Write: Normal case: write each cDataIm2D in a separate channel in image file
    void GdalWriteNtoN(const tvIm& aVecImV2, const cRect2& aRectIm, const cRect2& aRectFile, double aDyn)
    {
        GDalBuffer aBuffer(aRectIm);
        for (int aChan = 0; aChan < mGdalNbChan; aChan++)
        {
            auto aDataIm2D = aVecImV2[aChan];
            GDALRasterBand* aBand = mGdalDataset->GetRasterBand(aChan+1); // GDal channel begins at 1 (not 0)
            for (const auto &aPt : aRectIm)
            {
                aBuffer(aPt) = tNumTrait<TypeFile>::Trunc(aDataIm2D->GetV(aPt) * aDyn);
            }
            ReadWrite(GF_Write, aBand, aBuffer.addr(), aRectFile);
        }
    }

    // Write: Duplicate unique cDataIm2D in  each channel of image file
    void GdalWrite1toN(const tvIm& aVecImV2, const cRect2& aRectIm, const cRect2& aRectFile, double aDyn)
    {
        GDalBuffer aBuffer(aRectIm);
        for (int aChan = 0; aChan < mGdalNbChan; aChan++)
        {
            auto aDataIm2D = aVecImV2[0];
            GDALRasterBand* aBand = mGdalDataset->GetRasterBand(aChan+1); // GDal channel begins at 1 (not 0)
            for (const auto &aPt : aRectIm)
            {
                aBuffer(aPt) = tNumTrait<TypeFile>::Trunc(aDataIm2D->GetV(aPt) * aDyn);
            }
            ReadWrite(GF_Write, aBand, aBuffer.addr(), aRectFile);
        }
    }

    // Wrapper to GDAL API: RasterIO()
    void ReadWrite(GDALRWFlag aRWFlag, GDALRasterBand *aBand, void *aBuffer, const cRect2& aRectFile)
    {
        CPLErr aErr = aBand->RasterIO(
            aRWFlag,
            aRectFile.P0().x(), aRectFile.P0().y(), aRectFile.Sz().x(), aRectFile.Sz().y(),
            aBuffer,
            aRectFile.Sz().x(), aRectFile.Sz().y(),
            cGdalApi::FromMMVII(mDataFile->Type()),
            0, 0
            );
        if (aErr != 0 && aErr != 1)
        {
            MMVII_INTERNAL_ERROR(std::string("GDAL Error (") + (aRWFlag == GF_Read ? "read" : "write") + ") : " + CPLGetLastErrorMsg() + " [" + mDataFile->Name() + "]");
        }
    }

private:
    const cDataFileIm2D *mDataFile;
    GDALDataset *mGdalDataset;
    int mGdalNbChan;
    int mNbImg;
};



/*******************************************
 * cGDalAPi
 * Implementation
 ******************************************/

// cGdalApi : public methods


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
    InitGDAL();
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
    InitGDAL();
    MMVII_INTERNAL_ASSERT_always(aDataFileIm2D.IsCreateAtFirstWrite(),"GDAL: Invalid use of GDAL API for image file creation");
    auto aName = aDataFileIm2D.Name();
    auto aType = aDataFileIm2D.Type();
    auto aSz = aDataFileIm2D.Sz();
    auto aNbChannel = aDataFileIm2D.NbChannel();
    auto aDataset = OpenDataset(aName, GA_ReadOnly, eOnError::ReturnNullptr);
    if (aDataset != nullptr) {
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


// GDalAPi : the public Read/Write API
template <class TypeIm>
void cGdalApi::ReadWrite(
    IoMode aMode,
    const cDataIm2D<TypeIm>& aImV2,
    const cDataFileIm2D &aDF,
    const cPt2di & aP0File,
    double aDyn,
    const cRect2& aR2Init
    )
{
    std::vector<const cDataIm2D<TypeIm>*> aVIms({&aImV2});
    ReadWrite(aMode,aVIms,aDF,aP0File,aDyn,aR2Init);
}

template <class TypeIm>
void cGdalApi::ReadWrite(
    IoMode aMode,
    const cDataIm2D<TypeIm>& aImV2R,
    const cDataIm2D<TypeIm>& aImV2G,
    const cDataIm2D<TypeIm>& aImV2B,
    const cDataFileIm2D &aDF,
    const cPt2di & aP0File,
    double aDyn,
    const cRect2& aR2Init
    )
{
    std::vector<const cDataIm2D<TypeIm>*> aVIms({&aImV2R,&aImV2G,&aImV2B});
    ReadWrite(aMode,aVIms,aDF,aP0File,aDyn,aR2Init);
}


template <class TypeIm>
void cGdalApi::ReadWrite(
    IoMode aMode,
    const std::vector<const cDataIm2D<TypeIm>*>& aVecImV2,
    const cDataFileIm2D &aDF,
    const cPt2di & aP0File,
    double aDyn,
    const cRect2& aR2Init
    )
{
    MMVII_INTERNAL_ASSERT_strong(aVecImV2.size() > 0,"aVecImV2 is empty in GdalReadWrite");
    for (int aKIm=1 ; aKIm<int(aVecImV2.size()) ; aKIm++)
    {
        MMVII_INTERNAL_ASSERT_strong(aVecImV2.at(0)->Sz()==aVecImV2.at(aKIm)->Sz(),"Diff Sz in GdalReadWrite");
        MMVII_INTERNAL_ASSERT_strong(aVecImV2.at(0)->P0()==aVecImV2.at(aKIm)->P0(),"Diff P0 in GdalReadWrite");
    }

    switch (aDF.Type())
    {
    case eTyNums::eTN_INT1    : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_INT1") ; break ;
    case eTyNums::eTN_U_INT1  : GdalIO<TypeIm, tU_INT1>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
    case eTyNums::eTN_INT2    : GdalIO<TypeIm, tINT2>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init) ; break ;
    case eTyNums::eTN_U_INT2  : GdalIO<TypeIm, tU_INT2>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
    case eTyNums::eTN_INT4    : GdalIO<TypeIm, tINT4>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ;  break ;
    case eTyNums::eTN_U_INT4  : GdalIO<TypeIm, tU_INT4>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
    case eTyNums::eTN_INT8    : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_INT8") ; break ;
    case eTyNums::eTN_REAL4   : GdalIO<TypeIm, tREAL4>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
    case eTyNums::eTN_REAL8   : GdalIO<TypeIm, tREAL8>()(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
    case eTyNums::eTN_REAL16  : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_REAL16") ; break ;
    case eTyNums::eTN_UnKnown : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_UnKnown") ; break ;
    case eTyNums::eNbVals     : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eNbVals") ; break ;
    }
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

#endif // CGDALAPI_H
