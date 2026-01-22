#ifndef CGDALAPI_H
#define CGDALAPI_H

#include "MMVII_Sys.h"
#include "MMVII_enums.h"
#include "MMVII_Error.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Image2D.h"

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

    enum class eOnError {ReturnNullptr, RaiseError};
    static GDALDataset * OpenDataset(const std::string &aName, GDALAccess aAccess, eOnError onError);  // return nnullptr on error
    static void CloseDataset(GDALDataset *aGdalDataset);

    static std::map<std::string, std::vector<std::string>> GetMetadata(const std::string &aName, eOnError onError);
    static CPLStringList GetExifMetadata(const std::string &aName, eOnError onError);

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

    static eTyNums ToMMVII( GDALDataType aType );
    static GDALDataType FromMMVII( eTyNums aType );

    // Global Gdal Error Handler. Not multithread safe !
    // Avoid printing of error message => each API call must test and handle error case.
    static void GdalErrorHandler(CPLErr aErrorCat, CPLErrorNum aErrorNum, const char *aMesg);
    static GDALDriver* GetDriver(const std::string& aName);
    static const std::map<std::string, std::string> &SupportedDrivers();
    static bool GDalDriverCanCreate(GDALDriver *aGdalDriver);

    static CPLStringList GetCreateOptions(GDALDriver* aGdalDriver, const cDataFileIm2D::tOptions& aOptions);

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

        FileLock gdalLock;
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
            if (! notUpdatable)
                gdalLock.lock(aName);
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
        gdalLock.unlock();
    }

private:
    // Helper class: manage N (1 by default) image buffers read from/write to file with GDAL
    // "Inherits" from outer class (GdalIO) the template parameter "TypeFile"
    class GDalBuffer
    {
    public:
        explicit GDalBuffer(const cRect2& aRectIm, int nbChan=1)
            : mBuffer(nbChan)
            , mRectIm(aRectIm)
        {
            auto aSize = sizeof(TypeFile)*aRectIm.Sz().x()*aRectIm.Sz().y();
            std::generate(mBuffer.begin(), mBuffer.end(),[aSize](){return static_cast<TypeFile*>(cMemManager::Calloc(1,aSize));});
/*            for (auto& aBuf : mBuffer)
            {
                aBuf = static_cast<TypeFile*>(cMemManager::Calloc(1,aSize));
            }
*/
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

} // namespace MMVII

#endif // CGDALAPI_H
