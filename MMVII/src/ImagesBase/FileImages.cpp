// FIXME CM: clean list of headers
#include "V1VII.h"
#include "MMVII_Stringifier.h"
#include "MMVII_util.h"
#include "MMVII_Image2D.h"
#include "MMVII_DeclareCste.h"
#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "StdAfx.h"
#include "MMVII_Ptxd.h"
#include <gdal_priv.h>


using namespace MMVII;

#ifdef MMVII_KEEP_MMV1_IMAGE
bool mmvii_use_mmv1_image=false;
#endif

namespace{ // Private

   eTyNums TyGdalToMMVII( GDALDataType aType )
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


   GDALDataType TyMMVIIToGdal( eTyNums aType )
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
   
   enum class IoMode {Read, Write};

   // Global GDal Error Handler. Not multithread safe !
   // Avoid printing of error message => each API call must test and handle error case.
   void GDalErrorHandler(CPLErr aErrorCat, CPLErrorNum aErrorNum, const char *aMesg)
   {
       if (aErrorCat == CE_Fatal) {
           MMVII_INTERNAL_ERROR("GDal fatal Error #" + std::to_string(aErrorNum) + ": " +aMesg);
       }
   }
   
   void InitGDAL()
   {
       static bool isGdalInitialized = false;
       if (isGdalInitialized)
           return;
       GDALAllRegister();
       CPLSetErrorHandler(GDalErrorHandler);
       isGdalInitialized = true;
   }

   std::string ExtToGdalDriver( std::string aName )
   {
        auto aLowerName = ToLower(aName);
        if (ends_with(aLowerName,".tif") || ends_with(aLowerName,".tiff"))
            return "GTiff";
        if (ends_with(aLowerName,".jpg") || ends_with(aLowerName,".jpeg"))
            return "JPEG";
        if (ends_with(aLowerName,".png"))
            return "PNG";
        MMVII_INTERNAL_ERROR("MMVIITOGDal: Unsupported image format for " + aName);
         return "";
   }

   // Return nullptr if error
   GDALDataset * OpenDataset(std::string aName, GDALAccess aAccess)
   {
       auto aHandle = GDALOpen( aName.c_str(), aAccess);
       auto aDataSet = GDALDataset::FromHandle(aHandle);
       return aDataSet;
   }
   
   void CloseDataset(GDALDataset *aGdalDataset)
   {
      GDALClose( (GDALDatasetH) aGdalDataset );
   }


   //FIXME CM: Handle here: N<->N, 1->N, N->1
   template <typename TypeIn, typename TypeOut> void GdalReadWriteIO(IoMode aMode, const cDataFileIm2D &aDF, std::vector<const cDataIm2D<TypeIn>*>& aVecImV2, const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
   {
        GDALDataset * aDataset = OpenDataset(aDF.Name(), aMode==IoMode::Read ? GA_ReadOnly : GA_Update);
        GDALDataType aGDALDataType = TyMMVIIToGdal(aDF.Type());
        auto aDataIm2D = aVecImV2[0];
        cRect2 aRectFullIm (cPt2di(0,0),aDataIm2D->Sz());
        cRect2 aRectIm =  (aR2 == cRect2::TheEmptyBox) ? aRectFullIm : aR2;
        // aRectIm est dans le repère d'origine P0
        cPt2di aTrans  = aDataIm2D->P0() + aP0;
        TypeOut *aScanline = (TypeOut *) cMemManager::Calloc(1, sizeof(TypeOut)*aRectIm.Sz().x()*aRectIm.Sz().y());
        if (aMode == IoMode::Read) {
            for (auto aChan = 0; aChan < aDF.NbChannel(); aChan++)
            {
                auto aDataIm2D = const_cast<cDataIm2D<TypeIn>*>(aVecImV2[aChan]); // If IoMode::Read, the original parameter was not const. So we can de-const it.
                GDALRasterBand* aBand = aDataset->GetRasterBand(aChan+1);
                CPLErr cplErr2 = aBand->RasterIO( GF_Read, aRectIm.P0().x() + aTrans.x(), aRectIm.P0().y() + aTrans.y(), aRectIm.Sz().x(), aRectIm.Sz().y(), aScanline, aRectIm.Sz().x(), aRectIm.Sz().y(), aGDALDataType, 0, 0 );
                // FIXME CM: better error message
                MMVII_INTERNAL_ASSERT_strong(cplErr2 == 0 || cplErr2 == 1,"Error in reading image");

                for (int aY = 0; aY < aRectIm.Sz().y(); aY++)
                {
                    for (int aX = 0; aX < aRectIm.Sz().x(); aX++)
                    {
                        aDataIm2D->SetVTrunc(cPt2di(aX, aY) + aDataIm2D->P0(), aScanline[aY * aRectIm.Sz().x() + aX] * aDyn);
                    }
                }
            }
        } else {
            for (auto aChan = 0; aChan < aDF.NbChannel(); aChan++)
            {
                auto aDataIm2D = aVecImV2[aChan];
                GDALRasterBand* aBand = aDataset->GetRasterBand(aChan+1);
                for (int aY = 0; aY < aRectIm.Sz().y(); aY++)
                {
                    for (int aX = 0; aX < aRectIm.Sz().x(); aX++)
                    {
                        aScanline[aY * aRectIm.Sz().x() + aX] = tNumTrait<TypeOut>::Trunc(aDataIm2D->GetV(cPt2di(aX, aY) + aDataIm2D->P0() + aR2.P0()) * aDyn);
                    }
                }
                CPLErr cplErr2 = aBand->RasterIO( GF_Write, aRectIm.P0().x() + aTrans.x(), aRectIm.P0().y() + aTrans.y(), aRectIm.Sz().x(), aRectIm.Sz().y(), aScanline, aRectIm.Sz().x(), aRectIm.Sz().y(), aGDALDataType, 0, 0 );
                // FIXME CM: better error message
                MMVII_INTERNAL_ASSERT_strong(cplErr2 == 0 || cplErr2 == 1,"Error in writing image");
            }
        }
        cMemManager::Free(aScanline);
        CloseDataset(aDataset);
   }

// FIXME CM: special API to write a full image: Needed for JPEG, avoid empty useless image creation for TIFF (see: cDataFileIm2D::create and cDataIm2D::ToFile)
// FIXME CM: Read: if aVecImV2.size() != aDF.mNbChannel => Convert to gray if aVecImV2.size()==1, else error
// FIXME CM: test P0 and Sz is same on all aVecImV2
// FIXME CM: Write: if aVecImV2.size() != aDF.mNbChannel => write NnChannel times same channel if aVecImV2.size()==1, else error

    template <class Type> void GdalReadWrite
                            (
                                IoMode aMode,
                                std::vector<const cDataIm2D<Type>*>& aVecImV2,
                                const cDataFileIm2D &aDF,
                                const cPt2di & aP0File,
                                double aDyn,
                                const cRect2& aR2Init
                            )
    {
        switch (aDF.Type())
        {
        case eTyNums::eTN_INT1 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_INT1") ; break ;
        case eTyNums::eTN_U_INT1 : GdalReadWriteIO<Type, tU_INT1>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
        case eTyNums::eTN_INT2 : GdalReadWriteIO<Type, tINT2>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init) ; break ;
        case eTyNums::eTN_U_INT2 : GdalReadWriteIO<Type, tU_INT2>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
        case eTyNums::eTN_INT4 : GdalReadWriteIO<Type, tINT4>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ;  break ;
        case eTyNums::eTN_U_INT4 : GdalReadWriteIO<Type, tU_INT4>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
        case eTyNums::eTN_INT8 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_INT8") ; break ;
        case eTyNums::eTN_REAL4 : GdalReadWriteIO<Type, tREAL4>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
        case eTyNums::eTN_REAL8 : GdalReadWriteIO<Type, tREAL8>(aMode, aDF, aVecImV2, aP0File, aDyn, aR2Init)  ; break ;
        case eTyNums::eTN_REAL16 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_REAL16") ; break ;
        case eTyNums::eTN_UnKnown : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eTN_UnKnown") ; break ;
        case eTyNums::eNbVals : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::ReadWrite : case eTyNums::eNbVals") ; break ;
        }
    }

   template <class Type> void GdalReadWrite(
        IoMode aMode,
        const cDataIm2D<Type>& aImV2,
        const cDataFileIm2D &aDF,
        const cPt2di & aP0File,
        double aDyn,
        const cRect2& aR2Init
    )
    {
        std::vector<const cDataIm2D<Type> *> aVIms({&aImV2});
        GdalReadWrite(aMode,aVIms,aDF,aP0File,aDyn,aR2Init);
    }

   template <class Type> void GdalReadWrite(
        IoMode aMode,
        const cDataIm2D<Type>& aImV2R,
        const cDataIm2D<Type>& aImV2G,
        const cDataIm2D<Type>& aImV2B,
        const cDataFileIm2D &aDF,
        const cPt2di & aP0File,
        double aDyn,
        const cRect2& aR2Init
    )
    {
        std::vector<const cDataIm2D<Type> *> aVIms({&aImV2R,&aImV2G,&aImV2B});
        GdalReadWrite(aMode,aVIms,aDF,aP0File,aDyn,aR2Init);
    }

} // namespace Private


#ifdef MMVII_KEEP_MMV1_IMAGE
extern std::string MM3DFixeByMMVII; // Declared in MMV1 for its own stuff
#endif

namespace MMVII
{


std::string V1NameMasqOfIm(const std::string & aName)
{
   return LastPrefix(aName) + "_Masq.tif";
}

#ifdef MMVII_KEEP_MMV1_IMAGE
static void Init_mm3d_In_MMVII()
{
   // 
   static bool First= true;
   if (! First) return;
   First = false;

   // Compute mm3d location from relative position to MMVII
   // static std::string CA0 =  DirBin2007 + "../../bin/mm3d";
   char * A0= const_cast<char *>(cMMVII_Appli::MMV1Bin().c_str());
   MM3DFixeByMMVII = cMMVII_Appli::MMV1Bin();
   MMD_InitArgcArgv(1,&A0);
}
#endif


/* =========================== */
/*       cDataFileIm2D         */
/* =========================== */

cDataFileIm2D::cDataFileIm2D(const std::string & aName,eTyNums aType,const cPt2di & aSz,int aNbChannel,eForceGray isFG) :
    cPixBox<2> (cPt2di(0,0),aSz),
    mName       (aName),
    mType       (aType),
    mNbChannel  (aNbChannel),
    mForceGray  (isFG)
{
    
}


cDataFileIm2D cDataFileIm2D::Empty()
{
   return cDataFileIm2D( MMVII_NONE, eTyNums::eNbVals, cPt2di(1,1), -1,eForceGray::No);
}

bool cDataFileIm2D::IsEmpty() const
{
    return mNbChannel<=0;
}


void cDataFileIm2D::AssertNotEmpty() const
{
    MMVII_INTERNAL_ASSERT_strong((!IsEmpty()),"cDataFileIm2D was not initialized");
}


cDataFileIm2D cDataFileIm2D::Create(const std::string & aName,eForceGray isFG)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        // required because with jpg/raw mm1 may call itself, need some special stuff
        // as standar mmv1 by analyse of arg/argv would not work
        Init_mm3d_In_MMVII();

        bool aForce8B = false;
        std::string aNameTif = NameFileStd(aName,-1,!aForce8B ,true,true);
        Tiff_Im aTF = Tiff_Im::StdConvGen(aNameTif.c_str(),-1,!aForce8B ,true);

        return cDataFileIm2D(aName,ToMMVII(aTF.type_el()),ToMMVII(aTF.sz()), aTF.nb_chan(), isFG);
    } else {
#endif
    // Open a first time with gdal to have access to metadata and then to create a cDataFileIm2D object
    InitGDAL();
    auto aDataset = OpenDataset(aName, GA_ReadOnly);
    if (aDataset == nullptr) {
        MMVII_UserError(eTyUEr::eOpenFile,std::string("Can't open image file: ") + CPLGetLastErrorMsg());
        return Empty(); // Never executed
    }
    cPt2di aSz = cPt2di(aDataset->GetRasterXSize(), aDataset->GetRasterYSize());
    auto aNbChannel = aDataset->GetRasterCount();
    auto aType = TyGdalToMMVII( aDataset->GetRasterBand( 1 )->GetRasterDataType());
    CloseDataset(aDataset);
     // Create a cDataFileIm2D on an existing image
    return cDataFileIm2D(aName, aType, aSz, aNbChannel, isFG);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif    
}

cDataFileIm2D  cDataFileIm2D::Create(const std::string & aName,eTyNums aType,const cPt2di & aSz,int aNbChan)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
    if (mmvii_use_mmv1_image) {
        Tiff_Im::PH_INTER_TYPE aPIT = Tiff_Im::BlackIsZero;
        if (aNbChan==1)
            aPIT = Tiff_Im::BlackIsZero;
        else if (aNbChan==3)
            aPIT = Tiff_Im::RGB;
        else
        {
            MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
        }
        
        bool IsModified;
        Tiff_Im::CreateIfNeeded
        (
            IsModified,
            aName,
            ToMMV1(aSz),
            ToMMV1(aType),
            Tiff_Im::No_Compr,
            aPIT
        );
        return Create(aName,eForceGray::No);
    } else {
#endif
    if (aNbChan!=1 && aNbChan!=3)
    {
        MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
    }
    
    InitGDAL();
    auto aDataset = OpenDataset(aName, GA_ReadOnly);
    if (aDataset != nullptr) {
        cPt2di fileSz = cPt2di(aDataset->GetRasterXSize(), aDataset->GetRasterYSize());
        auto fileNbChan = aDataset->GetRasterCount();
        auto fileType = TyGdalToMMVII( aDataset->GetRasterBand( 1 )->GetRasterDataType());
        if (fileSz == aSz && fileNbChan == aNbChan && fileType == aType) {
            CloseDataset(aDataset);
            return cDataFileIm2D(aName, aType, aSz, aNbChan, eForceGray::No);
        }
    }
    remove(aName.c_str());
    
    auto pszFormat = ExtToGdalDriver(aName);
    GDALDriver *aGDALDriver = GetGDALDriverManager()->GetDriverByName(pszFormat.c_str());
    if (!aGDALDriver) {
        MMVII_INTERNAL_ERROR(std::string("GDAL can't handle file format : ") + pszFormat);
        return Empty(); // Never executed
    }
    auto aGDALType = TyMMVIIToGdal(aType);
    aDataset = aGDALDriver->Create( aName.c_str(), aSz.x(), aSz.y(), aNbChan, aGDALType, nullptr );
    if (!aDataset) {
        MMVII_UserError(eTyUEr::eOpenFile,std::string("Can't create image file: ") + CPLGetLastErrorMsg());
        return Empty(); // Never executed
    }
    size_t aSize = GDALGetDataTypeSizeBytes(aGDALType)*aSz.x()*aSz.y();
    void *abyRaster = cMemManager::Calloc(1, aSize);
    memset(abyRaster,0, aSize);          // cMemManager::Calloc fill allocated memory with a constant fixed debug value ...
    
    // Initialize the dataset
    GDALRasterBand *aBand;
    for (int aChan = 0; aChan < aNbChan; aChan++)
    {
        aBand = aDataset->GetRasterBand(aChan+1);
        CPLErr cplErr2 = aBand->RasterIO( GF_Write, 0, 0, aSz.x(), aSz.y(), abyRaster, aSz.x(), aSz.y(), aGDALType, 0, 0 );
        MMVII_INTERNAL_ASSERT_strong(cplErr2 == 0 || cplErr2 == 1,"Error in writing image");
    }
    cMemManager::Free(abyRaster);
    CloseDataset(aDataset);
    return cDataFileIm2D(aName, aType, aSz, aNbChan,eForceGray::No);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif    
  
}


cDataFileIm2D::~cDataFileIm2D()
{
}

const cPt2di &  cDataFileIm2D::Sz() const  {return  cPixBox<2>::Sz();}
const std::string &  cDataFileIm2D::Name() const { return mName; }
const int  & cDataFileIm2D::NbChannel ()  const { return mNbChannel; }
const eTyNums &   cDataFileIm2D::Type ()  const {return mType;}


bool cDataFileIm2D::IsPostFixNameImage(const std::string & aPost)
{
    static std::vector<std::string> aVNames({"jpg","jpeg","tif","tiff"});

    return UCaseMember(aVNames,aPost);
}

bool cDataFileIm2D::IsNameWith_PostFixImage(const std::string & aName)
{
   return IsPostFixNameImage(LastPostfix(aName));
}



#ifdef MMVII_KEEP_MMV1_IMAGE

template <class Type> void cMMV1_Conv<Type>::ReadWrite
                           (
                               bool ReadMode,
                               const std::vector<const tImMMVII *> & aVecImV2,
                               const cDataFileIm2D & aDF,
                               const cPt2di & aP0File,
                               double aDyn,
                               const cRect2& aR2Init
                           )
{

// StdOut() <<  "aP0File,aP0File, " << aP0File << "\n";
   Init_mm3d_In_MMVII();
   // C'est une image en originie (0,0) necessairement en MMV1
   const tImMMVII & aImV2 = *(aVecImV2.at(0));
   Fonc_Num aFoncImV1 = ImToMMV1(aImV2).in();
   Output   aOutImV1  = ImToMMV1(aImV2).out();
   for (int aKIm=1 ; aKIm<int(aVecImV2.size()) ; aKIm++)
   {
       MMVII_INTERNAL_ASSERT_strong(aImV2.Sz()==aVecImV2.at(aKIm)->Sz(),"Diff Sz in ReadWrite");
       MMVII_INTERNAL_ASSERT_strong(aImV2.P0()==aVecImV2.at(aKIm)->P0(),"Diff P0 in ReadWrite");
       aFoncImV1 = Virgule(aFoncImV1,ImToMMV1(*aVecImV2.at(aKIm)).in());
       aOutImV1  = Virgule( aOutImV1,ImToMMV1(*aVecImV2.at(aKIm)).out());
   }
   cRect2 aRectFullIm (cPt2di(0,0),aImV2.Sz());

   // Rectangle image / a un origine (0,0)
   cRect2 aRectIm =  (aR2Init== cRect2::TheEmptyBox)           ?  // Val par def
                     aRectFullIm                           :  // Rectangle en 00
                     aR2Init.Translate(-aImV2.P0())   ;  // Convention aR2Init tient compte de P0

   // It's a bit strange but in fact P0File en aImV2.P0() are redundant, so if both are used
   // it seems normal to add them
          
   Pt2di aTrans  = ToMMV1(aImV2.P0() + aP0File);
   Pt2di aP0Im = ToMMV1(aRectIm.P0());
   Pt2di aP1Im = ToMMV1(aRectIm.P1());

   cRect2 aRUsed(ToMMVII(aP0Im+aTrans),ToMMVII(aP1Im+aTrans));
   if (true)
   {
       MMVII_INTERNAL_ASSERT_strong(aRUsed.IncludedIn(aDF), "Read/write out of file");
       MMVII_INTERNAL_ASSERT_strong(aRectIm.IncludedIn(aRectFullIm), "Read/write out of Im");
   }

   Tiff_Im aTF=Tiff_Im::StdConvGen(aDF.Name(),-1,true);

   if (ReadMode)
   {
      Symb_FNum  aFIn = aTF.in();
      // If input is multi-channel and out single, we compute the average of all channel
      if ((aFIn.dimf_out()>1) && (aVecImV2.size()==1))
      {
         // if we have more than 3 channel, the 4th is generally an alpha channel, so dont want to use it
	 // btw it a bit basic, and certainly a more sophisticated rule will have to be used (as weighting of diff channels)
         int aNbCh = std::min(3,aFIn.dimf_out());

         Fonc_Num aNewF = aFIn.kth_proj(0);
         for (int aKF=1 ; aKF< aNbCh ; aKF++)
             aNewF = aNewF + aFIn.kth_proj(aKF);

           aFIn = aNewF / aNbCh;
      }
          
      ELISE_COPY
      (
           rectangle(aP0Im,aP1Im),
           trans(El_CTypeTraits<Type>::TronqueF(aFIn*aDyn),aTrans),
           aOutImV1
      );
   }
   else
   {
      ELISE_COPY
      (
           rectangle(aP0Im+aTrans,aP1Im+aTrans),
           trans(Tronque(aTF.type_el(),aFoncImV1*aDyn),-aTrans),
           aTF.out()
      );
   }
}

template <class Type> void cMMV1_Conv<Type>::ReadWrite
                           (
                               bool ReadMode,
                               const tImMMVII &aImV2,
                               const cDataFileIm2D & aDF,
                               const cPt2di & aP0File,
                               double aDyn,
                               const cRect2& aR2Init
                           )
{
     std::vector<const tImMMVII *> aVIms({&aImV2});
     ReadWrite(ReadMode,aVIms,aDF,aP0File,aDyn,aR2Init);
}
template <class Type> void cMMV1_Conv<Type>::ReadWrite
                           (
                               bool ReadMode,
                               const tImMMVII &aImV2R,
                               const tImMMVII &aImV2G,
                               const tImMMVII &aImV2B,
                               const cDataFileIm2D & aDF,
                               const cPt2di & aP0File,
                               double aDyn,
                               const cRect2& aR2Init
                           )
{

     std::vector<const tImMMVII *> aVIms({&aImV2R,&aImV2G,&aImV2B});
     ReadWrite(ReadMode,aVIms,aDF,aP0File,aDyn,aR2Init);
}


template <> void cMMV1_Conv<tREAL16>::ReadWrite
                 (bool,const tImMMVII &,const cDataFileIm2D &,const cPt2di &,double,const cRect2& )
{
   MMVII_INTERNAL_ASSERT_strong(false,"No ReadWrite of 16-Byte float");
}
template <> void cMMV1_Conv<tREAL16>::ReadWrite
                 (bool,const tImMMVII &,const tImMMVII &,const tImMMVII &,const cDataFileIm2D &,const cPt2di &,double,const cRect2& )
{
   MMVII_INTERNAL_ASSERT_strong(false,"No ReadWrite of 16-Byte float");
}
template <> void cMMV1_Conv<tU_INT4>::ReadWrite
                 (bool,const tImMMVII &,const tImMMVII &,const tImMMVII &,const cDataFileIm2D &,const cPt2di &,double,const cRect2& )
{
   MMVII_INTERNAL_ASSERT_strong(false,"No ReadWrite of 16-Byte float");
}
#endif

template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
   if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(true,*this,aFile,aP0,aDyn,aR2);
    } else {
#endif
    GdalReadWrite(IoMode::Read, *this, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,tIm &aImG,tIm &aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
#ifdef MMVII_KEEP_MMV1_IMAGE
   if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(true,*this,aImG,aImB,aFile,aP0,aDyn,aR2);
    } else {
#endif
    GdalReadWrite(IoMode::Read, *this, aImG, aImB, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}


template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
#ifdef MMVII_KEEP_MMV1_IMAGE
   if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(false,*this,aFile,aP0,aDyn,aR2);
    } else {
#endif
    GdalReadWrite(IoMode::Write, *this, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const tIm &aImG,const tIm &aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
#ifdef MMVII_KEEP_MMV1_IMAGE
   if (mmvii_use_mmv1_image) {
        cMMV1_Conv<Type>::ReadWrite(false,*this,aImG,aImB,aFile,aP0,aDyn,aR2);
    } else {
#endif
      GdalReadWrite(IoMode::Write, *this, aImG, aImB, aFile, aP0, aDyn, aR2);
#ifdef MMVII_KEEP_MMV1_IMAGE
    }
#endif
}

#ifdef MMVII_KEEP_MMV1_IMAGE
//  It's difficult to read unsigned int4 with micmac V1, wait for final implementation
template <>  void  cDataIm2D<tU_INT4>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
    if (mmvii_use_mmv1_image) {
        MMVII_INTERNAL_ASSERT_strong(false,"No read for unsigned int4 now");
    } else {
        GdalReadWrite(IoMode::Read, *this, aFile, aP0, aDyn, aR2);
    }
}

template <>  void  cDataIm2D<tU_INT4>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
    if (mmvii_use_mmv1_image) {
        MMVII_INTERNAL_ASSERT_strong(false,"No write for unsigned int4 now");
    } else {
        GdalReadWrite(IoMode::Write, *this, aFile, aP0, aDyn, aR2);
    }
}

template <>  void  cDataIm2D<tU_INT4>::Write(const cDataFileIm2D & aFile,const tIm& aImG,const tIm& aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
    if (mmvii_use_mmv1_image) {
        MMVII_INTERNAL_ASSERT_strong(false,"No write for unsigned int4 now");
    } else {
      GdalReadWrite(IoMode::Write, *this, aImG, aImB, aFile, aP0, aDyn, aR2);
    }
}
#endif


template <class Type>  void  cIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
    DIm().Read(aFile,aP0,aDyn,aR2);
}
template <class Type>  void  cIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
    DIm().Write(aFile,aP0,aDyn,aR2);
}



// FIXME CM: Remove/modify following code (until EOF)
GenIm::type_el ToMMV1(eTyNums aV2)
{
    switch (aV2)
    {
        case eTyNums::eTN_INT1 : return GenIm::int1  ;
        case eTyNums::eTN_INT2 : return GenIm::int2  ;
        case eTyNums::eTN_INT4 : return GenIm::int4  ;
        case eTyNums::eTN_U_INT1 : return GenIm::u_int1  ;
        case eTyNums::eTN_U_INT2 : return GenIm::u_int2  ;
        case eTyNums::eTN_U_INT4 : return GenIm::u_int4  ;
        case eTyNums::eTN_REAL4 : return GenIm::real4  ;
        case eTyNums::eTN_REAL8 : return GenIm::real8  ;
        default: ;
    }
    MMVII_INTERNAL_ERROR("GenIm::type_el ToMMV1(eTyNums)");
    return GenIm::int1;
}

eTyNums ToMMVII( GenIm::type_el aV1 )
{
    switch (aV1)
    {
        case  GenIm::int1 :  return eTyNums::eTN_INT1 ;
        case  GenIm::int2 :  return eTyNums::eTN_INT2 ;
        case  GenIm::int4 :  return eTyNums::eTN_INT4 ;

        case  GenIm::u_int1 :  return eTyNums::eTN_U_INT1 ;
        case  GenIm::u_int2 :  return eTyNums::eTN_U_INT2 ;
        case  GenIm::u_int4 :  return eTyNums::eTN_U_INT4 ;

        case  GenIm::real4 :  return eTyNums::eTN_REAL4 ;
        case  GenIm::real8 :  return eTyNums::eTN_REAL8 ;

        default: ;
    }
    return eTyNums::eTN_UnKnown ;
/*
    MMVII_INTERNAL_ASSERT_bench(false,"eTyNums ToMMVII( GenIm::type_el aV1 )");
    return eTyNums::eTN_INT1 ;
*/
}







double DifAbsInVal(const std::string & aN1,const std::string & aN2,double aDef)
{
   Tiff_Im aF1(aN1.c_str());
   Tiff_Im aF2(aN2.c_str());
   double aSom;

   if (aF1.sz()!=aF2.sz())
   {
       MMVII_INTERNAL_ASSERT_always(aDef!=0.0,"Diff sz and bad def in DifAbsInVal");
       return aDef;
   }

   ELISE_COPY(aF1.all_pts(),Abs(aF1.in()-aF2.in()),sigma(aSom));

   return aSom;
}

template <const int aNbBit>  cIm2D<tU_INT1>  BitsV1ToV2(const Im2D_Bits<aNbBit> & aImV1)
{
    cIm2D<tU_INT1> aImV2(ToMMVII(aImV1.sz()));
    cDataIm2D<tU_INT1>& aDImV2 = aImV2.DIm();

    for (const auto & aPixV2 : aDImV2)
    {
         aDImV2.SetV(aPixV2,aImV1.GetI(ToMMV1(aPixV2)));
    }


    return aImV2;
}


cIm2D<tU_INT1> ImageOfString_10x8(const std::string & aStr ,int aSpace)
{
    Im2D_Bits<1> aImV1 =  cElBitmFont::BasicFont_10x8().BasicImageString(aStr,aSpace);
    return BitsV1ToV2(aImV1);
}

cIm2D<tU_INT1> ImageOfString_DCT(const std::string & aStr ,int aSpace)
{
    Im2D_Bits<1> aImV1 =  cElBitmFont::FontCodedTarget().BasicImageString(aStr,aSpace);
    return BitsV1ToV2(aImV1);
}

void Convert_JPG(const std::string &  aNameIm,bool DeleteAfter,tREAL8 aQuality,const std::string & aPost)
{
    cParamCallSys aCom("convert","-quality",ToStr(aQuality),
	                   aNameIm,
                       LastPrefix(aNameIm) + "." + aPost
                       );

       int aResult = GlobSysCall(aCom,true);

       if ( (aResult==EXIT_SUCCESS) && DeleteAfter)
       {
           RemoveFile(aNameIm,false);
       }

}


//  INSTANTIATION 

#define MACRO_INSTANTIATE_READ_FILE(Type)\
template class cDataIm2D<Type>;\
template class cIm2D<Type>;

MACRO_INSTANTIATE_READ_FILE(tINT1)
MACRO_INSTANTIATE_READ_FILE(tINT2)
MACRO_INSTANTIATE_READ_FILE(tINT4)
// MACRO_INSTANTIATE_READ_FILE(tINT8)
MACRO_INSTANTIATE_READ_FILE(tU_INT1)
MACRO_INSTANTIATE_READ_FILE(tU_INT2)
MACRO_INSTANTIATE_READ_FILE(tU_INT4)
MACRO_INSTANTIATE_READ_FILE(tREAL4)
MACRO_INSTANTIATE_READ_FILE(tREAL8)
MACRO_INSTANTIATE_READ_FILE(tREAL16)
};
