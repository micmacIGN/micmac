#include "V1VII.h"
#include "MMVII_Stringifier.h"
#include "MMVII_util.h"
#include "MMVII_Image2D.h"
#include "MMVII_DeclareCste.h"
#include "cMMVII_Appli.h"
#include "MMVII_Sys.h"
#include "StdAfx.h"
#include "MMVII_Ptxd.h"
#include "gdal/gdal_priv.h"


using namespace MMVII;

namespace{ // Private
   eTyNums GdalToMMVII( GDALDataType aType )
   {
      switch (aType)
      {
         case GDT_Unknown : return eTyNums::eTN_UnKnown;
         case  GDT_Byte :  return eTyNums::eTN_U_INT1 ;
         case  GDT_UInt16 :  return eTyNums::eTN_U_INT2 ;
         case  GDT_Int16 :  return eTyNums::eTN_INT2 ;
         case  GDT_UInt32 :  return eTyNums::eTN_U_INT4 ;
         case  GDT_Int32 :  return eTyNums::eTN_INT4 ;
         case  GDT_Float32 :  return eTyNums::eTN_REAL4 ;
         case  GDT_Float64 :  return eTyNums::eTN_REAL8 ;

         case  GDT_CInt16 :  MMVII_INTERNAL_ERROR("GdalToMMVII : case GDT_CInt16") ;
         case  GDT_CInt32 :  MMVII_INTERNAL_ERROR("GdalToMMVII : case GDT_CInt32") ;
         case  GDT_CFloat32 :  MMVII_INTERNAL_ERROR("GdalToMMVII : case GDT_CFloat32") ;
         case  GDT_CFloat64 :  MMVII_INTERNAL_ERROR("GdalToMMVII : case GDT_CFloat64") ;
         case  GDT_TypeCount :  MMVII_INTERNAL_ERROR("GdalToMMVII : case GDT_TypeCount") ;
      }
      return eTyNums::eTN_UnKnown ;
   }


   GDALDataType MMVIIToGdal( eTyNums aType )
   {
      switch (aType)
      {

         case  eTyNums::eTN_INT1 :  MMVII_INTERNAL_ERROR("MMVIIToGdal : case eTyNums::eTN_INT1") ;
         case  eTyNums::eTN_U_INT1 :  return GDT_Byte ;
         case  eTyNums::eTN_INT2 :  return GDT_Int16 ;
         case  eTyNums::eTN_U_INT2 :  return GDT_UInt16 ;
         case  eTyNums::eTN_INT4 :  return GDT_Int32 ;
         case  eTyNums::eTN_U_INT4 :  return GDT_UInt32 ;
         case  eTyNums::eTN_INT8 :  MMVII_INTERNAL_ERROR("MMVIIToGdal : case eTyNums::eTN_INT8") ;
         case  eTyNums::eTN_REAL4 :  return GDT_Float32 ;
         case  eTyNums::eTN_REAL8 :  return GDT_Float64 ;
         case  eTyNums::eTN_REAL16 :  MMVII_INTERNAL_ERROR("MMVIIToGdal : case eTyNums::eTN_REAL16") ;
         case  eTyNums::eTN_UnKnown :  MMVII_INTERNAL_ERROR("MMVIIToGdal : case eTyNums::eTN_UnKnown") ;
         case  eTyNums::eNbVals :  MMVII_INTERNAL_ERROR("MMVIIToGdal : case eTyNums::eNbVals") ;
      }
      return GDT_Unknown ;
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


    GDALDataset * OpenDataset(std::string aName)
   {
      // Open image with Gdal
      const GDALAccess eAccess = GA_Update;
      GDALDataset * aGdalDataset = GDALDataset::FromHandle(GDALOpen( aName.c_str(), eAccess ));
      return aGdalDataset;
   }

   template <typename TypeIn, typename TypeOut> void GdalReadData(GDALDataset* aDataset,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2, int band, cDataIm2D<TypeIn> * aDataIm2D, GDALDataType aGDALDataType){

      // Get first band
      GDALRasterBand* aBand = aDataset->GetRasterBand(band);

      cRect2 aRectFullIm (cPt2di(0,0),aDataIm2D->Sz());
      cRect2 aRectIm =  (aR2== cRect2::TheEmptyBox)           ?  
                        aRectFullIm                           :  
                        aR2   ;
      // aRectIm est dans le repère d'origine P0
      cPt2di aTrans  = aDataIm2D->P0() + aP0;
      
      // Read image according to Gdal tutorial
      TypeOut *pafScanline;
      pafScanline = (TypeOut *) cMemManager::Calloc(1, sizeof(TypeOut)*aRectIm.Sz().x()*aRectIm.Sz().y());
      CPLErr cplErr2 = aBand->RasterIO( GF_Read, aRectIm.P0().x() + aTrans.x(), aRectIm.P0().y() + aTrans.y(), aRectIm.Sz().x(), aRectIm.Sz().y(), pafScanline, aRectIm.Sz().x(), aRectIm.Sz().y(), aGDALDataType, 0, 0 );   
      MMVII_INTERNAL_ASSERT_strong(cplErr2 == 0 || cplErr2 == 1,"Error in writing image");
      
      // Copy of data in mRawData2D. Is there a way to complete mRawData2D directly in aBand->RasterIO()?
      // Image is read in float 32, and the conversion is done here
      for (int aY = 0; aY < aRectIm.Sz().y(); aY++)
      {
         for (int aX = 0; aX < aRectIm.Sz().x(); aX++)
         {
            aDataIm2D->SetVTrunc(cPt2di(aX, aY) + aDataIm2D->P0(), pafScanline[aY * aRectIm.Sz().x() + aX] * aDyn);
         }
      }

      cMemManager::Free(pafScanline);

   }


   template <typename TypeIn, typename TypeOut> void GdalWriteData(GDALDataset * poDstDS,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2, int band, const cDataIm2D<TypeIn> * aDataIm2D, GDALDataType aGDALDataType){
         
         // get the raster band
         GDALRasterBand *poBand;
         poBand = poDstDS->GetRasterBand(band);

         cRect2 aRectFullIm (cPt2di(0,0),aDataIm2D->Sz());
         cRect2 aRectIm =  (aR2== cRect2::TheEmptyBox)           ?  
                           aRectFullIm                           :  
                           aR2   ;
         // aRectIm est dans le repère d'origine P0
         cPt2di aTrans  = aDataIm2D->P0() + aP0;
         
         TypeOut * abyRaster;
         abyRaster = (TypeOut *) cMemManager::Calloc(1, sizeof(TypeOut)*aRectIm.Sz().x()*aRectIm.Sz().y());

         for (int aY = 0; aY < aRectIm.Sz().y(); aY++)
         {
            for (int aX = 0; aX < aRectIm.Sz().x(); aX++)
            {
               abyRaster[aY * aRectIm.Sz().x() + aX] = tNumTrait<TypeOut>::Trunc(aDataIm2D->GetV(cPt2di(aX, aY) + aDataIm2D->P0() + aR2.P0()) * aDyn);
            }
         }
         // Write data in file
         CPLErr cplErr2 = poBand->RasterIO( GF_Write, aRectIm.P0().x() + aTrans.x(), aRectIm.P0().y() + aTrans.y(), aRectIm.Sz().x(), aRectIm.Sz().y(), abyRaster, aRectIm.Sz().x(), aRectIm.Sz().y(), aGDALDataType, 0, 0 );
         MMVII_INTERNAL_ASSERT_strong(cplErr2 == 0 || cplErr2 == 1,"Error in writing image");

         cMemManager::Free(abyRaster);

   }



   template <class Type> void GdalRead
                            (
                                std::vector<cDataIm2D<Type>*>& aVecImV2,
                                const cDataFileIm2D &aDF,
                                const cPt2di & aP0File,
                                double aDyn,
                                const cRect2& aR2Init
                            )
   {
      GDALDataset * aGdalDataset = OpenDataset(aDF.Name());
      int i = 0;
      for (auto & element : aVecImV2)
      {
               switch (aDF.Type())
               {
                  case eTyNums::eTN_INT1 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Read : case eTyNums::eTN_INT1") ; break ;
                  case eTyNums::eTN_U_INT1 : GdalReadData<Type, tU_INT1>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
                  case eTyNums::eTN_INT2 : GdalReadData<Type, tINT2>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type())) ; break ;
                  case eTyNums::eTN_U_INT2 : GdalReadData<Type, tU_INT2>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
                  case eTyNums::eTN_INT4 : GdalReadData<Type, tINT4>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ;  break ;
                  case eTyNums::eTN_U_INT4 : GdalReadData<Type, tU_INT4>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
                  case eTyNums::eTN_INT8 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Read : case eTyNums::eTN_INT8") ; break ;
                  case eTyNums::eTN_REAL4 : GdalReadData<Type, tREAL4>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
                  case eTyNums::eTN_REAL8 : GdalReadData<Type, tREAL8>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
                  case eTyNums::eTN_REAL16 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Read : case eTyNums::eTN_REAL16") ; break ;
                  case eTyNums::eTN_UnKnown : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Read : case eTyNums::eTN_UnKnown") ; break ;
                  case eTyNums::eNbVals : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Read : case eTyNums::eNbVals") ; break ;
               }
         i++;
      }
      GDALClose( (GDALDatasetH) aGdalDataset );
   }


   template <class Type> void GdalWrite
                              (
                                 std::vector<const cDataIm2D<Type>*>& aVecImV2,
                                 const cDataFileIm2D &aDF,
                                 const cPt2di & aP0File,
                                 double aDyn,
                                 const cRect2& aR2Init
                              )
   {
      GDALDataset * aGdalDataset = OpenDataset(aDF.Name());
      int i = 0;
      for (auto & element : aVecImV2)
      {
            switch (aDF.Type())
            {
               case eTyNums::eTN_INT1 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Write : case eTyNums::eTN_INT1") ; break ;
               case eTyNums::eTN_U_INT1 : GdalWriteData<Type, tU_INT1>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
               case eTyNums::eTN_INT2 : GdalWriteData<Type, tINT2>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type())) ; break ;
               case eTyNums::eTN_U_INT2 : GdalWriteData<Type, tU_INT2>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
               case eTyNums::eTN_INT4 : GdalWriteData<Type, tINT4>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
               case eTyNums::eTN_U_INT4 : GdalWriteData<Type, tU_INT4>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
               case eTyNums::eTN_INT8 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Write : case eTyNums::eTN_INT8") ; break ;
               case eTyNums::eTN_REAL4 : GdalWriteData<Type, tREAL4>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
               case eTyNums::eTN_REAL8 : GdalWriteData<Type, tREAL8>(aGdalDataset, aP0File, aDyn, aR2Init, i+1, element, MMVIIToGdal(aDF.Type()))  ; break ;
               case eTyNums::eTN_REAL16 : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Write : case eTyNums::eTN_REAL16") ; break ;
               case eTyNums::eTN_UnKnown : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Write : case eTyNums::eTN_UnKnown") ; break ;
               case eTyNums::eNbVals : MMVII_INTERNAL_ERROR("cDataIm2D<Type>::Write : case eTyNums::eNbVals") ; break ;
            }
         i++;
      }    
      GDALClose( (GDALDatasetH) aGdalDataset );
   }

}// Private




namespace MMVII
{


std::string V1NameMasqOfIm(const std::string & aName)
{
   return LastPrefix(aName) + "_Masq.tif";
}

/* =========================== */
/*       cDataFileIm2D         */
/* =========================== */

cDataFileIm2D::cDataFileIm2D(const std::string & aName,eTyNums aType,const cPt2di & aSz,int aNbChannel) :
    cPixBox<2> (cPt2di(0,0),aSz),
    mName       (aName),
    mType       (aType),
    mNbChannel  (aNbChannel)
{
    
}


cDataFileIm2D::cDataFileIm2D(const std::string & aName, const cPt2di & aSz) :
   cPixBox<2> (cPt2di(0,0), aSz),
   mName       (aName)
{
   const GDALAccess eAccess = GA_Update;
   GDALDataset * aDataset = GDALDataset::FromHandle(GDALOpen( aName.c_str(), eAccess ));
   mNbChannel = aDataset->GetRasterCount();
   mType = GdalToMMVII( aDataset->GetRasterBand( 1 )->GetRasterDataType());
}



cDataFileIm2D cDataFileIm2D::Empty()
{
   return cDataFileIm2D( MMVII_NONE, eTyNums::eNbVals, cPt2di(1,1), -1);
}

bool cDataFileIm2D::IsEmpty() const
{
    return mNbChannel<=0;
}


void cDataFileIm2D::AssertNotEmpty() const
{
    MMVII_INTERNAL_ASSERT_strong((!IsEmpty()),"cDataFileIm2D was not initialized");
}


cDataFileIm2D cDataFileIm2D::Create(const std::string & aName,bool aForceGray)
{

   // Create a cDataFileIm2D on an existing image
   
   // Open a first time with gdal to have access to the size and then to create a cDataFileIm2D object
   GDALAllRegister();
   const GDALAccess eAccess = GA_Update;
   GDALDataset * aDataset = GDALDataset::FromHandle(GDALOpen( aName.c_str(), eAccess ));
   cPt2di aSz = cPt2di(aDataset->GetRasterXSize(), aDataset->GetRasterYSize());

   return cDataFileIm2D(aName, aSz);
}

cDataFileIm2D  cDataFileIm2D::Create(const std::string & aName,eTyNums aType,const cPt2di & aSz,int aNbChan)
{

   // Create an image and a cDataFileIm2D

   remove(aName.c_str());

   // Check that aNbChan is 1 or 3
   if (aNbChan!=1 && aNbChan!=3)
   {
      MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
   } 

   // Create a driver
   GDALAllRegister();
   auto pszFormat = ExtToGdalDriver(aName);
   GDALDriver *poDriver;
   poDriver = GetGDALDriverManager()->GetDriverByName(pszFormat.c_str());

   // Create a dataset
   GDALDataset *poDstDS;
   char **papszOptions = NULL;
   poDstDS = poDriver->Create( aName.c_str(), aSz.x(), aSz.y(), aNbChan, MMVIIToGdal(aType), papszOptions );

   double * abyRaster;
   abyRaster = (double *) cMemManager::Calloc(1, sizeof(double)*aSz.x()*aSz.y());

   for (int aK = 0; aK < aSz.x()*aSz.y(); aK++)
   {
      abyRaster[aK] = 0;
   }
      
   // Initialize the dataset
   GDALRasterBand *poBand;
   for (int i = 1; i < aNbChan+1; i++)
   {
      poBand = poDstDS->GetRasterBand(i);
      CPLErr cplErr2 = poBand->RasterIO( GF_Write, 0, 0, aSz.x(), aSz.y(), abyRaster, aSz.x(), aSz.y(), MMVIIToGdal(aType), 0, 0 );
      MMVII_INTERNAL_ASSERT_strong(cplErr2 == 0 || cplErr2 == 1,"Error in writing image");
   }

   cMemManager::Free(abyRaster);
   
   GDALClose( (GDALDatasetH) poDstDS );

   return cDataFileIm2D(aName, aSz);
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





template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
    std::vector<tIm * > aVIms({this});
     GdalRead(aVIms, aFile, aP0, aDyn, aR2);

}

template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,tIm &aImG,tIm &aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
     std::vector<tIm * > aVIms({this,&aImG,&aImB});
     GdalRead(aVIms, aFile, aP0, aDyn, aR2);
}


template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
   std::vector<const tIm * > aVIms({this});
   GdalWrite(aVIms, aFile, aP0, aDyn, aR2);
}

template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const tIm &aImG,const tIm &aImB,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
      std::vector<const tIm * > aVIms({this,&aImG,&aImB});
      GdalWrite(aVIms, aFile, aP0, aDyn, aR2);
}




template <class Type>  void  cIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2)
{
      DIm().Read(aFile,aP0,aDyn,aR2);
}
template <class Type>  void  cIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cPixBox<2>& aR2) const
{
      DIm().Write(aFile,aP0,aDyn,aR2);
}



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
