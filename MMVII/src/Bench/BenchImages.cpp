#include "include/MMVII_all.h"

namespace MMVII
{

static int SomI(int i) {return (i * (i+1)) / 2;}
static int SomIntI(int i0,int i1) {return SomI(i1-1) - SomI(i0-1);}

static double FoncTestIm(const cPt2di & aP)
{
   return aP.x() - 1.234 * aP.y() + 1.0 * (1 +std::abs(aP.x()) + std::pow(std::abs(aP.y()),1.4));
}

/*====================== IMAGE BENCH ===========================================*/
/*====================== IMAGE BENCH ===========================================*/
/*====================== IMAGE BENCH ===========================================*/

/**  Test that an image do what an image must do, store the value of a pixel
   and restor them when we ask for it (i.e GetV and SetV)
*/

template <class Type> void TestOneImage2D(const cPt2di & aP0,const cPt2di & aP1)
{
    cIm2D<Type> aPIm(aP0,aP1);
    cDataIm2D<Type>  & aIm = aPIm.Im();

    // Check the rectangle (number of point ....)
    int aNb = 0;
    int aNbX = 0;
    int aSomX = 0;
    int aSomY = 0;
    // for (aP.x()=aP0.x() ; aP.x()<aP1.x() ; aP.x()++)
    for (cPt2di aP =aP0 ; aP.x()<aP1.x() ; aP.x()++)
    {
        aNbX++;
        aSomX += aP.x();
        for (aP.y()=aP0.y() ; aP.y()<aP1.y() ; aP.y()++)
        {
            Type aV = tNumTrait<Type>::Trunc(FoncTestIm(aP));
            aIm.SetV(aP,aV);
            aNb++;
            aSomY += aP.y();
        }
    }
    // Test sur le flux itere
    MMVII_INTERNAL_ASSERT_bench(aNbX==(aP1.x()-aP0.x()),"Bench image error");
    MMVII_INTERNAL_ASSERT_bench(aNb==(aP1.x()-aP0.x())*(aP1.y()-aP0.y()),"Bench image error");
    MMVII_INTERNAL_ASSERT_bench(aSomX==SomIntI(aP0.x(),aP1.x()),"Bench image error");
    MMVII_INTERNAL_ASSERT_bench(aSomY==(SomIntI(aP0.y(),aP1.y())*aNbX),"Bench image error");
    // std::cout << aSomY << " " << SomIntI(aP0.y(),aP1.y())*aNbX << "\n";

    double aDif0 = 0.0; // Som of dif on raw function
    double aDifTr = 0.0; // Som on truncated function
    double aSomFonc = 0.0; // Will be used for iterator
    for (cPt2di aP=aP0 ; aP.y()<aP1.y() ; aP.y()++)
    {
        for (aP.x()=aP0.x() ; aP.x()<aP1.x() ; aP.x()++)
        {
            double aV0 = FoncTestIm(aP);
            aSomFonc += aV0;
            Type aVTrunc = tNumTrait<Type>::Trunc(aV0);
            Type aVIm = aIm.GetV(aP);
            aDif0 += std::abs(aV0-aVIm);
            aDifTr += std::abs(aVTrunc-aVIm);
        }
    }
    aDif0 /= aNb;
    aDifTr /= aNb;
    MMVII_INTERNAL_ASSERT_bench(aDifTr<1e-10,"Bench image error");
    if (!tNumTrait<Type>::IsInt())
    {
         MMVII_INTERNAL_ASSERT_bench(aDif0<1e-5,"Bench image error");
    }

    // Test on iterator, we check that we get the same sum on FoncTestIm than with
    // standard loop

    double aSomFonc2 = 0.0;
    double aSomFonc3 = 0.0;
    for (cRectObjIterator<2> aP = aIm.begin() ; aP!=aIm.end() ; aP++)
    {
        aSomFonc2 += FoncTestIm(*aP);
        aSomFonc3 += FoncTestIm(cPt2di(aP->x(),aP->y()));
    }

    double aSomFonc4 = 0.0;
    for (const auto & aP : aIm)
    {
        aSomFonc4 += FoncTestIm(aP);
    }

    MMVII_INTERNAL_ASSERT_bench(std::abs(aSomFonc-aSomFonc2)<1e-10,"Bench image iterator");
    MMVII_INTERNAL_ASSERT_bench(std::abs(aSomFonc-aSomFonc3)<1e-10,"Bench image iterator");
    MMVII_INTERNAL_ASSERT_bench(std::abs(aSomFonc-aSomFonc4)<1e-10,"Bench image iterator");
}

template <class Type> void TestOneImage2D()
{


   TestOneImage2D<Type>(cPt2di(2,3),cPt2di(8,5));
   TestOneImage2D<Type>(cPt2di(5,3),cPt2di(8,5));

   TestOneImage2D<Type>(cPt2di(-2,-3),cPt2di(8,5));
   TestOneImage2D<Type>(cPt2di(-5,-3),cPt2di(8,5));

   TestOneImage2D<Type>(cPt2di(-35,-32),cPt2di(-20,-25));
   TestOneImage2D<Type>(cPt2di(-35,-32),cPt2di(-28,-25));
}

void BenchGlobImage2d()
{
    cMemState  aState = cMemManager::CurState() ;
    {
        TestOneImage2D<tREAL4>(cPt2di(2,2),cPt2di(4,5));

        TestOneImage2D<tREAL4>(cPt2di(0,0),cPt2di(10,10));
        TestOneImage2D<tREAL8>(cPt2di(0,0),cPt2di(10,10));


/*
*/
        TestOneImage2D<tINT1>();
        TestOneImage2D<tINT2>();
        TestOneImage2D<tINT4>();

        TestOneImage2D<tU_INT1>();
        TestOneImage2D<tU_INT2>();
        TestOneImage2D<tU_INT4>();
        TestOneImage2D<tREAL4>();
        TestOneImage2D<tREAL8>();
/*
*/
/*
        TestOneImage2D<tINT2>(cPt2di(2,3),cPt2di(8,5));
        TestOneImage2D<tINT2>(cPt2di(5,3),cPt2di(8,5));

        TestOneImage2D<tINT2>(cPt2di(-2,-3),cPt2di(8,5));
        TestOneImage2D<tINT2>(cPt2di(-5,-3),cPt2di(8,5));

        TestOneImage2D<tINT2>(cPt2di(-35,-32),cPt2di(-20,-25));
        TestOneImage2D<tINT2>(cPt2di(-35,-32),cPt2di(-28,-25));
*/
    }
    cMemManager::CheckRestoration(aState);
}


/*====================== FILE BENCH ===========================================*/
/*====================== FILE BENCH ===========================================*/
/*====================== FILE BENCH ===========================================*/

template <class TypeFile>  void TplBenchReadGlobIm(const std::string & aNameTiff,const cDataIm2D<TypeFile> & aDImDup)
{
   cIm2D<TypeFile> aICheck =  cIm2D<TypeFile>::FromFile(aNameTiff);
   MMVII_INTERNAL_ASSERT_bench(aICheck.Im().Sz()==aDImDup.Sz(),"Bench image error");
   // for (const auto & aP : aFileIm)
   for (const auto & aP : aDImDup)
   {
       TypeFile aV1 = aDImDup.GetV(aP);
       TypeFile aV2 = aICheck.Im().GetV(aP);
       MMVII_INTERNAL_ASSERT_bench(aV1==aV2,"Bench image error");
   }
}


template <class TypeImage,class tBase,class TypeFile>  void TplBenchFileImage(const cPt2di& aSz)
{
    std::string aNameTiff = cMMVII_Appli::TheAppli().TmpDirTestMMVII() + "Test.tif";
    eTyNums aTypeF2 = tElemNumTrait<TypeFile>::TyNum();
    cDataFileIm2D  aFileIm = cDataFileIm2D::Create(aNameTiff,aTypeF2,aSz,1);
    // This image will be a copy of file
    cIm2D<TypeFile> aIDup(aSz);

    
    cDataIm2D<TypeFile> & aDImDup = aIDup.Im();
    for (const auto & aP : aFileIm)
    {
        aDImDup.SetVTrunc(aP,FoncTestIm(aP));
    }
    aIDup.Write(aFileIm,cPt2di(0,0));  // Now File & Im contain the same data


    // Check read image
    TplBenchReadGlobIm(aNameTiff,aDImDup);

    for (int aK=0 ; aK<3 ; aK++)
    {
         cRect2 aR = aFileIm.GenerateRectInside(3.0);
         cPt2di aP0 = aR.P0();
         cIm2D<TypeImage> aIW(aR.Sz());
         aIW.Im().InitRandom();
         for (const auto & aP : aIW.Im())
         {
            aDImDup.SetVTrunc(aP+aP0,aIW.Im().GetV(aP));
         }
         aIW.Write(aFileIm,aP0);
        
    }
    TplBenchReadGlobIm(aNameTiff,aDImDup);
    

    std::cout << "Tiiiiiiiiiffff " << aNameTiff << "\n";
    getchar();
/*
    GenIm::type_el aTypeF1 = ToMMV1(aTypeF2);
    Tiff_Im aTif(aNameTiff.c_str(),ToMMV1(aSz),aTypeF1,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);

*/
/*

     cIm2D<TypeFile> aImV2(cPt2di(0,0),aSz);
     aImV2.Read(cDataFileIm2D::Create(aNameTiff),cPt2di(0,0));
*/
    // Im2D<TypeImage,tBase > aImV1(aSz.x(),aSz.y());
}

template <class TypeFile,class TypeImage>  void TplBenchFileImage()
{
    // TplBenchFileImage<tU_INT1,tINT4,tREAL4>(cPt2di(500,1000));
    TplBenchFileImage<tU_INT1,tINT4,tINT4>(cPt2di(1000,500));
    TplBenchFileImage<tINT4,tINT4,tINT2>(cPt2di(1000,500));
}

void BenchFileImage()
{
    TplBenchFileImage<tU_INT1,tINT4>();
}




};
