#include "MMVII_Image2D.h"
#include "MMVII_TplSimpleOperator.h"

namespace MMVII
{


void BenchFilterImage_000()
{
    // StdOut() << "BenchFilterImage1BenchFilterImage1BenchFilterImage1\n";
    cPt2di  aSzInd(6,9);
    cPt2di  aSz10 = aSzInd * 10;
    cPt2di  aSz = aSz10 - cPt2di(1,1);
    cRect2  aRCrop(cPt2di(20,22),cPt2di(50,62));
    cRect2  aRGlob(cPt2di(0,0),aSz);

    // Test of a copy image to image
    {
        cIm2D<tINT2> aI1(aSz,nullptr,eModeInitImage::eMIA_Rand);
        cIm2D<tINT1> aI2(aSz,nullptr,eModeInitImage::eMIA_Rand);
        cIm2D<tINT2> aIDest = aI1.Dup();

        TplCopy(aRCrop,aI2.DIm(),aIDest.DIm());
        for (const auto & aP : aRGlob)
        {
            tINT2  aVDest = aIDest.DIm().GetV(aP);
            tINT2  aVCmp  = (aRCrop.Inside(aP) ? aI2.DIm().GetV(aP) : aIDest.DIm().GetV(aP));
            MMVII_INTERNAL_ASSERT_bench(aVDest==aVCmp,"Bench TplCopy");
        }
     }
    // Test function coordinate +  Som neighboor + proj + interiority
     {
        cIm2D<tINT4> aIx(aSz,nullptr,eModeInitImage::eMIA_Rand);
        TplCopy(aRGlob,fCoord(0),aIx.DIm());  // aI1 contains x

        cIm2D<tINT4> aIy(aSz,nullptr,eModeInitImage::eMIA_Rand);
        TplCopy(aRGlob,fCoord(1),aIy.DIm());  // aIy contains y

        cPt2di aVign(2,3);
        cIm2D<tINT4> aISomX(aSz,nullptr,eModeInitImage::eMIA_Rand);
        TplCopy(aRGlob,fSomVign(fCoord(0),aVign),aISomX.DIm());  // aI2 contains som(x,aVign)

        cIm2D<tINT4> aISomImX(aSz,nullptr,eModeInitImage::eMIA_Rand);
        TplCopy(aRGlob,fSomVign(fProj(aIx.DIm()),aVign),aISomImX.DIm());  // aI2 contains som(Ix,aVign)

        cIm2D<tINT4> aISomImY(aSz,nullptr,eModeInitImage::eMIA_Rand);
        TplCopy(aRGlob,fSomVign(fProj(aIy.DIm()),aVign),aISomImY.DIm());  // aI2 contains som(Ix,aVign)

        for (const auto & aP : aRGlob)
        {
            int aVx = aP.x();
            int aVy = aP.y();
            int aVxI = aIx.DIm().GetV(aP);
            int aVyI = aIy.DIm().GetV(aP);
            int aVSomX = aISomX.DIm().GetV(aP);
            int aVSomIX = aISomImX.DIm().GetV(aP);

            MMVII_INTERNAL_ASSERT_bench(aVx==aVxI,"Bench TplCopy");
            MMVII_INTERNAL_ASSERT_bench(aVy==aVyI,"Bench TplCopy");
            MMVII_INTERNAL_ASSERT_bench(aVSomX==aVx*NbPixVign(aVign),"Bench TplCopy SomVign");

            int aInterX = aRGlob.WinInteriority(aP,aVign,0);
            if (aInterX>=0)
            {
               MMVII_INTERNAL_ASSERT_bench(aVSomX==aVSomIX,"Bench TplCopy");
            }
            else
            {
               int aD1= std::abs(aVSomX-aVSomIX);
               int aD2= BinomialCoeff(2,1+std::abs(aInterX)) * NbPixVign(aVign.y());
               MMVII_INTERNAL_ASSERT_bench(aD1==aD2,"Bench TplCopy");
            }

            int aVSomY = aP.y() * NbPixVign(aVign);
            int aVSomIY = aISomImY.DIm().GetV(aP);
            int aInterY = aRGlob.WinInteriority(aP,aVign,1);
            if (aInterY>=0)
            {
                MMVII_INTERNAL_ASSERT_bench(aVSomY==aVSomIY,"Bench TplCopy");
            }
            else
            {
               int aD1= std::abs(aVSomY-aVSomIY);
               int aD2= BinomialCoeff(2,1+std::abs(aInterY)) * NbPixVign(aVign.x());
               MMVII_INTERNAL_ASSERT_bench(aD1==aD2,"Bench TplCopy");
            }
        }

     // Test cParseBoxInOut
        cParseBoxInOut<2> aPBIO = cParseBoxInOut<2>::CreateFromSzMem(aRGlob,1e2);
        MMVII_INTERNAL_ASSERT_bench(aPBIO.BoxIndex().Sz() ==aSzInd,"cParseBoxInOut sz");
        cIm2D<tINT4> aISomImXParse(aSz,nullptr,eModeInitImage::eMIA_Rand);
        int aNbTot = 0;
        for (const auto & aP : aPBIO.BoxIndex())
        {

            cPixBox<2> aBI = aPBIO.BoxIn(aP,aVign);
            cPixBox<2> aBO = aPBIO.BoxOut(aP);
            aNbTot += aBO.NbElem();

            cIm2D<tINT4> aBufIn(aBI.Sz(),nullptr,eModeInitImage::eMIA_Rand);
            cIm2D<tINT4> aBufOut(aBI.Sz(),nullptr,eModeInitImage::eMIA_Rand);

            // Transfer aIx in aBufIn
            TplCopy
            (
                aBI,
                aIx.DIm(),
                oTrans(aBufIn.DIm(),-aBI.P0())
            ); 
            // Process image in aBufOut
            TplCopy
            (
                aBufIn.DIm(),
                fSomVign(fProj(aBufIn.DIm()),aVign),
                aBufOut.DIm()
            );
            TplCopy
            (
                aBO,
                fTrans(aBufOut.DIm(),-aBI.P0()),
                aISomImXParse.DIm()
            ); 
        }
        {
           double aDCheck= aISomImXParse.DIm().L2Dist(aISomImX.DIm());
           MMVII_INTERNAL_ASSERT_bench(aDCheck<1e-5,"Bench ParseBox");
           MMVII_INTERNAL_ASSERT_bench(aNbTot==aSz.x()*aSz.y(),"Bench ParseBox");
        }
     }
     // Op Bin + Decimate
     {
        cIm2D<tINT2> aI1(aSz,nullptr,eModeInitImage::eMIA_Rand);
        TplCopy (aI1.DIm(),fSum(fCoord(0),fCoord(1)),aI1.DIm());
        for (const auto & aP: aI1.DIm())
        {
           MMVII_INTERNAL_ASSERT_bench((aP.x()+aP.y())==aI1.DIm().GetV(aP),"Bench fSum");
        }
        int aFact = 3;
        cIm2D<tINT2> aI2 = aI1.Decimate(aFact);
        MMVII_INTERNAL_ASSERT_bench(aI2.DIm().Sz()==aSz/aFact,"Bench Decimate");
        for (const auto & aP: aI2.DIm())
        {
           MMVII_INTERNAL_ASSERT_bench((aP.x()+aP.y())*aFact==aI2.DIm().GetV(aP),"Bench fSum");
        }
     }
     {
        for (int aK=1 ; aK<3 ; aK++)
        {
             cIm2D<tINT2> aIm(aSz,nullptr,eModeInitImage::eMIA_Null);
             TplCopy(aIm.DIm().Border(aK),fCste(aK*10),aIm.DIm());
             for (const auto & aP : aIm.DIm())
             {
                  int aV1 = aIm.DIm().GetV(aP);
                  int aV2 = (aIm.DIm().Interiority(aP)<aK) ? (aK*10) : 0;
                  MMVII_INTERNAL_ASSERT_bench(aV1==aV2,"Bench Border");
             }
        }
     }
   
}

void BenchFilterImage1(cParamExeBench & aParam)
{
     if (! aParam.NewBench("ImageFilter1")) return;

     BenchFilterImage_000();

     aParam.EndBench();
}



};
