#define WITH_MMV1_BENCH false

#if (WITH_MMV1_BENCH)
#include "V1VII.h"
#endif 

#include "MMVII_Matrix.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_NonLinear2DFiltering.h"

// FIXME CM->MPD: Mail a MPD ! Qu'est ce qu'on fait ici pour remplacer ELISE_COPY ?

/** \file ImageFilterMMV1.cpp
    \brief file for using MMV1 filters

    MMV1 has a very powerfull and elegant image processing toolbox (maybe I am not 100% objective ;-)
    MMVII will probably not have this kind of library (at leats in first version) as it  quite
    complicated to maintain and understand. By the way,  for many filter as long as I do not know
    exactly what I want, it's much faster to implement them with MMV1. 
*/


namespace MMVII
{

void Bench_LabMaj(const cPt2di & aSz,const cBox2di & aBox,int aLabMin,int aLabMax)
{
    cIm2D<tINT2>  aILab(aSz);
    cDataIm2D<tINT2>& aDILab = aILab.DIm();

    for (const auto & aPt : aDILab)
       aDILab.SetV(aPt,RandUnif_M_N(aLabMin,aLabMax));

    cIm2D<tINT2> aV2Maj = LabMaj(aILab,aBox);
    cDataIm2D<tINT2>&  aDV2Maj  = aV2Maj.DIm();

    for (const auto & aPt : aDV2Maj)
    {
        std::vector<int> aHisto(aLabMax-aLabMin+1,0);
        for (const auto & aV : cRect2(aBox))
        {
            aHisto.at(aDILab.GetV(aDILab.Proj(aPt+aV))-aLabMin)++;
        }
        int aNb = aHisto.at(aDV2Maj.GetV(aPt)-aLabMin);
        for (const auto & aElem : aHisto)
            MMVII_INTERNAL_ASSERT_bench( aElem<=aNb,"Bench_LabMaj");
    }

}
void Bench_LabMaj()
{
    for (int aK=0 ; aK<100 ; aK++)
    {
        cPt2di aSz(RandUnif_M_N(20,30),RandUnif_M_N(20,30));
        int aVMin = RandUnif_M_N(-5,2);
        int aVMax = aVMin+1+RandUnif_N(7);
        cPt2di aP0(-RandUnif_N(3),-RandUnif_N(3));
        cPt2di aP1 = aP0 + cPt2di(1+RandUnif_N(6),1+RandUnif_N(6));
        Bench_LabMaj(aSz,cBox2di(aP0,aP1),aVMin,aVMax);
    }
}


#if (WITH_MMV1_BENCH)


// Not implemanted/used in MMVII 4 now
void MMV1_MakeImageDist(cIm2D<tU_INT1> aImIn,const std::string & aNameChamfer)
{
     const Chamfer & aChamf = Chamfer::ChamferFromName(aNameChamfer);
     aChamf.im_dist(cMMV1_Conv<tU_INT1>::ImToMMV1(aImIn.DIm())) ;
}
void ExportHomMMV1(const std::string & aIm1,const std::string & aIm2,const std::string & SH,const std::vector<cPt2dr> & aVP)
{
    ElPackHomologue aPackH;

    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
        Pt2dr aP = ToMMV1(aVP[aK]);
        ElCplePtsHomologues aCple(aP,aP,1.0);
        aPackH.Cple_Add(aCple);
    }

    std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ SH + "@dat";
    cInterfChantierNameManipulateur* anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");

    std::string aNameF = anICNM->Assoc1To2(aKeyH,aIm1,aIm2,true);

    aPackH.StdPutInFile(aNameF);

}
void ExportHomMMV1(const std::string & aIm1,const std::string & aIm2,const std::string & SH,const std::vector<cPt2di> & aVI)
{
    std::vector<cPt2dr> aVR;
    for (const auto & aPI: aVI)
       aVR.push_back(ToR(aPI));
    ExportHomMMV1(aIm1,aIm2,SH,aVR);
}
 

    //=================  V1 Solution ==================

template<class Type> cIm2D<Type> MMV1_CourbTgt(cIm2D<Type> aImIn)
{
    cIm2D<Type> aRes(aImIn.DIm().Sz());

    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    auto  aV1Res = cMMV1_Conv<Type>::ImToMMV1( aRes.DIm());

    ELISE_COPY(aV1In.all_pts(),courb_tgt(aV1In.in_proj(),0.5),aV1Res.out());

    return aRes;
}
template<class Type> void MMV1_SelfCourbTgt(cIm2D<Type> aImIn)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    ELISE_COPY(aV1In.all_pts(),courb_tgt(aV1In.in_proj(),0.5),aV1In.out());
}

cIm2D<tREAL4> MMV1_Lapl(cIm2D<tREAL4> aImIn)
{
    cIm2D<tREAL4> aRes(aImIn.DIm().Sz());

    Im2D<tREAL4,tREAL8>  aV1In  = cMMV1_Conv<tREAL4>::ImToMMV1(aImIn.DIm());
    Im2D<tREAL4,tREAL8>  aV1Res = cMMV1_Conv<tREAL4>::ImToMMV1( aRes.DIm());

    ELISE_COPY(aV1In.all_pts(),Laplacien(aV1In.in_proj()),aV1Res.out());

    return aRes;
}


    //=================  Global Solution ==================


template<class Type> void  BenchImFilterV1V2(cIm2D<Type> anI1,cIm2D<Type> anI2,tREAL8 aEps)
{

    tREAL8  aRD =   anI1.DIm().SafeMaxRelDif(anI2.DIm(),1e-2);
    if ( aRD > aEps )
    {
        MMVII_INTERNAL_ASSERT_bench( false,"BenchImFilterV1V2 RD="+ToStr(aRD));
    }
  
}

template <class Type> std::pair<Type,Type>   MMV1_ValExtre(cIm2D<Type> aImIn)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    double aVMin,aVMax;
    ELISE_COPY(aV1In.all_pts(),aV1In.in(),VMin(aVMin)|VMax(aVMax));
    return std::pair<Type,Type>((Type)aVMin,(Type)aVMax);
}

template<class Type> double  MMV1_MoyAbs(cIm2D<Type> aImIn)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    double aSom[2];
    ELISE_COPY(aV1In.all_pts(),Virgule(Abs(aV1In.in()),1),sigma(aSom,2));
    return aSom[0] / aSom[1];
}

template<class Type> void MMV1_ComputeDeriche(cImGrad<Type> & aResGrad,const cDataIm2D<Type> & aImIn,double aAlpha)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn);
    auto  aV1Gx  = cMMV1_Conv<Type>::ImToMMV1(aResGrad.mGx.DIm());
    auto  aV1Gy  = cMMV1_Conv<Type>::ImToMMV1(aResGrad.mGy.DIm());

    ELISE_COPY
    (
          aV1In.all_pts(),
          deriche(aV1In.in_proj(),aAlpha,10),
	  Virgule(aV1Gx.out(),aV1Gy.out())
    );
}
template<class Type> cImGrad<Type> MMV1_Deriche(const cDataIm2D<Type> & aImIn,double aAlpha)
{
    cImGrad<Type> aResGrad(aImIn.Sz());
    MMV1_ComputeDeriche(aResGrad,aImIn,aAlpha);
    return aResGrad;
}

void MMV1_MakeStdIm8BIts(cIm2D<tREAL4> aImIn,const std::string& aName)
{
    Im2D<tREAL4,tREAL8>  aV1In  = cMMV1_Conv<tREAL4>::ImToMMV1(aImIn.DIm());
    double aS0,aS1,aS2;
    ELISE_COPY
    (
        aV1In.all_pts(),
        Virgule(1,aV1In.in(),Square(aV1In.in())),
        Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
    );
    aS1 /= aS0;
    aS2 /= aS0;
    aS2 -= ElSquare(aS1);
    aS2 = sqrt(aS2);
    Fonc_Num aF = (aV1In.in()-aS1) / aS2;
    Tiff_Im::Create8BFromFonc
    (
         aName,
         aV1In.sz(),
         El_CTypeTraits<U_INT1>::TronqueF(256* erfcc(aF))
    );
}



template <class Type> void  Tpl_BenchImFilterV1V2(const cPt2di& aSz)
{
   cIm2D<Type>  aI1(aSz,nullptr,eModeInitImage::eMIA_Rand);
   tREAL8 aSigma = 5;
   aI1.DIm().InitDirac(aSz/2, Square(aSigma) * 10000);
   aI1=aI1.GaussFilter(aSigma,20);  // filter image, because corner on a pure random is too noisy

   BenchImFilterV1V2(MMV1_Lapl(aI1),Lapl(aI1),2e-3);

   cIm2D<Type> aCV1 = MMV1_CourbTgt(aI1);
   SelfCourbTgt(aI1);
   BenchImFilterV1V2(aCV1,aI1,1e-3);
}

template <class Type> void  Tpl_BenchDericheV1V2(const cPt2di& aSz,tREAL8 aAlpha)
{
   cIm2D<Type>  aI1(aSz,nullptr,eModeInitImage::eMIA_Rand);

   // we dont make check on diff V1/V2, because it's difficult and as its purely a visual tool
   // not important
   if (0)
   {
        MakeStdIm8BIts(aI1,"Grad-Norm-V2.tif");
        MMV1_MakeStdIm8BIts(aI1,"Grad-NormV1.tif");
   }

   tREAL8 aSigma = 5;
   aI1.DIm().InitDirac(aSz/2, Square(aSigma) * 10000);
   aI1=aI1.GaussFilter(aSigma,20);  // filter image, because corner on a pure random is too noisy

   cImGrad<Type> aDerV2 = Deriche(aI1.DIm(),aAlpha);
   cImGrad<Type> aDerV1 = MMV1_Deriche(aI1.DIm(),aAlpha); 

   MMVII_INTERNAL_ASSERT_bench(aDerV2.mDGx->LInfDist(*(aDerV1.mDGx))<1e-2,"BenchDericheVBenchDericheV");
   MMVII_INTERNAL_ASSERT_bench(aDerV2.mDGy->LInfDist(*(aDerV1.mDGy))<1e-2,"BenchDericheVBenchDericheV");

   // a Test to justify the  IGX[i*DX + j] /= p.mAmpl  in deriche_ll
   if (0)
   {
      aDerV2.mDGx->ToFile("GradX-V2.tif");
      aDerV2.mDGy->ToFile("GradY-V2.tif");
      aDerV1.mDGx->ToFile("GradX-V1.tif");
      aDerV1.mDGy->ToFile("GradY-V1.tif");
      aI1.DIm().ToFile("GradOri.tif");

      cPt2di aP= aSz/2 - cPt2di(1*aSigma,0);
      tREAL8 aDif = (aI1.DIm().GetV(aP+cPt2di(1,0)) -aI1.DIm().GetV(aP+cPt2di(-1,0))) / 2.0;
      tREAL8 aGx = aDerV2.mDGx->GetV(aP);
      StdOut() <<  " GGGGG " << aDif  << " " << aGx << "\n";
   }
}

template <class Type> void  Tpl_BenchInfoExtr_V1V2(const cPt2di& aSz)
{
   cIm2D<Type>  aI1(aSz,nullptr,eModeInitImage::eMIA_Rand);
   auto [aV1Min,aV1Max] =  MMV1_ValExtre(aI1);
   auto [aV2Min,aV2Max] =  ValExtre(aI1);

   MMVII_INTERNAL_ASSERT_bench(aV1Min == aV2Min,"Tpl_BenchInfoExtr VMin");
   MMVII_INTERNAL_ASSERT_bench(aV1Max == aV2Max,"Tpl_BenchInfoExtr VMax");

   MMVII_INTERNAL_ASSERT_bench(std::fabs(MMV1_MoyAbs(aI1) - MoyAbs(aI1))<=1e-8 ,"Tpl_BenchInfoExtr VMax");
}

void  BenchImFilterV1V2()
{
    Tpl_BenchImFilterV1V2<tREAL4>(cPt2di(100,110));

    Tpl_BenchInfoExtr_V1V2<tREAL4>(cPt2di(10,8));
    Tpl_BenchInfoExtr_V1V2<tINT2>(cPt2di(10,12));

    Bench_LabMaj();

    //Tpl_BenchDericheV1V2<tREAL4>(cPt2di(100,120),2.0);
    //Tpl_BenchDericheV1V2<tREAL4>(cPt2di(140,120),1.0);
    Tpl_BenchDericheV1V2<tREAL4>(cPt2di(240,220),2.0);
}

#else
void  BenchImFilterV1V2()
{
    Bench_LabMaj();
}
#endif


//==============================


};
