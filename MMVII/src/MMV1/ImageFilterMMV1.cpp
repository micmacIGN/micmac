#include "include/V1VII.h"

/** \file ImageFilterMMV1.cpp
    \brief file for using MMV1 filters

    MMV1 has a very powerfull and elegant image processing toolbox (maybe I am not 100% objective ;-)
    MMVII will probably not have this kind of library (at leats in first version) as it  quite
    complicated to maintain and understand. By the way,  for many filter as long as I do not know
    exactly what I want, it's much faster to implement them with MMV1. 
*/

// Test git

namespace MMVII
{

template<class Type> cIm2D<Type> CourbTgt(cIm2D<Type> aImIn)
{
    cIm2D<Type> aRes(aImIn.DIm().Sz());

    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    auto  aV1Res = cMMV1_Conv<Type>::ImToMMV1( aRes.DIm());

    ELISE_COPY(aV1In.all_pts(),courb_tgt(aV1In.in_proj(),0.5),aV1Res.out());

    return aRes;
}


template<class Type> void SelfCourbTgt(cIm2D<Type> aImIn)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    ELISE_COPY(aV1In.all_pts(),courb_tgt(aV1In.in_proj(),0.5),aV1In.out());
}

template<class Type> double  MoyAbs(cIm2D<Type> aImIn)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    double aSom[2];
    ELISE_COPY(aV1In.all_pts(),Virgule(Abs(aV1In.in()),1),sigma(aSom,2));
    return aSom[0] / aSom[1];
}


template<class Type> cImGrad<Type> Deriche(const cDataIm2D<Type> & aImIn,double aAlpha)
{
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn);

    cImGrad<Type> aResGrad(aImIn.Sz());
    auto  aV1Gx  = cMMV1_Conv<Type>::ImToMMV1(aResGrad.mGx.DIm());
    auto  aV1Gy  = cMMV1_Conv<Type>::ImToMMV1(aResGrad.mGy.DIm());

    ELISE_COPY
    (
          aV1In.all_pts(),
          deriche(aV1In.in_proj(),aAlpha,10),
	  Virgule(aV1Gx.out(),aV1Gy.out())
    );

    return aResGrad;
}


#define INSTANTIATE_TRAIT_AIME(TYPE)\
template cImGrad<TYPE> Deriche(const cDataIm2D<TYPE> &aImIn,double aAlpha);\
template double  MoyAbs(cIm2D<TYPE> aImIn);\
template cIm2D<TYPE> CourbTgt(cIm2D<TYPE> aImIn);\
template void SelfCourbTgt(cIm2D<TYPE> aImIn);

INSTANTIATE_TRAIT_AIME(tREAL4)
INSTANTIATE_TRAIT_AIME(tINT2)


cIm2D<tREAL4> Lapl(cIm2D<tREAL4> aImIn)
{
    cIm2D<tREAL4> aRes(aImIn.DIm().Sz());

    Im2D<tREAL4,tREAL8>  aV1In  = cMMV1_Conv<tREAL4>::ImToMMV1(aImIn.DIm());
    Im2D<tREAL4,tREAL8>  aV1Res = cMMV1_Conv<tREAL4>::ImToMMV1( aRes.DIm());

    ELISE_COPY(aV1In.all_pts(),Laplacien(aV1In.in_proj()),aV1Res.out());

    return aRes;
}

void MakeStdIm8BIts(cIm2D<tREAL4> aImIn,const std::string& aName)
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



template <class Type> cPt2dr   ValExtre(cIm2D<Type> aImIn)
{
    // cDataIm2D aDIm(aImIn.DIm());
    // Im2D<aDIm::tVal,aDIm::tBase>  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    double aVMin,aVMax;
    ELISE_COPY(aV1In.all_pts(),aV1In.in(),VMin(aVMin)|VMax(aVMax));
    return cPt2dr(aVMin,aVMax);
}


template <class Type> void SelfLabMaj(cIm2D<Type> aImIn,const cBox2di &  aBox)
{
    cPt2dr aVE = ValExtre(aImIn);
    int aMin = round_ni(aVE.x());
    int aMax = round_ni(aVE.y());
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn.DIm());
    ELISE_COPY
    (
        aV1In.all_pts(),
        aMin+label_maj(aV1In.in_proj()-aMin,1+aMax-aMin,ToMMV1(aBox)),
        aV1In.out()
    );
}


template cPt2dr ValExtre(cIm2D<tINT2> aImIn);
template void SelfLabMaj(cIm2D<tINT2> aImIn,const cBox2di &  aBox);

//==============================

void MakeImageDist(cIm2D<tU_INT1> aImIn,const std::string & aNameChamfer)
{
     const Chamfer & aChamf = Chamfer::ChamferFromName(aNameChamfer);
     aChamf.im_dist(cMMV1_Conv<tU_INT1>::ImToMMV1(aImIn.DIm())) ;
}




//==============================

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
 

};
