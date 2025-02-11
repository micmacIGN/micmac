#ifndef _V1V2_H_
#define _V1V2_H_

#ifndef MMVII_KEEP_LIBRARY_MMV1
// Maintain it for now, require for converting V1 TiePoints...
#define MMVII_KEEP_LIBRARY_MMV1 true
#endif 

#if (MMVII_KEEP_LIBRARY_MMV1)

#include "StdAfx.h"
#include "MMVII_Ptxd.h"
#include "MMVII_Stringifier.h"
#include "MMVII_MMV1Compat.h"



namespace MMVII
{

template <class Type> Pt2d<Type>  ToMMV1(const cPtxd<Type,2> &  aP) {return  Pt2d<Type>(aP.x(),aP.y());}
template <class Type> Pt3d<Type>  ToMMV1(const cPtxd<Type,3> &  aP) {return  Pt3d<Type>(aP.x(),aP.y(),aP.z());}

template <class Type> cPtxd<Type,2> ToMMVII(const Pt2d<Type> &  aP) {return cPtxd<Type,2>(aP.x,aP.y);}
template <class Type> cPtxd<Type,3> ToMMVII(const Pt3d<Type> &  aP) {return cPtxd<Type,3>(aP.x,aP.y,aP.z);}

cHomogCpleIm  ToMMVII(const cNupletPtsHomologues &  aNUp);


template <class Type> Box2d<Type>  ToMMV1(const cTplBox<Type,2> &  aBox) {return  Box2d<Type>(ToMMV1(aBox.P0()),ToMMV1(aBox.P1()));}



void ExportHomMMV1(const std::string & aIm1,const std::string & aIm2,const std::string & SH,const std::vector<cPt2di> & aVP);
void ExportHomMMV1(const std::string & aIm1,const std::string & aIm2,const std::string & SH,const std::vector<cPt2dr> & aVP);

void MakeStdIm8BIts(const cIm2D<tREAL4> &aImIn,const std::string& aName);



//FIXME CM->MPD: Mail MPD fichier xmmlV1.cpp Must replace cMMV1_Conv::ImToMMV1
template <class Type> class cMMV1_Conv
{
    public :
     typedef typename El_CTypeTraits<Type>::tBase   tBase;
     typedef  Im2D<Type,tBase>  tImMMV1;
     typedef  cDataIm2D<Type>       tImMMVII;

     static inline tImMMV1 ImToMMV1(const tImMMVII &  aImV2)  // To avoid conflict with global MMV1
     {
        Type * aDL = const_cast< tImMMVII &> (aImV2).RawDataLin();
        // return   tImMMV1(aImV2.RawDataLin(),nullptr,aImV2.Sz().x(),aImV2.Sz().y());
        return   tImMMV1(aDL,nullptr,aImV2.Sz().x(),aImV2.Sz().y());
     };

#ifdef MMVII_KEEP_MMV1_IMAGE
     // For gray level
     static void ReadWrite(bool ReadMode,const tImMMVII &aImV2,const cDataFileIm2D & aDF,const cPt2di & aP0File,double aDyn,const cRect2& aR2Init);

     // For RGB
     static void ReadWrite(bool ReadMode,const tImMMVII &,const tImMMVII &,const tImMMVII &,const cDataFileIm2D & aDF,const cPt2di & aP0File,double aDyn,const cRect2& aR2Init);
   private :
     // Generik function, real implemantation
     static void ReadWrite(bool ReadMode,const std::vector<const tImMMVII*>& ,const cDataFileIm2D &aDF,const cPt2di & aP0File,double aDyn,const cRect2& aR2Init);
#endif
};

// Call V1 Fast kth value extraction
double KthVal(std::vector<double> &, double aProportion);
// Idem but indicate a number and not a proportion
double  IKthVal(std::vector<double> & aV, int aK);


// Call V1 for roots of polynomials
template <class Type> std::vector<Type>  V1RealRoots(const std::vector<Type> &  aVCoef, Type aTol,int aNbMaxIter);

};

#endif // MMVII_KEEP_LIBRARY_MMV1
#endif // _V1V2_H_
