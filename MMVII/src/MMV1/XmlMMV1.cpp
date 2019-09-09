#include "include/V1VII.h"


#include "src/uti_image/NewRechPH/cParamNewRechPH.h"
#include "../CalcDescriptPCar/AimeTieP.h"

#include "include/im_tpl/cPtOfCorrel.h"


namespace MMVII
{




//=============  tNameRel ====================

void TestTimeV1V2()
{
    for (int aK= 1 ; aK<1000000 ; aK++)
    {
         int aSom=0;
         ElTimer aChronoV1;
         double aT1 = cMMVII_Appli::CurrentAppli().SecFromT0();
         for (int aI=0 ; aI<10000; aI++)
         {
              for (int aJ=0 ; aJ<10000; aJ++)
              {
                  aSom += 1/(1+aI+aJ);
              }
         }
         if (aSom==-1)
            return;
         double aT2 = cMMVII_Appli::CurrentAppli().SecFromT0();

         double aDV1 = aChronoV1.uval();
         double aDV2 = aT2-aT1;

         StdOut()  << "Ratio " << aDV1 / aDV2  << " TimeV1: " << aDV1 << "\n";
    }
}

//=============  tNameRel ====================

tNameRel  MMV1InitRel(const std::string & aName)
{
   tNameRel aRes;
   cSauvegardeNamedRel aSNR = StdGetFromPCP(aName,SauvegardeNamedRel);
   for (const auto & aCpl : aSNR.Cple())
   {
       aRes.Add(tNamePair(aCpl.N1(),aCpl.N2()));
   }

   return aRes;
}

/// Write a rel in MMV1 format
template<> void  MMv1_SaveInFile(const tNameRel & aSet,const std::string & aName)
{
   std::vector<const tNamePair *> aV;
   aSet.PutInVect(aV,true);

   cSauvegardeNamedRel aSNR;
   for (const auto & aPair : aV)
   {
      aSNR.Cple().push_back(cCpleString(aPair->V1(),aPair->V2()));
   }
   MakeFileXML(aSNR,aName);
}

//=============  tNameSet ====================

/// Read MMV1 Set
tNameSet  MMV1InitSet(const std::string & aName)
{
   tNameSet aRes ;
   cListOfName aLON = StdGetFromPCP(aName,ListOfName);
   for (const auto & el : aLON.Name())
       aRes.Add(el);
   return aRes;
}

/// Write a set in MMV1 format
template<> void  MMv1_SaveInFile(const tNameSet & aSet,const std::string & aName)
{
    std::vector<const std::string *> aV;
    aSet.PutInVect(aV,true);

    cListOfName aLON;
    for (const auto & el : aV)
        aLON.Name().push_back(*el);
    MakeFileXML(aLON,aName);
}

/********************************************************/

#define AC_RHO  5.0
#define AC_SZW  3

#define AC_CutInt 0.70      // Less, after integer correl, considered as not self  correl
#define AC_CutReal 0.80     // Less in real correl considered as not self  corre
#define AC_Threshold  0.90  // Threshold, Any value overs this, will be considered as self correl


#define  VAR_RHO  6.0

/// Class implementing services promized by cInterf_ExportAimeTiep

/**  This class use MMV1 libray to implement service described in cInterf_ExportAimeTiep
*/
template <class Type> class cImplem_ExportAimeTiep : public cInterf_ExportAimeTiep<Type>
{
     public :
         cImplem_ExportAimeTiep(bool IsMin,int ATypePt,const std::string & aName,bool ForInspect);
         virtual ~cImplem_ExportAimeTiep();

         void AddAimeTieP(const cProtoAimeTieP & aPATP ) override;
         void Export(const std::string &) override;
         void SetCurImages(cIm2D<Type>,cIm2D<Type>,double aScaleInO) override;

     private :
          typedef typename tElemNumTrait<Type>::tBase tBase;
          typedef Im2D<Type,tBase>           tImV1;
          typedef TIm2D<Type,tBase>          tTImV1;
          typedef cCutAutoCorrelDir<tTImV1>  tCACD;
          typedef std::unique_ptr<cFastCriterCompute> tFCC;
 
          cXml2007FilePt  mPtsXml;
          cIm2D<Type>     mIm0V2;  ///< Image "init",  MMVII version
          tImV1           mIm0V1;  ///<  Image "init", MMV1 Version
          tTImV1          mTIm0V1; ///< T Image V1
          cIm2D<Type>     mImStdV2;  ///<  Imagge Std eq Lapl, Corner ...
          tImV1           mImStdV1;
          tTImV1          mTImStdV1;
          std::shared_ptr<tCACD> mCACD;
          tFCC                   mFCC;
          bool                   mForInspect;
          
};
/* ================================= */
/*    cInterf_ExportAimeTiep         */
/* ================================= */

template <class Type> cInterf_ExportAimeTiep<Type>::~cInterf_ExportAimeTiep()
{
}

template <class Type> cInterf_ExportAimeTiep<Type> * cInterf_ExportAimeTiep<Type>::Alloc(bool IsMin,int ATypePt,const std::string & aName,bool ForInspect)
{
    return new cImplem_ExportAimeTiep<Type>(IsMin,ATypePt,aName,ForInspect);
}



/* ================================= */
/*    cImplem_ExportAimeTiep         */
/* ================================= */

template <class Type> cImplem_ExportAimeTiep<Type>::cImplem_ExportAimeTiep(bool IsMin,int ATypePt,const std::string & aNameType,bool ForInspect) :
    mIm0V2    (cPt2di(1,1)),
    mIm0V1    (1,1),
    mTIm0V1   (mIm0V1),
    mImStdV2  (cPt2di(1,1)),
    mImStdV1  (1,1),
    mTImStdV1 (mImStdV1),
    mCACD  (nullptr),
    mFCC   (cFastCriterCompute::Circle(3.0)),
    mForInspect (ForInspect)
{
    mPtsXml.IsMin() = IsMin;
    mPtsXml.TypePt() = IsMin;
    mPtsXml.NameTypePt() = aNameType;
    
}
template <class Type> cImplem_ExportAimeTiep<Type>::~cImplem_ExportAimeTiep()
{
}

template <class Type> void cImplem_ExportAimeTiep<Type>::AddAimeTieP(const cProtoAimeTieP & aPATP)
{

    Pt2di  aPIm = round_ni(ToMMV1(aPATP.Pt()) / double(1<<aPATP.NumOct()));
    bool  aAutoCor = mCACD->AutoCorrel(aPIm,AC_CutInt,AC_CutReal,AC_Threshold);
    
    cXml2007Pt aPXml;

/*
static int aNbOut = 0;
static int aNbIn = 0;
if (mCACD->mCorOut==-1)
   aNbOut ++;
else
   aNbIn++;
StdOut() << "PropOut= " << aNbOut / double(aNbIn+aNbOut) << "\n";
*/

    aPXml.Pt() = ToMMV1(aPATP.Pt());
    aPXml.NumOct() = aPATP.NumOct();
    aPXml.NumIm() = aPATP.NumIm();
    aPXml.ScaleInO() = aPATP.ScaleInO();
    aPXml.ScaleAbs() = aPATP.ScaleAbs();
    aPXml.OKAc() = ! aAutoCor;
    aPXml.AutoCor() = mCACD->mCorOut;
    aPXml.NumChAC() = mCACD->mNumOut;
     

    if (aPXml.OKAc() || mForInspect)
    {
        Pt2dr aFQ =  FastQuality(mTImStdV1,aPIm,*mFCC,! mPtsXml.IsMin() ,Pt2dr(0.75,0.85));
        aPXml.FastStd() = aFQ.x;
        aPXml.FastConx() = aFQ.y;
        aPXml.Var() = CubGaussWeightStandardDev(mIm0V2.DIm(),ToI(aPATP.Pt()),aPATP.ScaleInO()*VAR_RHO);
        
        mPtsXml.Pts().push_back(aPXml);
    }
}
template <class Type> void cImplem_ExportAimeTiep<Type>::Export(const std::string & aName)
{
     MakeFileXML(mPtsXml,aName);
}

template <class Type> void cImplem_ExportAimeTiep<Type>::SetCurImages(cIm2D<Type> anIm0,cIm2D<Type> anImStd,double aScaleInO) 
{
   mIm0V2 = anIm0;
   mIm0V1 = cMMV1_Conv<Type>::ImToMMV1(mIm0V2.DIm());
   mTIm0V1 =  tTImV1(mIm0V1);

   int aSzW = round_ni(AC_SZW*aScaleInO);
   double aFact  = aSzW / double(AC_SZW);
   double aRho = AC_RHO * aFact;


   mCACD = std::shared_ptr<tCACD>(new tCACD(mTIm0V1,Pt2di(0,0),aRho,aSzW));

   mImStdV2 = anImStd;
   mImStdV1 = cMMV1_Conv<Type>::ImToMMV1(mImStdV2.DIm());
   mTImStdV1 =  tTImV1(mImStdV1);


}

template class cInterf_ExportAimeTiep<tREAL4>;
template class cInterf_ExportAimeTiep<tINT2>;
template class cImplem_ExportAimeTiep<tREAL4>;
template class cImplem_ExportAimeTiep<tINT2>;


};
