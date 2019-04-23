#include "include/V1VII.h"


namespace MMVII
{

/* =========================== */
/*       cDataFileIm2D         */
/* =========================== */

cDataFileIm2D::cDataFileIm2D(const std::string & aName,eTyNums aType,const cPt2di & aSz,int aNbChannel) :
    cRectObj<2> (cPt2di(0,0),aSz),
    mName       (aName),
    mType       (aType),
    mNbChannel  (aNbChannel)
{
}

cDataFileIm2D cDataFileIm2D::Create(const std::string & aName)
{
   Tiff_Im aTF=Tiff_Im::StdConvGen(aName,-1,true);

   return cDataFileIm2D(aName,ToMMVII(aTF.type_el()),ToMMVII(aTF.sz()), aTF.nb_chan());
}

cDataFileIm2D  cDataFileIm2D::Create(const std::string & aName,eTyNums aType,const cPt2di & aSz,int aNbChan)
{
   Tiff_Im::PH_INTER_TYPE aPIT = Tiff_Im::BlackIsZero;
   if (aNbChan==1) 
      aPIT = Tiff_Im::BlackIsZero;
   else if (aNbChan==3) 
      aPIT = Tiff_Im::RGB;
   else 
   {
      MMVII_INTERNAL_ASSERT_strong(false,"Incoherent channel number");
   }
   Tiff_Im
   (
      aName.c_str(),
      ToMMV1(aSz),
      ToMMV1(aType),
      Tiff_Im::No_Compr,
      aPIT
   );
   return Create(aName);
}


// eTyNums ToMMVII( GenIm::type_el aV1 )
cDataFileIm2D::~cDataFileIm2D()
{
}

const cPt2di &  cDataFileIm2D::Sz() const  {return  cRectObj<2>::Sz();}
const std::string &  cDataFileIm2D::Name() const { return mName; }
const int  & cDataFileIm2D::NbChannel ()  const { return mNbChannel; }





/********************************************************/
/********************************************************/
/********************************************************/
/********************************************************/


// template <class Type> Pt2d<Type> ToMMV1(const cPtxd<Type,2> &

template <class Type> class cMMV1_Conv
{
    public :
     typedef typename El_CTypeTraits<Type>::tBase   tBase;
     typedef  Im2D<Type,tBase>  tImMMV1;
     typedef  cDataIm2D<Type>       tImMMVII;

     static tImMMV1 ImToMMV1(tImMMVII &  aImV2)  // To avoid conflict with global MMV1
     {
        return   tImMMV1(aImV2.DataLin(),nullptr,aImV2.Sz().x(),aImV2.Sz().y());
     };

      
     static void ReadWrite(bool ReadMode,tImMMVII &aImV2,const cDataFileIm2D & aDF,const cPt2di & aP0File,double aDyn,const cRect2& aR2Init)
     {
          // C'est une image en originie (0,0) necessairement en MMV1
          tImMMV1 aImV1 = ImToMMV1(aImV2);
          cRect2 aRectFullIm (cPt2di(0,0),aImV2.Sz());

          // Rectangle image / a un origine (0,0)
          cRect2 aRectIm =  (aR2Init== cRect2::Empty00)           ?  // Val par def
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

// std::cout << "GGGGGG " << aP0Im << " " << aP1Im << " " << aTrans << "\n";

          if (ReadMode)
          {
             ELISE_COPY
             (
                  rectangle(aP0Im,aP1Im),
                  trans(El_CTypeTraits<Type>::TronqueF(aTF.in()*aDyn),aTrans),
                  aImV1.out()
             );
          }
          else
          {
             ELISE_COPY
             (
                  rectangle(aP0Im+aTrans,aP1Im+aTrans),
                  trans(Tronque(aTF.type_el(),aImV1.in()*aDyn),-aTrans),
                  aTF.out()
             );
/*
*/
          }
     }
};


template <class Type>  void  cDataIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cRectObj<2>& aR2)
{
     cMMV1_Conv<Type>::ReadWrite(true,*this,aFile,aP0,aDyn,aR2);
}
template <class Type>  void  cDataIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cRectObj<2>& aR2)
{
     cMMV1_Conv<Type>::ReadWrite(false,*this,aFile,aP0,aDyn,aR2);
}


//  It's difficult to read unsigned int4 with micmac V1, wait for final implementation
template <>  void  cDataIm2D<tU_INT4>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cRectObj<2>& aR2)
{
   MMVII_INTERNAL_ASSERT_strong(false,"No read for unsigned int4 now");
}
template <>  void  cDataIm2D<tU_INT4>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cRectObj<2>& aR2)
{
   MMVII_INTERNAL_ASSERT_strong(false,"No write for unsigned int4 now");
}

template <class Type>  void  cIm2D<Type>::Read(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cRectObj<2>& aR2)
{
      Im().Read(aFile,aP0,aDyn,aR2);
}
template <class Type>  void  cIm2D<Type>::Write(const cDataFileIm2D & aFile,const cPt2di & aP0,double aDyn,const cRectObj<2>& aR2)
{
     Im().Write(aFile,aP0,aDyn,aR2);
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
    MMVII_INTERNAL_ASSERT_bench(false,"GenIm::type_el ToMMV1(eTyNums)");
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
    MMVII_INTERNAL_ASSERT_bench(false,"eTyNums ToMMVII( GenIm::type_el aV1 )");
    return eTyNums::eTN_INT1 ;
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
};
