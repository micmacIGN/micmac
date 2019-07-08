#ifndef  _MMVII_Tpl_TplSimpleOperator_H_
#define  _MMVII_Tpl_TplSimpleOperator_H_
namespace MMVII
{


/** \file  MMVII_TplSimpleOperator.h
    \brief    Very basic and unoptimzed filter, may be usefull to test other functionnality
              Maybe used as a "bac a sable" 4 further implementation
*/


     //  ===  Image value safe  =====
template <class TypeIm> class cImInputProj
{
    public :
       typedef typename TypeIm::tBase  tBase;
       // typedef typename TypeIm::tVal   tBase;

       cImInputProj(const TypeIm & aI2) :
          mI2(aI2)
       {  
       }
       tBase  GetV(const cPt2di & aP) const {return mI2.GetV(mI2.Proj(aP));}
    private :
       const TypeIm & mI2;
};
template <class TypeIm> cImInputProj<TypeIm> fProj(const TypeIm & aF)
{
    return cImInputProj<TypeIm>(aF);
}

     //  ===  Coodinates X,Y, Z... =====

class fCoord
{
    public :
       typedef int  tBase;

       fCoord(int aK) : mK(aK) { }

       template<class TypeP> tBase  GetV(const TypeP & aP) const {return aP[mK];}
    private :
       int  mK;
};

     //  ===  Som on Window =====

template <class TypeFctr> class cSomVign
{
    public :
       typedef typename TypeFctr::tBase  tBase;

       cSomVign(const TypeFctr & aF,const cPt2di aVign) :
          mF(aF),
          mBox (-aVign,aVign+cPt2di(1,1))
       {  
       }
       tBase  GetV(const cPt2di & aP)  const
       {
            tBase aRes=0;
            for (const auto & aDP : mBox)
            {
                aRes += mF.GetV(aP+aDP);
            }
            return aRes;
       }
    private :
       const TypeFctr & mF;
       cPixBox<2>       mBox;
};

template <class TypeFctr> cSomVign<TypeFctr> fSomVign(const TypeFctr & aF,const cPt2di aVign)
{
    return cSomVign<TypeFctr>(aF,aVign);
}

  // ==== Out Translate =====

template <class TypeOut,const int Dim> class cOutTrans
{
    public :
       cOutTrans(TypeOut & aOut,const cPtxd<int,Dim> & aTr) :
          mOut     (aOut),
          mTrans   (aTr)
       {
       }
       template<class TOut> void SetV(const cPtxd<int,Dim> & aP,const TOut & aV) const {return mOut.SetV(aP+mTrans,aV);}
    private :
       TypeOut &        mOut;
       cPtxd<int,Dim> mTrans;
};

template <class TypeO,const int Dim> cOutTrans<TypeO,Dim> oTrans(TypeO & aF,const cPtxd<int,Dim> &aP)
{
    return cOutTrans<TypeO,Dim>(aF,aP);
}

  // ==== Fonc Translate =====

template <class TypeF,const int Dim> class cFoncTrans
{
    public :
       typedef typename TypeF::tBase  tBase;

       cFoncTrans (const TypeF & aF,const cPtxd<int,Dim> aTr) :
          mF(aF) ,
          mTrans (aTr)
       {
       }
       tBase  GetV(const cPtxd<int,Dim> & aP) const {return mF.GetV(aP+mTrans);}
    private :
       const TypeF & mF;
       cPtxd<int,Dim> mTrans;
};

template <class TypeF,const int Dim> cFoncTrans<TypeF,Dim>
   fTrans(const TypeF & aF,const cPtxd<int,Dim> &aP)
{
    return cFoncTrans<TypeF,Dim>(aF,aP);
}

    //  ============= Constant fonctions ============

template <class TypeV> class cFoncCste
{
    public :
       typedef TypeV  tBase;

       cFoncCste (const TypeV & aV) :
          mV(aV)
       {
       }
       template <class TypeP> tBase  GetV(const TypeP &) const {return mV;}
    private :
       TypeV  mV;
};

template <class TypeV> cFoncCste<TypeV> fCste(const TypeV & aV)
{
    return cFoncCste<TypeV>(aV);
}



    //  ============= Copy function ============

template<class TypeFlux,class TypeFonc,class TypeOut>
   void TplCopy(const TypeFlux & aFlux,const TypeFonc & aFonc,const TypeOut & aCOut)
{  
    TypeOut & aOut = const_cast<TypeOut &>(aCOut);
    for (const auto & aP : aFlux)
    {
         aOut.SetV(aP,aFonc.GetV(aP));
    }    
}   

};

#endif  //  _MMVII_Tpl_TplSimpleOperator_H_
