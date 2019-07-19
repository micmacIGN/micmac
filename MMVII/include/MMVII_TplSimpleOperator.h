#ifndef  _MMVII_Tpl_TplSimpleOperator_H_
#define  _MMVII_Tpl_TplSimpleOperator_H_
namespace MMVII
{


/** \file  MMVII_TplSimpleOperator.h
    \brief    Very basic and unoptimzed filter, may be usefull to test other functionnality

     Formalisation close to MMV1 : Flux/Fonction/OutPut  but :
       * template (vs virtual) implementation
       * non buffered implementation (i.e. computation point by point)

     These caracteristic do it much easier to use/increase but also not very efficient.

     Maybe used as a "bac a sable" 4 further implementation (buffered, template or virtual)

     Flux must be iterators on point, usable with for (const auto : )
     Fonction  must defin  GetV
     Out       must define SetV
*/

/*=======================================================*/
/*                                                       */
/*                 FLUX Pts                              */
/*                                                       */
/*=======================================================*/

/*  See template  MMVII_Images.h for

      - cPixBoxIterator / cPixBox  , for iterating on rectangle
      - cBorderPixBoxIterator / cBorderPixBox , for iterating on border
*/

/*
template <class TFlux,class TFonc> class cIterFluxSel
{
    public :
         tIter &  operator ++()
         {
         };

    private :
};

template <class TFlux,class TFonc> class   cFluxSel
{
    public :
    private :
};
*/

/*=======================================================*/
/*                                                       */
/*          FUNCTIONS                                    */
/*                                                       */
/*=======================================================*/


     /**   Image value "safe"  : do not generate error as when point is out,
         give the value of projection inside de validity domai
     */
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

     /**  Coordinates functions */

class fCoord
{
    public :
       typedef int  tBase;

       fCoord(int aK) : mK(aK) { }

       template<class TypeP> tBase  GetV(const TypeP & aP) const {return aP[mK];}
    private :
       int  mK;
};

// extern fCoord  fX();
// extern fCoord  fY();
// extern fCoord  fZ();

     /** Sum on a Windows of another function */

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
       const TypeFctr & mF;    ///< Summed function
       cPixBox<2>       mBox;  ///< Windows on witch function is summed
};

template <class TypeFctr> cSomVign<TypeFctr> fSomVign(const TypeFctr & aF,const cPt2di aVign)
{
    return cSomVign<TypeFctr>(aF,aVign);
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

    //  ============= Binary operators ============
template <class TypeF1,class TypeF2,class TypeOp> class cFoncBinaryOp
{
    public :
       typedef double  tBase; /// To Change

       cFoncBinaryOp (const TypeF1 & aF1,const TypeF2 & aF2,const TypeOp & aOp) :
          mF1(aF1),
          mF2(aF2),
          mOp (aOp)
       {
       }
       template <class TypeP> tBase  GetV(const TypeP & aP) const {return mOp(mF1.GetV(aP),mF2.GetV(aP));}
    private :
       const TypeF1 & mF1;
       const TypeF2 & mF2;
       const TypeOp & mOp;
};

template <class TypeF1,class TypeF2,class TypeOp> 
           cFoncBinaryOp<TypeF1,TypeF2,TypeOp> fBinaryOp(const TypeF1 & aF1,const TypeF2 & aF2,const TypeOp & aOp)
{
    return cFoncBinaryOp<TypeF1,TypeF2,TypeOp>(aF1,aF2,aOp);
}

double BasSom(const double &aV1,const double & aV2) {return aV1+aV2;}
template <class TypeF1,class TypeF2> auto fSum(const TypeF1 & aF1,const TypeF2 & aF2) {return fBinaryOp(aF1,aF2,BasSom);}


/*=======================================================*/
/*                                                       */
/*               OUTPUT                                  */
/*                                                       */
/*=======================================================*/

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
