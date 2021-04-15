#ifndef  _MMVII_MAPPINGS_H_
#define  _MMVII_MAPPINGS_H_

namespace MMVII
{

/** \file MMVII_Mappings.h
    \brief contain interface class for continuous mapping

   Most probably this will evolve a lot, with several reengenering 
  phases. 

*/

template <class Type,const int Dim> class cDataBoundedSet : public cMemCheck
{
    public :
      typedef  cPtxd<Type,Dim>  tPt;

      /// Does it belong to the set;  default true
      virtual bool Inside(const tPt &) const;
       
    private :
       cTplBox<Type,Dim> mBox;
};

/// Class that represent a continous mapping R^k -> R^n

template <class Type,const int DimIn,const int DimOut> class cDataMapping : public cMemCheck
{
    public :
      typedef  cPtxd<Type,DimOut> tPtOut;
      typedef  cPtxd<Type,DimIn>  tPtIn;
      typedef  std::vector<tPtIn> tVecIn;
      typedef  cDenseMatrix<Type> tGrad;  ///< For each 

      cDataMapping(int aSzBuf);
      /// Has it a diffenrentiable method : default false
      virtual  bool    HasValAndGrad() const;
           // ========== Bufferized mode ==============
      void     AddBufIn(const tPtIn &) const;
      virtual  void  ComputeDirect() const;
      virtual  void ComputeValAndGrad() const;
      /// Compute image in direct sens
      virtual  const std::pair<tPtOut,tGrad> &  ComputeValAndGrad(const tPtIn &) const;
      virtual  const tPtOut &  Val(const tPtIn &) const;

      /// compute diffenrentiable method , default = erreur
    public :
       tVecIn              mBufIn;
       tVecIn              mBufOut;
       std::vector<tGrad>  mGrads;
};


/**  We coul expect that DimIn=DimOut, but in fact in can be used
to represent a mapping from a part of the plane to a part of the sphere */

/*

template <class Type,const int DimIn,const int DimOut> 
         class cInvertibleMapping : public cMapping<Type,DimIn,DimOut>
{
    public :
      typedef  cPtxd<Type,DimOut> tPtOut;
      typedef  cPtxd<Type,DimIn>  tPtIn;
      typedef  cDenseMatrix<Type> tGrad;  ///< For each 
      virtual  tPtOut  Inverse(const tPtOut &) const = 0;
};

template <class Type> class  cImageSensor : public  cMapping<Type,3,2>
{
    public :
};

template <class Type> class cImagePose : public cInvertibleMapping<Type,3,3>
{
    public :
      typedef  cPtxd<Type,3>  tPt;
      /// Coordinate Cam -> Word ; Pt =>  mC + mOrient * Pt
      tPt  Direct(const tPt &)  const override;  
      /// Coordinate Cam -> Word ; Pt =>  (Pt-mC) * mOrient 
      tPt  Inverse(const tPt &) const override;  // 
      // cImagePose();

    private :
       cDenseMatrix<Type>  mOrient;
       tPt                 mC;
};
*/

/*
Avec R=N(x,y,z) et r=N(x,y)

  Ter -> Cam

       Proj Cam normale :  x/z y/z   ou  
       Proj fisheye equiline :   atan(r/z) * (x/r,y/r) = z/r atan(r/z) (x/z,y/z)

            ou Asin(r/R) *(x/r,y/r) = (Asin(r/R)*R/r) (x/R,y/R)

       Si appel AsinC(X) = asin(X)/X, fonction prolongeable en 0 par AsinC(0) =1 et Cinfini,
       alors
 
         Proj = AsinC(r/R) * (x/R,y/R)   avec l'avantage de n'avoir que des terme borne
         (en fait compris entre -1 et 1) et bien defini meme en x,y=0,0  
         Mais pb car deriv infini en 1 et ne gere pas bien aude la de pi/2

         Qhi(x,y,z) = atan2(r,z)/r  * (x,y)

      Autre method   ATC(A,B) = atan2(A,B)/A     =>  ATC(r,z) * (x,y)

     Encore une autre :
          Si z>r     (Asin(r/R)*R/r) (x/R,y/R)
          Si r>z>-r  (Acos(z/R) (x/r,y/r)
          Si -r>     (Asin(r/R)*R/r) (x/R,y/R)
      Cela fait intervenir les fonc
           *  AsinC pronlogeable en 0 , et

*/


};

#endif  //  _MMVII_MAPPINGS_H_
