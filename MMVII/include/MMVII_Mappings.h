#ifndef  _MMVII_MAPPINGS_H_
#define  _MMVII_MAPPINGS_H_

namespace MMVII
{

/** \file MMVII_Mappings.h
    \brief contain interface class for continuous mapping

   Most probably this will evolve a lot, with several reengenering 
  phases. 

*/


/// Class that represent a continous mapping R^k -> R^n

template <class Type,const int DimIn,const int DimOut> class cMapping : public cMemCheck
{
    public :
      typedef  cPtxd<Type,DimOut> tPtOut;
      typedef  cPtxd<Type,DimIn>  tPtIn;

      virtual  tPtOut  Direct(const tPtIn &) const = 0;
};



template <class Type,const int Dim> class cBijMapping : public cMapping<Type,Dim,Dim>
{
    public :
      typedef  cPtxd<Type,Dim>  tPt;
      virtual  tPt  Inverse(const tPt &) const = 0;
};

template <class Type> class  cImageSensor : public  cMapping<Type,3,2>
{
    public :
};

template <class Type> class cImagePose : public cBijMapping<Type,3>
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

     Si r<z  (Asin(r/R)*R/r) (x/R,y/R)
     Si r>z  (Acos(z/R)*z/r) (x/R,y/R)

*/


};

#endif  //  _MMVII_MAPPINGS_H_
