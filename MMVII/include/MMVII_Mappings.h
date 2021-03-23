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

template <const int DimIn,const int DimOut> class cMapping : public cMemCheck
{
    public :
      typedef  cPtxd<double,DimOut> tPtOut;
      typedef  cPtxd<double,DimIn>  tPtIn;

      virtual  tPtOut  Direct(const tPtIn &) const = 0;
};



template <const int Dim> class cBijMapping : public cMapping<Dim,Dim>
{
    public :
      typedef  cPtxd<double,Dim>  tPt;
      virtual  tPt  Inverse(const tPt &) const = 0;
};

class cImageSensor : public  cMapping<3,2>
{
    public :
};


class cImagePose : public cBijMapping<3>
{
    public :

      /// Coordinate Cam -> Word ; Pt =>  mC + mOrient * Pt
      cPt3dr  Direct(const cPt3dr &)  const override;  
      /// Coordinate Cam -> Word ; Pt =>  (Pt-mC) * mOrient 
      cPt3dr  Inverse(const cPt3dr &) const override;  // 
    private :

       cDenseMatrix<double>  mOrient;
       cPt3dr                mC;
};



};

#endif  //  _MMVII_MAPPINGS_H_
