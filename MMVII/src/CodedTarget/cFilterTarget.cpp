#include "CodedTarget.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "include/MMVII_Tpl_Images.h"


namespace MMVII
{

       /* ================================================== */
       /*               SCALITY                              */
       /* ================================================== */

/*
template<class TypeEl> double ScalInv
                              (
                                  const  cDataIm2D<TypeEl> & aDImIn,
                                  const  cDataIm2D<TypeEl> & aDFiltr,
                                  const  cPt2dr & aPt,
                                  double aScale,
                                  double aR0,
                                  double aR1,
                                  double Epsilon
                              )
{
    
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,true);
    for (const auto & aV  : aVectVois)
    {
		  TypeEl aV1 = aDImIn.GetV(aP+aV);
		  TypeEl aV2 = aDImIn.GetV(aP-aV);
		  aSM.Add(aV1,aV2);
    }
}
*/

/*
template<class TypeEl> cIm2D<TypeEl> ImScalab(const  cDataIm2D<TypeEl> & aDImIn,double aR0,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,true);

    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  StdOut() << "SYMMMM\n";

    int aD = round_up(aR1);
    cPt2di aPW(aD,aD);

    cPt2di aSz = aDImIn.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    for (const auto & aP : cRect2(aPW,aSz-aPW))
    {
          cSymMeasure<float> aSM;
          for (const auto & aV  : aVectVois)
	  {
		  TypeEl aV1 = aDImIn.GetV(aP+aV);
		  TypeEl aV2 = aDImIn.GetV(aP-aV);
		  aSM.Add(aV1,aV2);
	  }
	  aDImOut.SetV(aP,aSM.Sym(Epsilon));
    }

    return aImOut;
}
*/



       /* ================================================== */
       /*               BINARITY                             */
       /* ================================================== */

template<class TypeEl> 
   double IndBinarity(const  cDataIm2D<TypeEl> & aDIm,const cPt2di & aP0,const std::vector<cPt2di> & aVectVois)
{
    double aPRop = 0.15;
    std::vector<float> aVVal;
    for (int aKV=0; aKV<int(aVectVois.size()) ; aKV++)
    {
         cPt2di aPV = aP0 + aVectVois[aKV];
         aVVal.push_back(aDIm.GetV(aPV));
    }
    std::sort(aVVal.begin(),aVVal.end());
    int aNb = aVVal.size();

    int aIndBlack = round_ni(aPRop * aNb);
    int aIndWhite =  aNb-1- aIndBlack;
    float aVBlack = aVVal[aIndBlack];
    float aVWhite = aVVal[aIndWhite];
    float aVThreshold = (aVBlack+aVWhite)/2.0;

    int aNbBlack = 0;
    double aSumDifBlack = 0;
    double aSumDifWhite = 0;
    for (const auto & aV : aVVal)
    {
              bool IsBLack = (aV <aVThreshold ) ;
              float aVRef = (IsBLack ) ?  aVBlack : aVWhite;
              float aDif = std::abs(aVRef-aV);
              if (IsBLack)
              {
                 aNbBlack++;
                 aSumDifBlack += aDif;
              }
              else
              {
                 aSumDifWhite += aDif;
              }
    }
    double aValue=1.0;
    int aNbWhite = aNb -aNbBlack;
    if ((aNbBlack!=0) && (aNbWhite!= 0)  && (aVBlack!=aVWhite))
    {
              double aDifMoy  =  aSumDifBlack/ aNbBlack + aSumDifWhite/aNbWhite;
              aValue = aDifMoy  / (aVWhite - aVBlack);
    }
	  
    return aValue;
}

template<class TypeEl> cIm2D<TypeEl> ImBinarity(const  cDataIm2D<TypeEl> & aDIm,double aR0,double aR1,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,false);
    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  

    int aD = round_up(aR1);
    cPt2di aPW(aD,aD);


    cPt2di aSz = aDIm.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    // double aPRop = 0.15;
    for (const auto & aP0 : cRect2(aPW,aSz-aPW))
    {
        double aValue = IndBinarity(aDIm,aP0,aVectVois);
        aDImOut.SetV(aP0,aValue);
    }

    return aImOut;
}

       /* ================================================== */
       /*               STARITY                              */
       /* ================================================== */

std::vector<cPt2dr> VecDir(const  std::vector<cPt2di>&  aVectVois)
{
    std::vector<cPt2dr>  aVecDir;
    for (const auto & aPV : aVectVois)
    {
           aVecDir.push_back(VUnit(ToR(aPV)));
    }
    return aVecDir;
}


template<class TypeEl> double Starity
                              (
                                  const  cImGrad<TypeEl> & aImGrad,
                                  const cPt2dr & aP0,
                                  const  std::vector<cPt2di>&  aVectVois ,
                                  const  std::vector<cPt2dr>&  aVecDir,
                                  double Epsilon
                              )
{
    const cDataIm2D<TypeEl> & aIGx = aImGrad.mGx.DIm();
    const cDataIm2D<TypeEl> & aIGy = aImGrad.mGy.DIm();

    double aSomScal = 0;
    double aSomG2= 0;
    for (int aKV=0; aKV<int(aVectVois.size()) ; aKV++)
    {
         cPt2dr aVR =  ToR(aVectVois[aKV]);
         cPt2dr aPV = aP0 + aVR;
         cPt2dr aGrad(aIGx.GetVBL(aPV),aIGy.GetVBL(aPV));

         aSomScal += Square(Scal(aGrad,aVecDir[aKV]));
         aSomG2 += SqN2(aGrad);

    }
    double aValue = (aSomScal+ Epsilon) / (aSomG2+Epsilon);
 
    return aValue;
}


template<class TypeEl> cIm2D<TypeEl> ImStarity(const  cImGrad<TypeEl> & aImGrad,double aR0,double aR1,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,false);
    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  
    std::vector<cPt2dr>  aVecDir = VecDir(aVectVois);

    int aDist = round_up(aR1)+2;
    cPt2di aPW(aDist,aDist);

    const cDataIm2D<TypeEl> & aIGx = aImGrad.mGx.DIm();
    // const cDataIm2D<TypeEl> & aIGy = aImGrad.mGy.DIm();

    cPt2di aSz = aIGx.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    for (const auto & aP0 : cRect2(aPW,aSz-aPW))
    {
         double aValue = Starity(aImGrad,ToR(aP0),aVectVois,aVecDir,Epsilon);
/*
          double aSomScal = 0;
          double aSomG2= 0;
          for (int aKV=0; aKV<int(aVectVois.size()) ; aKV++)
	  {
		  cPt2di aPV = aP0 + aVectVois[aKV];
		  cPt2dr aGrad(aIGx.GetV(aPV),aIGy.GetV(aPV));

		  aSomScal += Square(Scal(aGrad,aVecDir[aKV]));
		  aSomG2 += SqN2(aGrad);

		  // StdOut () << " GR=" << aGrad << " S=" << Square(Scal(aGrad,aVecDir[aKV])) << " G2=" << Norm2(aGrad) << "\n";
	  }
	  double aValue = (aSomScal+ Epsilon) / (aSomG2+Epsilon);
*/
	  aDImOut.SetV(aP0,aValue);
    }

    return aImOut;
}

/** This filter caracetrize how an image is symetric arround  each pixel; cpmpute som diff arround
 * oposite pixel, normamlized by standard deviation (so contrast invriant)
*/

/*
class  cSymetricityCalc
{
      public :
          cSymetricityCalc(double aR0,double aR1,double Epsilon);
      protected :
};
*/

       /* ================================================== */
       /*               SYMETRY                              */
       /* ================================================== */

template<class TypeEl> cIm2D<TypeEl> ImSymetricity(const  cDataIm2D<TypeEl> & aDImIn,double aR0,double aR1,double Epsilon)
{
    std::vector<cPt2di>  aVectVois = VectOfRadius(aR0,aR1,true);

    // aVectVois = GetPts_Circle(cPt2dr(0,0),aR0,true);  StdOut() << "SYMMMM\n";

    int aD = round_up(aR1);
    cPt2di aPW(aD,aD);

    cPt2di aSz = aDImIn.Sz();
    cIm2D<TypeEl> aImOut(aSz,nullptr,eModeInitImage::eMIA_V1);
    cDataIm2D<TypeEl> & aDImOut = aImOut.DIm();

    for (const auto & aP : cRect2(aPW,aSz-aPW))
    {
          cSymMeasure<float> aSM;
          for (const auto & aV  : aVectVois)
	  {
		  TypeEl aV1 = aDImIn.GetV(aP+aV);
		  TypeEl aV2 = aDImIn.GetV(aP-aV);
		  aSM.Add(aV1,aV2);
	  }
	  aDImOut.SetV(aP,aSM.Sym(Epsilon));
    }

    return aImOut;
}

#define INSTANTIATE_FILTER_TARGET(TYPE)\
template double Starity(const cImGrad<TYPE> &,const cPt2dr &,const std::vector<cPt2di>&,const  std::vector<cPt2dr>&,double Epsilon);\
template cIm2D<TYPE> ImBinarity(const  cDataIm2D<TYPE> & aDIm,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImStarity(const  cImGrad<TYPE> & aImGrad,double aR0,double aR1,double Epsilon);\
template cIm2D<TYPE> ImSymetricity(const  cDataIm2D<TYPE> & aDImIn,double aR0,double aR1,double Epsilon);

INSTANTIATE_FILTER_TARGET(tREAL4)


};
