#include "include/MMVII_all.h"
#include "include/SymbDer/SymbolicDerivatives.h"
#include "include/SymbDer/SymbDer_GenNameAlloc.h"


using namespace NS_SymbolicDerivative;


namespace MMVII
{

template <class tUk,class tObs>
std::vector<Type> Dist2DConservation
                  (
                      const std::vector<tUk> & aVUk,
                      const std::vector<tObs> & aVObs
                  )
{
    const Type & x1 = aVUk[0];
    const Type & y1 = aVUk[1];
    const Type & x2 = aVUk[2];
    const Type & y2 = aVUk[3];

    const Type & d  = aVObs[0];  // Warn the data I got were in order y,x ..

    return { sqrt(square(x1-x2) + square(y1-y2)) - d } ;
}


void GenerateCodeGeom2D(const  std::string &aDirGenCode,bool WithDerive)
{
	/*
   int aSzBuf=1;

   NS_SymbolicDerivative::cCoordinatorF<double> aCEq(std::string("Dist2DCon_") +(WithDerive ? "d" :"f"),aSzBuf,{"x1","y1","x2","y2"},{"d"});

      // Set header in a place to compilation path of MMVII
   aCEq.SetHeaderIncludeSymbDer("include/SymbDer/SymbDer_Common.h");
   aCEq.SetDirGenCode(aDirGenCode);

   auto aXY= Dist2DConservation(aCEq.VUk(),aCEq.VObs()); // Give ths list of atomic formula
   if (WithDerive)
      aCEq.SetCurFormulasWithDerivative(aXY);
   else
      aCEq.SetCurFormulas(aXY);
   auto [aClassName,aFileName] = aCEq.GenerateCode("CodeGen_");
   cGenNameAlloc::Add(aClassName,aFileName);
   */
}



};//  namespace MMVII

