#ifndef  _MMVII_Bench_H_
#define  _MMVII_Bench_H_

namespace MMVII
{


/** \file MMVII_Bench.h
    \brief Declare function that will be called for bench

    When MMVII will grow, most bench function will be done in file/folde
   close to the functionality. So it will be necessary to declare these function.

*/


void Bench_0000_Ptxd();  ///< Basic Ptxd
void Bench_0000_SysDepString(); ///< String split (dir, path, file ...)
void Bench_0000_Memory(); ///< Bench on memory integrity
void Bench_0000_Param();  ///< Bench on param line processing (elementary)
// void cAppli_MMVII_Bench::Bench_0000_String(); => Bench on string-split
void BenchSerialization(const std::string & aDirOut,const std::string & aDirIn); ///< Bench on seriaization function



void BenchGlob();      ///< All Bench


void BenchSet(const std::string & aDir); ///< Bench on cExtSet (set "en extension")
void BenchSelector(const std::string & aDir); ///< Bench on selecto, (set "en comprehension")
void BenchEditSet(); ///< Bench on commands : EditSet  
void BenchEditRel(); ///< Bench on commands : EditRel

void BenchEnum(); ///< Bench on Str2E / E2Str

void Bench_Nums(); ///< Bench on rounding, modulo ... basic numeric service
void Bench_Random(); ///< Bench on random generator

void BenchGlobImage(); ///< Global bench on image
void BenchRectObj(); ///< Global bench on image
void BenchBaseImage(); ///< Global bench on image
void BenchGlobImage2d(); ///< Global bench on image
void BenchFileImage(); ///< Global bench on image



};

#endif  //  _MMVII_Bench_H_
