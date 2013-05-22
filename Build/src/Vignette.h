#ifndef VIGNETTE_H
#define VIGNETTE_H
#include "StdAfx.h"

class CPP_Vignette
{
public:
    CPP_Apero2PMVS();
    void Apero2PMVS(string aPattern,string aOri);
    void Apero2PMVS_Banniere();
private:
    ElMatrix<double> OriMatrixConvertion(CamStenope * aCS);
};

#endif // CPP_APERO2PMVS_H