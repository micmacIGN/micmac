#include "include/MMVII_all.h"
//using namespace MMVII;




void  TestLE2()
{
    std::cout << "TEST LE2 \n";
    cPt1d<double>  aP1(3.0);
    aP1+aP1;
    aP1-aP1;

    cPt2d<double>  aP2(1.0,3.0);
    aP2 + aP2*2.0;

    cPtxd<double,2> aQ2(aP2);
    cPt2d<double>   aR2(aQ2);
    std::cout << "Q2 " << aQ2.Data()[0] << " " << aQ2.Data()[1] << "\n";
    std::cout << "R2 " << aR2.x()  << " " << aR2.y() << "\n";
}

