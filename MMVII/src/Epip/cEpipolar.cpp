#include "cEpipolar.h"

namespace MMVII
{


    auto [centre1,centre2, dir1, dir2, aListH] = GenerateData(mSens1, mSens2);

//    GenSegBundle(mSens1,mSens2,List1.SetH()[0]);

    // TODOCM: Check almost epip
    // if ((dir2.x+dir1.x) <0)
    // {
    //     dir1 = -dir1;
    //     dir2 = -dir2;
    // }

    // TODOCM: Check d1 and d2 /= 0

    StdOut() << "dir2: " << dir2 << " " << Norm2(dir2) << "\n" ;

//    auto prevV1 = cPt2dr(0,0);
    for (auto& pair : aListH.SetH()) {
        pair.mP1 = (pair.mP1 - centre1) / dir1;
        pair.mP2 = (pair.mP2 - centre2) / dir2;
    }

    auto mNbCoeff1 = (mDegree+1)*(mDegree+2)/2 - (mDegree+1);
    auto mNbCoeff2 = (mDegree+1)*(mDegree+2)/2;
    auto mNbUkn = mNbCoeff1 + mNbCoeff2;

    cPolyXY_N<double> V1(mDegree);
    cPolyXY_N<double> V2(mDegree);
    auto aLeastSq = cLeasSqtAA<double>(mNbUkn);
    cDenseVect<double> coeffs(mNbUkn);
    for (int j=0; j<=mDegree; j++)
    {
        V1.AddFixedK(0,j, j==1 ? 1.0 : 0.0);
    }

    double aWeight = 1;
    for (const auto& aPair : aListH.SetH())
    {
        auto aDiffObs = V1.VarToCoeffs(aPair.mP1,coeffs.RawData(),1);
        V2.VarToCoeffs(aPair.mP2,coeffs.RawData()+mNbCoeff1,-1);
        aLeastSq.PublicAddObservation(aWeight, coeffs, - aDiffObs);
    }

    auto params = aLeastSq.PublicSolve();
    StdOut() << "LeastSq Var: " << aLeastSq.VarCurSol() << "\n";

    V1.SetVK(params.RawData());
    V2.SetVK(params.RawData()+mNbCoeff1);

    auto joe = aListH.SetH()[0];
    StdOut() << "V1(p1):" << V1(joe.mP1) << " V2(p2):" << V2(joe.mP2) << " Diff:" << V1(joe.mP1) - V2(joe.mP2) << "\n";



} // MMVII

