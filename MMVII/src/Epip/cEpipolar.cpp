#include "cEpipolar.h"
#include "MMVII_MeasuresIm.h"
#include "MMVII_Sensor.h"

namespace MMVII
{

namespace {     // anonymous
cPt3dr GenRandomHomol(const cSensorImage *aSensor1,const cSensorImage *aSensor2)
{
    for (int aKTest=0 ; aKTest< 10000 ; aKTest++)
    {
        tPt2dr aPIm1 = aSensor1->RandomVisiblePIm();
        tPt2dr aPIm2 = aSensor2->RandomVisiblePIm();

        tPt3dr aResult = aSensor1->PInterBundle(cHomogCpleIm(aPIm1,aPIm2),*aSensor2);
        if ( aSensor1->IsVisible(aResult)  && aSensor2->IsVisible(aResult))
        {
            return aResult;
        }
    }

    MMVII_INTERNAL_ERROR("Cannot compute RandomVisiblePGround");
    return tPt3dr(0.0,0.0,0.0);
}

cSetHomogCpleIm GenRandomHomolSet(const cSensorImage *aSensor1,const cSensorImage *aSensor2, int aNb)
{
    cSetHomogCpleIm aSetHomogCpleIm;
    for (int i=0; i < aNb; i++) {
        auto aPtGround = GenRandomHomol(aSensor1,aSensor2);
        aSetHomogCpleIm.Add(cHomogCpleIm(aSensor1->Ground2Image(aPtGround),aSensor2->Ground2Image(aPtGround)));
    }
    return aSetHomogCpleIm;
}


void GenTraverseHomolSet(const cSensorImage *aSensor1,const cSensorImage *aSensor2, int aNbXY, cSetHomogCpleIm& aSetHomogCpleIm)
{
    auto aBoxIm1 = aSensor1->PixelDomain().Box();
    auto aXMin = aBoxIm1.P0().x();
    auto aXMax = aBoxIm1.P1().x();
    auto aYMin = aBoxIm1.P0().y();
    auto aYMax = aBoxIm1.P1().y();
    int deltaX = (aXMax - aXMin) / aNbXY;
    int deltaY = (aYMax - aYMin) / aNbXY;

    
    for (int x = aXMin ; x < aXMax; x += deltaX) {
        for (int y = aYMin; y < aYMax; y += deltaY) {
            AddHomolSetIfVisible(cPt2dr(x,y)
        }

    return aSetHomogCpleIm;
}
} // namespce anonymous

static std::pair<std::optional<cPt2dr>,cPt2dr> toto(const cSegmentCompiled<double,3>& aBundle, double anAbscissa1, double anAbscissa2, const cSensorImage *aSensor)
{
    auto p2g = aBundle.PtOfAbscissa(anAbscissa1);
    auto p2gp = aBundle.PtOfAbscissa(anAbscissa2);
    if (! aSensor->IsVisible(p2g) || !aSensor->IsVisible(p2gp))
        return {std::nullopt,{}};
    auto p2 = aSensor->Ground2Image(p2g);
    auto p2p = aSensor->Ground2Image(p2gp);
    return {p2, p2p};
}


template<typename T>
std::tuple<cPt2dr, cPt2dr,cPt2dr, cPt2dr, cSetHomogCpleIm> cEpipolarCouple<T>::GenerateData(
    const cSensorImage *aSensor1,
    const cSensorImage *aSensor2,
    const cSetHomogCpleIm& aSetHomogCpleIm
    
    )
{
    auto c1 = cPt2dr(0,0);
    auto c2 = cPt2dr(0,0);
    auto d1 = cPt2dr(0,0);
    auto d2 = cPt2dr(0,0);
    double aZStep = 0.1;

    int n1 = 0;
    int n2 = 0;
    cSetHomogCpleIm aList;
    for (const auto& aHPair : aSetHomogCpleIm.SetH()) {
        auto aPtInter = aSensor1->PInterBundle(aHPair,*aSensor2);
        auto aBundle1 = cSegmentCompiled(aSensor1->Image2Bundle(aHPair.mP1));
        auto aBundle2 = cSegmentCompiled(aSensor2->Image2Bundle(aHPair.mP2));
        auto anAbscissa1 = aBundle1.Abscissa(aPtInter);
        auto anAbscissa2 = aBundle2.Abscissa(aPtInter);

        for (double aKz = 0.9; aKz <= 1.1; aKz += aZStep)
        {
            auto [p2,p2p] = toto(aBundle1, anAbscissa1 * aKz, anAbscissa1 *(aKz+aZStep), aSensor2);
            if (p2)
            {
                aList.Add(cHomogCpleIm(aHPair.mP1,*p2));
                d2 += VUnit(p2p - *p2);
                c1 += aHPair.mP1;
                c2 += aHPair.mP2;
                n2++;

            }
            auto [p1,p1p] = toto(aBundle2, anAbscissa2 * aKz, anAbscissa2 *(aKz+aZStep), aSensor1);
            if (p1)
            {
                aList.Add(cHomogCpleIm(*p1,aHPair.mP2));
                d1 += VUnit(p1p - *p1);
                c1 += aHPair.mP1;
                c2 += aHPair.mP2;
                n1++;

            }
        }
    }
    c1 = c1 / double(n1+n2);
    c2 = c2 / double(n1+n2);

    d1 = VUnit(d1 / double(n1));
    d2 = VUnit(d2 / double(n2));

    return std::tuple(c1,c2,d1,d2,aList);
}


template<typename T>
cEpipolarCouple<T> cEpipolarCouple<T>::FromSensors(
    const cSensorImage *aSensor1, 
    const cSensorImage *aSensor2,
    int aDegree
    )
{
    auto [aCentre1, aCentre2, aDir1, aDir2, aListH] = GenerateData(aSensor1, aSensor2);

    auto anEpipolarCouple = std::make_unique<cEpipolarCouple<T>(aCentre1,aDir1,aCentre2,aDir2,aDegree));
            
    // TODOCM: Check almost epip
    // if ((dir2.x+dir1.x) <0)
    // {
    //     dir1 = -dir1;
    //     dir2 = -dir2;
    // }

    // TODOCM: Check d1 and d2 /= 0

//    auto prevV1 = cPt2dr(0,0);
    for (auto& pair : aListH.SetH()) {
        pair.mP1 = (pair.mP1 - aCentre1) / aDir1;
        pair.mP2 = (pair.mP2 - aCentre2) / aDir2;
    }

    cPolyXY_N<double> V1(aDegree);
    cPolyXY_N<double> V2(aDegree);
    auto mNbUkn = V1.NbCoeffs() - (aDegree+1) + V2.NbCoeffs();

    auto aLeastSq = cLeasSqtAA<double>(mNbUkn);
    cDenseVect<double> coeffs(mNbUkn);
    for (int j=0; j<=aDegree; j++)
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

 }

} // MMVII

