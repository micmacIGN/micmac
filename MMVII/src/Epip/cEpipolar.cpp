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

#if 0
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
            AddHomolSetIfVisible(cPt2dr(x,y));
        }
    }
    cSetHomogCpleIm aSetHomogCpleIm;
    auto c1 = cPt2dr(0,0);
    auto d2  = cPt2dr(0,0);

    auto C1 = cPt2dr(1000,750);
    auto C2 = cPt2dr(1000,750);

    auto inter_g = aSensor1->PInterBundle(cHomogCpleIm(C1,C2),*aSensor2);
    aZ = inter_g.z();
    auto aMinZ = aZ * .8;
    auto aMaxZ = aZ * 1.2;

    int t = 0;
    unsigned n = 0;
    double deltaZ = (aMaxZ - aMinZ) / mNbZ;
    int deltaX = aRect1.Sz().x() / mNbXY;
    int deltaY = aRect1.Sz().y() / mNbXY;

    for (int x = aRect1.P0().x(); x < aRect1.P1().x(); x += deltaX) {
        for (int y = aRect1.P0().y(); y < aRect1.P1().y(); y += deltaY) {
            auto p1 = cPt2dr(x,y);
            auto bundle = aSensor1->Image2Bundle(p1);
            auto vect = VUnit(bundle.P2() - bundle.P1());
            for (double z = aMinZ; z < aMaxZ; z += deltaZ) {
                t++;
                auto p1g = bundle.P1() + vect * z;
                auto p2 = aSensor2->Ground2Image(p1g);
                if (! aRect2.InsideBL(p2))
                    continue;
                auto p1pg = bundle.P1() + vect * (z + deltaZ);
                auto p2p = aSensor2->Ground2Image(p1pg);
                if (! aRect2.InsideBL(p2p))
                    continue;
                n ++;
                aSetHomogCpleIm.Add(cHomogCpleIm(p1,p2));
                c1 += p1;
                d2 += VUnit(p2p - p2);
            }
        }
    }
    c1 = c1 / double(n);
    d2 = d2 / double(n);


    return aSetHomogCpleIm;
}
#endif

struct GenDataCtxt
{
    enum class eBundleFrom{FIRST,SECOND};

    void Generate(
        const cSensorImage *aSensor1,
        const cSensorImage *aSensor2,
        const cSetHomogCpleIm& aSetHomogCpleIm
    )
    {
        double aZStep = 0.1;

        for (const auto& aHPair : aSetHomogCpleIm.SetH()) {
            auto aPtInter = aSensor1->PInterBundle(aHPair,*aSensor2);
            auto aBundle1 = cSegmentCompiled(aSensor1->Image2Bundle(aHPair.mP1));
            auto aBundle2 = cSegmentCompiled(aSensor2->Image2Bundle(aHPair.mP2));
            double aDist1 = aBundle1.Abscissa(aPtInter);
            double aDist2 = aBundle2.Abscissa(aPtInter);

            for (double aKz = 0.9; aKz <= 1.1; aKz += aZStep)
            {
                AddCouple(aBundle1, aSensor2, aHPair, aDist1 * aKz, aDist1 * (aKz+aZStep), eBundleFrom::FIRST);
                AddCouple(aBundle2, aSensor1, aHPair, aDist2 * aKz, aDist2 * (aKz+aZStep), eBundleFrom::SECOND);
            }
        }
        Normalize();
    }

    void AddCouple(const tSegComp3dr& aBundle, const cSensorImage *aSensor, const cHomogCpleIm& aHPair, double aDist1, double aDist2, eBundleFrom aBundleFrom)
    {
        auto p2g = aBundle.PtOfAbscissa(aDist1);
        auto p2gp = aBundle.PtOfAbscissa(aDist2);
        if (! aSensor->IsVisible(p2g) || !aSensor->IsVisible(p2gp))
            return;
        auto p2 = aSensor->Ground2Image(p2g);
        auto p2p = aSensor->Ground2Image(p2gp);
        c1 += aHPair.mP1;
        c2 += aHPair.mP2;
        if (aBundleFrom == eBundleFrom::FIRST) {
            homogSet.Add(cHomogCpleIm(aHPair.mP1,p2));
            d2 += VUnit(p2p - p2);
            n2++;
        } else {
            homogSet.Add(cHomogCpleIm(p2,aHPair.mP2));
            d1 += VUnit(p2p - p2);
            n1++;
        }
    }

    void Normalize()
    {
        c1 = c1 / double(n1+n2);
        c2 = c2 / double(n1+n2);
        d1 = VUnit(d1 / double(n1));
        d2 = VUnit(d2 / double(n2));

        // TODOCM: Check almost epip
        // if ((dir2.x+dir1.x) <0)
        // {
        //     dir1 = -dir1;
        //     dir2 = -dir2;
        // }
        // TODOCM: Check d1 and d2 /= 0 ?

        for (auto& pair : homogSet.SetH()) {
            pair.mP1 = (pair.mP1 - c1) / d1;
            pair.mP2 = (pair.mP2 - c2) / d2;
        }
    }


    cSetHomogCpleIm homogSet;
    cPt2dr c1{0,0};
    cPt2dr c2{0,0};
    cPt2dr d1{0,0};
    cPt2dr d2{0,0};
    int n1;
    int n2;
};


} // namespace anonymous



template<typename T>
cEpipolarCouple<T> cEpipolarCouple<T>::FromSensors(
    const cSensorImage *aSensor1,
    const cSensorImage *aSensor2,
    int aDegree
    )
{
    auto aSetHomol = GenRandomHomolSet(aSensor1, aSensor2, 10000);
    GenDataCtxt aCtxt;
    aCtxt.Generate(aSensor1, aSensor2, aSetHomol);


    cPolyXY_N<double> V1(aDegree);
    cPolyXY_N<double> V2(aDegree);
    auto mNbUkn = V1.NbCoeffs() - (aDegree+1) + V2.NbCoeffs();

    auto aLeastSq = cLeasSqtAA<double>(mNbUkn);
    cDenseVect<double> coeffs(mNbUkn);
    double aWeight = 1;
    for (const auto& aPair : aCtxt.homogSet.SetH())
    {
        int n=0;
        auto P1 = aPair.mP1;
        auto P2 = aPair.mP2;
        T X_n = P1.x();
        for (int i=1; i<= V1.Degree(); i++)
        {
            T Y_n = 1;
            for (int j=0; j<=V1.Degree()-i; j++)
            {
                coeffs(n++) = X_n * Y_n;
                Y_n *= P1.y();
            }
            X_n *= P1.x();;
        }
        X_n = 1;
        for (int i=0; i<= V2.Degree(); i++)
        {
            T Y_n = 1;
            for (int j=0; j<=V2.Degree()-i; j++)
            {
                coeffs(n++) = - X_n * Y_n;
                Y_n *= P2.y();
            }
            X_n *= P2.x();;
        }
        aLeastSq.PublicAddObservation(aWeight, coeffs, - P1.y());
    }

    auto params = aLeastSq.PublicSolve();
    int n = 0;
    int m = 0;
    for (int j=0; j<= V1.Degree(); j++)
    {
        V1.K(m++) = (j == 1 ? 1 : 0);
    }
    for (int i=0; i<= V1.Degree(); i++)
    {
        for (int j=0; j<=V1.Degree()-i; j++)
        {
            V1.K(m++) = params(n++);
        }
    }
    m = 0;
    for (int i=0; i<= V2.Degree(); i++)
    {
        for (int j=0; j<=V2.Degree()-i; j++)
        {
            V2.K(m) = params(n++);
        }
    }


    auto joe = aCtxt.homogSet.SetH()[0];
    StdOut() << "V1(p1):" << V1(joe.mP1) << " V2(p2):" << V2(joe.mP2) << " Diff:" << V1(joe.mP1) - V2(joe.mP2) << "\n";
    return cEpipolarCouple<T>(aCtxt, V1, V2);
 }

template class cEpipolarCouple<tREAL4>;
template class cEpipolarCouple<tREAL8>;
template class cEpipolarCouple<tREAL16>;


} // MMVII

