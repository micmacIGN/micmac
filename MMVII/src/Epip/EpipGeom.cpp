#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "cPolyXY_N.h"

#include <vector>

/**
   \file EpipGeom.cpp


 */


namespace MMVII
{


static cPt3dr GenGround(const cSensorImage *aSensor1,const cSensorImage *aSensor2)
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
/*
static tPt2dr GenSegBundle(const cSensorImage *aSensor1, const cSensorImage *aSensor2, const cHomogCpleIm &aHomog, double aKz)
{
    auto aPtInter = aSensor1->PInterBundle(aHomog,*aSensor2);
    auto bundle = cSegmentCompiled(aSensor1->Image2Bundle(aHomog.mP1));
    auto pg = bundle.PtOfAbscissa(bundle.Abscissa(aPtInter) * aKz);
    auto p2 = aSensor2->Ground2Image(pg);
    return p2;
}
*/

class cAppli_EpipGeom : public cMMVII_Appli
{
public :

    cAppli_EpipGeom(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

private :
    std::tuple<cPt2dr, cPt2dr, cSetHomogCpleIm> GenerateData(
        const cRect2 &aRect1,
        const cSensorImage *aSensor1,
        const cRect2 &aRect2,
        const cSensorImage *aSensor2,
        double aZ
        ) const;

    std::tuple<cPt2dr, cPt2dr, cPt2dr, cPt2dr, cSetHomogCpleIm> GenerateData(const cSensorImage *aSensor1, const cSensorImage *aSensor2) const;

    std::pair<cPt2dr, cPt2dr> GenerateData(
        const cRect2 &aRect1,
        const cSensorImage *aSensor1,
        const cRect2 &aRect2,
        const cSensorImage *aSensor2,
        const cSetHomogCpleIm &aSetHomogCpleIm
        ) const;
    cPhotogrammetricProject  mPhProj;
    std::string  mImName1;
    std::string  mImName2;
    cSensorImage *mSens1 = nullptr;
    cSensorImage *mSens2 = nullptr;
    int mDegree = 5;
    int mDegreeInv = mDegree + 4;
    int mNbXY = 100;
    int mNbZ = 3;
    int mMinZ = 100;
    int mMaxZ = 1000;
};

cAppli_EpipGeom::cAppli_EpipGeom (
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli & aSpec
    )
    : cMMVII_Appli  (aVArgs,aSpec)
    , mPhProj       (*this)
{
}

double f(double x, double y)
{
    return 10 * sin(x) + sin(y);
}

int cAppli_EpipGeom::Exe()
{
    mPhProj.FinishInit();
    if (! IsInit(&mDegreeInv))
        mDegreeInv = mDegree + 4;

#if 1
    cPolyXY_N<double> P(mDegree);

    for (double x = -3.0; x<=3.0; x+= 0.1) {
        for (double y = -3.0; y<=3.0; y+= 0.1) {
            P.AddObs(x,y, f(x,y));
        }
    }
    P.Fit();
    printf("P var %lf\n",P.VarCurSol());
    for (int i=0; i< 10; i++) {
        cPt2dr p(RandInInterval(-3,3),RandInInterval(-3,3));
        auto v = f(p.x(),p.y());
        auto po = P(p);
        StdOut() << std::setw(11) << po << " " << std::setw(11) << v << " " << std::setw(11) << v - po  << " " << p << "\n";
    }

    return 0;
#endif

/*
    // mIm1 = cDataFileIm2D::Create(mIm1);
    auto mImRect1 = cRect2(cDataFileIm2D::Create(mImName1,eForceGray::No).Sz());
    auto mImRect2 = cRect2(cDataFileIm2D::Create(mImName1,eForceGray::No).Sz());
    mSens1 = mPhProj.ReadSensor(mImName1,true);
    mSens2 = mPhProj.ReadSensor(mImName2,true);
*/
    auto aCaptSize = cPt2di(2000,1500);
//    auto aImRect1 = cRect2(cPt2di(0,0),aCaptSize);
//    auto aImRect2 = cRect2(cPt2di(0,0),aCaptSize);


    cPt3dr aCenterGround(10.0,5.0,1500.0);
    cCamSimul aCamSimul(aCenterGround,1000,2000);
    auto aCalibStenope = cPerspCamIntrCalib::SimpleCalib("Stenopee",aCaptSize,2500.0);
    aCamSimul.AddCam(aCalibStenope,true);
    aCamSimul.AddCam(aCalibStenope,true);
    const auto *mSens1 = aCamSimul.ListCam()[0];
    const auto *mSens2 = aCamSimul.ListCam()[1];

    auto p1 = mSens1->Ground2Image(aCenterGround);
    auto p2 = mSens2->Ground2Image(aCenterGround);
    auto b1 = mSens1->Image2Bundle(p1);

    StdOut() << "GroundCenter in image: " << p1 << "  " << p2 << std::endl;
    StdOut() << "GroundCenter: " << aCenterGround << std::endl;


//    auto [centre1,dir2,List12] = GenerateData(aImRect1, mSens1, aImRect2, mSens2, z1);
//    auto [centre2,dir1,List21] = GenerateData(aImRect2, mSens2, aImRect1, mSens1, z2);

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


    // TODOCM: Utilisation de  v1 O R et de R-1 O W1

    // TODOCM: jeu test !!
    delete aCalibStenope;
    return EXIT_SUCCESS;
}


cCollecSpecArg2007 & cAppli_EpipGeom::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
//           << Arg2007(mImName1,"name first image",{eTA2007::FileImage})
//           << Arg2007(mImName2,"name second image",{eTA2007::FileImage})
//           << mPhProj.DPOrient().ArgDirInMand()
        ;
}


cCollecSpecArg2007 & cAppli_EpipGeom::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return anArgOpt
           << AOpt2007(mDegree,"Degree","Poly degree",{eTA2007::HDV})
           << AOpt2007(mDegreeInv,"DegreeInv","Inv Poly degree",{eTA2007::HDV})
           << AOpt2007(mNbXY,"XYSteps","Nb XY steps",{eTA2007::HDV})
           << AOpt2007(mNbZ,"ZSteps","Nb Z steps",{eTA2007::HDV})
           << AOpt2007(mMinZ,"minZ","Z start",{eTA2007::HDV})
           << AOpt2007(mMaxZ,"maxZ","Z stop",{eTA2007::HDV})
        ;
}


std::tuple<cPt2dr, cPt2dr, cSetHomogCpleIm> cAppli_EpipGeom::GenerateData(
    const cRect2& aRect1,
    const cSensorImage *aSensor1,
    const cRect2& aRect2,
    const cSensorImage *aSensor2,
    double aZ
    ) const
{

    // TODOCM: Test d'epipolarisabilite ...

    // TODOCM: Determiner min & max Z (stenopee, RPC)
    // TODOCM: X & Y pas trop pres des bords ?

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

    std::cout << n << "/" << t << std::endl;

    return {c1,d2,std::move(aSetHomogCpleIm)};
}

std::pair<std::optional<cPt2dr>,cPt2dr> toto(const cSegmentCompiled<double,3>& aBundle, double anAbscissa1, double anAbscissa2, const cSensorImage *aSensor)
{
    auto p2g = aBundle.PtOfAbscissa(anAbscissa1);
    auto p2gp = aBundle.PtOfAbscissa(anAbscissa2);
    if (! aSensor->IsVisible(p2g) || !aSensor->IsVisible(p2gp))
        return {std::nullopt,{}};
    auto p2 = aSensor->Ground2Image(p2g);
    auto p2p = aSensor->Ground2Image(p2gp);
    return {p2, p2p};
}


std::tuple<cPt2dr, cPt2dr,cPt2dr, cPt2dr, cSetHomogCpleIm> cAppli_EpipGeom::GenerateData(
    const cSensorImage *aSensor1,
    const cSensorImage *aSensor2
    ) const
{
    cSetHomogCpleIm aSetHomogCpleIm;
    for (int i=0; i < 10000; i++) {
        auto aPtGround = GenGround(aSensor1,aSensor2);
        aSetHomogCpleIm.Add(cHomogCpleIm(aSensor1->Ground2Image(aPtGround),aSensor2->Ground2Image(aPtGround)));
    }
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
/*
            auto p2g = aBundle1.PtOfAbscissa(anAbscissa1 * aKz);
            auto p2gp = aBundle1.PtOfAbscissa(anAbscissa1 * (aKz+aZStep));
            if (aSensor2->IsVisible(p2g) && aSensor2->IsVisible(p2gp)) {
                auto p2 = aSensor2->Ground2Image(p2g);
                auto p2p = aSensor2->Ground2Image(p2gp);
                aList.Add(cHomogCpleIm(aHPair.mP1,p2));
                d2 += VUnit(p2p - p2);
                c1 += aHPair.mP1;
                c2 += aHPair.mP2;
                n2++;
            }
            auto p1g = aBundle2.PtOfAbscissa(anAbscissa2 * aKz);
            auto p1gp = aBundle2.PtOfAbscissa(anAbscissa2 * (aKz+aZStep));
            if (aSensor1->IsVisible(p1g) && aSensor1->IsVisible(p1gp)) {
                auto p1 = aSensor1->Ground2Image(p1g);
                auto p1p = aSensor1->Ground2Image(p1gp);
                aList.Add(cHomogCpleIm(p1,aHPair.mP2));
                d1 += VUnit(p1p - p1);
                c1 += aHPair.mP1;
                c2 += aHPair.mP2;
                n1++;
            }
*/
        }
    }
    c1 = c1 / double(n1+n2);
    c2 = c2 / double(n1+n2);

    d1 = VUnit(d1 / double(n1));
    d2 = VUnit(d2 / double(n2));

    return std::tuple(c1,c2,d1,d2,aList);
}

/* ==================================================== */

tMMVII_UnikPApli Alloc_EpipGeom(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_EpipGeom(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_EpipGeom
(
     "EpipGeom",
      Alloc_EpipGeom,
      "Epipolar geometry of two images",
      {eApF::Ori},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);

}; // MMVII

