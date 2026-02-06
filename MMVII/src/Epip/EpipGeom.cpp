#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "cPolyXY_N.h"

#include <vector>

/**
   \file EpipGeom.cpp


 */

namespace MMVII
{

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
#if 0
    cPolyXY_N<double> P(mDegree);

    cSetHomogCpleIm h;
    for (double x = -3.0; x<=3.0; x+= 0.1) {
        for (double y = -3.0; y<=3.0; y+= 0.1) {
            h.Add(cHomogCpleIm(cPt2dr(x,y),cPt2dr(x,f(x,y))));
        }
    }
/*
    for (int i=0; i<=PSin.Degree(); i++) {
        PSin.AddFixedK(0,i,0.0);
    }
    PSin.AddFixedK(0,1,1.0);
*/

    for (const auto& pair : h.SetH()) {
        P.AddObs(pair.mP1, pair.mP2.y());
    }
    P.Fit();
    printf("P var %lf\n",P.VarCurSol());
    for (int i=0; i< 10; i++) {
        cPt2dr p(RandInInterval(-3,3),RandInInterval(-3,3));
        auto v = f(p.x(),p.y());
        auto po = P(p);
        StdOut() << std::setw(11) << po << " " << std::setw(11) << v << " " << std::setw(11) << v - po  << " " << p << "\n";
    }

    cPolyXY_N<double> Q(mDegreeInv);
    for (auto& pair : h.SetH()) {
        Q.AddObs(pair.mP1.x(),P(pair.mP1),pair.mP1.y());
    }
    Q.Fit();
    printf("Q var %lf\n",Q.VarCurSol());
    for (int i=0; i< 10; i++) {
        cPt2dr p(RandInInterval(-3,3),RandInInterval(-3,3));
        cPt2dr p2(p.x(),Q(p.x(),P(p)));
        StdOut() << std::setw(11) << p << " " << std::setw(11) << p2 << " " << std::setw(11) << p - p2 << "\n";
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


    cPt3dr aCenterGround(10.0,5.0,20.0);
    cCamSimul aCamSimul(aCenterGround,10,20);
    auto aCalibStenope = cPerspCamIntrCalib::SimpleCalib("Stenopee",aCaptSize,1.0);
    aCamSimul.AddCam(aCalibStenope,false);
    aCamSimul.AddCam(aCalibStenope,false);
    const auto *mSens1 = aCamSimul.listCam()[0];
    const auto *mSens2 = aCamSimul.listCam()[1];

    auto p1 = mSens1->Ground2Image(aCenterGround);
    auto b1 = mSens1->Image2Bundle(p1);
    auto c1 = b1.P1() + (b1.P2() - b1.P1())*Norm2(b1.P1() - aCenterGround);
    auto c1p = b1.P1() + (b1.P2() - b1.P1())*(Norm2(b1.P1() - aCenterGround) + 1);

    auto P1 = mSens1->Ground2ImageAndDepth(aCenterGround);
    auto C1 = mSens1->ImageAndDepth2Ground(P1);
    auto C1p = mSens1->ImageAndDepth2Ground(P1+cPt3dr(0,0,1));

    auto p2 = mSens2->Ground2ImageAndDepth(aCenterGround);
    auto c2 = mSens2->ImageAndDepth2Ground(p2);
    auto c2p = mSens2->ImageAndDepth2Ground(p2+cPt3dr(0,0,1));

    StdOut() << "GroundCenter in image: " << p1 << "  " << P1 << std::endl;
    StdOut() << "GroundCenter: " << aCenterGround <<  "  " << c1 << "  " << C1 << std::endl;
    StdOut() << "GroundCenter': " << aCenterGround <<  "  " << c1p << "  " << C1p << std::endl;

    StdOut() << "GroundCenter in image: " << p1 << "  " << p2 << std::endl;
    StdOut() << "GroundCenter: " << aCenterGround <<  "  " << c1 << "  " << c2 << std::endl;
    StdOut() << "GroundCenter': " << aCenterGround <<  "  " << c1p << "  " << c2p << std::endl;

    auto z1 = Norm2(mSens1->Center() - aCenterGround);
    auto z2 = Norm2(mSens2->Center() - aCenterGround);

    StdOut() << "Z: " << z1 << "," << z2 << std::endl;

//    auto [centre1,dir2,List1] = GenerateData(aImRect1, mSens1, aImRect2, mSens2, z1);
//    auto [centre2,dir1,List2] = GenerateData(aImRect2, mSens2, aImRect1, mSens1, z2);

    auto [centre1,centre2, dir1, dir2, List1] = GenerateData(mSens1, mSens2);
    auto List2 = List1;

    StdOut() << "Centre1, dir1: " << centre1 << " " << dir1 << " " << Norm2(dir1) << std::endl;
    StdOut() << "Centre2, dir2: " << centre2 << "," << dir2 << " " << Norm2(dir2) << std::endl;

    // TODOCM: Check almost epip
    // if ((dir2.x+dir1.x) <0)
    // {
    //     dir1 = -dir1;
    //     dir2 = -dir2;
    // }

    // TODOCM: Check d1 and d2 /= 0
    // pair.p1 from Im1 and pair.p2 from Im2
    for (auto& pair : List1.SetH()) {
        pair.mP1 = (pair.mP1 - centre1) / dir1;
        pair.mP2 = (pair.mP2 - centre2) / dir2;
    }

    // pair.p1 from Im2 and pair.p2 from Im1
//    for (auto& pair : List2.SetH()) {
//        pair.mP1 = (pair.mP1 - centre2) / dir2;
//        pair.mP2 = (pair.mP2 - centre1) / dir1;
//    }

/*
    std::cout <<  "Norm List p1,p2: " << std::endl;
    for (const auto& pair : List1)
        std::cout << pair.p1 << " -> " << pair.p2 << std::endl;
  */

    /* V1 Calculus */
    /* v1(x,y) = y + S(i=1->d, S(j=0->d-i; C(i,j) * x^i * y^j ) )
     * x = p1.x(); y = p1.y(); v1(x,y) = p2.y()
     */
    cPolyXY_N<double> V1(mDegree);
    for (int i=0; i<=V1.Degree(); i++) {
        V1.AddFixedK(0,i,0.0);
    }
    V1.AddFixedK(0,1,1.0);
    for (auto& pair : List1.SetH()) {
        V1.AddObs(pair.mP1, pair.mP2.y());
    }
    V1.Fit();
    printf("V1 var %lf\n",V1.VarCurSol());

    /* V2 Calculus */
    /* v2(x,y) = S(i=0->d; S(j->d-i; C(i,j) * x^i * y^j ) )
     * x = p1.x(); y = p1.y(); v2(x,y) = p2.y()
     */
    cPolyXY_N<double> V2(mDegree);
    for (auto& pair : List2.SetH()) {
        V2.AddObs(pair.mP1,pair.mP2.y());
    }
    V2.Fit();
    printf("V2 var %lf\n",V2.VarCurSol());


    // TODOCM: Calcul fonctions inverses W1 W2
    cPolyXY_N<double> W1(mDegreeInv);
    for (auto& pair : List1.SetH()) {
        W1.AddObs(pair.mP1.x(),V1(pair.mP1),pair.mP1.y());
    }
    W1.Fit();
    printf("W1 var %lf\n",W1.VarCurSol());

    cPolyXY_N<double> W2(mDegreeInv);
    for (auto& pair : List2.SetH()) {
        W2.AddObs(pair.mP1.x(),V2(pair.mP1),pair.mP1.y());
    }
    W2.Fit();
    printf("W2 var %lf\n",W2.VarCurSol());

    // TODOCM: Utilisation de  v1 O R et de R-1 O W1

    // TODOCM: jeu test !!
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
    // TODOCM: Determiner min & max Z (stenopee, RPC)
    // TODOCM: X & Y pas trop pres des bords ?

    cSetHomogCpleIm aSetHomogCpleIm;
    auto c1 = cPt2dr(0,0);
    auto d2  = cPt2dr(0,0);

    auto C1 = cPt2dr(1000,750);
    auto C2 = cPt2dr(1000,750);

    auto inter_g = aSensor1->PInterBundle(cHomogCpleIm(C1,C2),*aSensor2);
    auto inter_i = aSensor1->Ground2ImageAndDepth(inter_g);
    aZ = inter_i.z();
    auto aMinZ = aZ - 10;
    auto aMaxZ = aZ + 10;

    StdOut() << "aZ : " << aZ << " " << aSensor1->ImageAndDepth2Ground(inter_i) << "\n";

    int t = 0;
    unsigned n = 0;
    double deltaZ = (aMaxZ - aMinZ) / mNbZ;
    int deltaX = aRect1.Sz().x() / mNbXY;
    int deltaY = aRect1.Sz().y() / mNbXY;

    for (int x = aRect1.P0().x(); x < aRect1.P1().x(); x += deltaX) {
        for (int y = aRect1.P0().y(); y < aRect1.P1().y(); y += deltaY) {
            auto p1 = cPt2dr(x,y);
            for (double z = aMinZ; z < aMaxZ; z += deltaZ) {
                t++;
                auto p2 = aSensor2->Ground2Image(aSensor1->ImageAndDepth2Ground(p1,z));
                if (! aRect2.InsideBL(p2))
                    continue;
                auto p2p = aSensor2->Ground2Image(aSensor1->ImageAndDepth2Ground(p1,z+deltaZ));
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

std::pair<cPt2dr, cPt2dr> cAppli_EpipGeom::GenerateData(
    const cRect2& aRect1,
    const cSensorImage *aSensor1,
    const cRect2& aRect2,
    const cSensorImage *aSensor2,
    const cSetHomogCpleIm &aSetHomogCpleIm
    ) const
{
    // TODOCM: Determiner min & max Z (stenopee, RPC)
    // TODOCM: X & Y pas trop pres des bords ?

    auto c1 = cPt2dr(0,0);
    auto c2  = cPt2dr(0,0);
    auto d2  = cPt2dr(0,0);
    for (const auto& aHomog : aSetHomogCpleIm.SetH()) {
        c1 += aHomog.mP1;
        c2 += aHomog.mP2;
//        tPt3dr aResult = aSensor1->PInterBundle(aHomog,*aSensor2);
    }
/*    int t = 0;
    unsigned n = 0;
    double deltaZ = (aMaxZ - aMinZ) / mNbZ;
    int deltaX = aRect1.Sz().x() / mNbXY;
    int deltaY = aRect1.Sz().y() / mNbXY;
    for (int x = aRect1.P0().x(); x < aRect1.P1().x(); x += deltaX) {
        for (int y = aRect1.P0().y(); y < aRect1.P1().y(); y += deltaY) {
            for (double z = aMinZ; z < aMaxZ; z += deltaZ) {
                t++;
                auto p1 = cPt2dr(x,y);
                auto p1_z = cPt3dr(x,y,z);
                auto p1p_z = cPt3dr(x,y,z+deltaZ);
                auto p2 = aSensor2->Ground2Image(aSensor1->ImageAndZ2Ground(p1_z));
                if (! aRect2.InsideBL(p2))
                    continue;
                auto p2p = aSensor2->Ground2Image(aSensor1->ImageAndZ2Ground(p1p_z));
                if (! aRect2.InsideBL(p2p))
                    continue;
                n ++;
                aSetHomogCpleIm.Add(cHomogCpleIm(p1,p2));
                c1 += p1;
                auto v2 = p2p - p2;
                d2 += VUnit(v2);
            }
        }
    }
    c1 = c1 / double(n);
    d2 = d2 / double(n);

    std::cout << n << "/" << t << std::endl;
*/
    return {c1,d2};
}

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
    double stheta2 =0;
    double vtheta2 =0;
    double stheta1 =0;
    double vtheta1 =0;

    for (const auto &aHPair : aSetHomogCpleIm.SetH()) {
        c1 += aHPair.mP1;
        c2 += aHPair.mP2;
        tPt3dr aPtInter= aSensor1->PInterBundle(aHPair,*aSensor2);

        auto p1i3 = aSensor1->Ground2ImageAndDepth(aPtInter);
        auto p2i2 = aSensor2->Ground2Image(aSensor1->ImageAndDepth2Ground(p1i3 + cPt3dr(0,0,-1)));
        auto v2 = aHPair.mP2 - p2i2;
        d2 += VUnit(v2);
        auto theta = std::atan2(v2.y(),v2.x());
        stheta2 += theta;
        vtheta2 += theta*theta;
        p2i2 = aSensor2->Ground2Image(aSensor1->ImageAndDepth2Ground(p1i3 + cPt3dr(0,0,1)));
        v2 = p2i2 - aHPair.mP2;
        d2 += VUnit(v2);
        stheta2 += theta;
        vtheta2 += theta*theta;

        auto p2i3 = aSensor2->Ground2ImageAndDepth(aPtInter);
        auto p1i2 = aSensor1->Ground2Image(aSensor2->ImageAndDepth2Ground(p2i3 + cPt3dr(0,0,-1)));
        auto v1 = aHPair.mP1 - p1i2;
        d1 += VUnit(v1);
        theta = std::atan2(v1.y(),v1.x());
        stheta1 += theta;
        vtheta1 += theta*theta;
        p1i2 = aSensor2->Ground2Image(aSensor1->ImageAndDepth2Ground(p2i3 + cPt3dr(0,0,1)));
        v1 = p1i2 - aHPair.mP1;
        d1 += VUnit(v1);
        theta = std::atan2(v1.y(),v1.x());
        stheta1 += theta;
        vtheta1 += theta*theta;
    }

    auto n = (double)aSetHomogCpleIm.SetH().size() * 2;
    c1 = c1 / n;
    c2 = c2 / n;
    d1 = d1 / n;
    d2 = d2 / n;


    StdOut() << "Theta1 : " << stheta1 / n << " Sig: " << sqrt((vtheta1 - (stheta1*stheta1) / n) /n) << "\n";
    StdOut() << "Theta2 : " << stheta2 / n << " Sig: " << sqrt((vtheta2 - (stheta2*stheta2) / n) /n) << "\n";
    return std::tuple(c1,c2,d1,d2,aSetHomogCpleIm);
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

