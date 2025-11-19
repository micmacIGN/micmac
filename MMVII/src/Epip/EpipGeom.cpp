#include "cMMVII_Appli.h"
#include "MMVII_Sensor.h"


/**
   \file EpipGeom.cpp


 */

namespace MMVII
{

struct cHCompat
{
    cPt2dr p1;
    cPt2dr p2;
};

typedef std::vector<cHCompat> cHCompatList;

class cAppli_EpipGeom : public cMMVII_Appli
{
public :
    
    cAppli_EpipGeom(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;
    
private :
    std::tuple<cPt2dr, cPt2dr, cHCompatList> GenerateData(
        const cRect2 &aRect1,
        cSensorImage *aSensor1,
        const cRect2 &aRect2,
        cSensorImage *aSensor2
    );
    
    cPhotogrammetricProject  mPhProj;
    std::string  mImName1;
    std::string  mImName2;
    cSensorImage *mSens1;
    cSensorImage *mSens2;
    int mDegre;
    int mNbXY;
    int mNbZ;
    int mMinZ;
    int mMaxZ;
};

cAppli_EpipGeom::cAppli_EpipGeom (
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli & aSpec
    )
    : cMMVII_Appli  (aVArgs,aSpec)
    , mPhProj       (*this)
    , mDegre(5)
    , mNbXY(100)
    , mNbZ(3)
    , mMinZ(100)
    , mMaxZ(1000)
{
}

int cAppli_EpipGeom::Exe()
{
    mPhProj.FinishInit();
    
    // mIm1 = cDataFileIm2D::Create(mIm1);
    auto mImRect1 = cRect2(cDataFileIm2D::Create(mImName1,eForceGray::No).Sz());
    auto mImRect2 = cRect2(cDataFileIm2D::Create(mImName1,eForceGray::No).Sz());
    mSens1 = mPhProj.ReadSensor(mImName1,true);
    mSens2 = mPhProj.ReadSensor(mImName2,true);
    
    StdOut() << mImRect1 << " " <<  mImRect2 << std::endl;
    
    auto x = mSens1->Image2Bundle(cPt2dr(0,0));
    StdOut() << x.P1() << "," << x.P2() << std::endl;
    StdOut() << mSens1->Ground2Image(x.P2()) << std::endl;
    
    auto [centre1,dir2,List1] = GenerateData(mImRect1, mSens1, mImRect2, mSens2);
    auto [centre2,dir1,List2] = GenerateData(mImRect2, mSens2, mImRect1, mSens1);
    
    // TODOCM: Check almost epip
    // if ((dir2.x+dir1.x) <0)
    // {
    //     dir1 = -dir1;
    //     dir2 = -dir2;
    // }
   
    // TODOCM: Check d1 and d2 /= 0
    // pair.p1 from Im1 and pair.p2 from Im2
    for (auto& pair : List1) {
        pair.p1 = (pair.p1 - centre1) / dir1;
        pair.p2 = (pair.p2 - centre2) / dir2;
    }
    
    // pair.p1 from Im2 and pair.p2 from Im1
    for (auto& pair : List2) {
        pair.p1 = (pair.p1 - centre2) / dir2;
        pair.p2 = (pair.p2 - centre1) / dir1;
    }
    
/*    
    std::cout <<  "Norm List p1,p2: " << std::endl;
    for (const auto& pair : List1)
        std::cout << pair.p1 << " -> " << pair.p2 << std::endl;
  */
    
    /* V1 Calculus */
    /* v1(x,y) = y + S(i=1->d, S(j=0->d-i; C(i,j) * x^i * y^j ) )
     * x = p1.x(); y = p1.y(); v1(x,y) = p2.y()
     */
    int nbVar1 = (mDegre+1) * (mDegre+2) / 2  - (mDegre + 1);
    cDenseVect<tREAL8> v1(nbVar1);
    cLeasSqtAA<tREAL8> leastSq1(nbVar1);
    cDenseVect<tREAL8> coeffs1(nbVar1);
    
    for (auto& pair : List1) {
        int n = 0;
        tREAL8 x = pair.p1.x();
        for (int i=1; i<=mDegre; i++) {
            tREAL8 y = 1;
            for (int j=0; j<=mDegre-i; j++) {
                coeffs1(n) = x*y;
                y *= pair.p1.y();
                n++;
            }
            x *= pair.p1.x();
        }
        leastSq1.PublicAddObservation(1,coeffs1,pair.p2.y() - pair.p1.y());
    }
    v1 = leastSq1.PublicSolve();
    printf("V1 var %lf\n",leastSq1.VarCurSol());
    
    /* V2 Calculus */
    /* v2(x,y) = S(i=0->d; S(j->d-i; C(i,j) * x^i * y^j ) )
     * x = p1.x(); y = p1.y(); v2(x,y) = p2.y()
     */
    int nbVar2 = (mDegre+1) * (mDegre+2) / 2 ;
    cDenseVect<tREAL8> v2(nbVar2);
    cLeasSqtAA<tREAL8> leastSq2(nbVar2);
    cDenseVect<tREAL8> coeffs2(nbVar2);
    
    for (auto& pair : List2) {
        int n = 0;
        tREAL8 x = 1;
        for (int i=0; i<=mDegre; i++) {
            tREAL8 y = 1;
            for (int j=0; j<=mDegre-i; j++) {
                coeffs2(n) = x*y;
                y *= pair.p1.y();
                n++;
            }
            x *= pair.p1.x();
        }
        leastSq2.PublicAddObservation(1,coeffs2,pair.p2.y());
    }
    v2 = leastSq2.PublicSolve();
    printf("V2 var %lf\n",leastSq2.VarCurSol());
    
    
    // TODOCM: Calcul fonctions inverses W1 W2
    
    // TODOCM: Utilisation de  v1 O R et de R-1 O W1
    
    // TODOCM: jeu test !!
    return EXIT_SUCCESS;
}


cCollecSpecArg2007 & cAppli_EpipGeom::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           << Arg2007(mImName1,"name first image",{eTA2007::FileImage})
           << Arg2007(mImName2,"name second image",{eTA2007::FileImage})
           << mPhProj.DPOrient().ArgDirInMand()
        ;
}


cCollecSpecArg2007 & cAppli_EpipGeom::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    
    return anArgOpt
           << AOpt2007(mDegre,"Degre","Poly degre",{eTA2007::HDV})
           << AOpt2007(mNbXY,"XYSteps","Nb XY steps",{eTA2007::HDV})
           << AOpt2007(mNbZ,"ZSteps","Nb Z steps",{eTA2007::HDV})
           << AOpt2007(mMinZ,"minZ","Z start",{eTA2007::HDV})
           << AOpt2007(mMaxZ,"maxZ","Z stopt",{eTA2007::HDV})
        ;
}


std::tuple<cPt2dr, cPt2dr, cHCompatList> cAppli_EpipGeom::GenerateData(
    const cRect2& aRect1,
    cSensorImage *aSensor1,
    const cRect2& aRect2,
    cSensorImage *aSensor2
    )
{
    // TODOCM: Determiner min & max Z (stenopee, RPC)
    // TODOCM: X & Y pas trop pres des bords ?
    
    
    cHCompatList HCompatList;
    auto c1 = cPt2dr(0,0);
    auto d2  = cPt2dr(0,0);
    int t = 0;
    unsigned n = 0;
    int deltaZ = (mMaxZ - mMinZ) / mNbZ;
    int deltaX = aRect1.Sz().x() / mNbXY;
    int deltaY = aRect1.Sz().y() / mNbXY;
    for (int x = aRect1.P0().x(); x < aRect1.P1().x(); x += deltaX) {
        for (int y = aRect1.P0().y(); y < aRect1.P1().y(); y += deltaY) {
            for (int z = mMinZ; z < mMaxZ; z += deltaZ) {
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
                HCompatList.push_back({p1,p2});
                c1 += p1;
                auto v2 = p2p - p2;
                d2 += v2 / Norm2(v2);
            }
        }
    }
    c1 = c1 / double(n);
    d2 = d2 / double(n);
    
    std::cout << n << "/" << t << std::endl;

    return {c1,d2,std::move(HCompatList)};
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

