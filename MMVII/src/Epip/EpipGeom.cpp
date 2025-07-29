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
    int mDeltaXY;
    int mDeltaZ;
    int mMinZ;
    int mMaxZ;
};

cAppli_EpipGeom::cAppli_EpipGeom (
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli & aSpec
    )
    : cMMVII_Appli  (aVArgs,aSpec)
    , mPhProj       (*this)
    , mDeltaXY(100)
    , mDeltaZ(100)
    , mMinZ(100)
    , mMaxZ(1000)
{
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
           << AOpt2007(mDeltaXY,"deltaXY","XY step",{eTA2007::HDV})
           << AOpt2007(mDeltaZ,"deltaXY","XY step",{eTA2007::HDV})
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
    cHCompatList HCompatList;
    auto c1 = cPt2dr(0,0);
    auto d2  = cPt2dr(0,0);
    int t = 0;
    unsigned n = 0;
    for (int x = aRect1.P0().x(); x < aRect1.P1().x(); x += mDeltaXY) {
        for (int y = aRect1.P0().y(); y < aRect1.P1().y(); y += mDeltaXY) {
            for (int z = mMinZ; z < mMaxZ; z += mDeltaZ) {
                t++;
                auto p1 = cPt2dr(x,y);
                auto p1_z = cPt3dr(x,y,z);
                auto p1p_z = cPt3dr(x,y,z+mDeltaZ);
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
    
    auto [c1,d2,List1] = GenerateData(mImRect1, mSens1, mImRect2, mSens2);
    auto [c2,d1,List2] = GenerateData(mImRect2, mSens2, mImRect1, mSens1);
    
    
    std::cout <<  "List p1,p2: " << std::endl;
    for (const auto& pair : List1)
        std::cout << pair.p1 << " -> " << pair.p2 << std::endl;
    
    // TODO: Check d1 and d2 /= 0
    // pair.p1 from Im1 and pair.p2 from Im2
    for (auto& pair : List1) {
        pair.p1 = (pair.p1 - c1) / d1;
        pair.p2 = (pair.p2 - c2) / d2;
    }
    
    // pair.p1 from Im2 and pair.p2 from Im1
    for (auto& pair : List2) {
        pair.p1 = (pair.p1 - c2) / d2;
        pair.p2 = (pair.p2 - c1) / d1;
    }
    
    
    std::cout <<  "Norm List p1,p2: " << std::endl;
    for (const auto& pair : List1)
        std::cout << pair.p1 << " -> " << pair.p2 << std::endl;
    
    
    return EXIT_SUCCESS;
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

