#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"

#include <vector>

/**
   \file EpipGeom.cpp


 */

namespace MMVII
{
// /////////////////////////////////// POLY
template<typename T>
class cPolyXY_N
{
public:
    explicit cPolyXY_N(int degree);
    explicit cPolyXY_N(std::initializer_list<T> aVK);
    explicit cPolyXY_N(const std::vector<T>& aVK);

    T operator()(const T& x, const T& y);
    T operator()(const cPtxd<T,2> aPt);

    const T& K(int i, int j) const;
    T& K(int i, int j);
    const std::vector<T>& VK() const;
    void SetVK(const std::vector<T>& aVK);

    int Degree() const;
    int NbCoeffs() const;

    void ResetFit();
    void AddFixedK(int i, int j, const T& k);
    void AddObs(const T& x, const T& y, const T& v, const T& aWeight=1);
    void AddObs(const cPtxd<T,2> aPt, const T& v, const T& aWeight=1);
    void Fit();
    T VarCurSol() const;

private:
    int idx(int i, int j);
    int mDegree;
    std::vector<T> mVK;
    std::unique_ptr<cLeasSqtAA<T>> mLeastSq;
    std::map<std::pair<int,int>, T> mFixedK;
};


template<typename T>
cPolyXY_N<T>::cPolyXY_N(int degree)
: mDegree(degree)
, mVK((degree+1) * (degree+2) / 2)
{
}

template<typename T>
cPolyXY_N<T>::cPolyXY_N(std::initializer_list<T> aVK)
    : mDegree(static_cast<int>(std::sqrt(aVK.size()*2)) - 1)
    , mVK(aVK)
{
    MMVII_INTERNAL_ASSERT_medium((mDegree+1)*(mDegree+2)/2 == (int)aVK.size(),"Incorrect vector for cPolyXY_N coefficients");
}

template<typename T>
cPolyXY_N<T>::cPolyXY_N(const std::vector<T>& aVK)
    : mDegree(static_cast<int>(std::sqrt(aVK.size()*2)) - 1)
    , mVK(aVK)
{
    MMVII_INTERNAL_ASSERT_medium((mDegree+1)*(mDegree+2)/2 == (int)aVK.size(),"Incorrect vector for cPolyXY_N coefficients");
}

template<typename T>
T cPolyXY_N<T>::operator()(const T &x, const T &y)
{
    T X_n;
    T Y_n;
    T result;
    int n;

    X_n = 1;            // X^0
    n = 0;
    result = T{};
    for (int i=0; i<=mDegree; i++) {
        Y_n = 1;        // Y^0
        for (int j=0; j<=mDegree-i; j++) {
            result += mVK[n] * X_n * Y_n;
            Y_n *= y;   // Y^(j+1)
            n++;
        }
        X_n *= x;       // X^(i+1)
    }
    return result;
}

template<typename T>
T cPolyXY_N<T>::operator()(const cPtxd<T, 2> aPt)
{
    return this->operator()(aPt.x(),aPt.y());
}

template<typename T>
const T &cPolyXY_N<T>::K(int i, int j) const
{
    return mVK[idx(i,j)];
}

template<typename T>
T& cPolyXY_N<T>::K(int i, int j)
{
    return mVK[idx(i,j)];
}

template<typename T>
const std::vector<T> &cPolyXY_N<T>::VK() const
{
    return mVK;
}

template<typename T>
void cPolyXY_N<T>::SetVK(const std::vector<T>& aVK)
{
    MMVII_INTERNAL_ASSERT_medium(aVK.size() == mVK.size(),"Incorrect vector for cPolyXY_N coefficients");
    mVK = aVK;
}

template<typename T>
int cPolyXY_N<T>::Degree() const
{
    return mDegree;
}

template<typename T>
int cPolyXY_N<T>::NbCoeffs() const
{
    return mVK.size();
}


template<typename T>
void cPolyXY_N<T>::ResetFit()
{
    mLeastSq.reset();
    mFixedK.clear();
}

template<typename T>
void cPolyXY_N<T>::AddFixedK(int i, int j, const T &k)
{
    MMVII_INTERNAL_ASSERT_medium(i>=0 && j>=0 && i+j<=mDegree,"Bad usage of cPolyXY_N::AddFixedK()");
    MMVII_INTERNAL_ASSERT_medium(! mLeastSq,"Can't add fixed K after obs in cPolyXY_N::AddFixedK()");
    mFixedK.insert_or_assign(std::make_pair(i,j),k);
}

template<typename T>
void cPolyXY_N<T>::AddObs(const T &x, const T &y, const T &v, const T& aWeight)
{
    if (! mLeastSq)
        mLeastSq = std::make_unique<cLeasSqtAA<T>>(NbCoeffs() - mFixedK.size());
    cDenseVect<T> coeffs(NbCoeffs() - mFixedK.size());
    int n = 0;
    T aDiffObs = 0;
    T X_n = 1;
    for (int i=0; i<=mDegree; i++) {
        T Y_n = 1;
        for (int j=0; j<=mDegree-i; j++) {
            auto itFixedK = mFixedK.find({i,j});
            if ( itFixedK == mFixedK.end()) {
                coeffs(n) = X_n * Y_n;
                n++;
            } else {
                aDiffObs += itFixedK->second * X_n * Y_n;
            }
            Y_n *= y;
        }
        X_n *= x;
    }
    mLeastSq->PublicAddObservation(aWeight,coeffs, v - aDiffObs);
}

template<typename T>
void cPolyXY_N<T>::AddObs(const cPtxd<T, 2> aPt, const T &v, const T &aWeight)
{
    AddObs(aPt.x(),aPt.y(),v,aWeight);
}

template<typename T>
void cPolyXY_N<T>::Fit()
{
    auto aVK = mLeastSq->PublicSolve();
    int n=0;
    int m=0;
    for (int i=0; i<= mDegree; i++) {
        for (int j=0; j<=mDegree-i; j++) {
            auto itFixedK = mFixedK.find({i,j});
            if ( itFixedK == mFixedK.end()) {
                mVK[n] = aVK(m);
                m++;
            } else {
                mVK[n] = itFixedK->second;
            }
            n++;
        }
    }
}

template<typename T>
T cPolyXY_N<T>::VarCurSol() const
{
    return mLeastSq->VarCurSol();
}

template<typename T>
int cPolyXY_N<T>::idx(int i, int j)
{
    MMVII_INTERNAL_ASSERT_medium(i>=0 && j>=0 && i+j<=mDegree,"Bad usage of cPolyXY_N");
    return j + ((mDegree + 1) * (mDegree + 2) - (mDegree - i + 1) * (mDegree - i + 2)) / 2;
}

// /////////////////////////////////// END POLY



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
        );

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
    cPolyXY_N<double> P(mDegre);

    cHCompatList l;
    for (double x = -3.0; x<=3.0; x+= 0.1) {
        for (double y = -3.0; y<=3.0; y+= 0.1) {
            l.push_back({cPt2dr(x,y),cPt2dr(x,f(x,y))});
        }
    }
/*
    for (int i=0; i<=PSin.Degree(); i++) {
        PSin.AddFixedK(0,i,0.0);
    }
    PSin.AddFixedK(0,1,1.0);
*/

    for (const auto& pair : l) {
        P.AddObs(pair.p1, pair.p2.y());
    }
    P.Fit();
    printf("P var %lf\n",P.VarCurSol());
    for (int i=0; i< 10; i++) {
        cPt2dr p(RandInInterval(-3,3),RandInInterval(-3,3));
        auto v = f(p.x(),p.y());
        auto po = P(p);
        StdOut() << std::setw(11) << po << " " << std::setw(11) << v << " " << std::setw(11) << v - po  << " " << p << "\n";
    }

    cPolyXY_N<double> Q(mDegreInv);
    for (auto& pair : l) {
        Q.AddObs(pair.p1.x(),P(pair.p1),pair.p1.y());
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
    auto aImRect1 = cRect2(cPt2di(0,0),aCaptSize);
    auto aImRect2 = cRect2(cPt2di(0,0),aCaptSize);


    cPt3dr aCenterGround(10.0,5.0,20.0);
    cCamSimul aCamSimul(aCenterGround,110,200);
    auto aCalibStenope = cPerspCamIntrCalib::SimpleCalib("Stenopee",aCaptSize,1.0);
    aCamSimul.AddCam(aCalibStenope,false);
    aCamSimul.AddCam(aCalibStenope,false);
    const auto *mSens1 = aCamSimul.listCam()[0];
    const auto *mSens2 = aCamSimul.listCam()[1];

    auto p1 = mSens1->Ground2Image(aCenterGround);
    auto p2 = mSens2->Ground2Image(aCenterGround);
    StdOut() << "Center gound in image: " << p1 << "  " << p2 << std::endl;

    auto z1 = Norm2(mSens1->Center() - aCenterGround);
    auto z2 = Norm2(mSens2->Center() - aCenterGround);

    StdOut() << "Z: " << z1 << "," << z2 << std::endl;

    auto [centre1,dir2,List1] = GenerateData(aImRect1, mSens1, aImRect2, mSens2, z1);
    auto [centre2,dir1,List2] = GenerateData(aImRect2, mSens2, aImRect1, mSens1, z2);

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
    for (auto& pair : List2.SetH()) {
        pair.mP1 = (pair.mP1 - centre2) / dir2;
        pair.mP2 = (pair.mP2 - centre1) / dir1;
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
    )
{
    // TODOCM: Determiner min & max Z (stenopee, RPC)
    // TODOCM: X & Y pas trop pres des bords ?

    auto aMinZ = aZ - 20.0;
    auto aMaxZ = aZ + 20.0;
    cSetHomogCpleIm aSetHomogCpleIm;
    auto c1 = cPt2dr(0,0);
    auto d2  = cPt2dr(0,0);
    int t = 0;
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
                d2 += v2 / Norm2(v2);
            }
        }
    }
    c1 = c1 / double(n);
    d2 = d2 / double(n);

    std::cout << n << "/" << t << std::endl;

    return {c1,d2,std::move(aSetHomogCpleIm)};
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
}

/* ======================================================================================================== */
/* ======================================================================================================== */
/* ======================================================================================================== */
#include "MMVII_PCSens.h"

namespace  MMVII {

class cAppli_SimulEpip : public cMMVII_Appli
{
public :

    cAppli_SimulEpip(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

private :
    int CreateCam(const std::string& aNameImage);
    std::string  mImName1;
    std::string  mImName2;
    cPt2di mSize = cPt2di(2000,2000);
    tREAL8 mFocale = 2000;
    cPerspCamIntrCalib *mCalibStenope;
};

cAppli_SimulEpip::cAppli_SimulEpip (
    const std::vector<std::string> &  aVArgs,
    const cSpecMMVII_Appli & aSpec
    )
    : cMMVII_Appli  (aVArgs,aSpec)
{
}

int cAppli_SimulEpip::CreateCam(const std::string& aNameImage)
{
    cSensorCamPC::tPose aPose;
    cSensorCamPC Cam(aNameImage,aPose,mCalibStenope);
    return 0;
}

int cAppli_SimulEpip::Exe()
{

    return 0;
}

cCollecSpecArg2007 & cAppli_SimulEpip::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           << Arg2007(mImName1,"name first image",{eTA2007::FileImage})
           << Arg2007(mImName2,"name second image",{eTA2007::FileImage})
        ;
}


cCollecSpecArg2007 & cAppli_SimulEpip::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return anArgOpt
//           << AOpt2007(mDegree,"Degree","Poly degree",{eTA2007::HDV})
    ;
}



/* ==================================================== */

tMMVII_UnikPApli Alloc_SimulEpip(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_SimulEpip(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_SimulEpip
(
     "EpipSimul",
      Alloc_SimulEpip,
      "Create simulated dataset for EpipGeom",
      {eApF::Test},
      {eApDT::Orient},
      {eApDT::Orient},
      __FILE__
);



}; // MMVII

