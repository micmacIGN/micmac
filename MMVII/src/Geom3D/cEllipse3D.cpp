#include "MMVII_Geom3D.h"

namespace MMVII
{

/* *********************************************************** */
/*                                                             */
/*                         cEllipse3D                          */
/*                                                             */
/* *********************************************************** */

cEllipse3D::cEllipse3D() :
    mCDG(cPt3dr(0,0,0)),
    mSxx(0),
    mSyy(0),
    mSzz(0),
    mSxy(0),
    mSxz(0),
    mSyz(0),
    mPds(0),
    mNorm(false)
{}

void cEllipse3D::AddData(const cPt3dr& aP, double aPds)
{
    MMVII_INTERNAL_ASSERT_strong(!mNorm, "Ellipse3D is normalised");
    mCDG +=  aP*aPds;

    mSxx += aPds *aP.x() * aP.x();
    mSyy += aPds *aP.y() * aP.y();
    mSzz += aPds *aP.z() * aP.z();
    mSxy += aPds *aP.x() * aP.y();
    mSxz += aPds *aP.x() * aP.z();
    mSyz += aPds *aP.y() * aP.z();

    mPds += aPds;
}

void cEllipse3D::Normalise()
{
    MMVII_INTERNAL_ASSERT_strong(!mNorm, "Ellipse3D is normalised");

    mNorm = true;

    double aPds = mPds;
    mCDG = mCDG/aPds;
    cPt3dr aCdg = mCDG;

    mSxx = mSxx/aPds - aCdg.x() * aCdg.x();
    mSyy = mSyy/aPds - aCdg.y() * aCdg.y();
    mSzz = mSzz/aPds - aCdg.z() * aCdg.z();
    mSxy = mSxy/aPds - aCdg.x() * aCdg.y();
    mSxz = mSxz/aPds - aCdg.x() * aCdg.z();
    mSyz = mSyz/aPds - aCdg.y() * aCdg.z();

}

void cEllipse3D::Reset()
{
    mCDG = cPt3dr(0,0,0);
    mSxx = 0;
    mSyy = 0;
    mSzz = 0;
    mSxy = 0;
    mSxz = 0;
    mSyz = 0;
    mPds = 0;
    mNorm = false;
}

cPt3dr & cEllipse3D::CDG()
{
    return mCDG;
}

const cPt3dr & cEllipse3D::CDG()const
{
    return mCDG;
}


double & cEllipse3D::Sxx()
{
    return mSxx;
}

const double & cEllipse3D::Sxx()const
{
    return mSxx;
}


double & cEllipse3D::Syy()
{
    return mSyy;
}

const double & cEllipse3D::Syy()const
{
    return mSyy;
}


double & cEllipse3D::Szz()
{
    return mSzz;
}

const double & cEllipse3D::Szz()const
{
    return mSzz;
}


double & cEllipse3D::Sxy()
{
    return mSxy;
}

const double & cEllipse3D::Sxy()const
{
    return mSxy;
}


double & cEllipse3D::Sxz()
{
    return mSxz;
}

const double & cEllipse3D::Sxz()const
{
    return mSxz;
}


double & cEllipse3D::Syz()
{
    return mSyz;
}

const double & cEllipse3D::Syz()const
{
    return mSyz;
}
double & cEllipse3D::Pds()
{
    return mPds;
}

const double & cEllipse3D::Pds()const
{
    return mPds;
}


bool & cEllipse3D::Norm()
{
    return mNorm;
}

const bool & cEllipse3D::Norm()const
{
    return mNorm;
}

// value that corrects for the fact that the function is an approximation
static const double A = 0.147;

double InvErrFonc(double x)
{
    static double DePiSA = 2/(M_PI*A);

    double X2 = x * x;
    double Log1MX2 = log(1-X2);

    double aV1 = Square(DePiSA + Log1MX2/2)  - Log1MX2/A;
    aV1 = sqrt(std::max(0.0,aV1)) - (DePiSA+Log1MX2/2);

    aV1 = sqrt(std::max(0.0,aV1));


    return (x>0) ? aV1 : (-aV1);
}

double InvErrFoncRationel(int P,int Q)
{
    int aSign = (P>0 ? 1 : -1) * (Q>0 ? 1 : -1);
    P = abs(P);
    Q = abs(Q);
    static std::vector<std::vector<double>  > mBuf;

    for (int q=mBuf.size() ; q<=Q ; q++)
    {
        mBuf.push_back(std::vector<double>());
        for (int p=0 ; p<q ; p++)
            mBuf.back().push_back(InvErrFonc(p/double(q)));
    }

    return mBuf[Q][P] * aSign;
}
static double FactCorrectif(int aNb)
{
    return sqrt(2.) / (1 - 0.3333/(aNb+0.5));
}

/* *********************************************************** */
/*                                                             */
/*                         cGenGaus3D                          */
/*                                                             */
/* *********************************************************** */

cGenGauss3D::cGenGauss3D(const cEllipse3D & aEl ) :
    mCDG(aEl.CDG()),
    mVP(cDenseVect<tREAL8>(3,eModeInitImage::eMIA_Null)),
    mVecP(cDenseMatrix<tREAL8>(3,3,eModeInitImage::eMIA_Null))
{
    MMVII_INTERNAL_ASSERT_strong(aEl.Norm(), "Ellipse3D is normalised");

    cDenseMatrix<double> aCov(3,3,eModeInitImage::eMIA_Null);

    aCov.SetElem(0,0,aEl.Sxx());
    aCov.SetElem(1,1,aEl.Syy());
    aCov.SetElem(2,2,aEl.Szz());

    aCov.SetElem(0,1,aEl.Sxy());
    aCov.SetElem(1,0,aEl.Sxy());
    aCov.SetElem(0,2,aEl.Sxz());
    aCov.SetElem(2,0,aEl.Sxz());
    aCov.SetElem(1,2,aEl.Syz());
    aCov.SetElem(2,1,aEl.Syz());

    // eigenvalue decomposition
    cResulSymEigenValue<tREAL8> aRSEV = aCov.SymEigenValue();

    // update member variables
    mVP = aRSEV.EigenValues();
    mVecP = aRSEV.EigenVectors();

}

void cGenGauss3D::GetDistribGaus(std::vector<cPt3dr> & aVPts,int aN1,int aN2,int aN3)
{ //GetCol(mVGa,aRSEV.EigenVectors(),0);
    cPt3dr aVec0,aVec1,aVec2;
    GetCol(aVec0,mVecP,0);
    GetCol(aVec1,mVecP,1);
    GetCol(aVec2,mVecP,2);

    aVPts.clear();

    cPt3dr aFact1 = aVec0 * (FactCorrectif(aN1) * Sqrt(mVP(0)));
    cPt3dr aFact2 = aVec1 * (FactCorrectif(aN2) * Sqrt(mVP(1)));
    cPt3dr aFact3 = aVec2 * (FactCorrectif(aN3) * Sqrt(mVP(2)));

    for (int aK1 =-aN1 ; aK1<=aN1 ; aK1++)
    {
        for (int aK2 =-aN2 ; aK2<=aN2 ; aK2++)
        {
            for (int aK3 =-aN3 ; aK3<=aN3 ; aK3++)
            {
                cPt3dr aP  =   mCDG
                           + aFact1 * InvErrFoncRationel(2*aK1,2*aN1+1)
                           + aFact2 * InvErrFoncRationel(2*aK2,2*aN2+1)
                           + aFact3 * InvErrFoncRationel(2*aK3,2*aN3+1)
                    ;
                aVPts.push_back(aP);
            }
        }
    }

}
/*
 *     _____
 * P1 *     /|
 *   /     / |
 *  /___P2*  |
 *  | P4x |  *P3   x - pt in the middle
 *  |     | /      * - pts in the corners
 *P5*_____|/
 *
 *
 * */

void cGenGauss3D::GetDistrib5Pts(std::vector<cPt3dr> & aVPts,double aScale)
{
    // Correction factor: the 5-point scheme reconstructs variance as (4/5)*v^2*lambda,
    // so scale by sqrt(5)/(2*v) to recover exact eigenvalues.
    const double v = InvErrFonc(2.0/3.0);
    const double aCorr = std::sqrt(5.0)/(2.0*v);

    cPt3dr aVec0,aVec1,aVec2;
    GetCol(aVec0,mVecP,0);
    GetCol(aVec1,mVecP,1);
    GetCol(aVec2,mVecP,2);

    cPt3dr aFact0 = aVec0 * (aCorr * Sqrt(mVP(0)));
    cPt3dr aFact1 = aVec1 * (aCorr * Sqrt(mVP(1)));
    cPt3dr aFact2 = aVec2 * (aCorr * Sqrt(mVP(2)));

    aVPts.clear();

    cPt3dr aP1;
    aP1 = mCDG + aFact0 * InvErrFoncRationel(-1*2,2+1)
               + aFact1 * InvErrFoncRationel(1*2,2+1)
               + aFact2 * InvErrFoncRationel(1*2,2+1);
    aVPts.push_back(aP1);

    cPt3dr aP2;
    aP2 = mCDG + aFact0 * InvErrFoncRationel(1*2,2+1)
               + aFact1 * InvErrFoncRationel(-1*2,2+1)
               + aFact2 * InvErrFoncRationel(1*2,2+1);
    aVPts.push_back(aP2);

    cPt3dr aP3;
    aP3 = mCDG + aFact0 * InvErrFoncRationel(1*2,2+1)
               + aFact1 * InvErrFoncRationel(1*2,2+1)
               + aFact2 * InvErrFoncRationel(-1*2,2+1);
    aVPts.push_back(aP3);

    cPt3dr aP4;
    aP4 = mCDG + aFact0 * InvErrFoncRationel(0,2+1)
               + aFact1 * InvErrFoncRationel(0,2+1)
               + aFact2 * InvErrFoncRationel(0,2+1);
    aVPts.push_back(aP4);

    cPt3dr aP5;
    aP5 = mCDG + aFact0 * InvErrFoncRationel(-1*2,2+1)
               + aFact1 * InvErrFoncRationel(-1*2,2+1)
               + aFact2 * InvErrFoncRationel(-1*2,2+1);
    aVPts.push_back(aP5);

}

void cEllipse3D::Bench()
{
    int aNbIter=5;

    for (int aKI=0; aKI<aNbIter; aKI++)
    {
        StdOut() << "==== Iter " << aKI << std::endl;
        int aNbPts = 5 + RandUnif_N(20);

        cEllipse3D aEllipse;

        cPt3dr aC0 (RandUnif_C(),RandUnif_C(),RandUnif_C());
        cPt3dr aU0 (RandUnif_C(),RandUnif_C(),RandUnif_C());
        cPt3dr aU1 (RandUnif_C(),RandUnif_C(),RandUnif_C());
        cPt3dr aU2 (RandUnif_C(),RandUnif_C(),RandUnif_C());
        //StdOut() << aC0 << " " << aU0 << " " << aU1 << " " << aU2 << " " << aNbPts << std::endl;
        //getchar();
        for (int aK=0; aK<aNbPts; aK++)
        {
            cPt3dr aP = aC0 + aU0 * RandUnif_C() + aU1 * RandUnif_C() + aU2 * RandUnif_C();
            aP.x() *= 10;
            aP.y() *= 10;
            aP.z() *= 10;
            aEllipse.AddData(aP,1.0);
        }
        aEllipse.Normalise();

        cGenGauss3D aG3D(aEllipse);

        for (int aKMethod=0; aKMethod<2; aKMethod++)
        {
            std::vector<cPt3dr> aVPts;

            if (aKMethod==0)
            {
                StdOut() << "GetDistribGaus" << std::endl;
                aG3D.GetDistribGaus(aVPts,1+RandUnif_N(2),2+RandUnif_N(2),3+RandUnif_N(2));
            }
            else if (aKMethod==1)
            {
                StdOut() << "GetDistrib5Pts" << std::endl;
                aG3D.GetDistrib5Pts(aVPts,1.0);
            }

            aEllipse.Reset();

            for (size_t aK=0; aK<aVPts.size(); aK++)
                aEllipse.AddData(aVPts.at(aK),1.0);

            aEllipse.Normalise();
            cGenGauss3D aG3D2(aEllipse);

            for(int aK=0; aK<3; aK++)
            {
                double aDistVec = Sqrt(Square(aG3D.VecP(aK)(0)-aG3D2.VecP(aK)(0)) +
                                       Square(aG3D.VecP(aK)(1)-aG3D2.VecP(aK)(1)) +
                                       Square(aG3D.VecP(aK)(2)-aG3D2.VecP(aK)(2)));
                StdOut() << "Ratio=" << aG3D.ValP(aK) / aG3D2.ValP(aK) << " "
                         << " dist=" << aDistVec << " "
                         << " ValP=" << aG3D.ValP(aK) << std::endl;
            }
        }
    }

}

};
