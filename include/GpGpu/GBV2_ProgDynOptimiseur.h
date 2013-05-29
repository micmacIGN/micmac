#ifndef H_GBV2_PROGDUNOPTIMISEUR
#define H_GBV2_PROGDUNOPTIMISEUR

#include "GpGpu/GpGpu.h"

namespace NS_ParamMICMAC
{

/**************************************************/
/*                                                */
/*       cGBV2_CelOptimProgDyn                         */
/*                                                */
/**************************************************/
typedef unsigned int tCost;
class cGBV2_CelOptimProgDyn
{
public :
    cGBV2_CelOptimProgDyn() :
        mCostFinal (0)
    {
    }
    typedef enum
    {
        eAvant = 0,
        eArriere = 1
    } eSens;
    void SetCostInit(int aCost)
    {
        mCostInit = aCost;
    }
    void SetBeginCumul(eSens aSens)
    {
        mCostCum[aSens] = mCostInit;
    }
    void SetCumulInitial(eSens aSens)
    {
        mCostCum[aSens] = int(1e9) ;
    }
    void UpdateCost
    (
            const cGBV2_CelOptimProgDyn& aCel2,
            eSens aSens,
            int aCostTrans
            )
    {
        ElSetMin
                (
                    mCostCum[aSens],
                    mCostInit + aCostTrans + aCel2.mCostCum[aSens]
                    );
    }

    tCost CostPassageForce() const
    {
        return mCostCum[eAvant] + mCostCum[eArriere] - mCostInit;
    }
    tCost GetCostInit() const
    {
        return mCostInit;
    }
    const tCost & CostFinal() const { return mCostFinal; }
    tCost & CostFinal() { return mCostFinal; }
	void	AddToCostFinal(tCost cost ) { mCostFinal+= cost; }
private :
    cGBV2_CelOptimProgDyn(const cGBV2_CelOptimProgDyn &);
    tCost   mCostCum[2];
    tCost   mCostInit;
    tCost   mCostFinal;
};

typedef  cSmallMatrixOrVar<cGBV2_CelOptimProgDyn>   tCGBV2_tMatrCelPDyn;


/**************************************************/
/*                                                */
/*               cProgDynOptimiseur               */
/*                                                */
/**************************************************/


class cGBV2_TabulCost
{
public :

    static int  CostR2I(double aCost) { return round_ni(aCost*1e4); }

    void Reset(double aCostL1,double aCostL2)
    {
        mNb=0;
        mTabul = std::vector<int>();
        mCostL1 = aCostL1;
        mCostL2 = aCostL2;
    }
    inline int Cost(int aDx)
    {
        aDx = ElAbs(aDx);
        for (;mNb<=aDx; mNb++)
        {
            mTabul.push_back
                    (
                        CostR2I(
                        (
                            mCostL1 * mNb
                            + mCostL2 * ElSquare(mNb)
                            )));
        }
        return mTabul[aDx];
    }

    double           mCostL1;
    double           mCostL2;
    int              mNb;
    std::vector<int> mTabul;
};

class cGBV2_ProgDynOptimiseur : public cSurfaceOptimiseur
{
public :
    cGBV2_ProgDynOptimiseur
    (
        cAppliMICMAC &    mAppli,
        cLoadTer&         mLT,
        const cEquiv1D &        anEqX,
        const cEquiv1D &        anEqY,
            Im2D_INT2  aPxMin,
            Im2D_INT2  aPxMax
            );
    ~cGBV2_ProgDynOptimiseur() {}
    void Local_SetCout(Pt2di aPTer,int *aPX,REAL aCost,int aLabel);
    void Local_SolveOpt(Im2D_U_INT1 aImCor);


    // Im2D_INT2     ImRes() {return mImRes;}

private :

    void BalayageOneDirection(Pt2dr aDir);
    void BalayageOneLine(const std::vector<Pt2di> & aVPt);
    void BalayageOneLineGpu(const std::vector<Pt2di> & aVPt);
    void BalayageOneSens
    (
            const std::vector<Pt2di> & aVPt,
            cGBV2_CelOptimProgDyn::eSens,
            int anIndInit,
            int aDelta,
            int aLimite
            );

    void BalayageOneSensGpu
    (
            const std::vector<Pt2di> & aVPt,
            cGBV2_CelOptimProgDyn::eSens,
            int anIndInit,
            int aDelta,
            int aLimite
            );
    void SolveOneEtape(int aNbDir);

#ifdef CUDA_ENABLED
    void SolveAllDirectionGpu(int aNbDir);
    template<bool dir> void copyCells(const std::vector<Pt2di>* aVPt,Pt2di aDirI, CuHostData3D<uint> &h_strCostVolume, CuHostData3D<uint3> &rStrPar, CuHostData3D<short2> &h_strIndex, uint* h_OutForceCostVol);
#endif

    Im2D_INT2                          mXMin;
    Im2D_INT2                          mXMax;
    Pt2di                              mSz;
    int                                mNbPx;
    Im2D_INT2                          mYMin;
    Im2D_INT2                          mYMax;
    cMatrOfSMV<cGBV2_CelOptimProgDyn>  mMatrCel;
    cLineMapRect                       mLMR;
    cGBV2_TabulCost                    mTabCost[theDimPxMax];
    // int                             mCostActu[theDimPxMax];
    int                                mMaxEc[theDimPxMax];
    eModeAggregProgDyn                 mModeAgr;
    int                                mNbDir;
    double                             mPdsProgr;

    InterfMicMacOptGpGpu               IGpuOpt;

    // Im2D_INT2     mImRes;
    // INT2 **       mDataImRes;

};
}
#endif //H_GBV2_PROGDUNOPTIMISEUR
