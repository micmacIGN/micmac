#include "Topo.h"
#include "MMVII_PhgrDist.h"
#include "../BundleAdjustment/BundleAdjustment.h"

namespace MMVII
{


void cMMVII_BundleAdj::InitItereTopo()
{
    if (mTopo)
    {
        //... TODO
    }
}

cBA_Topo::cBA_Topo
(const cPhotogrammetricProject &aPhProj, const std::string & aTopoFilePath)  :
    mPhProj  (aPhProj),
    mTopoObsType2equation
    {
        {eTopoObsType::eDist, EqDist3D(true,1)},
        {eTopoObsType::eSubFrame, EqTopoSubFrame(true,1)},
        {eTopoObsType::eDistParam, EqDist3DParam(true,1)},
    },
    mTopoData(aTopoFilePath, this),
    mInFile(aTopoFilePath)
{

}

cBA_Topo::~cBA_Topo()
{
    for (auto& [_, aEq] : mTopoObsType2equation)
                  delete aEq;
}

void cBA_Topo::Save()
{
    mTopoData.ToFile(mInFile+"-out.json");
}

tTopoPtUK& cBA_Topo::getPointWithUK(const std::string & aName)
{
    // search if already in map
    if (auto search = mPts_UK.find(aName); search != mPts_UK.end())
            return search->second;

    // search among cameras
    cSensorCamPC * aCam = mPhProj.ReadCamPC(aName, true, false);
    std::cout<<"cSensorCamPC: "<<aCam<<std::endl;  //TODO : crash if not found for now
    return mPts_UK[aName] = {aCam, &aCam->Center()};

    // search among gcp

    // add new pure topo point
}

void cBA_Topo::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
    //for (const auto& [p_uk, p_3d]: mPts_UK)
    //    aSet.AddOneObj(p_uk);
    // add only unknowns not handled by anything else?
}

/**  In a bundle adjusment its current that some variable are "hard" frozen, i.e they are
 *   considered as constant.  We could write specific equation, but generally it's more
 *   economic (from software devlopment) to just indicate this frozen part to the system.
 *
 *   Here the frozen unknown are the poses of the master camera of each bloc because the
 *   calibration is purely relative
 *
 */
void cBA_Topo::SetFrozenVar(cResolSysNonLinear<tREAL8> & aSys)
{
     // for now, no frozen var for topo
}

    // =========================================================
    //       The Equation itself
    // =========================================================

/*double cBA_Topo::AddEquation_Dist3d(cResolSysNonLinear<tREAL8> & aSys)
{
    return 0.0;
    // returns residual

    //obs: 3 DSCF3297_L.jpg DSCF3298_L.jpg 0.3170 0.001
    double val = 0.3170;
    double sigma = 0.0001;
    auto & [ptFromUK, ptFrom3d] = mPts_UK.at(0);
    auto & [ptToUK, ptTo3d] = mPts_UK.at(1);
    //double dist = sqrt(SqN2(*ptFrom3d-*ptTo3d));
    std::cout<<"Obs: "<<ptFrom3d<<" "<<ptTo3d<<" "<<val<<" "<<sigma<<"\n";

    // We must create the observation/context of the equation; here we will push the coeef of matrix
    // for linearization
    std::vector<double> aVObs;
    aVObs.push_back(val);

    // We must create a vector that contains all the global num of unknowns
    std::vector<int>  aVInd;
    ptFromUK->PushIndexes(aVInd, *ptFrom3d);
    ptToUK->PushIndexes(aVInd, *ptTo3d);


    // now we are ready to add the equation
    auto& equation = mTopoObsType2equation.at(eTopoObsType::eDist);
    aSys.R_CalcAndAddObs
    (
          equation,  // the equation itself
          aVInd,
          aVObs,
          cResidualWeighterExplicit<tREAL8>(true,{sigma})
    );


    double residual = equation->ValComp(0,0);
    StdOut() << "  topo resid: " << residual << std::endl;

    return residual;
}*/

cCalculator<double>*  cBA_Topo::getEquation(eTopoObsType tot) const {
    auto eq = mTopoObsType2equation.find(tot);
    if (eq != mTopoObsType2equation.end())
        return mTopoObsType2equation.at(tot);
    else
    {
        MMVII_INTERNAL_ERROR("unknown equation for obs type")
        return nullptr;
    }
}

void cBA_Topo::AddTopoEquations(cResolSysNonLinear<tREAL8> & aSys)
{
    for (auto &obsSet: mTopoData.allObsSets)
        for (size_t i=0;i<obsSet->nbObs();++i)
        {
            cTopoObs* obs = obsSet->getObs(i);
            auto equation = getEquation(obs->getType());
            aSys.CalcAndAddObs(equation, obs->getIndices(this), obs->getVals(), obs->getWeights());
            double residual = equation->ValComp(0,0);
            StdOut() << "  topo resid: " << residual << std::endl;
        }
}

};

