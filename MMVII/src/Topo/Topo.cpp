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
        {TopoObsType::dist, EqDist3D(true,1)},
        {TopoObsType::subFrame, EqTopoSubFrame(true,1)},
        {TopoObsType::distParam, EqDist3DParam(true,1)},
    },
    mOk(true)
{
    StdOut()<<"Read topo file "<<aTopoFilePath<<"\n";

    // TODO
    auto from_name = "DSCF3297_L.jpg";
    auto to_name = "DSCF3298_L.jpg";
    cSensorCamPC * aCamFrom = aPhProj.ReadCamPC(from_name,true,false);
    mPts_UK.push_back({aCamFrom, &aCamFrom->Center()});
    cSensorCamPC * aCamTo = aPhProj.ReadCamPC(to_name,true,false);
    mPts_UK.push_back({aCamTo, &aCamTo->Center()});


    //...
    mOk = true;
}

cBA_Topo::~cBA_Topo()
{
    for (auto& [_, aEq] : mTopoObsType2equation)
                  delete aEq;
}

void cBA_Topo::Save()
{

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

double cBA_Topo::AddEquation_Dist3d(cResolSysNonLinear<tREAL8> & aSys)
{
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
    auto& equation = mTopoObsType2equation.at(TopoObsType::dist);
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
}



void cBA_Topo::AddTopoEquations(cResolSysNonLinear<tREAL8> & aSys)
{
    AddEquation_Dist3d(aSys);
}

};

