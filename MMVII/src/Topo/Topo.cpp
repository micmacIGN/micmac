#include "Topo.h"
#include "MMVII_PhgrDist.h"
#include "../BundleAdjustment/BundleAdjustment.h"

namespace MMVII
{

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
    //obs: 3 DSCF3297_L.jpg DSCF3298_L.jpg 0.3170 0.001

    auto from_name = "DSCF3297_L.jpg";
    auto to_name = "DSCF3298_L.jpg";
    double val = 0.3170;
    double sigma = 0.001;
    cSensorCamPC * aCamTmp;
    aCamTmp = aPhProj.ReadCamPC(from_name,true,false);
    auto& ptFrom = aCamTmp->Pose().Tr();
    aCamTmp = aPhProj.ReadCamPC(to_name,true,false);
    auto& ptTo = aCamTmp->Pose().Tr();
    double dist = sqrt(SqN2(ptFrom-ptTo));
    std::cout<<"Obs: "<<&ptFrom<<" "<<&ptTo<<" "<<val-dist<<" "<<sigma<<"\n";

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
    //  "cSetInterUK_MultipeObj" is a structure that contains a set of
    //  unknowns, here we "declarate" all the unknwon, the object
    //  declared must derive from "cObjWithUnkowns". The "declaration" is
    //  made by calling "AddOneObj" in aSet
    //
    //  For each bloc, the unkowns are the "cPoseWithUK" contained in "mMapPoseUKInBloc"
    //

     //  .....
     // map all bloc
     /*for (const auto & aBloc : mBlocs)
     {
          for (auto & aPair : aBloc->MapStrPoseUK())
          {
              aSet.AddOneObj(&aPair.second);
          }
     }*/
     //   map all pair of MapStrPoseUK
     //        add cPoseWithUK
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
     // ... parse all bloc
     /*for (const auto & aBloc : mBlocs)
     {
         //  ... extract the master
	 cPoseWithUK &  aMPose = aBloc->MasterPoseInBl()  ;

	 // The system handle the unknwon as integers,  the "object" (here aPose)
	 // "knows" the association between its member and local integers, that's why
	 // we pass object and members to do the job
         
         aSys.SetFrozenVarCurVal(aMPose,aMPose.Center());
         aSys.SetFrozenVarCurVal(aMPose,aMPose.Omega());
	 //
	 //  ...  freeze the center
	 //  ... freez omega
     }*/
}

    // =========================================================
    //       The Equation itself
    // =========================================================

void cBA_Topo::AddEquation_Dist3d(cResolSysNonLinear<tREAL8> & aSys)
{
/*
    //  extract the sensor corresponding to the time and num of bloc
    cSensorCamPC* aCam1 = aBloc.CamKSyncKInBl(aKS,aKBl1);
    cSensorCamPC* aCam2 = aBloc.CamKSyncKInBl(aKS,aKBl2);


    // it may happen that some image are absent, non oriented ...
    if ((aCam1==nullptr) || (aCam2==nullptr)) return cPt3dr(0,0,0);

    // extract the unknown pose for each cam
    cPoseWithUK &  aPBl1 =  aBloc.PoseUKOfNumBloc(aKBl1);
    cPoseWithUK &  aPBl2 =  aBloc.PoseUKOfNumBloc(aKBl2);
     
    //  FakeUseIt(aPBl1); FakeUseIt(aPBl2);

    // We must create the observation/context of the equation; here we will push the coeef of matrix
    // for linearization 
    std::vector<double> aVObs;

    aCam1->Pose_WU().PushObs(aVObs,false);
    aCam2->Pose_WU().PushObs(aVObs,false);
    aPBl1.PushObs(aVObs,false);
    aPBl2.PushObs(aVObs,false);

    // We must create a vector that contains all the global num of unknowns
    std::vector<int>  aVInd;
    aCam1->PushIndexes(aVInd);
    aCam2->PushIndexes(aVInd);
    aPBl1.PushIndexes(aVInd);
    aPBl2.PushIndexes(aVInd);

    // now we are ready to add the equation
    aSys.R_CalcAndAddObs
    (
          mEqBlUK,  // the equation itself
	  aVInd,
	  aVObs,
	  cResidualWeighterExplicit<tREAL8>(false,mWeight)
    );


    cPt3dr  aRes(0,0,1);

    //  Now compute the residual  in Tr and Rot,
    //  ie agregate   (Rx,Ry,Rz, m00 , m01 ...)
    //
    for (size_t aKU=0 ; aKU<12 ;  aKU++)
    {
         aRes[aKU>=3] += Square(mEqBlUK->ValComp(0,aKU));
    }

    return cPt3dr(aRes.x()/3.0,aRes.y()/9.0,1.0);*/
}



void cBA_Topo::AddTopoEquations(cResolSysNonLinear<tREAL8> & aSys)
{
    std::vector<int> indices;

    auto fromIndices = mPts[0]->getIndices();
    indices.insert(std::end(indices), std::begin(fromIndices), std::end(fromIndices));

    auto toIndices = mPts[1]->getIndices();
    indices.insert(std::end(indices), std::begin(toIndices), std::end(toIndices));

}

};

