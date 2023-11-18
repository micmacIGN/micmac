#include "BundleAdjustment.h"
#include "MMVII_BlocRig.h"


namespace MMVII
{

cBA_BlocRig::cBA_BlocRig
(
     const cPhotogrammetricProject &aPhProj,
     const std::vector<double> & aSigma
)  :
    mPhProj  (aPhProj),
    mBlocs   (aPhProj.ReadBlocCams()),
    mSigma   (aSigma),
    mAllPair (false),
    mEqBlUK  (EqBlocRig(true,1,true))
{
    // push the weigth for the 3 equation on centers
    for (int aK=0 ; aK<3 ; aK++)
        mWeight.push_back(Square(1/mSigma.at(0)));
    
    // push the weigth for the 9 equation on rotation
    for (int aK=0 ; aK<9 ; aK++)
        mWeight.push_back(Square(1/mSigma.at(1)));
}

cBA_BlocRig::~cBA_BlocRig()
{
    DeleteAllAndClear(mBlocs);
}

void cBA_BlocRig::Save()
{
     for (const auto & aBloc : mBlocs)
	 mPhProj.SaveBlocCamera(*aBloc);
}

void cBA_BlocRig::AddCam (cSensorCamPC * aCam)
{
     size_t aNbAdd = 0;
     for (const auto & aBloc : mBlocs)
     {
         aNbAdd += aBloc->AddSensor(aCam);
     }
     if (aNbAdd>1)
     {
         MMVII_UnclasseUsEr("Multiple bloc for "+ aCam->NameImage());
     }

}


void cBA_BlocRig::AddToSys(cSetInterUK_MultipeObj<tREAL8> & aSet)
{
     for (auto & aBloc : mBlocs) // Parse the blocs
     {
         for (auto & aPair : aBloc->MapStrPoseUK())  // Parse the Unkown poses inside each bloc
         {
		 aSet.AddOneObj(&aPair.second);
         }
     }
}

/**  In a bundle adjusment its current that some variable are "hard" frozen, i.e they are
 *   considered as constant.  We could write specific equation, but generally it's more
 *   economic (from software devlopment) to just indicate this frozen part to the system.
 *
 *   Here the frozen unknown are the poses of the master camera of each blokc
 *
 */
void cBA_BlocRig::SetFrozenVar(cResolSysNonLinear<tREAL8> & aSys)  
{
     for (const auto & aBloc : mBlocs)
     {
	 cPoseWithUK &  aPose = aBloc->MasterPoseInBl() ;

	 // The system handle the unknwon as integers,  the "object" (here aPose)
	 // "knows" the association between its member and local integers, that's why
	 // we pass object and members to do the job
	 aSys.SetFrozenVarCurVal(aPose,aPose.Center());   // freeze the center
	 aSys.SetFrozenVarCurVal(aPose,aPose.Omega());    // freeze the differential rotation
     }
}

    // =========================================================
    //       The RigidityEquation itself
    // =========================================================

cPt3dr cBA_BlocRig::OnePairAddRigidityEquation(size_t aKS,size_t aKBl1,size_t aKBl2,cBlocOfCamera& aBloc,cResolSysNonLinear<tREAL8> & aSys)
{
    OrderMinMax(aKBl1,aKBl2); // not sure necessary, but prefer to fix arbitrary order
			    
    //  extract the sensor
    cSensorCamPC* aCam1 = aBloc.CamKSyncKInBl(aKS,aKBl1);
    cSensorCamPC* aCam2 = aBloc.CamKSyncKInBl(aKS,aKBl2);


    // it may happen that some image are absent, non oriented ...
    if ((aCam1==nullptr) || (aCam2==nullptr)) return cPt3dr(0,0,0);

    cPoseWithUK &  aPBl1 =  aBloc.PoseUKOfNumBloc(aKBl1);
    cPoseWithUK &  aPBl2 =  aBloc.PoseUKOfNumBloc(aKBl2);

    // We must create the observation/context of the equation; here we will push the coeef of matrix
    // for linearization 
    std::vector<double> aVObs;
    aCam1->Pose_WU().PushObs(aVObs,false);
    aCam2->Pose_WU().PushObs(aVObs,false);
    aPBl1.PushObs(aVObs,false);
    aPBl2.PushObs(aVObs,false);

    // We must create a vector that contains all the unknowns
    std::vector<int>  aVInd;
    aCam1->PushIndexes(aVInd);
    aCam2->PushIndexes(aVInd);
    aPBl1.PushIndexes(aVInd);
    aPBl2.PushIndexes(aVInd);

    // now we are ready to add the equation
    aSys.R_CalcAndAddObs(mEqBlUK,aVInd,aVObs,cResidualWeighterExplicit<tREAL8>(false,mWeight));
    // aSys.R_CalcAndAddObs(mEqBlUK,aVInd,aVObs,0.1);

    cPt2dr aResTrW(0,0);
    for (size_t aKU=0 ; aKU<12 ; aKU++)
    {
	aResTrW[aKU>=3] += Square(mEqBlUK->ValComp(0,aKU));
    }
    aResTrW = cPt2dr(std::sqrt(aResTrW.x()/3.0),std::sqrt(aResTrW.y()/9.0));

    return cPt3dr(aResTrW.x(),aResTrW.y(),1.0);
}

void cBA_BlocRig::OneBlAddRigidityEquation(cBlocOfCamera& aBloc,cResolSysNonLinear<tREAL8> & aSys)
{
     cPt3dr aRes(0,0,0);
     for (size_t  aKSync=0 ; aKSync<aBloc.NbSync() ; aKSync++)
     {
         // case "AllPair", to symetrise the problem we process all pair w/o distinguish master
         if (mAllPair)
	 {
            for (size_t aKBl1=0 ; aKBl1<aBloc.NbInBloc() ; aKBl1++)
            {
                for (size_t aKBl2=aKBl1+1 ; aKBl2<aBloc.NbInBloc() ; aKBl2++)
                {
                     aRes += OnePairAddRigidityEquation(aKSync,aKBl1,aKBl2,aBloc,aSys);
                }
            }
	 }
         else
         {
            size_t aKM = aBloc.IndexMaster();
            for (size_t aKBl=0 ; aKBl<aBloc.NbInBloc() ; aKBl++)
                if (aKBl!=aKM)
                   aRes += OnePairAddRigidityEquation(aKSync,aKM,aKBl,aBloc,aSys);
         }
     }

     aRes = aRes/aRes.z();

     StdOut() << "  Residual for Bloc : "  <<  aBloc.Name() << ", Tr=" << aRes.x() << ", Rot=" << aRes.y() << std::endl;
}

void cBA_BlocRig::AddRigidityEquation(cResolSysNonLinear<tREAL8> & aSys)
{

     for (const auto & aBloc : mBlocs)
         OneBlAddRigidityEquation(*aBloc,aSys);
}

};

