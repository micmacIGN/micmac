#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"


/**
   \file GCPQuality.cpp

 */

namespace MMVII
{

typedef std::pair<std::string,std::string> tPairStr;
typedef std::pair<int,int>                 tPairInd;

class cSetSensSameId;     //  .....
class cBlocMatrixSensor;  // ....
class cBlocCamComputeInd;    // class for computing index of Bloc/Sync 
			  

/**  store a set of "cSensorCamPC" sharing the same identifier, can be ident of time or ident  of camera, this
 * class is a an helper for implementation of set aqcuired at same time or of same camera */

class cSetSensSameId
{
      public :
         friend class cBlocMatrixSensor;

	 ///  Construct with a number of sensor + the id common to all
         cSetSensSameId(size_t aNbCam,const std::string & anIdSync);
	 void Resize(size_t); ///<  Extend size with nullptr

	 const std::vector<cSensorCamPC*>&   VCams() const;  ///< Accessor
	 const std::string & Id() const;  ///< Accessor
      private :
	 /* Identifier common to same sensors like  "Im_T01_CamA.JPG","Im_T01_CamB.JPG"  => "T01"
	  *                                   or    "Im_T01_CamA.JPG","Im_T02_CamA.JPG"  => "CamA" */
	 std::string                 mId;        
	 std::vector<cSensorCamPC*>  mVCams;     ///< vector of camera sharing same  id
};						 

/** class to represent the "matricial" organization of bloc of sensor for example with a
 *  bloc of 2 camera "A" and "B" acquired at 5 different times, we can  will organize
 *
 *        A1  A2 A3 A4 A5
 *        B1  B2 B3 B4 B5
 *
 *  The class is organize to alloxw a dynamic creation, we permanently maintain the matrix structure
 *
 * */
class cBlocMatrixSensor
{
      public :

	   size_t NbSet() const;
	   size_t NbInSet() const;
	   const std::string & NameKthSet(size_t) const;
	   const std::string & NameKthInSet(size_t) const;

           /// return the num of the set associated to a string (possibly new)
           size_t NumString(const std::string &) ;

	   /// Add a sensor in the matrix at given "Num Of Set" and given "Num inside Set"
	   void AddNew(cSensorCamPC*,size_t aNumSet,size_t aNumInSet);

	   /// Creator 
	   cBlocMatrixSensor();
	   /// Show the structure, tuning process
	   void ShowMatrix() const;

	   /// extract the camera for a given "Num Of Set" and a given "Num inside set"
           cSensorCamPC* &  GetCam(size_t aNumSet,size_t aNumInSet);

	   const cSetSensSameId &  KthSet(size_t aKth) const;
      private :
	   size_t                        mMaxSzSet;  ///< max number of element in Set
           t2MapStrInt                   mMapInt2Id; ///< For string/int conversion : Bijective map  SyncInd <--> int
	   std::vector<cSetSensSameId>   mMatrix;    /// the matrix itself
};



/** Class for  computing indexes */

class cBlocCamComputeInd
{
      public :
           cBlocCamComputeInd(const std::string & aPattern,size_t aKBloc,size_t aKSync);

           /// Compute the index of a sensor inside a bloc, pose must have  same index "iff" they are correpond to a position in abloc
           std::string  CalculIdBloc(cSensorCamPC * ) const ;
           /// Compute the synchronisation index of a sensor, pose must have  same index "iff" they are acquired at same time
           std::string  CalculIdSync(cSensorCamPC * ) const ;
	   /// it may happen that processing cannot be made
           bool  CanProcess(cSensorCamPC * ) const ;
      private :
           std::string mPattern;
           size_t      mKBloc;  ///< Num of expression
           size_t      mKSync;  ///< Num of bloc
};


/* ************************************************** */
/*              cSetSensSameId                        */
/* ************************************************** */

cSetSensSameId::cSetSensSameId(size_t aNbCam,const std::string & anIdSync) :
    mId     (anIdSync),
    mVCams  (aNbCam,nullptr)
{
}

void cSetSensSameId::Resize(size_t aSize)
{
	mVCams.resize(aSize);
}

const std::vector<cSensorCamPC*>&   cSetSensSameId::VCams() const {return mVCams;}

const std::string & cSetSensSameId::Id() const {return mId;}

/* *********************************************************** */
/*                                                             */
/*                    cBlocMatrixSensor                        */
/*                                                             */
/* *********************************************************** */

cBlocMatrixSensor::cBlocMatrixSensor() :
    mMaxSzSet (0)
{
}

size_t cBlocMatrixSensor::NumString(const std::string & anId) 
{
     int anInd = mMapInt2Id.Obj2I(anId,true);  // Get index, true because OK non exist
     if (anInd<0)
     {
         anInd = mMatrix.size();
	 mMapInt2Id.Add(anId);
	 mMatrix.push_back(cSetSensSameId(mMaxSzSet,anId));
     }
     return anInd;
}

cSensorCamPC* &  cBlocMatrixSensor::GetCam(size_t aNumSet,size_t aNumInSet)
{
     return  mMatrix.at(aNumSet).mVCams.at(aNumInSet);
}

void cBlocMatrixSensor::AddNew(cSensorCamPC* aPC,size_t aNumSet,size_t aNumInSet)
{
     if (aNumInSet >= mMaxSzSet)
     {
         mMaxSzSet = aNumInSet+1;
	 for (auto  & aSet : mMatrix)
             aSet.Resize(mMaxSzSet);
     }

     cSensorCamPC* & aLocPC = GetCam(aNumSet,aNumInSet); //  mMatrix.at(aNumSet).mVCams.at(aNumInSet);

     if (aLocPC!=nullptr)
     {
         MMVII_UnclasseUsEr("Bloc Matrix, cam already exists, detected at  image : " + aPC->NameImage());
     }
     aLocPC = aPC;
}

void cBlocMatrixSensor::ShowMatrix() const
{
    for (size_t aKMat=0 ; aKMat<mMatrix.size() ; aKMat++)
    {
        StdOut() <<  "==================== " <<  mMatrix[aKMat].mId << " =====================" << std::endl;
	for (const auto & aPtrCam : mMatrix[aKMat].mVCams)
	{
            if (aPtrCam)
               StdOut() << "   * " << aPtrCam ->NameImage() << std::endl;
	    else
               StdOut() << "   * xxxxxxxxxxxxxxxxxxxxxxxxxxxx"  << std::endl;
	}

    }
}

const cSetSensSameId &  cBlocMatrixSensor::KthSet(size_t aKth) const
{
	return mMatrix.at(aKth);
}

size_t cBlocMatrixSensor::NbSet()   const {return mMaxSzSet;}
size_t cBlocMatrixSensor::NbInSet() const {return mMatrix.size();}

const std::string & cBlocMatrixSensor::NameKthSet(  size_t aKTh) const { return *mMapInt2Id.I2Obj(aKTh); }
const std::string & cBlocMatrixSensor::NameKthInSet(size_t aKTh) const { return mMatrix.at(aKTh).Id(); }

/* ************************************************** */
/*              cBlocCamComputeInd                    */
/* ************************************************** */

cBlocCamComputeInd::cBlocCamComputeInd(const std::string & aPattern,size_t aKBloc,size_t aKSync) :
     mPattern (aPattern),
     mKBloc   (aKBloc),
     mKSync   (aKSync)
{
}

bool  cBlocCamComputeInd::CanProcess(cSensorCamPC * aCam) const
{
    return MatchRegex(aCam->NameImage(),mPattern);
}

std::string  cBlocCamComputeInd::CalculIdBloc(cSensorCamPC * aCam) const
{
     return PatternKthSubExpr(mPattern,mKBloc,aCam->NameImage());
}

std::string  cBlocCamComputeInd::CalculIdSync(cSensorCamPC * aCam) const
{
     return PatternKthSubExpr(mPattern,mKSync,aCam->NameImage());
}

/**  Interface class for computing the identifier of bloc from a given camera.
 *
 *  Use is done in three step :
 *
 *     - in first step we "Add" camera, essentially we memorize all the cam
 *     - when we have finis addding we indicate it by "SetFinishAdd" so  that some indexation are made
 *     - after we can use it ...
 *
 *  The bool "mFinishAdd" control that we dont mix all the steps
 *    
*/
class cBlocOfCamera
{
      public :
           void Add(cSensorCamPC *);

	   void Show();
	   cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync);

	   size_t  NbInBloc() const;
	   size_t  NbSync() const;

	   const std::string & NameKthInBloc(size_t) const;
	   const std::string & NameKthSync(size_t) const;

	   const cBlocMatrixSensor & MatSyncBloc() const ; ///< Acessor
	   const cBlocMatrixSensor & MatBlocSync() const ; ///< Acessor

	   cSensorCamPC *  CamKInBlKSync(size_t aKInBloc,size_t aKSync) const;

           tPoseR  EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli & anAppli,const std::string & aReportGlob);
      private :
	   cBlocCamComputeInd            mCompIndexes;

	   cBlocMatrixSensor             mMatSyncBloc;
	   cBlocMatrixSensor             mMatBlocSync;

};


cBlocOfCamera::cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync) :
    mCompIndexes (aPattern,aKBloc,aKSync)
{
}

const cBlocMatrixSensor & cBlocOfCamera::MatSyncBloc()  const {return mMatSyncBloc;}
const cBlocMatrixSensor & cBlocOfCamera::MatBlocSync()  const {return mMatBlocSync;}

size_t  cBlocOfCamera::NbInBloc() const  {return mMatSyncBloc.NbSet();}
size_t  cBlocOfCamera::NbSync() const  {return mMatSyncBloc.NbInSet();}

const std::string & cBlocOfCamera::NameKthInBloc(size_t aKBloc) const {return mMatSyncBloc.NameKthSet(aKBloc);}
const std::string & cBlocOfCamera::NameKthSync(size_t aKSync) const {return mMatSyncBloc.NameKthInSet(aKSync);}


/*
cSensorCamPC *   cBlocOfCamera::CamKInBlKSync(size_t aKInBloc,size_t aKSync) const
{
}
cSensorCamPC *  CamKInBlKSync(size_t aKBloc,size_t aKSync)
{
}
*/
/*
const std::string & cBlocOfCamera::NameKthBloc(size_t) const 
{
}
*/
//const std::string & NameKthSync(size_t) const;


void cBlocOfCamera::Add(cSensorCamPC * aCam)
{

     if (! mCompIndexes.CanProcess(aCam))
     {
         MMVII_UnclasseUsEr("Cant process bloc/ident for " + aCam->NameImage());
     }

     std::string aIdInBoc = mCompIndexes.CalculIdBloc(aCam);
     std::string aIdSync  = mCompIndexes.CalculIdSync(aCam);

     size_t  aKSync = mMatSyncBloc.NumString(aIdSync);
     size_t  aKBloc = mMatBlocSync.NumString(aIdInBoc);

     mMatSyncBloc.AddNew(aCam,aKSync,aKBloc);
     mMatBlocSync.AddNew(aCam,aKBloc,aKSync);
}



tPoseR  cBlocOfCamera::EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli & anAppli,const std::string & anIdReportGlob)
{

    std::string  anIdReport =  "Detail_" +  NameKthInBloc(aKB1)  + "_" +   NameKthInBloc(aKB2) ;

    anAppli.InitReport(anIdReport,"csv",false);
    anAppli.AddOneReportCSV(anIdReport,{"SyncId","x","y","z","w","p","k"});

    cPt3dr aSomTr;  //  average off translation

     const std::vector<cSensorCamPC*> &  aVC1 = mMatBlocSync.KthSet(aKB1).VCams();
     const std::vector<cSensorCamPC*> &  aVC2 = mMatBlocSync.KthSet(aKB2).VCams();
     size_t aNbC = aVC1.size();

     cPt3dr aAvgTr = cPt3dr::PCste(0.0);
     cDenseMatrix<tREAL8> aAvgMat(3,3,eModeInitImage::eMIA_Null);
     int aNbOk = 0;

     for (size_t aKC=0 ; aKC<aNbC ; aKC++)
     {
         if (aVC1[aKC] && aVC2[aKC])
         {
            tPoseR  aP2To1 =  aVC1[aKC]->RelativePose(*(aVC2[aKC]));
	    cPt3dr aTr = aP2To1.Tr();
            cRotation3D<tREAL8>  aRot = aP2To1.Rot();

            aAvgTr += aTr;
	    aAvgMat = aAvgMat + aRot.Mat();
            aNbOk++;

            cPt3dr aWPK = aRot.ToWPK();

	    std::vector<std::string>  aVReport{
                                              NameKthSync(aKC),
                                              ToStr(aTr.x()),ToStr(aTr.y()),ToStr(aTr.z()),
		                              ToStr(aWPK.x()),ToStr(aWPK.y()),ToStr(aWPK.z())
	                                   };
             anAppli.AddOneReportCSV(anIdReport,aVReport);
	 }
     }

     aAvgTr =  aAvgTr / tREAL8(aNbOk);
     aAvgMat = aAvgMat * (1.0/tREAL8(aNbOk));
     cRotation3D<tREAL8>  aAvgRot(aAvgMat,true);

     tREAL8 aSigmTr=0;
     tREAL8 aSigmRot=0;

     for (size_t aKC=0 ; aKC<aNbC ; aKC++)
     {
         if (aVC1[aKC] && aVC2[aKC])
         {
	     tPoseR  aP2To1 =  aVC1[aKC]->RelativePose(*(aVC2[aKC]));
	     aSigmTr +=  SqN2(aAvgTr-aP2To1.Tr());

	     aSigmRot += aAvgRot.Mat().SqL2Dist(aP2To1.Rot().Mat());
	 }
     }

     aSigmTr  = std::sqrt( aSigmTr/tREAL8(aNbOk-1));
     aSigmRot = std::sqrt(aSigmRot/tREAL8(aNbOk-1));

     StdOut() << " STr=" << aSigmTr << " SRot=" << aSigmRot << "\n";

     /*
     if (anIdReportGlob!="")
        anAppli.AddOneReportCSV(anIdReportGlob,{mMatBlocSync.KthSet(aKB1).Id(),mMatBlocSync.KthSet(aKB2).Id(),
	*/
     
     return tPoseR(aAvgTr,aAvgRot);
}

//=======================================================================================
//=======================================================================================
//=======================================================================================
//=======================================================================================
//=======================================================================================
//=======================================================================================


class cBlocCam
{
      public :
          void Add(cSensorCamPC *);
          cBlocCam(const std::string & aPattern,cPt2di aNum);

          void Finish();

          void Show() const;

	  /// Make the initial estimation for all pair
          void EstimateInit(cMMVII_Appli &,const std::string & aPrefReport);

	  ///  Estimate the initial  value of relative pose between  Camera K1 & K2
          void EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli &,const std::string & anIdGlob,const std::string & aPrefReport);
      private :


	  typedef std::pair<std::string,std::string> tPairStr;
	  typedef std::pair<int,int>                 tPairInd;

	  const std::string & NamePoseId(size_t) const;
	  const std::string & NameSyncId(size_t) const;


	  // typedef cBijectiveMapI2O<cSensorCamPC *>  tMap;
	  void AssertFinish();
	  void AssertNotFinish();

	  tPairStr   ComputeStrPoseIdSyncId(cSensorCamPC*);
	  tPairInd   ComputeIndPoseIdSyncId(cSensorCamPC*);

	  cSensorCamPC *& CamOfIndexes(const tPairInd&);

	  bool         mClosed;
	  std::string  mPattern;
	  cPt2di       mNumSub;

          mutable t2MapStrInt  mMapIntSyncId;
          mutable t2MapStrInt  mMapIntPoseId;
	  size_t       mNbSyncId;
	  size_t       mNbPoseId;


	  std::vector<cSensorCamPC*>  mVAllCam;
	  std::vector<std::vector<cSensorCamPC*> >  mV_TB;  // mV_TB[KSyncId][KBlock]
};

cBlocCam::cBlocCam(const std::string & aPattern,cPt2di aNum):
    mClosed  (false),
    mPattern (aPattern),
    mNumSub  (aNum)
{
}

const std::string & cBlocCam::NamePoseId(size_t anInd) const { return *mMapIntPoseId.I2Obj(anInd,false); }

const std::string & cBlocCam::NameSyncId(size_t anInd) const { return *mMapIntSyncId.I2Obj(anInd,false); }


void cBlocCam::Finish()
{
    mNbSyncId = mMapIntSyncId.size();
    mNbPoseId = mMapIntPoseId.size();
    AssertNotFinish();
    mClosed = true;

    for (size_t aKSyncId=0 ; aKSyncId<mNbSyncId ; aKSyncId++)
    {
        mV_TB.push_back(std::vector<cSensorCamPC*>(mMapIntPoseId.size(),nullptr));
    }

    for (const auto & aCam : mVAllCam)
    {
        CamOfIndexes(ComputeIndPoseIdSyncId(aCam)) = aCam;
    }
}

cSensorCamPC *& cBlocCam::CamOfIndexes(const tPairInd & aPair)
{
	return  mV_TB.at(aPair.second).at(aPair.first);
}
void cBlocCam::Show() const
{
    for (const auto & aVBl : mV_TB)
    {
        for (const auto  &aCam : aVBl)
	{
            if (aCam==nullptr)
               StdOut() <<  "    00000000000000000" << std::endl;
	    else
               StdOut() <<  "    " << aCam->NameImage() << std::endl;
	}
        StdOut() << "============================================================" << std::endl;
    }
}

void cBlocCam::AssertNotFinish() {MMVII_INTERNAL_ASSERT_tiny(!mClosed,"cBlocCam::AssertNotFinish");}
void cBlocCam::AssertFinish() {MMVII_INTERNAL_ASSERT_tiny(mClosed,"cBlocCam::AssertFinish");}

void cBlocCam::Add(cSensorCamPC * aCam)
{
    AssertNotFinish();

    if (MatchRegex(aCam->NameImage(),mPattern))
    {
	auto [aStrBlock,aStrSyncId] = ComputeStrPoseIdSyncId(aCam);

        mMapIntPoseId.Add(aStrBlock,true);
        mMapIntSyncId.Add(aStrSyncId,true);

        mVAllCam.push_back(aCam);
    }
}

cBlocCam::tPairStr  cBlocCam::ComputeStrPoseIdSyncId(cSensorCamPC* aCam)
{
    std::string aNameCam = aCam->NameImage();

    std::string aStrBlock = PatternKthSubExpr(mPattern,mNumSub.x(),aNameCam);
    std::string aStrSyncId  = PatternKthSubExpr(mPattern,mNumSub.y(),aNameCam);

    return std::pair<std::string,std::string>(aStrBlock,aStrSyncId);
}

cBlocCam::tPairInd cBlocCam::ComputeIndPoseIdSyncId(cSensorCamPC* aCam)
{
    auto [aStrBlock,aStrSyncId] = ComputeStrPoseIdSyncId(aCam);

    return std::pair<int,int>(mMapIntPoseId.Obj2I(aStrBlock),mMapIntSyncId.Obj2I(aStrSyncId));
}

void cBlocCam::EstimateInit(cMMVII_Appli & anAppli,const std::string & aPrefReport)
{
    std::string  anIdGlob =  aPrefReport + "StatGlob";
    anAppli.InitReport(anIdGlob,"csv",false);
    anAppli.AddOneReportCSV(anIdGlob,{"Id1","Id2","SigmaTr","SigmaRot"});

     for (size_t aKB1=0 ; aKB1<mNbPoseId ; aKB1++)
     {
         for (size_t aKB2 =aKB1+1 ; aKB2<mNbPoseId ; aKB2++)
         {
             EstimateInit(aKB1,aKB2,anAppli,anIdGlob,aPrefReport);
         }
     }
}

void cBlocCam::EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli & anAppli,const std::string & anIdGlob,const std::string & aPrefReport)
{
    std::string  anIdReport =  aPrefReport + "Detail_" +  NamePoseId(aKB1) + "_" +   NamePoseId(aKB2) ;

    anAppli.InitReport(anIdReport,"csv",false);
    anAppli.AddOneReportCSV(anIdReport,{"SyncId","x","y","z","w","p","k"});

    cPt3dr aSomTr;  //  average off translation
    cPt4dr a4;      // average of quaternions -> not sure good idea because Q ~ -Q, maybe replace with good old matrices
    size_t aNb=0;

    //  Parse all "times/synchronisation-id"
    for (size_t aKT=0 ; aKT<mNbSyncId ; aKT++)
    {
        cSensorCamPC * aCam1 = CamOfIndexes(tPairInd(aKB1,aKT));  // camera [KB1,KT]
        cSensorCamPC * aCam2 = CamOfIndexes(tPairInd(aKB2,aKT));  // camera [KB2,KT]

	// P1(C1)=W   P2(C2) =  W   P1-1 P2 (C2)  = C1
        if ((aCam1!=nullptr) && (aCam2!=nullptr)) // not all cam have a value
        {
	   cIsometry3D<tREAL8>  aP2To1 =  aCam1->RelativePose(*aCam2); // relative pose 
	   const auto & aRot = aP2To1.Rot();
	   const auto & aMat = aRot.Mat();
	   cPt3dr aTr = aP2To1.Tr();
	   aSomTr += aTr;
	   cPt3dr aWPK = aRot.ToWPK();

	   a4  +=  MatrRot2Quat(aMat);
	   aNb++;

	   std::vector<std::string>  aVReport{
		                              NameSyncId(aKT),
                                              ToStr(aTr.x()),ToStr(aTr.y()),ToStr(aTr.z()),
		                              ToStr(aWPK.x()),ToStr(aWPK.y()),ToStr(aWPK.z())
	                                   };
           anAppli.AddOneReportCSV(anIdReport,aVReport);
        }
    }

    aSomTr = aSomTr / tREAL8(aNb);
    a4  = a4  / tREAL8(aNb);
    a4 = VUnit(a4);

    tREAL8 aSomEcTr = 0;  // will store std dev on translation
    tREAL8 aSomEc4 = 0;   // will store std dev on rotation

    for (size_t aKT=0 ; aKT<mNbSyncId ; aKT++)
    {
        cSensorCamPC * aCam1 = CamOfIndexes(tPairInd(aKB1,aKT));
        cSensorCamPC * aCam2 = CamOfIndexes(tPairInd(aKB2,aKT));

        if ((aCam1!=nullptr) && (aCam2!=nullptr))
        {
	   cIsometry3D<tREAL8>  aP2To1 =  aCam1->RelativePose(*aCam2);

	   aSomEcTr += SqN2(aSomTr-aP2To1.Tr());
	   aSomEc4  += SqN2(a4- MatrRot2Quat(aP2To1.Rot().Mat()));
        }
    }
    tREAL8 aSigmaTr =  std::sqrt(aSomEcTr/(aNb-1));
    tREAL8 aSigmaRot =  std::sqrt(aSomEc4/(aNb-1));

    if (anIdGlob !="")
    {
        std::vector<std::string> aVReport {NamePoseId(aKB1),NamePoseId(aKB2),ToStr(aSigmaTr),ToStr(aSigmaRot)};
        anAppli.AddOneReportCSV(anIdGlob,aVReport);
    }
    else
    {
        StdOut() << " KKKKK " << aSomTr << " " << a4 << std::endl;
        StdOut()  << "DISPTR " << aSigmaTr  << " DISPROT=" << aSigmaRot << std::endl;
        StdOut() << " BBB " << NamePoseId(aKB1) <<  " " << NamePoseId(aKB2) << std::endl;
    }
}

/**  Structure of block data
 
         - there can exist several block, each block has its name
	 - the result are store in a folder, one file by block
	 - 

     This command will create a block init  and save it in a given folder


     For now, just check the rigidity => will implemant in detail with Celestin ...

*/


/* ==================================================== */
/*                                                      */
/*          cAppli_CalibratedSpaceResection             */
/*                                                      */
/* ==================================================== */

class cAppli_BlockCamInit : public cMMVII_Appli
{
     public :

        cAppli_BlockCamInit(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli &);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override;

     private :
        std::string              mSpecImIn;   ///  Pattern of xml file
        cPhotogrammetricProject  mPhProj;
	std::string              mPattern;
	cPt2di                   mNumSub;

	bool                     mShowByBloc;
	bool                     mShowByCam;
	std::string              mPrefixReport;
};

cAppli_BlockCamInit::cAppli_BlockCamInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mShowByBloc   (false),
     mShowByCam    (false)
{
}



cCollecSpecArg2007 & cAppli_BlockCamInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
              <<  Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
              <<  mPhProj.DPOrient().ArgDirInMand()
              <<  Arg2007(mPattern,"Pattern for images specifing sup expr")
              <<  Arg2007(mNumSub,"Num of sub expr for x:block and  y:image")
           ;
}


cCollecSpecArg2007 & cAppli_BlockCamInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	     << AOpt2007(mShowByBloc,"ShowByBloc","Show structure, grouping pose of same bloc",{{eTA2007::HDV}})
	     << AOpt2007(mShowByCam,"ShowByCam","Show structure, grouping pose of same camera",{{eTA2007::HDV}})
    ;
}

int cAppli_BlockCamInit::Exe()
{
    mPhProj.FinishInit();

    {
       cBlocOfCamera aBloc(mPattern,mNumSub.x(),mNumSub.y());
       for (const auto & anIm : VectMainSet(0))
       {
	   cSensorCamPC * aCam = mPhProj.ReadCamPC(anIm,true);
	   aBloc.Add(aCam);
       }

       if (mShowByBloc) 
           aBloc.MatSyncBloc().ShowMatrix();
       if (mShowByCam) 
           aBloc.MatBlocSync().ShowMatrix();

       aBloc.EstimateInit(0,1,*this,"");
       return EXIT_SUCCESS;
    }


    mPrefixReport = "Ori_" +  mPhProj.DPOrient().DirIn();

    cBlocCam aBloc(mPattern,mNumSub);
    for (const auto & anIm : VectMainSet(0))
    {
	cSensorCamPC * aCam = mPhProj.ReadCamPC(anIm,true);
	aBloc.Add(aCam);
    }
    aBloc.Finish();
    aBloc.Show();

    aBloc.EstimateInit(0,1,*this,"",mPrefixReport);
    aBloc.EstimateInit(*this,mPrefixReport);

    return EXIT_SUCCESS;
}                                       

/* ==================================================== */
/*                                                      */
/*               MMVII                                  */
/*                                                      */
/* ==================================================== */


tMMVII_UnikPApli Alloc_BlockCamInit(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_BlockCamInit(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_BlockCamInit
(
     "BlockCamInit",
      Alloc_BlockCamInit,
      "Initialisation of bloc camera",
      {eApF::GCP,eApF::Ori},
      {eApDT::Orient},
      {eApDT::Xml},
      __FILE__
);

/*
*/




}; // MMVII

