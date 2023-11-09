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
class cDataBlocCam;       // class for computing index of Bloc/Sync 
			  

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
	   // const std::string & NameKthInSet(size_t) const;

           /// return the num of the set associated to a string (possibly new)
           size_t NumStringCreate(const std::string &) ;
           /// return the num of an "existing" set associated to a string (-1 if dont exist)
           int NumStringExist(const std::string &) const ;

	   /// Add a sensor in the matrix at given "Num Of Set" and given "Num inside Set"
	   void AddNew(cSensorCamPC*,size_t aNumSet,size_t aNumInSet);

	   /// Creator 
	   cBlocMatrixSensor();
	   /// Show the structure, tuning process
	   void ShowMatrix() const;

	   /// extract the camera for a given "Num Of Set" and a given "Num inside set"
           cSensorCamPC* &  GetCam(size_t aNumSet,size_t aNumInSet);
           cSensorCamPC*  GetCam(size_t aNumSet,size_t aNumInSet) const;

	   const cSetSensSameId &  KthSet(size_t aKth) const;
      private :
	   size_t                        mMaxSzSet;  ///< max number of element in Set
           t2MapStrInt                   mMapInt2Id; ///< For string/int conversion : Bijective map  SyncInd <--> int
	   std::vector<cSetSensSameId>   mMatrix;    /// the matrix itself
};



/** Class for  computing indexes */

class cDataBlocCam
{
      public :
           cDataBlocCam(const std::string & aPattern,size_t aKPatBloc,size_t aKPatSync);

           /** Compute the index of a sensor inside a bloc, pose must have  
               same index "iff" they are correpond to a position in abloc */
           std::string  CalculIdBloc(cSensorCamPC * ) const ;
           /** Compute the synchronisation index of a sensor, pose must have  
               same index "iff" they are acquired at same time */
           std::string  CalculIdSync(cSensorCamPC * ) const ;
	   /// it may happen that processing cannot be made
           bool  CanProcess(cSensorCamPC * ) const ;
      private :
           std::string mPattern; ///< Regular expression for extracting "BlocId/SyncId"
           size_t      mKPatBloc;   ///< Num of expression
           size_t      mKPatSync;   ///< Num of bloc
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

size_t cBlocMatrixSensor::NumStringCreate(const std::string & anId) 
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

int cBlocMatrixSensor::NumStringExist(const std::string & anId) const
{
     return  mMapInt2Id.Obj2I(anId,true);  // Get index, true because OK non exist
} 

cSensorCamPC* &  cBlocMatrixSensor::GetCam(size_t aNumSet,size_t aNumInSet)
{
     return  mMatrix.at(aNumSet).mVCams.at(aNumInSet);
}
cSensorCamPC*  cBlocMatrixSensor::GetCam(size_t aNumSet,size_t aNumInSet) const 
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

/* ************************************************** */
/*              cDataBlocCam                    */
/* ************************************************** */

cDataBlocCam::cDataBlocCam(const std::string & aPattern,size_t aKPatBloc,size_t aKPatSync) :
     mPattern   (aPattern),
     mKPatBloc  (aKPatBloc),
     mKPatSync  (aKPatSync)
{
}

bool  cDataBlocCam::CanProcess(cSensorCamPC * aCam) const
{
    return MatchRegex(aCam->NameImage(),mPattern);
}

std::string  cDataBlocCam::CalculIdBloc(cSensorCamPC * aCam) const
{
     return PatternKthSubExpr(mPattern,mKPatBloc,aCam->NameImage());
}

std::string  cDataBlocCam::CalculIdSync(cSensorCamPC * aCam) const
{
     return PatternKthSubExpr(mPattern,mKPatSync,aCam->NameImage());
}

/**  Interface class for computing the identifier of bloc from a given camera.
 *
 *
 *    
*/
class cBlocOfCamera
{
      public :
           void Add(cSensorCamPC *);

	   void ShowByBloc() const;
	   void ShowBySync() const;

	   cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync);

	   size_t  NbInBloc() const;
	   size_t  NbSync() const;

           int NumInBloc(const std::string &)  const;

	   const std::string & NameKthInBloc(size_t) const;
	   const std::string & NameKthSync(size_t) const;

	   const cBlocMatrixSensor & MatSyncBloc() const ; ///< Acessor
	   const cBlocMatrixSensor & MatBlocSync() const ; ///< Acessor

	   cSensorCamPC *  CamKSyncKInBl(size_t aKInBloc,size_t aKSync) const;
           tPoseR  EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli & anAppli,const std::string & aReportGlob);
           void    EstimateInit(cMMVII_Appli & anAppli);

      private :
	   cDataBlocCam                  mData;
	   cBlocMatrixSensor             mMatSyncBloc;
	   cBlocMatrixSensor             mMatBlocSync;
};


cBlocOfCamera::cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync) :
    mData (aPattern,aKBloc,aKSync)
{
}

void cBlocOfCamera::ShowByBloc() const {mMatSyncBloc.ShowMatrix();}
void cBlocOfCamera::ShowBySync() const {mMatBlocSync.ShowMatrix();}

size_t  cBlocOfCamera::NbInBloc() const  {return mMatSyncBloc.NbSet();}
size_t  cBlocOfCamera::NbSync() const  {return mMatSyncBloc.NbInSet();}

const std::string & cBlocOfCamera::NameKthSync(size_t   aKSync)   const {return mMatSyncBloc.NameKthSet(aKSync);}
const std::string & cBlocOfCamera::NameKthInBloc(size_t aKInBloc) const {return mMatBlocSync.NameKthSet(aKInBloc);}

int cBlocOfCamera::NumInBloc(const std::string & aName)  const { return mMatBlocSync.NumStringExist(aName); }

cSensorCamPC *   cBlocOfCamera::CamKSyncKInBl(size_t aKSync,size_t aKInBloc) const
{
  return mMatSyncBloc.GetCam(aKSync,aKInBloc);
}


void cBlocOfCamera::Add(cSensorCamPC * aCam)
{

     if (! mData.CanProcess(aCam))
     {
         MMVII_UnclasseUsEr("Cant process bloc/ident for " + aCam->NameImage());
     }

     std::string aIdInBoc = mData.CalculIdBloc(aCam);
     std::string aIdSync  = mData.CalculIdSync(aCam);

     size_t  aKSync = mMatSyncBloc.NumStringCreate(aIdSync);
     size_t  aKBloc = mMatBlocSync.NumStringCreate(aIdInBoc);

     mMatSyncBloc.AddNew(aCam,aKSync,aKBloc);
     mMatBlocSync.AddNew(aCam,aKBloc,aKSync);
}



tPoseR  cBlocOfCamera::EstimateInit(size_t aKB1,size_t aKB2,cMMVII_Appli & anAppli,const std::string & anIdReportGlob)
{
    std::string  anIdReport =  "Detail_" +  NameKthInBloc(aKB1)  + "_" +   NameKthInBloc(aKB2) ;

    anAppli.InitReport(anIdReport,"csv",false);
    anAppli.AddOneReportCSV(anIdReport,{"SyncId","x","y","z","w","p","k"});

    cPt3dr aSomTr;  //  average off translation

    cPt3dr aAvgTr = cPt3dr::PCste(0.0);
    cDenseMatrix<tREAL8> aAvgMat(3,3,eModeInitImage::eMIA_Null);
    int aNbOk = 0;

    for (size_t aKC=0 ; aKC<NbSync() ; aKC++)
    {
         cSensorCamPC * aCam1 =  CamKSyncKInBl(aKC,aKB1);
         cSensorCamPC * aCam2 =  CamKSyncKInBl(aKC,aKB2);
         if (aCam1 &&  aCam2)
         {
            tPoseR  aP2To1 =  aCam1->RelativePose(*aCam2);
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

     tREAL8 aSigmTr  = 0;
     tREAL8 aSigmRot = 0;

     for (size_t aKC=0 ; aKC<NbSync() ; aKC++)
     {
         cSensorCamPC * aCam1 =  CamKSyncKInBl(aKC,aKB1);
         cSensorCamPC * aCam2 =  CamKSyncKInBl(aKC,aKB2);
         if (aCam1 &&  aCam2)
         {
	     tPoseR  aP2To1 =  aCam1->RelativePose(*aCam2);
	     aSigmTr +=  SqN2(aAvgTr-aP2To1.Tr());

	     aSigmRot += aAvgRot.Mat().SqL2Dist(aP2To1.Rot().Mat());
	 }
     }

     aSigmTr  = std::sqrt( aSigmTr/tREAL8(aNbOk-1));
     aSigmRot = std::sqrt(aSigmRot/tREAL8(aNbOk-1));

     StdOut() << " STr=" << aSigmTr << " SRot=" << aSigmRot << "\n";

     if (anIdReportGlob!="")
     {
        anAppli.AddOneReportCSV(anIdReportGlob,{NameKthInBloc(aKB1),NameKthInBloc(aKB2),ToStr(aSigmTr),ToStr(aSigmRot)});
     }
     
     return tPoseR(aAvgTr,aAvgRot);
}

void  cBlocOfCamera::EstimateInit(cMMVII_Appli & anAppli)
{
     std::string  anIdGlob =  "Glob";
     anAppli.InitReport(anIdGlob,"csv",false);
     anAppli.AddOneReportCSV(anIdGlob,{"Id1","Id2","SigmaTr","SigmaRot"});

     for (size_t aKB1=0 ; aKB1<NbInBloc() ; aKB1++)
     {
         for (size_t aKB2=aKB1+1 ; aKB2<NbInBloc() ; aKB2++)
         {
              EstimateInit(aKB1,aKB2,anAppli,anIdGlob);
         }
     }
}


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
	bool                     mShowBySync;
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
     mShowBySync   (false)
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
	     << AOpt2007(mShowBySync,"ShowBySync","Show structure, grouping pose of same camera",{{eTA2007::HDV}})
    ;
}

int cAppli_BlockCamInit::Exe()
{
    mPhProj.FinishInit();

    cBlocOfCamera aBloc(mPattern,mNumSub.x(),mNumSub.y());
    for (const auto & anIm : VectMainSet(0))
    {
	cSensorCamPC * aCam = mPhProj.ReadCamPC(anIm,true);
	aBloc.Add(aCam);
    }

    if (mShowByBloc) aBloc.ShowByBloc();
    if (mShowBySync ) aBloc.ShowBySync();

    aBloc.EstimateInit(*this);

   StdOut()  << " HHH " << aBloc.NumInBloc("toto") << "\n";
   StdOut()  << " HHH " << aBloc.NumInBloc("043") << "\n";
   StdOut()  << " HHH " << aBloc.NumInBloc("949") << "\n";
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

