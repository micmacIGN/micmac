#include "MMVII_BlocRig.h"

#include "MMVII_Ptxd.h"
#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_2Include_Serial_Tpl.h"

//  RIGIDBLOC


/**
 
  Apprendre a faire une commande/application
  Lire les arguments de la commande 
  Générer des rapports
  Ecrire et relire des données structurées en utilisant les mécanisme de sérialization

  Manipuler le gestionnaire de chantier

  Generer des equation non lineaire à integrer dans le l'optimisation (bundle adjsument)

      * creer la formule symbolique
      * generer le code et charger les classes associees
      * integrer comme contrainte dans le bundle
          # comuniquer l'environnement de la formule  (inconnues , "observation=contexte")
          # ajouter la formule elle même

   \file cBlockCamInit.cpp

 */

namespace MMVII
{

class cSetSensSameId;     //  .....
class cBlocMatrixSensor;  // ....
class cDataBlocCam;       // class for computing index of Bloc/Sync 
			  

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

int cBlocMatrixSensor::NumStringExist(const std::string & anId,bool SVP) const
{
     return  mMapInt2Id.Obj2I(anId,SVP);  // Get index, true because OK non exist
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
/*              cDataBlocCam                          */
/* ************************************************** */

cDataBlocCam::cDataBlocCam(const std::string & aPattern,size_t aKPatBloc,size_t aKPatSync,const std::string & aName) :
     mName      (aName),
     mPattern   (aPattern),
     mKPatBloc  (aKPatBloc),
     mKPatSync  (aKPatSync)
{
}

/// dummy value
cDataBlocCam::cDataBlocCam() :
     cDataBlocCam ("",0,0,"")
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


void cDataBlocCam::AddData(const  cAuxAr2007 & anAuxInit)
{
      cAuxAr2007 anAux("RigidBlocCam",anAuxInit);

      MMVII::AddData(cAuxAr2007("Name",anAux)    ,mName);
      MMVII::AddData(cAuxAr2007("Pattern",anAux) ,mPattern);
      MMVII::AddData(cAuxAr2007("KPatBloc",anAux),mKPatBloc);
      MMVII::AddData(cAuxAr2007("KPatSync",anAux),mKPatSync);
      MMVII::AddData(cAuxAr2007("RelPoses",anAux),mMapPoseInBloc); 
}

void AddData(const  cAuxAr2007 & anAux,cDataBlocCam & aBloc) 
{
     aBloc.AddData(anAux);
}

/* ************************************************** */
/*              cBlocOfCamera                         */
/* ************************************************** */

cBlocOfCamera::cBlocOfCamera(const std::string & aPattern,size_t aKBloc,size_t aKSync,const std::string & aName) :
    mForInit  (true),
    mData     (aPattern,aKBloc,aKSync,aName)
{
}

cBlocOfCamera::cBlocOfCamera() :
     cBlocOfCamera("",0,0,"")
{
}

void  cBlocOfCamera::Set4Compute()
{
    mForInit  = false;
    for (const auto  & aPair : mData.mMapPoseInBloc)
    {
        mMapPoseInBlUK[aPair.first] = cPoseWithUK(aPair.second);
	//  we force the creation a new Id in the bloc because later we will not accept new bloc in compute mode
	mMatBlocSync.NumStringCreate(aPair.first);
    }

    // Now we make a vector of PoseUk* for fast indexed access , we could not do in previous loop , because when maps
    // grow they copy object, invalidating previous pointers on these objects
    for (size_t aKInBl=0 ; aKInBl<NbInBloc() ; aKInBl ++)
    {
         const std::string &  aNameBl = NameKthInBloc(aKInBl);
         mVecPUK.push_back(&(mMapPoseInBlUK[aNameBl]));
    }
}

void cBlocOfCamera::ToFile(const std::string & aNameFile) const
{
    for (const auto  & aPair : mMapPoseInBlUK)
    {
        // when the transfert is done , "omega" (the differential rot)  should have been transfered 
	// in rot, else there is a lost of information
        MMVII_INTERNAL_ASSERT_tiny(IsNull(aPair.second.Omega()),"cBlocOfCamera::TransfertFromUK Omega not null");
        const_cast<cBlocOfCamera*>(this)->mData.mMapPoseInBloc[aPair.first] = aPair.second.Pose();
    }
    SaveInFile(mData,aNameFile);
}

cBlocOfCamera *  cBlocOfCamera::FromFile(const std::string & aNameFile)
{
   cBlocOfCamera *aRes = new cBlocOfCamera;
   ReadFromFile(aRes->mData,aNameFile);
   aRes->Set4Compute();  // put in unknown the initial value

   return aRes;
}

cPoseWithUK & cBlocOfCamera::MasterPoseInBl()  
{
     auto  anIter = mMapPoseInBlUK.find(mData.mMaster);
     MMVII_INTERNAL_ASSERT_tiny(anIter!=mMapPoseInBlUK.end(),"cBlocOfCamera::MasterPose none for master=" + mData.mMaster);

     return anIter->second;
}

const std::string &  cBlocOfCamera::NameMaster() const  { return mData.mMaster; }
size_t cBlocOfCamera::IndexMaster() const {return NumInBloc(NameMaster());}





void cBlocOfCamera::ShowByBloc() const {mMatSyncBloc.ShowMatrix();}
void cBlocOfCamera::ShowBySync() const {mMatBlocSync.ShowMatrix();}

size_t  cBlocOfCamera::NbInBloc() const  {return mMatSyncBloc.NbSet();}
size_t  cBlocOfCamera::NbSync() const  {return mMatSyncBloc.NbInSet();}

const std::string & cBlocOfCamera::NameKthSync(size_t   aKSync)   const {return mMatSyncBloc.NameKthSet(aKSync);}
const std::string & cBlocOfCamera::NameKthInBloc(size_t aKInBloc) const {return mMatBlocSync.NameKthSet(aKInBloc);}

const std::string & cBlocOfCamera::Name() const {return mData.mName;}

int cBlocOfCamera::NumInBloc(const std::string & aName,bool SVP)  const { return mMatBlocSync.NumStringExist(aName,SVP); }

cBlocOfCamera::tMapStrPoseUK& cBlocOfCamera::MapStrPoseUK() {return mMapPoseInBlUK;}

cPoseWithUK &  cBlocOfCamera::PoseOfIdBloc(size_t aKBl) {return *mVecPUK.at(aKBl);}



cSensorCamPC *   cBlocOfCamera::CamKSyncKInBl(size_t aKSync,size_t aKInBloc) const
{
  return mMatSyncBloc.GetCam(aKSync,aKInBloc);
}


bool cBlocOfCamera::AddSensor(cSensorCamPC * aCam)
{

     if (! mData.CanProcess(aCam))
     {
         MMVII_UnclasseUsEr("Cant process bloc/ident for " + aCam->NameImage());
	 // maybe be more lenient later with multiple bloc
	 return false;
     }

     std::string aIdInBoc = mData.CalculIdBloc(aCam);
     std::string aIdSync  = mData.CalculIdSync(aCam);

     size_t  aKSync = mMatSyncBloc.NumStringCreate(aIdSync);

     size_t  aKBloc =  mForInit                                ?
	               mMatBlocSync.NumStringCreate(aIdInBoc)  :
		       mMatBlocSync.NumStringExist(aIdInBoc,false) ;  // if in compute mode, dont accept new Id In bloc

     mMatSyncBloc.AddNew(aCam,aKSync,aKBloc);
     mMatBlocSync.AddNew(aCam,aKBloc,aKSync);

     return true;
}



tPoseR  cBlocOfCamera::EstimatePoseRel1Cple(size_t aKB1,size_t aKB2,cMMVII_Appli * anAppli,const std::string & anIdReportGlob)
{
    std::string aNB1 = NameKthInBloc(aKB1);
    std::string aNB2 = NameKthInBloc(aKB2);
    std::string  anIdReport =  "Detail_" +  NameKthInBloc(aKB1)  + "_" +   NameKthInBloc(aKB2) ;

    if (anAppli)
    {
        anAppli->InitReport(anIdReport,"csv",false);
        anAppli->AddOneReportCSV(anIdReport,{"SyncId","x","y","z","w","p","k"});
    }

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
	     if (anAppli)
                anAppli->AddOneReportCSV(anIdReport,aVReport);
	 }
     }

     if (aNbOk==0)
     {
         MMVII_UnclasseUsEr("No pair of image found fof bloc with Ids :" + aNB1 + " " + aNB2 );
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

     std::string sSigmTr  = (aNbOk>1) ? ToStr(std::sqrt( aSigmTr/tREAL8(aNbOk-1))) : "xxxx" ;
     std::string sSigmRot = (aNbOk>1) ? ToStr(std::sqrt(aSigmRot/tREAL8(aNbOk-1))) : "xxxx" ;

     StdOut() << " STr=" << sSigmTr << " SRot=" << sSigmRot << "\n";

     if (anIdReportGlob!="")
     {
        anAppli->AddOneReportCSV(anIdReportGlob,{aNB1,aNB2,sSigmTr,sSigmRot});
     }
     
     return tPoseR(aAvgTr,aAvgRot);
}

void  cBlocOfCamera::StatAllCples(cMMVII_Appli * anAppli)
{
     std::string  anIdGlob =  "Glob";
     anAppli->InitReport(anIdGlob,"csv",false);
     anAppli->AddOneReportCSV(anIdGlob,{"Id1","Id2","SigmaTr","SigmaRot"});

     for (size_t aKB1=0 ; aKB1<NbInBloc() ; aKB1++)
     {
         for (size_t aKB2=aKB1+1 ; aKB2<NbInBloc() ; aKB2++)
         {
              EstimatePoseRel1Cple(aKB1,aKB2,anAppli,anIdGlob);
         }
     }
}


void cBlocOfCamera::EstimateBlocInit(size_t aKMaster)
{
    mData.mMaster = NameKthInBloc(aKMaster);

    std::vector<tPoseR> aVP;

    for (size_t aKInB=0 ;  aKInB<NbInBloc() ; aKInB++)
    {
        std::string aName = NameKthInBloc(aKInB);
        tPoseR  aPose =  EstimatePoseRel1Cple(aKMaster,aKInB,nullptr,"");
	aVP.push_back(aPose);
	mData.mMapPoseInBloc[aName] = aPose;
    }

    Set4Compute();

    if (0)
    {
        SaveInFile(mData,"toto.xml");
	cDataBlocCam aBX;
        ReadFromFile(aBX,"toto.xml");
        SaveInFile(aBX,"tata.xml");

        SaveInFile(mData,"toto.json");
	cDataBlocCam aBJ;
        ReadFromFile(aBJ,"toto.json");
        SaveInFile(aBJ,"tata.json");

	StdOut() <<  "HX==HJ= " << HashValue(aBX,true) << " " <<  HashValue(aBJ,true) << "\n";
	aBX.mMapPoseInBloc["toto"]  = tPoseR();
	StdOut() <<  "HJ!=HX " << HashValue(aBX,true) << " " << HashValue(aBJ,true) << "\n";

        SaveInFile(mData,"toto.dmp");
	cDataBlocCam aBD;
        ReadFromFile(aBD,"toto.dmp");
        SaveInFile(aBD,"tata_dmp.xml");

        SaveInFile(mData,"toto.txt");
	cDataBlocCam aBT;
        ReadFromFile(aBT,"toto.txt");
        SaveInFile(aBT,"tata_txt.xml");


	StdOut() <<  "HJ!=HD " << HashValue(aBD,true) << " " << HashValue(aBJ,true) << "\n";


         StdOut() <<  aBD.mMapPoseInBloc.begin()->first  << " " <<  aBJ.mMapPoseInBloc.begin()->first << "\n";
	 StdOut() <<  aBD.mMapPoseInBloc["949"].Rot().AxeI()   << aBJ.mMapPoseInBloc["949"].Rot().AxeI()  << "\n";
	 StdOut() <<  (aBD.mMapPoseInBloc["949"].Rot().AxeI()   - aBJ.mMapPoseInBloc["949"].Rot().AxeI())  << "\n";

	 SpecificationSaveInFile<cDataBlocCam>("toto_specif.xml");
	 SpecificationSaveInFile<cDataBlocCam>("toto_specif.json");

	 // test the effectiveness of cMemCheck
	 cBlocOfCamera * aBOC= new  cBlocOfCamera("",0,0,"");
         FakeUseIt(aBOC);
	 delete aBOC;


	 this->ToFile("toto_myfile.xml");
	 cBlocOfCamera * aB2 = cBlocOfCamera::FromFile("toto_myfile.xml");
	 aB2->ToFile("toto_myfile2.xml");

	 delete aB2;
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
        std::string              mSpecImIn;   ///  Pattern or xml file
        cPhotogrammetricProject  mPhProj;
	std::string              mPattern;
	cPt2di                   mNumSub;

	bool                     mShowByBloc;  ///< Do we show the structure by bloc of image
	bool                     mShowBySync;  ///< Do we show structure by synchronization
	std::string              mMaster;      ///< If we enforce the master cam in the bloc
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
	      <<  mPhProj.DPRigBloc().ArgDirOutMand()
           ;
}


cCollecSpecArg2007 & cAppli_BlockCamInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
	     << AOpt2007(mShowByBloc,"ShowByBloc","Show structure, grouping pose of same bloc",{{eTA2007::HDV}})
	     << AOpt2007(mShowBySync,"ShowBySync","Show structure, grouping pose of same camera",{{eTA2007::HDV}})
	     << AOpt2007(mMaster,"MasterCam","Fix the master cam in the bloc(else arbitrary)")
    ;
}

int cAppli_BlockCamInit::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    cBlocOfCamera aBloc(mPattern,mNumSub.x(),mNumSub.y(),"toto");
    for (const auto & anIm : VectMainSet(0))
    {
	cSensorCamPC * aCam = mPhProj.ReadCamPC(anIm,true);
	aBloc.AddSensor(aCam);
    }

    if (mShowByBloc) aBloc.ShowByBloc();
    if (mShowBySync ) aBloc.ShowBySync();

    aBloc.StatAllCples(this);

    int aNumMaster = 0; // arbitrary if not specified

    if (IsInit(&mMaster))  // IsInit(void*) => indicate if a value was set by user
    {
        aNumMaster = aBloc.NumInBloc(mMaster,true); // true=SVP, becausewe handle ourself the case dont exist
        if (aNumMaster<0)
        {
            StdOut()<< "- Allowed blocs : " << std::endl;
            for (size_t aK=0 ; aK< aBloc.NbInBloc()   ; aK++)
                StdOut() << "  * " << aBloc.NameKthInBloc(aK) << std::endl;
            MMVII_UnclasseUsEr("Name master = " +mMaster + " is not an existing bloc");
        }
    }

    aBloc.EstimateBlocInit(aNumMaster);
    mPhProj.SaveBlocCamera(aBloc);

    if (1)
    {
         aBloc.ToFile("titi.xml");
         cBlocOfCamera * aBL = cBlocOfCamera::FromFile("titi.xml");

         for (const auto & anIm : VectMainSet(0))
         {
	      cSensorCamPC * aCam = mPhProj.ReadCamPC(anIm,true);
	      aBL->AddSensor(aCam);
         }

	 delete aBL;
    }


    StdOut()  << " NumMaster " <<  aNumMaster  << "\n";
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

