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
class cCalibBlocCam;       // class for computing index of Bloc/Sync 
			  

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
    //  ...
    //  Parse matrix id (0 to mMatrix.size())
        for (size_t aKSet=0 ; aKSet<mMatrix.size() ; aKSet++)
        {
            //    print the Id
            StdOut() <<  "========   "  << mMatrix[aKSet].mId <<  " ======= " << std::endl;
            for (const auto & aPtrCam : mMatrix[aKSet].mVCams)
            {
                 if (aPtrCam!=nullptr)
                    StdOut ()  <<  "  * " << aPtrCam->NameImage()   << std::endl;
                 else
                    StdOut ()  <<  "  * xxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
            }
        }
    //
    //
    //    parse  the cams in a set
    //        
    //        print the name of cam or "xxxxx" if cams is null
    //
    //
    //    Should get something like 
    //
    //      ...
    //    ==============   0103 ======
    //    043_0103.JPG
    //    671_0103.JPG 
    //    948_0103.JPG
    //    949_0103.JPG
    //    ==============   0104 ======
    //    043_0104.JPG
    //    xxxxxxxxxxxx
    //    948_0104.JPG
    //    949_0104.JPG
    //    ==============   0105 ======
    //      ...
    //
    //
}

const cSetSensSameId &  cBlocMatrixSensor::KthSet(size_t aKth) const
{
	return mMatrix.at(aKth);
}

size_t cBlocMatrixSensor::NbSet()   const      {return mMaxSzSet;}
size_t cBlocMatrixSensor::NbInSet() const      {return mMatrix.size();}
size_t cBlocMatrixSensor::SetCapacity() const  {return mMapInt2Id.size();}

const std::string & cBlocMatrixSensor::NameKthSet(  size_t aKTh) const { return *mMapInt2Id.I2Obj(aKTh); }

/* ************************************************** */
/*              cCalibBlocCam                         */
/* ************************************************** */

cCalibBlocCam::cCalibBlocCam(const std::string & aPattern,size_t aKPatBloc,size_t aKPatSync,const std::string & aName) :
     mName      (aName),
     mPattern   (aPattern),
     mKPatBloc  (aKPatBloc),
     mKPatSync  (aKPatSync)
{
}

/// dummy value
cCalibBlocCam::cCalibBlocCam() :
     cCalibBlocCam ("",0,0,"")
{
}

bool  cCalibBlocCam::CanProcess(cSensorCamPC * aCam) const
{
    return MatchRegex(aCam->NameImage(),mPattern);
}

std::string  cCalibBlocCam::CalculIdBloc(cSensorCamPC * aCam) const
{
     return PatternKthSubExpr(mPattern,mKPatBloc,aCam->NameImage());
}

std::string  cCalibBlocCam::CalculIdSync(cSensorCamPC * aCam) const
{
     return PatternKthSubExpr(mPattern,mKPatSync,aCam->NameImage());
}


void cCalibBlocCam::AddData(const  cAuxAr2007 & anAuxInit)
{
     cAuxAr2007 anAux("CalibBlocCam",anAuxInit);
     // ...
     // Put the data in  tag "RigidBlocCam"

     // Add data for
     //    mName
     //    mPattern
     //    mKPatBloc
     //    mKPatSync
     //    mKPatSync
     //    mMapPoseInBloc
     // MMVII::AddData(cAuxAr2007("Name",anAuxInit) ,mName);
     MMVII::AddData(cAuxAr2007("Name",anAux),mName);
     MMVII::AddData(cAuxAr2007("Master",anAux),mMaster);
     MMVII::AddData(cAuxAr2007("Pattern",anAux),mPattern);
     MMVII::AddData(cAuxAr2007("KBloc",anAux),mKPatBloc);
     MMVII::AddData(cAuxAr2007("KSync",anAux),mKPatSync);
     MMVII::AddData(cAuxAr2007("PoseRel",anAux),mMapPoseUKInBloc);

     //  cAuxAr2007(const std::string& ,const  cAuxAr2007 &)
     //   MMVII::AddData(cAuxAr2007("Name",anAux)    ,mName);
}

void AddData(const  cAuxAr2007 & anAux,cCalibBlocCam & aBloc) 
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

    for (const auto  & [aName, aPoseUk] : mData.mMapPoseUKInBloc)
    {
        MMVII_INTERNAL_ASSERT_tiny(IsNull(aPoseUk.Omega()),"cBlocOfCamera::TransfertFromUK Omega not null");
        mMapPoseInit[aName] = aPoseUk.Pose();
	//  we force the creation a new Id in the bloc because later we will not accept new bloc in compute mode
	mMatBlocSync.NumStringCreate(aName);
    }
}

cBlocOfCamera::~cBlocOfCamera()
{
}

void cBlocOfCamera::ToFile(const std::string & aNameFile) const
{
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
     return PoseUKOfIdBloc(mData.mMaster);
}
cPoseWithUK &  cBlocOfCamera::PoseUKOfIdBloc(const std::string& anId) 
{
     auto  anIter = mData.mMapPoseUKInBloc.find(anId);
     MMVII_INTERNAL_ASSERT_tiny(anIter!=mData.mMapPoseUKInBloc.end(),"cBlocOfCamera::PoseUKOfIdBloc none for:" + anId);

     return anIter->second;

}


const std::string &  cBlocOfCamera::NameMaster() const  { return mData.mMaster; }
size_t cBlocOfCamera::IndexMaster() const {return NumInBloc(NameMaster());}





void cBlocOfCamera::ShowByBloc() const {mMatBlocSync.ShowMatrix();}
void cBlocOfCamera::ShowBySync() const {mMatSyncBloc.ShowMatrix();}



size_t  cBlocOfCamera::NbInBloc() const  {return mMatSyncBloc.NbSet();}
size_t  cBlocOfCamera::NbSync() const  {return mMatSyncBloc.NbInSet();}
size_t  cBlocOfCamera::BlocCapacity() const {return mMatBlocSync.SetCapacity(); }

const std::string & cBlocOfCamera::NameKthSync(size_t   aKSync)   const {return mMatSyncBloc.NameKthSet(aKSync);}
const std::string & cBlocOfCamera::NameKthInBloc(size_t aKInBloc) const {return mMatBlocSync.NameKthSet(aKInBloc);}

const std::string & cBlocOfCamera::Name() const {return mData.mName;}

int cBlocOfCamera::NumInBloc(const std::string & aName,bool SVP)  const { return mMatBlocSync.NumStringExist(aName,SVP); }

cBlocOfCamera::tMapStrPoseUK& cBlocOfCamera::MapStrPoseUK() {return mData.mMapPoseUKInBloc;}

cPoseWithUK &  cBlocOfCamera::PoseUKOfNumBloc(size_t aKBl) 
{
     return PoseUKOfIdBloc(NameKthInBloc(aKBl));
}

const tPoseR &  cBlocOfCamera::PoseInitOfNumBloc(size_t aKBl)  const
{
    auto anIter = mMapPoseInit.find(NameKthInBloc(aKBl));

    MMVII_INTERNAL_ASSERT_tiny(anIter!=mMapPoseInit.end(),"PoseInitOfNumBloc cannot find ");
    return anIter->second;
}


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

    //  create an identifier for report on  bloc1 and bloc2
    std::string  anIdReport =  "Detail_" +  aNB1  + "_" +   aNB2 ;

    //  extract the name of the 2 bloc

    if (anAppli)
    {
         //  Init the raport,  false mean that we are doin it in the main application, not in a sub-process
         anAppli->InitReport(anIdReport,"csv",false);
         //  Add one header  "SyncId","x","y","z","w","p","k"
         anAppli->AddOneReportCSV(anIdReport,{"SyncId","x","y","z","w","p","k"});
    }

    // ============= [1]  Compute, for all relative orientation, average of Translation and rotation

    cPt3dr aAvgTr = cPt3dr::PCste(0.0);//  => accumulate som of translatio,
    cDenseMatrix<tREAL8> aAvgMat(3,3,eModeInitImage::eMIA_Null); // => accumulate sum of translation matrixes
    int aNbOk = 0; //  count the number of pair where we could make the computation

    for (size_t aKSync=0 ; aKSync<NbSync() ; aKSync++) // parse all pair timee
    {
        // extract Cam1 and Cam2
        cSensorCamPC *   aCam1 = CamKSyncKInBl(aKSync,aKB1);
        cSensorCamPC *   aCam2 = CamKSyncKInBl(aKSync,aKB2);

         // if  they are not null
        if ((aCam1!=nullptr) && (aCam2!=nullptr))
        {
            // compute relative pose
             tPoseR aPose = aCam1->RelativePose(*aCam2);

             cPt3dr aTr = aPose.Tr();
             cPt3dr aWPK = aPose.Rot().ToWPK();
            // sum translation and rotation
             aAvgTr += aTr;
             aNbOk++;
             aAvgMat = aAvgMat + aPose.Rot().Mat();
            // eventually make a report
             // StdOut() << " Tr=" << aPose.Tr()  << " WPK=" <<  aPose.Rot().ToWPK() << "\n"; 
             if (anAppli)
             {
                anAppli->AddOneReportCSV
                (
                     anIdReport,
                     {    NameKthSync(aKSync),
                          ToStr(aTr.x()),ToStr(aTr.y()),ToStr(aTr.z()),
                          ToStr(aWPK.x()),ToStr(aWPK.y()),ToStr(aWPK.z())
                     }
                );
             }
        }
     }

     // if no pair OK we cannot compute an average
     if (aNbOk==0)
     {
          MMVII_UnclasseUsEr("No pair of image found fof bloc with Ids :" + aNB1 + " " + aNB2 );
     }
     aAvgTr =  aAvgTr / tREAL8(aNbOk);
     aAvgMat = aAvgMat * (1.0/tREAL8(aNbOk));
     cRotation3D<tREAL8>  aAvgRot(aAvgMat,true);  // true-> compute the closest orthogonal matrix

    // ============= [2]  Compute, standard deviati,on


     tREAL8 aSigmTr  = 0;  // som of square dif for translation
     tREAL8 aSigmRot = 0;  // som of square dif for rotation

     for (size_t aKSync=0 ; aKSync<NbSync() ; aKSync++)
     {
        // extract Cam1 and Cam2
        cSensorCamPC *   aCam1 = CamKSyncKInBl(aKSync,aKB1);
        cSensorCamPC *   aCam2 = CamKSyncKInBl(aKSync,aKB2);

         // if  they are not null
        if ((aCam1!=nullptr) && (aCam2!=nullptr))
        {
             tPoseR aPose = aCam1->RelativePose(*aCam2);
             // Add the square difference to tran& rotation average
             aSigmTr  += SqN2(aPose.Tr()-aAvgTr);
             aSigmRot += aAvgRot.Mat().SqL2Dist(aPose.Rot().Mat());
        }
     }

     std::string sSigmTr  = (aNbOk>1) ? ToStr(std::sqrt( aSigmTr/tREAL8(aNbOk-1))) : "xxxx" ;
     std::string sSigmRot = (aNbOk>1) ? ToStr(std::sqrt(aSigmRot/tREAL8(aNbOk-1))) : "xxxx" ;
     StdOut() << " STr=" << sSigmTr << " SRot=" << sSigmRot << std::endl;

     if ((anIdReportGlob!="") && anAppli)
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
    // for all num bloc
    for (size_t aKB=0 ; aKB<NbInBloc() ; aKB++)
    {
          //    * get name
         std::string  aName = NameKthInBloc(aKB);
          //    * estimate  relative pose with KMaster
         tPoseR  aPoseR =  EstimatePoseRel1Cple(aKMaster,aKB,nullptr,"");
          //    * update mMapPoseUKInBloc
         // mData.mMapPoseUKInBloc[aName]  = cPoseWithUK(aPoseR);
         mData.mMapPoseUKInBloc.try_emplace(aName,aPoseR);
    }

    Set4Compute(); //  now can be used in computation
}

void cBlocOfCamera::TestReadWrite(bool OmitDel) const
{
     StdOut() << "Test read/write xml " << std::endl;
     SaveInFile(mData,"BR.xml");
     cCalibBlocCam aBRXml;
     ReadFromFile(aBRXml,"BR.xml");
     SaveInFile(aBRXml,"BR_2.xml");
     StdOut() << "inspect BR.xml/BR_2.xml " << std::endl;
     getchar();

     StdOut() << "Test read/write json " << std::endl;
     SaveInFile(mData,"BR.json");
     cCalibBlocCam aBRJson;
     ReadFromFile(aBRJson,"BR.json");
     SaveInFile(aBRJson,"BR_2.json");
     StdOut() << "inspect BR.json/BR_2.json " << std::endl;
     getchar();

     StdOut() << "Test Hash code" << std::endl;
     StdOut() <<  "HX==HJ= " << HashValue(aBRXml,true) << " " <<  HashValue(aBRJson,true) << std::endl;
     aBRXml.mMapPoseUKInBloc.try_emplace("toto"); // , tPoseR();
     StdOut() <<  "HJ!=HX " << HashValue(aBRXml,true) << " " << HashValue(aBRJson,true) << std::endl;
     getchar();

     StdOut() << "Test read/write dmp " << std::endl;
     SaveInFile(mData,"BR.dmp");
     cCalibBlocCam aBRDump;
     ReadFromFile(aBRDump,"BR.dmp");
     SaveInFile(aBRDump,"BR_dmp.xml");
     StdOut() << "inspect BR_dmp.xml " << std::endl;
     getchar();

     StdOut() << "Test read/write txt " << std::endl;
     SaveInFile(mData,"BR.txt");
     cCalibBlocCam aBT;
     ReadFromFile(aBT,"BR.txt");
     SaveInFile(aBT,"BR_txt.xml");
     StdOut() << "inspect BR_dmp.txt " << std::endl;
     getchar();

     StdOut() << "Test dump vx xml/json " << std::endl;
     StdOut() <<  "HJ!=HD " << HashValue(aBRDump,true) << " " << HashValue(aBRJson,true) << std::endl;
     StdOut() <<  aBRDump.mMapPoseUKInBloc.begin()->first  << " " <<  aBRJson.mMapPoseUKInBloc.begin()->first << std::endl;
     StdOut() <<  aBRDump.mMapPoseUKInBloc["949"].AxeI()   << aBRJson.mMapPoseUKInBloc["949"].AxeI()  << std::endl;
     StdOut() <<  (aBRDump.mMapPoseUKInBloc["949"].AxeI()   - aBRJson.mMapPoseUKInBloc["949"].AxeI())  << std::endl;
     StdOut() << "Difference ~ epsilon machine " << std::endl;
     getchar();

     StdOut() << "Test specifications " << std::endl;
     SpecificationSaveInFile<cCalibBlocCam>("BR_specif.xml");
     SpecificationSaveInFile<cCalibBlocCam>("BR_specif.json");
     StdOut() << "inspect BR_specif.xml/BR_specif.json " << std::endl;
     getchar();



     this->ToFile("BR_Tofile.xml");
     cBlocOfCamera * aB2 = cBlocOfCamera::FromFile("BR_Tofile.xml");
     aB2->ToFile("BR_Tofile.xml");
     // test the effectiveness of cMemCheck
     if (! OmitDel)
         delete aB2;
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
	std::string              mNameBloc;    ///< Name of the bloc
	bool                     mTestRW;      ///< Do we do a test on read/write
	bool                     mTestNoDel;   ///< Do force an error on memory management to illustrate the 
};

cAppli_BlockCamInit::cAppli_BlockCamInit
(
     const std::vector<std::string> &  aVArgs,
     const cSpecMMVII_Appli & aSpec
) :
     cMMVII_Appli  (aVArgs,aSpec),
     mPhProj       (*this),
     mShowByBloc   (false),
     mShowBySync   (false),
     mNameBloc     ("TheBloc"),
     mTestRW       (false),
     mTestNoDel    (false)
{
}



cCollecSpecArg2007 & cAppli_BlockCamInit::ArgObl(cCollecSpecArg2007 & anArgObl)
{
      return anArgObl
             <<  Arg2007(mSpecImIn,"Pattern/file for images", {{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}}  )
             <<  mPhProj.DPOrient().ArgDirInMand()
             <<  Arg2007(mPattern,"Pattern for images specifing sup expr")
             <<  Arg2007(mNumSub,"Num of sub expr for x:block and  y:image")
             <<  mPhProj.DPRigBloc().ArgDirOutMand()
           ;
}


cCollecSpecArg2007 & cAppli_BlockCamInit::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{

    return    anArgOpt
             << AOpt2007(mNameBloc,"NameBloc","Set the name of the bloc ",{{eTA2007::HDV}})
             << AOpt2007(mMaster,"Master","Set the name of the master bloc, is user wants to enforce it ")
             << AOpt2007(mShowByBloc,"ShowByBloc","Show matricial organization by bloc ",{{eTA2007::HDV}})
             << AOpt2007(mShowBySync,"ShowBySync","Show matricial organization by sync ",{{eTA2007::HDV}})
             << AOpt2007(mTestRW,"TestRW","Call test en Read-Write ",{{eTA2007::HDV}})
             << AOpt2007(mTestNoDel,"TestNoDel","Force a memory leak error ",{{eTA2007::HDV}})
    ;
}

int cAppli_BlockCamInit::Exe()
{
    mPhProj.FinishInit();  // the final construction of  photogrammetric project manager can only be done now

    // creat the bloc, for now no cam,just the info to insert them
    cBlocOfCamera aBloc(mPattern,mNumSub.x(),mNumSub.y(),mNameBloc);

    //  parse all images : create the sensor and add it  to the bloc
    for (const auto & aNameIm :  VectMainSet(0))
    {
        cSensorCamPC * aCamPC  = mPhProj.ReadCamPC(aNameIm,true);
        aBloc.AddSensor(aCamPC);
    }


    // eventually show the bloc structure
    if (mShowByBloc)  aBloc.ShowByBloc();
    if (mShowBySync ) aBloc.ShowBySync();

    ///aBloc.EstimatePoseRel1Cple(0,1,this,"Global");
    // Show the statistics
     aBloc.StatAllCples(this);



    /*  Fix the master bloc if specicied by user */

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
    StdOut()  << " NumMaster " <<  aNumMaster  << std::endl;

    //  Do the estimation of calibration
    aBloc.EstimateBlocInit(aNumMaster);

    //  Save the bloc of camera
    mPhProj.SaveBlocCamera(aBloc);


    if (mTestRW)
    {
        aBloc.TestReadWrite(mTestNoDel);
    }

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
      "Compute initial calibration of rigid bloc cam",
      {eApF::Ori},
      {eApDT::Orient}, 
      {eApDT::Xml}, 
      __FILE__
);
/*
*/



}; // MMVII

