#include "cMMVII_Appli.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_PhgrDist.h"
#include "../Serial/Serial.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "cExternalSensor.h"

using namespace NS_SymbolicDerivative;

namespace MMVII
{

class cChangCoordSensImage ;
class cDataChcSI;
class cChSysSensImage;


/* *************************************************** */
/*                                                     */
/*            cChangCoordSensImage                     */
/*                                                     */
/* *************************************************** */

class cChangCoordSensImage : public cSensorImage
{
      public :
         cChangCoordSensImage();
         virtual ~cChangCoordSensImage();

         void CCSI_SetSensorAndMap(const cSensorImage * aSI,cDataInvertibleMapping<tREAL8,3>  *aChSys,bool DeleteSI);
      protected :

	 // ====  Methods overiiding for being a cSensorImage =====
         tSeg3dr  Image2Bundle(const cPt2dr &) const override;
         /// Basic method  GroundCoordinate ->  image coordinate of projection
         cPt2dr Ground2Image(const cPt3dr &) const override;
	 
	 //    To see if  we can do more accurate, but for now maintain method with bundles 
         //  cPt3dr ImageAndZ2Ground(const cPt3dr &) const override;


	  /// Indicate how much a point belongs to sensor visibilty domain
         double DegreeVisibility(const cPt3dr &) const  override;

	 ///  
         const cPixelDomain & PixelDomain() const override;
	 std::string  V_PrefixName() const override;
         cPt3dr  EpsDiffGround2Im(const cPt3dr &) const override;


         ///  Return center of footprint (used for example in RTL)
         const cPt3dr *  CenterOfFootPrint() const override;


         tProjImAndGrad  DiffGround2Im(const cPt3dr &) const override;

         bool  HasIntervalZ()  const override;
         cPt2dr GetIntervalZ() const override;
         cPt3dr  PseudoCenterOfProj() const override;

	 bool               mDeleteSI;
	 const cSensorImage *  mSensorInit;
	 cDataInvertibleMapping<tREAL8,3> *  mChCoord;
	 cPt2dr             mIntZ;
	 bool               mHasCOFP;
	 cPt3dr             mCenterOfFootPrint;
};

cChangCoordSensImage::cChangCoordSensImage() :
   cSensorImage    (""),
   mDeleteSI       (false),
   mSensorInit     (nullptr),
   mChCoord        (nullptr),
   mHasCOFP        (false)
{
}

void cChangCoordSensImage::CCSI_SetSensorAndMap(const cSensorImage * aSI,cDataInvertibleMapping<tREAL8,3> * aChSys,bool DeleteSI)
{
    mSensorInit = aSI;
    mChCoord = aChSys;
    mDeleteSI = DeleteSI;

    SetNameImage(aSI->NameImage());
    // Compute interval Z 
    if (aSI->HasIntervalZ())
    {
         cPt2dr  aIntZ = aSI->GetIntervalZ();
         cSet2D3D  aVCor =  aSI->SyntheticsCorresp3D2D(5,2,aIntZ.x(),aIntZ.y(),false,0.0);

	 cBoundVals<tREAL8> aBounds;
	 for (const auto  & aCorresp : aVCor.Pairs())
	 {
             aBounds.Add(mChCoord->Value(aCorresp.mP3).z());
	 }
	 mIntZ.x() = aBounds.VMin();
	 mIntZ.y() = aBounds.VMax();
    }

    // Compute center of foot print
    const cPt3dr *  aCOFP = mSensorInit->CenterOfFootPrint();
    if (aCOFP)
    {
       mHasCOFP = true;
       mCenterOfFootPrint = mChCoord->Value(*aCOFP);
    }
}

cChangCoordSensImage::~cChangCoordSensImage()
{
   if (mDeleteSI)
      delete mSensorInit;
   delete mChCoord;
}

std::string cChangCoordSensImage::V_PrefixName() const {return "ChangCoord";}

tSeg3dr  cChangCoordSensImage::Image2Bundle(const cPt2dr & aPixIm) const
{
     tSeg3dr aSeg =   mSensorInit->Image2Bundle(aPixIm);

     return tSeg3dr(mChCoord->Value(aSeg.P1()), mChCoord->Value(aSeg.P2()));
}


cPt2dr cChangCoordSensImage::Ground2Image(const cPt3dr & aPGround) const 
{
     return mSensorInit->Ground2Image(mChCoord->Inverse(aPGround));
}



double cChangCoordSensImage::DegreeVisibility(const cPt3dr & aPGround) const
{
     return mSensorInit->DegreeVisibility(mChCoord->Inverse(aPGround));
}

const cPixelDomain & cChangCoordSensImage::PixelDomain() const 
{
   return mSensorInit->PixelDomain();
}


cPt3dr  cChangCoordSensImage::EpsDiffGround2Im(const cPt3dr &) const 
{
	return mChCoord->EpsJac();
}

bool  cChangCoordSensImage::HasIntervalZ()  const {return mSensorInit->HasIntervalZ();}
cPt2dr cChangCoordSensImage::GetIntervalZ() const {return mIntZ;}

tProjImAndGrad  cChangCoordSensImage::DiffGround2Im(const cPt3dr & aPt) const
{
     // return mSensorInit->DiffG2IByFiniteDiff(aPt);
     return DiffG2IByFiniteDiff(aPt);
}

const cPt3dr *  cChangCoordSensImage::CenterOfFootPrint() const
{
    if  (mHasCOFP) 
        return & mCenterOfFootPrint;
    return nullptr;
}

cPt3dr  cChangCoordSensImage::PseudoCenterOfProj() const 
{
	return mChCoord->Value(mSensorInit->PseudoCenterOfProj());
}

/* *************************************************** */
/*                                                     */
/*            cChSysSensImage                          */
/*                                                     */
/* *************************************************** */


class cChSysSensImage : public cChangCoordSensImage
{
	public :
           cChSysSensImage();
           cChSysSensImage(const cSensorImage *,const std::string& aDirSens,const cChangeSysCo &);
	   void FinishInit(const cPhotogrammetricProject & aPhProj);

	   static std::string    StaticPrefixName() ;
           void ToFile(const std::string & aNameFile) const override;
	   void AddData(const  cAuxAr2007 & );

	   static cChSysSensImage * TryRead(const cPhotogrammetricProject &,const std::string&aNameImage);

	private :
	   std::string  V_PrefixName() const override;
	   // -----------------------  Data Part to read/write -------------------
              std::string         mDirSensInit;
              std::string         mNameImage;
              std::string         mSysCoOri;
	   // -----------------------  End data -------------------
	   cChangeSysCo *  mChSys;
};


cChSysSensImage::cChSysSensImage() :
     cChangCoordSensImage  (),
     mChSys                (nullptr)
{
}

cChSysSensImage::cChSysSensImage
(const cSensorImage *           aSensInit,
     const std::string&       aDirSens,
     const cChangeSysCo &aChSys
) :
     cChangCoordSensImage  ()
{
    mDirSensInit = aDirSens;
    mNameImage   = aSensInit->NameImage();
    mSysCoOri    = aChSys.SysOrigin()->Def();
    SetCoordinateSystem(aChSys.SysTarget()->Def());

    cChangCoordSensImage::CCSI_SetSensorAndMap(aSensInit,new cChangeSysCo(aChSys),false /* delete sens*/);
}

void cChSysSensImage::AddData(const  cAuxAr2007 & anAux0)
{
    cAuxAr2007 anAux("ChangSystemCoord",anAux0);

    MMVII::AddData(cAuxAr2007("FolderOri",anAux),mDirSensInit);
    MMVII::AddData(cAuxAr2007("NameImage",anAux),mNameImage);

    MMVII::AddData(cAuxAr2007("Input_"+TagCoordSys,anAux),mSysCoOri);

    if (anAux.Input())
    {
       SetCoordinateSystem("");
    }
    MMVII::AddData(cAuxAr2007(TagCoordSys,anAux),OptCoordinateSystem().value());
}

void AddData(const  cAuxAr2007 & anAux,cChSysSensImage & aChS)
{
     aChS.AddData(anAux);
}

void cChSysSensImage::ToFile(const std::string & aNameFile) const 
{
     SaveInFile(*this,aNameFile);
}


std::string cChSysSensImage::StaticPrefixName() {return "ChangSys";}
std::string cChSysSensImage::V_PrefixName() const {return StaticPrefixName();}

void cChSysSensImage::FinishInit(const cPhotogrammetricProject & aPhProj)
{
     mChSys = new cChangeSysCo(aPhProj.ChangSysCo(mSysCoOri,GetCoordinateSystem()));
     cSensorImage* aSensor = aPhProj.ReadSensorFromFolder(mDirSensInit,mNameImage,true,false);
     cChangCoordSensImage::CCSI_SetSensorAndMap(aSensor,mChSys,false /*DeleteSens*/);
}


cChSysSensImage * cChSysSensImage::TryRead(const cPhotogrammetricProject & aPhProj,const std::string&aNameImage)
{
     bool AlreadyExist=false;
     cChSysSensImage * aResult =  SimpleTplTryRead<cChSysSensImage>(aPhProj, aNameImage,AlreadyExist);

     if ((aResult==nullptr) || AlreadyExist)  return aResult;

     aResult->FinishInit(aPhProj);

     return aResult;
}

///  Global interface creator to cChSysSensImage
cSensorImage * SensorTryReasChSys(const cPhotogrammetricProject & aPhProj,const std::string & aNameImage)
{
    return cChSysSensImage::TryRead(aPhProj,aNameImage);
}

/* *************************************************** */
/*                                                     */
/*            cSensorImage                             */
/*                                                     */
/* *************************************************** */

cSensorImage * cSensorImage::SensorChangSys(const std::string & aDir,cChangeSysCo & aChSys) const
{
    return new cChSysSensImage(this,aDir,aChSys);
} 


};
