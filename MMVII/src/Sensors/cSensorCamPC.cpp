#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"


/**
   \file cSensorCamPC.cpp  

   \brief file for implementing class sensor for perspective central
*/


namespace MMVII
{

/* ******************************************************* */
/*                                                         */
/*                   cSensorCamPC                          */
/*                                                         */
/* ******************************************************* */

cSensorCamPC::cSensorCamPC(const std::string & aNameImage,const tPose & aPose,cPerspCamIntrCalib * aCalib) :
   cSensorImage     (aNameImage),
   mPose            (aPose),
   mInternalCalib   (aCalib),
   mOmega           (0,0,0)
{
}

std::vector<cObjWithUnkowns<tREAL8> *>  cSensorCamPC::GetAllUK() 
{
    return std::vector<cObjWithUnkowns<tREAL8> *> {this,mInternalCalib};
}


void cSensorCamPC::PutUknowsInSetInterval()
{
    mSetInterv->AddOneInterv(mPose.Tr());
    mSetInterv->AddOneInterv(mOmega);
}

cPt2dr cSensorCamPC::Ground2Image(const cPt3dr & aP) const
{
        //  mPose(0,0,0) = Center, then mPose Cam->Word, then we use Inverse, BTW Inverse is as efficient as direct
     return mInternalCalib->Value(mPose.Inverse(aP));
}

double cSensorCamPC::Visibility(const cPt3dr & aP) const
{
     return mInternalCalib->Visibility(mPose.Inverse(aP));
}

double cSensorCamPC::VisibilityOnImFrame(const cPt2dr & aP) const
{
     return mInternalCalib->VisibilityOnImFrame(aP);
}

cPt3dr cSensorCamPC::Ground2ImageAndDepth(const cPt3dr & aP) const
{
    cPt3dr aPCam = mPose.Inverse(aP);  // P in camera coordinate
    cPt2dr aPIm = mInternalCalib->Value(aPCam);

    return cPt3dr(aPIm.x(),aPIm.y(),aPCam.z());
}

const cPt2di & cSensorCamPC::SzPix() const {return  mInternalCalib->SzPix();}

cPt3dr cSensorCamPC::ImageAndDepth2Ground(const cPt3dr & aP) const
{
    cPt3dr aPCam = mInternalCalib->Inverse(Proj(aP));

    return mPose.Value(aPCam * (aP.z() / aPCam.z()));

}

const cPt3dr * cSensorCamPC::CenterOfPC() { return  & Center(); }
         /// Return the calculator, adapted to the type, for computing colinearity equation
cCalculator<double> * cSensorCamPC::EqColinearity(bool WithDerives,int aSzBuf) 
{
   return mInternalCalib->EqColinearity(WithDerives,aSzBuf);
}

void cSensorCamPC::PushOwnObsColinearity(std::vector<double> & aVObs)
{
     Pose().Rot().Mat().PushByCol(aVObs);
}



/*
size_t  cSensorCamPC::NumXCenter() const
{
   return IndOfVal(&(mPose.Tr().x()));
}
*/

cPerspCamIntrCalib * cSensorCamPC::InternalCalib() {return mInternalCalib;}

const cPt3dr & cSensorCamPC::Center() const {return mPose.Tr();}
const cPt3dr & cSensorCamPC::Omega()  const {return mOmega;}
cPt3dr cSensorCamPC::AxeI()   const {return mPose.Rot().AxeI();}
cPt3dr cSensorCamPC::AxeJ()   const {return mPose.Rot().AxeJ();}
cPt3dr cSensorCamPC::AxeK()   const {return mPose.Rot().AxeK();}
const cIsometry3D<tREAL8> & cSensorCamPC::Pose() const {return mPose;}

/*   Let R be the rotation of pose  P=(C,P= : Cam-> Word, what is optimized in colinearity for a ground point G
 *   is Word->Cam  :
 *
 *          tR(G-C)
 *
 * So the optimal rotation R' with get satisfy the equation :
 *
 *          (1+^W) tR0 =tR'   
 *
 * The we have the formula :
 *
 *          --->   R'=R0 t(1+^W)
 *
 *  And note that for axiators :
 *
 *       t Axiator(W) = Axiator(-W)
 *
 */

void cSensorCamPC::OnUpdate()
{
	//  used above formula to modify  rotation
     mPose.SetRotation(mPose.Rot() * cRotation3D<tREAL8>::RotFromAxiator(-mOmega));
        // now this have modify rotation, the "delta" is void :
     mOmega = cPt3dr(0,0,0);
}


     // =================  READ/WRITE on files ===================

void cSensorCamPC::AddData(const cAuxAr2007 & anAux)
{
     std::string aNameImage = NameImage();
     cPt3dr aC = Center();
     cPt3dr aI = AxeI();
     cPt3dr aJ = AxeJ();
     cPt3dr aK = AxeK();
     cPtxd<tREAL8,4>  aQuat =  MatrRot2Quat(mPose.Rot().Mat());
     std::string      aNameCalib = (anAux.Input() ? "" : mInternalCalib->Name());


     MMVII::AddData(cAuxAr2007("NameImage",anAux),aNameImage);
     MMVII::AddData(cAuxAr2007("NameInternalCalib",anAux),aNameCalib);
     MMVII::AddData(cAuxAr2007("Center",anAux),aC);

    {
        cAuxAr2007 aAuxRot("RotMatrix",anAux);
        MMVII::AddData(cAuxAr2007("AxeI",aAuxRot),aI);
        MMVII::AddData(cAuxAr2007("AxeJ",aAuxRot),aJ);
        MMVII::AddData(cAuxAr2007("AxeK",aAuxRot),aK);
    }
    MMVII::AddData(cAuxAr2007("EQ",anAux),aQuat);
    AddComment(anAux.Ar(),"EigenQuaternion, for information");

    cPt3dr aWPK = mPose.Rot().ToWPK() *  (180.0/M_PI);
    MMVII::AddData(cAuxAr2007("WPK",anAux),aWPK);
    AddComment(anAux.Ar(),"Omega Phi Kapa in degree, for information");


    cPt3dr aYPR = mPose.Rot().ToYPR() *  (180.0/M_PI);
    MMVII::AddData(cAuxAr2007("YPR",anAux),aYPR);
    AddComment(anAux.Ar(),"Yaw Pitch Roll in degree, for information");



    if (anAux.Input())
    {
         SetNameImage(aNameImage);
         mPose = tPose(aC,cRotation3D<tREAL8>(MatFromCols(aI,aJ,aK),false));
	 mTmpNameCalib = aNameCalib;
	 mOmega = cPt3dr(0,0,0);
    }
}

void AddData(const cAuxAr2007 & anAux,cSensorCamPC & aPC)
{
    aPC.AddData(anAux);
}

void cSensorCamPC::ToFile(const std::string & aNameFile) const
{
    SaveInFile(const_cast<cSensorCamPC &>(*this),aNameFile);
    std::string aNameCalib = DirOfPath(aNameFile) + mInternalCalib->Name() + ".xml";

    mInternalCalib->ToFileIfFirstime(aNameCalib);
}

cSensorCamPC * cSensorCamPC::FromFile(const std::string & aFile)
{
   cSensorCamPC * aPC = new cSensorCamPC("NONE",tPose::Identity(),nullptr);
   ReadFromFile(*aPC,aFile);

   aPC->mInternalCalib =  cPerspCamIntrCalib::FromFile(DirOfPath(aFile) + aPC->mTmpNameCalib + ".xml");
   aPC->mTmpNameCalib = "";

   return aPC;
}

std::string  cSensorCamPC::NameOri_From_Image(const std::string & aNameImage)
{
   return cSensorImage::NameOri_From_PrefixAndImage(PrefixName(),aNameImage);
}

std::vector<cPt2dr>  cSensorCamPC::PtsSampledOnSensor(int aNbByDim) const 
{
     return  mInternalCalib->PtsSampledOnSensor(aNbByDim);
}


std::string  cSensorCamPC::V_PrefixName() const { return PrefixName() ; }
std::string  cSensorCamPC::PrefixName()  { return "PerspCentral";}

void  cSensorCamPC::GetAdrInfoParam(cGetAdrInfoParam<tREAL8> & aGAIP)
{
   aGAIP.TestParam(this, &( mPose.Tr().x()),"Cx");
   aGAIP.TestParam(this, &( mPose.Tr().y()),"Cy");
   aGAIP.TestParam(this, &( mPose.Tr().z()),"Cz");

   aGAIP.TestParam(this, &( mOmega.x())    ,"Wx");
   aGAIP.TestParam(this, &( mOmega.y())    ,"Wy");
   aGAIP.TestParam(this, &( mOmega.z())    ,"Wz");
}

}; // MMVII


