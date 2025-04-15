#include "CodedTarget.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_Interpolators.h"
#include "MMVII_Sensor.h"
#include "MMVII_Random.h"
#include "MMVII_PCSens.h"



// Test git branch

namespace MMVII
{

template <class Type>   
   std::pair<cPt2di,cIm2D<Type>>
       cDataIm2D<Type>::ReSample
               (
                   const cInterpolator1D & anInterpol,
                   const cDataInvertibleMapping<tREAL8,2> & aMap,
                   Type aDefValOut
               ) const
{
   cBox2dr aBoxOut = aMap.BoxOfFrontier(this->ToR(),1.0);

   cPt2di aP0 = Pt_round_down(aBoxOut.P0());
   cPt2di aP1 = Pt_round_up  (aBoxOut.P1());

   cIm2D<Type> aResult(cPt2di(0,0),aP1-aP0);
   cDataIm2D<Type>&  aDRes = aResult.DIm();

   for (auto & aPixOut : aDRes)
   {
       cPt2dr aPixIn = aMap.Inverse(MMVII::ToR(aPixOut+aP0));
       if (this->Inside(MMVII::ToI(aPixIn)))
       {
          aDRes.SetVTrunc(aPixOut,ClipedGetValueInterpol(anInterpol,aPixIn));
       }
       else
          aDRes.SetV(aPixOut,aDefValOut);
   }

   return  std::pair<cPt2di,cIm2D<Type>> (aP0,aResult);
}

template class cIm2D<tU_INT1>;


class cMapRessampleCam : public cDataInvertibleMapping<tREAL8,2>
{
    public :
       cPt2dr Value(const cPt2dr & aPt) const ;
       cPt2dr Inverse(const cPt2dr & ) const ;
       cMapRessampleCam(cPerspCamIntrCalib * aCalibIn, cPerspCamIntrCalib * aCalibOut);
    private :
        cPerspCamIntrCalib *       mCalibIn;  
        cPerspCamIntrCalib *       mCalibOut; 
};

cPt2dr cMapRessampleCam::Value(const cPt2dr & aPt) const 
{
   return mCalibOut->Value(mCalibIn->DirBundle(aPt));
}

cPt2dr cMapRessampleCam::Inverse(const cPt2dr & aPt) const 
{
   return mCalibIn->Value(mCalibOut->DirBundle(aPt));
}

cMapRessampleCam::cMapRessampleCam(cPerspCamIntrCalib * aCalibIn, cPerspCamIntrCalib * aCalibOut) :
   mCalibIn   (aCalibIn) ,
   mCalibOut  (aCalibOut) 
{
}

/*  *********************************************************** */
/*                                                              */
/*              cAppliSimulSphere                           */
/*                                                              */
/*  *********************************************************** */

/// Class for representing the sphere
class cSphSimul
{
     public :
        cPt2dr mC;   ///< center of the proj in image
        tREAL8 mRay; ///< Ray in pixel, so approxim

        bool AreSeparate(const cSphSimul &,tREAL8 aMargin = 2.0) const;
};

bool cSphSimul::AreSeparate(const cSphSimul & aSph2,tREAL8 aMargin) const
{
    return Norm2(mC-aSph2.mC) > aMargin * (mRay+aSph2.mRay);
}


class cAppliSimulSphere : public cMMVII_Appli
{
     public :
        typedef tU_INT1           tElem;
        typedef cIm2D<tElem>      tIm;
        typedef cDataIm2D<tElem>  tDIm;

        cAppliSimulSphere(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        // =========== overridding cMMVII_Appli::methods ============
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        bool IsSeparate(const cSphSimul & ) const;
        void  AddOneSphere(const cSphSimul & ) ;
        bool IsOk(const cPt2dr & ) const;
        bool IsOk(const cPt3dr & ) const;

        // in case we use real calib
        cPhotogrammetricProject     mPhProj;

        // =========== other methods ============

	std::string mNameIm;       ///< Name of background image

            


        // =========== Internal param ============
        tIm                        mImIn;        ///< Input global image
        cBox2dr                    mBoxIm;
        cPt2di                     mSzIm;
        cPerspCamIntrCalib *       mCalib;       ///< Calibration used 
        int                        mNbSphere;
        cPt2dr                     mIntervRay;
         
        std::vector<cSphSimul>     mVSph;
        tREAL8                     mBlurSz;
        eProjPC                    mTypeProj;
        tREAL8                     mFocPDiag;
};


/* *************************************************** */
/*                                                     */
/*              cAppliSimulSphere                  */
/*                                                     */
/* *************************************************** */

cAppliSimulSphere::cAppliSimulSphere(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli     (aVArgs,aSpec),
   mPhProj          (*this),
   mImIn            (cPt2di(1,1)),
   mBoxIm           (cPt2dr(1,1)),
   mNbSphere        (20),
   mIntervRay       (30.0,100.0),
   mBlurSz          (3.0),
   mTypeProj        (eProjPC::eStenope),
   mFocPDiag        (0.3)
{
}


cCollecSpecArg2007 & cAppliSimulSphere::ArgObl(cCollecSpecArg2007 & anArgObl)
{
   return  anArgObl
             <<   Arg2007(mNameIm,"Pattern of files",{{eTA2007::MPatFile,"0"},eTA2007::FileImage})
   ;
}


cCollecSpecArg2007 & cAppliSimulSphere::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
	          anArgOpt
             <<   mPhProj.DPOrient().ArgDirInOpt()
             <<   AOpt2007(mBlurSz,"BlurSz","Size of blurring at frontier",{eTA2007::HDV})
             <<   AOpt2007(mNbSphere,"NbSph","Number of spheres",{eTA2007::HDV})
             <<   AOpt2007(mTypeProj,"TProj","Type of projection",{{eTA2007::HDV},{AC_ListVal<eProjPC>()}})
             <<   AOpt2007(mFocPDiag,"FDiag","Focal as a prop of diag",{{eTA2007::HDV},{AC_ListVal<eProjPC>()}})
   ;
}

bool cAppliSimulSphere::IsSeparate(const cSphSimul & aSph) const
{
    for (const auto & aSph2 : mVSph)
       if (! aSph.AreSeparate(aSph2))
          return false;
    if (mBoxIm.Insideness(aSph.mC) < aSph.mRay +10.0)
       return false;
    return true;
}

bool cAppliSimulSphere::IsOk(const cPt2dr &  aPt) const
{
    return mCalib->DegreeVisibilityOnImFrame(aPt) >= 0.0;
}



bool cAppliSimulSphere::IsOk(const cPt3dr &  aPt) const
{
    return mCalib->DegreeVisibility(aPt) >= 0.0;
}


void  cAppliSimulSphere::AddOneSphere(const cSphSimul & aSph) 
{
   if (! IsOk(aSph.mC))
      return ;

   cPt3dr aAxeCone =   VUnit(mCalib->DirBundle(aSph.mC)); // direction of cone
   tREAL8 aRayCone = 0.0;
   int aNbS = 1000;

   // compute Ray of cone to have ray of proj +- equal  to aSph.mRay
   //  => average of distance of image of circle to axe
   {
       for (int aK=0 ; aK<aNbS ; aK++)
       {
           cPt2dr aPImK = aSph.mC + FromPolar(aSph.mRay,(aK*2*M_PI)/aNbS); // point on circle
           if (!IsOk(aPImK)) 
              return;
//StdOut() << "DdDV " << mCalib->DegreeVisibilityOnImFrame(aPImK) << "\n";

           cPt3dr aDirK =   VUnit(mCalib->DirBundle(aPImK)); // would be a point on cone w/o persp
           aRayCone += Norm2(aAxeCone-aDirK);
       } 
       aRayCone /= aNbS;
   }

   // compute box image => image of projection of cone
   cStdStatRes aStat;
   cTplBoxOfPts<tREAL8,2>  aBoxPtsIm;
   {
       // complete axe in ortho normal repair
       tRotR aRot= tRotR::CompleteRON(aAxeCone);
       cPt3dr aDir1 = aRot.AxeJ();
       cPt3dr aDir2 = aRot.AxeK();

       // parse point of cone and project
       for (int aK=0 ; aK<aNbS ; aK++)
       {
            tREAL8 aTeta = (aK*2*M_PI)/aNbS;
            cPt3dr aPOnC = VUnit(aAxeCone + (aDir1*cos(aTeta) + aDir2*sin(aTeta)) * aRayCone);
            if (! IsOk(aPOnC))
               return;
            aBoxPtsIm.Add(mCalib->Value(aPOnC));
            aStat.Add(Norm2(aPOnC-aAxeCone));
       }
    }

    tDIm & aDImIn = mImIn.DIm();
    
    cBox2di  aBoxIm = aBoxPtsIm.CurBox().Dilate(mBlurSz+20.0).ToI();
    aBoxIm = aBoxIm.Inter(aDImIn);

    // StdOut() << "CcCCC=" << aSph.mC << " "<< aAxeCone << " RayC=" << aRayCone << " " << ToR(aBoxIm.Sz()) / aSph.mRay << "\n";
    // StdOut() << " EcT=" << aStat.DevStd() << " " << aStat.Avg() - aRayCone << "\n";

    aRayCone = aStat.Avg();

    for (const auto & aPix : cRect2(aBoxIm))
    {
        if (IsOk(ToR(aPix)))
        {
            cPt3dr aDirBundle =   VUnit(mCalib->DirBundle(ToR(aPix))); // would be a point on cone w/o persp
            tREAL8 aSignedDist = (aRayCone -Norm2(aAxeCone-aDirBundle) )*  (aSph.mRay/aRayCone) ;

            tREAL8 aWSph = IntegrLinear(aSignedDist/mBlurSz); 

            aDImIn.SetV(aPix,255*aWSph);
         }
    }

/*
    for (const auto & aPix : aBoxIm)
    {
           cPt3dr aDirK =   VUnit(mCalib->DirBundle(aPImK)); // would be a point on cone w/o persp
    }
*/
    // cPtd3dr
}




int  cAppliSimulSphere::Exe()
{
   mPhProj.FinishInit();
   mImIn = tIm::FromFile(mNameIm);
   mSzIm = mImIn.DIm().Sz();
   mBoxIm = cBox2dr(cPt2dr(0,0),ToR(mSzIm));


   if (mPhProj.DPOrient().DirInIsInit())
   {
      mCalib =  mPhProj.InternalCalibFromImage(mNameIm);
   }
   else
   {
      tREAL8 aDiag = Norm2(mSzIm);

      mCalib =   cPerspCamIntrCalib::SimpleCalib
                 (
                     "SimulSphere",
                     // eProjPC::eStereroGraphik,
                     mTypeProj,
                     mSzIm,
                     cPt3dr(mSzIm.x()/2.0,mSzIm.y()/2.0,aDiag*mFocPDiag),
                     cPt3di(3,1,1)
                  );

   }

   tDIm & aDImIn = mImIn.DIm();
   for (const auto & aPix : aDImIn)
   {
      if (! IsOk(ToR(aPix)))
         aDImIn.SetV(aPix,0);
   }



   // generate the sphere such that don't intersect and dont touch border
   for (int aKpt = 0 ; aKpt< mNbSphere ; aKpt++)
   {
        int aNbT = 50;
        for (int aKTest=0 ; aKTest < aNbT ; aKTest++)
        {
             cSphSimul aSph;
             aSph.mC = mBoxIm.GeneratePointInside();
             aSph.mRay = RandInInterval(mIntervRay);
             if (IsSeparate(aSph))
             {
                aNbT = aKTest;
                aKTest = 1e9;
                mVSph.push_back(aSph);
             }
        }
   }


   for (const auto & aSph : mVSph)
   {
       AddOneSphere(aSph);
   }


   
   aDImIn.ToFile("toto.tif");

   // cSim2D<tREAL8> aSim(cPt2dr(-1000,1000),cPt2dr(0.9,0.5));
   // cInvertMappingFromElem<cSim2D<tREAL8>> aMap(aSim);
   cPt2di aSzOut = mSzIm;
   cPerspCamIntrCalib * aCalSter  =
       cPerspCamIntrCalib::SimpleCalib
       (
            "SimulSphereStereo",
             eProjPC::eStereroGraphik,
             aSzOut,
             cPt3dr(aSzOut.x()/2.0,aSzOut.y()/2.0,mCalib->F()),
             cPt3di(0,0,0)
        );


   // cTabulatedInterpolator aTabI(cSinCApodInterpolator(5.0,5.0),1000,true);
   cTabulatedInterpolator aTabI(cCubicInterpolator(-0.5),1000,true);

   cMapRessampleCam  aMap(mCalib,aCalSter);

   auto [aPt, aImRes]  =  aDImIn.ReSample(aTabI,aMap,128);

   aImRes.DIm().ToFile("toto_Resampled.tif");


   delete aCalSter;
   if (!mPhProj.DPOrient().DirInIsInit())
      delete mCalib;
   return EXIT_SUCCESS;
}


/* =============================================== */
/*                                                 */
/*                       ::                        */
/*                                                 */
/* =============================================== */

tMMVII_UnikPApli Alloc_SimulSphere(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliSimulSphere(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecSimulSphere
(
     "SimulImageSphere",
      Alloc_SimulSphere,
      "Simulate images of spheres taking into account perspectives & distortions",
      {eApF::CodedTarget,eApF::ImProc},
      {eApDT::Image,eApDT::Xml},
      {eApDT::Image,eApDT::Xml},
      __FILE__
);

#if (0)
#endif

};
