#include "cMMVII_Appli.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_Geom3D.h"
#include "MMVII_Sensor.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Tpl_Images.h"
#include "MMVII_PointCloud.h"
#include "MMVII_Linear2DFiltering.h"
#include "MMVII_Interpolators.h"
#include "MMVII_PCSens.h"


namespace MMVII
{
/*
    To do mark :
*/

class cCamOrthoC;
class cOrthoProj;



/* ------------------  cOrthoProj ------------------- */

/**  The orthographic projection as a bijective mapping  R3 <-> R3 .
 *    Used as element to build a  cCamOrthoC.  */


class cOrthoProj  :  public  tIMap_R3
{
    public :
       friend cCamOrthoC;
       typedef std::vector<cPt3dr> tVecP3;

       cOrthoProj (const tRotR & aRot ,const cPt3dr& aC,const cPt2dr& aPP ,tREAL8 aResol,bool profIsZ0) ;
       cOrthoProj (const cPt3dr & aDir,const cPt3dr& aC =cPt3dr(0,0,0),const cPt2dr& aPP= cPt2dr(0,0) ,tREAL8 aResol=1.0,bool profIsZ0=false) ;
       tSeg3dr  BundleInverse(const cPt2dr &) const ;

       cOrthoProj(const cOrthoProj&);

        
    private  :

       /// the proection PP + Pi(W->L(P-C))/Resol  , the we add Z

       const  tVecP3 &  Values   (tVecP3 &,const tVecP3 & ) const override;
       const  tVecP3 &  Inverses (tVecP3 &,const tVecP3 & ) const override;

    
       tRotR  mRL2W;  ///< Axes, rotation Local->Word
       cPt3dr mC;     ///< Center
       cPt2dr mPP;    ///< Principal point
       tREAL8 mResol;    ///< Resol in ground unit
       bool   mProfIsZ0; ///< Prof is Z Init (not local), used for an old bug ...
};


/* ------------------  cCamOrthoC ------------------- */

class cCamOrthoC  :  public  cSensorImage
{
    public :
       cCamOrthoC(const std::string &aName,const cOrthoProj & aProj,const cPt2di & aSz);

       cPt2dr Ground2Image(const cPt3dr &) const override;
       const cPixelDomain & PixelDomain() const override;
       tSeg3dr  Image2Bundle(const cPt2dr &) const override;
       std::string  V_PrefixName() const   override;
       cPt3dr  PseudoCenterOfProj() const override;
       double DegreeVisibility(const cPt3dr &) const override;

       bool  HasImageAndDepth() const override;
       cPt3dr Ground2ImageAndDepth(const cPt3dr &) const override;
       cPt3dr ImageAndDepth2Ground(const cPt3dr &) const override;

    private :
       cOrthoProj         mProj;
       cDataPixelDomain   mDataPixDom;
       cPixelDomain       mPixelDomain;
};

/* ------------------  cProjPointCloud  ------------------- */


///  Class for computing projection of a point cloud

class cResImagesPPC
{
   public :
      cResImagesPPC(const cPt2di & aSz);

      cIm2D<tU_INT1>   mImRadiom;
      cIm2D<tU_INT1>   mImWeight;
      cIm2D<tREAL4>    mImDepth;
};


class cProjPointCloud
{
     public :
         typedef tREAL8 tImageDepth;
         static constexpr int NoIndex = -1;

         /// constructor : memoriez PC, inialize accum, allocate mem
         cProjPointCloud(cPointCloud & aParam,tREAL8 aWeightInit );

	 /// Process on projection for  OR  (1) modify colorization of points (2) 
         void ProcessOneProj
              (
                    tREAL8 aSurResol,
                    const cSensorImage &,
                    tREAL8 aW,
                    bool ModeImage,
                    const std::string& aMsg,
                    bool  ShowMsg,
                    bool  ExportIm
              );
         
         cResImagesPPC ProcessImage(tREAL8 aSurResol,const cSensorImage &);

	 // export the average of radiomeries (in mSumRad) as a field of mPC
         void ColorizePC(); 
     cCamOrthoC * PPC_CamOrtho(int aK,bool  ProfIsZ0,const cPt3dr & aDir,tREAL8 aMulResol=1.0, tREAL8 aMulSz = 1.0);
     cCamOrthoC * PPC_CamOrtho(int aK,bool  ProfIsZ0,const tRotR & aDir,tREAL8 aMulResol=1.0, tREAL8 aMulSz = 1.0);

     // cCamOrthoC * PPC_CamEpip(cCamOrthoC *,const tRotR & aDir);


     private :
	 // --------- Processed at initialization ----------------
         cPointCloud&           mPC;       ///< memorize cloud point
         const int              mNbPtsGlob;    ///< store number of points
	 // int                    mNbPts;    ///<  Dynamic, change with SetOk
         std::vector<cPt3dr>    mGlobPtsInit; ///< initial point cloud (stores once  for all in 64-byte, for efficienciency)
         std::vector<cPt3dr> *  mVPtsInit;     /// Dynamic, change with SetOk
         // const tREAL8           mSurResol;
	 const tREAL8           mAvgD;       ///< Avg 2D-Distance between points in 3D Cloud
         //const tREAL8           mStepProf;  ///< Step for computing depth-images
	 // --------- Updated  with  "ProcessOneProj"  ----------------
         tREAL8                 mSumW;      ///< accumulate sum of weight on radiometries
         std::vector<tREAL4>    mSumRad;    ///< accumulate sum of radiometry
	 // --------- Computed at each run of "ProcessOneProj"  ------------------
         std::vector<cPt3dr>    mVPtsProj;  ///< memorize projections 
         std::vector<cPt2di>    mVPtImages; ///< Projection in image of given 3D Point
         cTplBoxOfPts<int,2>    mBoxInd;    ///< Compute for of mVPtImages
					    
         cPt2di                  mSzIm;
         cIm2D<tImageDepth>      mImDepth;
         cDataIm2D<tImageDepth>* mDImDepth;
         cIm2D<tREAL4>            mImRad;
         cDataIm2D<tREAL4>*       mDImRad;
         cIm2D<tREAL4>            mImWeigth;
         cDataIm2D<tREAL4>*       mDImWeigth;
         cIm2D<int>               mImIndex;
         cDataIm2D<int>*          mDImIndex;
};

};
