#ifndef  _MMVII_CLINO_H_
#define  _MMVII_CLINO_H_

#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"
#include "MMVII_memory.h"


namespace MMVII
{

/** \file MMVII_Clino.h 
    \brief Classes for handling clinometers
 *  
 *
*/


/*  3 classes for representing the calibration of set of clinometer,
 *  as saved by the command "ClinoInit"
 */

class cOneCalibRelClino;   // Relative calibration of 2 clinometers,
class cOneCalibClino;      // Calibration of 1 clinometer
class cCalibSetClino;      // Set of calibration of "N" clinometers relatively to a Cam
class cOneMesureClino;     // Represent the  set of angular measures associated to one measurement time
class cSetMeasureClino;   // Represent a set of cOneMesureClino (used for ex in a measure or in a compensation)


/**  Class for Representing the  set of angular measures associated to one measurement time */
class cOneMesureClino
{
      public :
              cOneMesureClino(); ///< Default constructor, required for serialization
              /// Constructor used at import step
              cOneMesureClino(const std::string &,const std::vector<tREAL8> &,const std::optional<std::vector<tREAL8>>&);

              void AddData(const  cAuxAr2007 & anAux);  ///< Serialization
              const std::string &  Ident() const;          ///< Accessor
              const std::vector<tREAL8> & Angles() const;  ///< Accessor
              const std::optional<std::vector<tREAL8>> & VSigma() const; ///< Accessor
      private :
              std::string mIdent;  ///< Identifier of time
              std::vector<tREAL8> mAngles;  ///< Value of angles measures
              std::optional<std::vector<tREAL8>> mVSigma; ///< Optional values of sigma on measures
};
/// External function for serialization
void AddData(const  cAuxAr2007 & anAux, cOneMesureClino & aMesClino);

/** Class for Representing  a set of "cOneMesureClino" */
class cSetMeasureClino
{
       public :
          cSetMeasureClino(const std::string& aPatMatch,const std::string &aPatReplace,const std::vector<std::string> & aVNames = {});
          cSetMeasureClino();

          void Add1Mesures(const cOneMesureClino &);
          void AddData(const  cAuxAr2007 & anAux);

          std::string NameOfIm(const cOneMesureClino & ) const;

	  const std::vector<cOneMesureClino>&  SetMeasures() const;
          const std::vector<std::string> &     NamesClino() const;
          const  cOneMesureClino *  MeasureOfId(const std::string & anId,bool SVP=false) const;
          const  cOneMesureClino *  MeasureOfImage(const std::string & aNameImage,bool SVP=false) const;
	 

          void SetNames(const  std::vector<std::string> &);

          void FilterByPatIdent(const std::string& aPat);

       private :
          std::vector<std::string>      mNamesClino;
          std::string                   mPatMatch;
          std::string                   mPatReplace;
          std::vector<cOneMesureClino>  mSetMeasures;
};

/// External function for serialization
void AddData(const  cAuxAr2007 & anAux, cSetMeasureClino & aSet);



/** Relative calibration of 2 clinometers : Orient + name of reference */
class cOneCalibRelClino
{
      public :
         cOneCalibRelClino();  ///< Defaut constructor for serialization
         std::string    mNameRef; ///< Name of reference clinometer
         tRotR          mRot;     ///< Value of relative rotation
      private :
};
void AddData(const  cAuxAr2007 & anAux,cOneCalibRelClino & aSet);

/**   Calibrarion of 1 Clino :  Orient to camera + optional relative calib */
class cOneCalibClino
{
      public :
         cOneCalibClino();  ///< Defaut constructor for serialization
         cOneCalibClino(const std::string aNameClino);

         /// fix the interpretation  of mRot
         cPt3dr  CamToClino(const cPt3dr & aPt) const  {return  mRot.Value(aPt);}

         std::string    mNameClino;  ///< Name of clinometer
         tRotR           mRot;       ///< Value of rotation
         std::string    mCameraName; /// Name of camera with the relative orientation
         std::optional<cOneCalibRelClino>   mLinkRel;  ///< Possible relative calib
         tRotR Rot() const {return mRot;};
         std::string NameClino() const {return mNameClino;};
         std::string NameCamera() const {return mCameraName;};
};
void AddData(const  cAuxAr2007 & anAux,cOneCalibClino & aSet);

/** Global calibration : name of the camera + vector of all individual calibs */
class cCalibSetClino : public cMemCheck
{
     public :

         cCalibSetClino();  ///< Defaut constructor for serialization
         cCalibSetClino(std::string aNameCam, std::vector<cOneCalibClino> aClinosCal);
         /// Name of the camera where the calibration, but at least for tracability
         std::string NameCam(){return mNameCam;};
         const std::vector<cOneCalibClino> & ClinosCal() const {return mClinosCal;};

         // Set clinometers calibration
         void setClinosCal(std::vector<cOneCalibClino>  aClinosCal){mClinosCal=aClinosCal;}


         std::string mNameCam;

	      /// Set of all clinometers calibration
         std::vector<cOneCalibClino>  mClinosCal  ;        
};
void AddData(const  cAuxAr2007 & anAux,cCalibSetClino & aSet);


/** In case we are working in non verticalized system (for ex no target, no gps ...) , it may
    be necessary to extract vertical in the camera coordinate  system, using the boresight calibration
    and the clino measures
*/


class cGetVerticalFromClino
{
    public :
       cGetVerticalFromClino(const cCalibSetClino &,const std::vector<tREAL8> & aVAngle);
       tREAL8 ScoreDir3D(const cPt3dr & aDir) const;

       std::pair<tREAL8,cPt3dr> OptimGlob(int aNbStep0,tREAL8 StepEnd) const;

       cPt3dr OptimInit(int aNbStepInSphere) const;
       cPt3dr Refine(cPt3dr aP0,tREAL8 StepInit,tREAL8 StepEnd) const;
    private :
       const cCalibSetClino & mCalibs;
       std::vector<cPt2dr>    mDirs;
};




};

#endif  //  _MMVII_CLINO_H_
