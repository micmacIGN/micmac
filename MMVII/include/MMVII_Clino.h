#ifndef  _MMVII_CLINO_H_
#define  _MMVII_CLINO_H_

#include "MMVII_Geom3D.h"
#include "MMVII_PCSens.h"


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


/** Relative calibration of 2 clinometers : Orient + name of reference */
class cOneCalibRelClino
{
      public :
         cOneCalibRelClino();  ///< Defaut constructor for serialization
         std::string    mNameRef; ///< Name of reference clinometer
         tRotR          mRot;     ///< Value of relative rotation
      private :
};

/**   Calibrarion of 1 Clino :  Orient to camera + optional relative calib */
class cOneCalibClino
{
      public :
         cOneCalibClino();  ///< Defaut constructor for serialization
         std::string    mNameClino;  ///< Name of clinometer
         tRotR           mRot;       ///< Value of rotation
         std::optional<cOneCalibRelClino>   mLinkRel;  ///< Possible relative calib
};

/** Global calibration : name of the camera + vector of all individual calibs */
class cCalibSetClino
{
     public :

         cCalibSetClino();  ///< Defaut constructor for serialization
         /// Name of the camera where the calibration, but at least for tracability
         std::string mNameCam;

	 /// Set of all clinometers calibration
         std::vector<cOneCalibClino>  mClinosCal  ;
};



};

#endif  //  _MMVII_CLINO_H_
