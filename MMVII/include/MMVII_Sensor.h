#ifndef  _MMVII_SENSOR_H_
#define  _MMVII_SENSOR_H_

namespace MMVII
{

/** \file MMVII_Sensor.h
    \brief Interface class for sensors

   Most probably this will evolve a lot, with several reengenering 
  phases. 

   For now, it will essentially contain interfaces to existing MMV1 cameras.

*/

/// Base class for all image geometry, laser
/*
class cGeomSensor : public cMapping<3,2>
{
   public :
       cGeomSensor();
   private :
};


/// Base class for all image geometry, laser
class cPoseStenope : public cGeomSensor
{
   public :
       /// To make a general mapping, just a synomym of Proj
       cPtxd<double,2> Direct(const cPtxd<double,3> &) const override;
       /// 
       cPt2dr  Proj(const cPt3dr &);

       cPoseStenope(const cPt3dr & aC);
       
       const cPt3dr & C() const;
   private :
       cPt3dr  mC;
};
*/


};

#endif  //  _MMVII_SENSOR_H_
