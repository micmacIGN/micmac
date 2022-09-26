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

class cDefProjPerspC
{
       public :
          /// signed belonging function for 3-d points
          virtual tREAL8  P3DIsDef(const cPt3dr &) const = 0 ;
          /// signed belonging function for 2-d points
          virtual tREAL8  P2DIsDef(const cPt2dr &) const =0 ;

          static const cDefProjPerspC & ProjOfType(eProjPC);
};


/// Base class for all image geometry, laser
/*
class cDataGeomSensor : public cDataMapping<tREAL8,3,2>
{
   public :
       cDataGeomSensor();
   private :
};
*/


/*
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
