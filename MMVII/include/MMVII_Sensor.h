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


/**  Interface class for specification where of a projection is defined: */
class cDefProjPerspC
{
       public :
          /// signed belonging function for 3-d points
          virtual tREAL8  P3DIsDef(const cPt3dr &) const = 0 ;
          /// signed belonging function for 2-d points
          virtual tREAL8  P2DIsDef(const cPt2dr &) const =0 ;

          /// Radial symetry, true for physcicall based model, false for ex with equirect
          virtual bool  HasRadialSym() const ;
          static const cDefProjPerspC & ProjOfType(eProjPC);
};

/** class for representing  a 3D point paired with it 2d image projection */
 
struct  cPair2D3D
{
     public :
          cPair2D3D(const cPt2dr &,const cPt3dr &);
          cPt2dr mP2;
          cPt3dr mP3;
};

/**  class for representing  set of pairs 2-3  */
struct cSet2D3D
{
     public :
         typedef std::vector<cPair2D3D>   tCont2D3D;

         void AddPair(const cPair2D3D &);
         const tCont2D3D &  Pairs() const;
         void  Clear() ;

     private :
        tCont2D3D  mPairs;
};


class cSensorImage  :  public cObjWithUnkowns<tREAL8>
{
     public :
         virtual cPt2dr Ground2Image(const cPt3dr &) const = 0;
         virtual double SqResidual(const cPair2D3D &) const;
         virtual double AvgResidual(const cSet2D3D &) const;
};


};

#endif  //  _MMVII_SENSOR_H_
