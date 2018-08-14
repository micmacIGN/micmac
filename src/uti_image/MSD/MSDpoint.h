#ifndef MSDPOINT_H
#define MSDPOINT_H
#include "StdAfx.h"
#include "../../uti_image/Digeo/Digeo.h"
#include "DescriptorExtractor.h"



class MSDPoint
{
	public:
    MSDPoint(Pt2df P, float size, float angle, float response);
    MSDPoint(float x, float y, float size, float angle, float response);
    MSDPoint();
    ~MSDPoint();
    //setters
     void setPoint(Pt2dr Point)
     {
		 m_Point.x=Point.x;
		 m_Point.y=Point.y;
     }

     void setPointx(float x)
     {
         m_Point.x=x;
     }

     void setPointy(float y)
     {
         m_Point.y=y;
     }


     void setSize(float sz) {m_size=sz;}
     void setScale(float sc) {m_scale=sc;}    
     void setAngle(float angle, unsigned int index){
         if (index<m_angle.size()) {m_angle.at(index)=angle;}
         else {std::cout << "MSD: index " << index << " for angle " << angle << "is not an option\n";}}
     void addAngle(float angle){m_angle.push_back(angle);}
     void setResponse(float response){m_response=response;}
     void undist(CamStenope * aCalib);
     //getters
     Pt2df getPoint(){return m_Point;}
     float getAngle(unsigned int index){return m_angle.at(index);}
     float getAngle(){return m_angle.at(0);}
     std::vector<float> getAngles(){return m_angle;}
     float getSize() {return m_size ;}
     float getScale() {return m_scale ;}
     float getResponse(){return m_response;}
	 
	private:
        Pt2df m_Point;
        // size in pixel
		float m_size;
        // Dezoom at wich the point has been detected
        float m_scale;
        // radians
        std::vector<float> m_angle;
		float m_response;
 };

template <class tData, class tComp>
std::vector<DigeoPoint> ToDigeo(std::vector<MSDPoint> & aVMSD, Im2D<tData, tComp> & Image);

template <class tData, class tComp>
DigeoPoint ToDigeo(MSDPoint & aPt,DescriptorExtractor<tData, tComp> & aDesc);

#endif
