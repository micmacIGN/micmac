#ifndef KEYPOINT_H
#define KEYPOINT_H
#include "StdAfx.h"

class KeyPoint
{
	public:
    KeyPoint(Pt2df P, float size, float angle, float response);
    KeyPoint(float x, float y, float size, float angle, float response);
    KeyPoint();
    ~KeyPoint();
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
	 void setAngle(float angle){m_angle=angle;}
     void setResponse(float response){m_response=response;}
     void undist(CamStenope * aCalib);
     //getters
     Pt2df getPoint(){return m_Point;}
     float getAngle(){return m_angle;}
     float getSize() {return m_size ;}
     float getResponse(){return m_response;}
	 
	private:
        Pt2df m_Point;
		float m_size;
		float m_angle;
		float m_response;
    };



#endif
