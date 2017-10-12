#include "StdAfx.h"
#include "Keypoint.h"


KeyPoint::KeyPoint(Pt2df P, float size, float angle, float response)
{
    m_Point.x=P.x;
    m_Point.y=P.y;
    m_size=size;
    m_angle=angle;
    m_response=response;
}

KeyPoint::KeyPoint(float x, float y, float size, float angle, float response)
{
    m_Point.x=x;
    m_Point.y=y;
    m_size=size;
    m_angle=angle;
    m_response=response;
}

KeyPoint::KeyPoint(){}

KeyPoint::~KeyPoint(){}
