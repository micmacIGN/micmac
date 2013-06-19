#ifndef DATA_H
#define DATA_H

#include "StdAfx.h"
#include "general/ptxd.h"
#include "Cloud.h"

using namespace Cloud_;

class cData
{
    public:

        cData();
        ~cData();

        void addCamera(CamStenope *);
        void centerCloud(Cloud *);

        void addCameras(vector <CamStenope *>);

        int NbCameras(){return m_Cameras.size();}
        int NbClouds(){return m_Clouds.size();}

        CamStenope * & getCamera(int aK) {return m_Cameras[aK];}
        Cloud * & getCloud(int aK) {return m_Clouds[aK];}
        Cloud * & getOriginalCloud(int aK) {return m_oClouds[aK];}

        //Bounding box, center and diameter of all clouds
        double m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_cX, m_cY, m_cZ, m_diam;

   private:

        vector <CamStenope *> m_Cameras;
        vector <Cloud *>    m_Clouds;  //centered and scaled clouds
        vector <Cloud *>    m_oClouds; //original clouds
};

#endif // DATA_H
