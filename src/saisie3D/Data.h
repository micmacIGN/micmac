#ifndef DATA_H
#define DATA_H

#include "StdAfx.h"
#include "general/ptxd.h"
#include "private/cElNuage3DMaille.h"
#include "Cloud.h"

using namespace Cloud_;

class cData
{
    public:

        cData();
        ~cData();

        void addCamera(cElNuage3DMaille *);
        void centerCloud(Cloud *);

        void addCameras(vector <cElNuage3DMaille *>);

        int NbCameras(){return m_Cameras.size();}
        int NbClouds(){return m_Clouds.size();}

        cElNuage3DMaille * & getCamera(int aK) {return m_Cameras[aK];}
        Cloud * & getCloud(int aK) {return m_Clouds[aK];}

        //Bounding box, center and diameter of all clouds
        double m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_cX, m_cY, m_cZ, m_diam;

   private:

        vector <cElNuage3DMaille *> m_Cameras;
        vector <Cloud *> m_Clouds;
};

#endif // DATA_H
