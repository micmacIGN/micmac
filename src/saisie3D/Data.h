#ifndef DATA_H
#define DATA_H

#include "StdAfx.h"
#include "Cloud.h"
#include <QSettings>
#include <QVector>

using namespace Cloud_;

class cData
{
    public:

        cData();
        ~cData();

        void addCamera(CamStenope *);
        void addCloud(Cloud *);
        void centerCloud(Cloud *);

        void addCameras(vector <CamStenope *>);
        void addClouds(vector <Cloud *>);

        int NbCameras(){return m_Cameras.size();}
        int NbClouds(){return m_Clouds.size();}

        CamStenope * & getCamera(int aK) {return m_Cameras[aK];}
        Cloud * & getCloud(int aK) {return m_Clouds[aK];}
        Cloud * & getOriginalCloud(int aK) {return m_oClouds[aK];}

        //Bounding box, center and diameter of all clouds
        double m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_cX, m_cY, m_cZ, m_diam;

   private:

        vector <CamStenope *> m_Cameras;
        vector <Cloud *>      m_Clouds;  //centered and scaled clouds
        vector <Cloud *>      m_oClouds; //original clouds
};

class cSaisieInfos
{
    public:

        cSaisieInfos();
        ~cSaisieInfos();

        //! Selection mode
        enum SELECTION_MODE { INSIDE,
                              OUTSIDE,
                              ADD
        };

        cSaisieInfos(float rotX, float rotY, float translation[3], float scale, QVector <QPoint> polyline, SELECTION_MODE);

    private:
         //Ortho camera infos
         float               m_rotationX;
         float               m_rotationY;
         float*              m_translation;
         float               m_scale;

         //polyline infos
         QVector <QPoint>    m_poly;
         SELECTION_MODE      m_selection_mode;
};

#endif // DATA_H
