#ifdef __USE_ORIENTATIONIGN__

#include <ign/orientation/io/driver/ImageModelReaderCON.h>
#include <ign/geodesy/ProjEngine.h>

#include "cOrientationCon.h"

OrientationCon::OrientationCon(std::string const &nom):ModuleOrientation(nom)
{
    // Pas de gestion des projections (identite)
    ign::geodesy::ProjEngine::SetInstance(new ign::geodesy::ProjEngine(NULL));
    ign::orientation::io::driver::ImageModelReaderCON reader;
    
    try{
        ori.reset((ign::orientation::ImageModelConical*)reader.newFromFile(nom));
    }
    catch (std::exception &e) {
        std::cout << "exception : "<<e.what()<<std::endl;
    }
    IGN_ASSERT(ori.get()!=NULL);
    
}

///
///
///
void OrientationCon::ImageAndPx2Obj(double c, double l, const double *aPx,
							double &x, double &y)const
{
    ign::transform::Vector pt1(c,l,aPx[0]),pt2(3);
    try {
        ori->direct(pt1,pt2);
    } catch (std::exception &e) {
        std::cout << "exception : "<<e.what()<<std::endl;
    }
    x = pt2.x();
    y = pt2.y();
}

///
///
///
void OrientationCon::Objet2ImageInit(double x, double y, const double *aPx,
							 double &c, double &l)const
{
    ign::transform::Vector pt1(x,y,aPx[0]),pt2(3);
    
    try {
        ori->inverse(pt1,pt2);
    }
    catch (std::exception &e) {
        std::cout << "exception : "<<e.what()<<std::endl;
    }
    c = pt2.x();
    l = pt2.y();
}

#endif
