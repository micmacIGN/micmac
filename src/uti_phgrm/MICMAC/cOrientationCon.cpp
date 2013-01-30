#ifdef __USE_ORIENTATIONIGN__

#include <ign/orientation/io/driver/ImageModelReaderCON.h>
#include <ign/geodesy/ProjEngine.h>

#include "cOrientationCon.h"
//#include "ign/matis/orientation/modeleprojection.hpp"

OrientationCon::OrientationCon(std::string const &nom):ModuleOrientation(nom)
{
    // Pas de gestion des projections (identite)
    //if (ign::geodesy::ProjEngine::Instance())
    //ign:: geodesy::SystemCoordProjection* srs=ign::transform::SystemRegistry::Create<ign::geodesy::SystemCoordProjection>("LAMBERT93", *ign::numeric::unit::SysUnitRegistry::Instance().getSystemById(ign::numeric::unit::kUndefined));
    ign::geodesy::ProjEngine::SetInstance(new ign::geodesy::ProjEngine(NULL));
    ign::orientation::io::driver::ImageModelReaderCON reader;
    
    try{
        ori.reset((ign::orientation::ImageModelConical*)reader.newFromFile(nom));
    }
    catch (std::exception &e) {
        std::cout << "exception : "<<e.what()<<std::endl;
    }
    IGN_ASSERT(ori.get()!=NULL);
    
    
    /*    
	ign::matis::orientation::ModeleProjection::InitAllIO();
	ori = ign::matis::orientation::ModeleProjection::ReadFile(nom);
	std::string systemGeodesie = ori->GetSystemGeodesie();
	projection=systemGeodesie;
 */
	/*
	std::string initStr("+init=");
	int pos=-1;
	if (systemGeodesie.size()>initStr.size())
	{
		for(int i=0;(i<((int)systemGeodesie.size()-(int)initStr.size()))&&(pos==-1);++i)
		{
			std::string ext(systemGeodesie,i,initStr.size());
			if (ext==initStr)
				pos=i;
		}
	}
	if (pos==-1) 
		projection = std::string("TERRAIN");
	else
	{
		std::cout << "pos : "<<pos<<std::endl;
		projection.assign(systemGeodesie.begin()+pos,systemGeodesie.end());
	}
	*/
	//std::cout << "CONIQUE == "<<projection<<std::endl;
}

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
/*    
	double xLocal,yLocal,z;
	ori->ImageAndZToLocal(c,l,aPx[0],xLocal,yLocal);
	ori->LocalToWorld(xLocal,yLocal,aPx[0],projection.c_str(),x,y,z);
*/
 //std::cout << "OrientationCon::ImageAndPx2Obj "<<c<<" "<<l<<" "<<aPx[0]<<" "<<projection<<" -> "<<x<<" "<<y<<std::endl;
}

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
    //std::cout << "Verfication inverse : "<<x<<" "<<y<<" "<<pt1.z()<<" -> "<<c<<" "<<l<<" "<<pt2.z()<<std::endl;
	//ori->WorldToImage(projection.c_str(),x,y,aPx[0],c,l);
	//std::cout << "OrientationCon::Objet2ImageInit "<<projection<<" "<<x<<" "<<y<<" "<<aPx[0]<<" -> "<<c<<" "<<l<<std::endl;
}

#endif
