#ifdef __USE_ORIENTATIONMATIS__

#include "cOrientationCon.h"
#include "ign/matis/orientation/modeleprojection.hpp"

OrientationCon::OrientationCon(std::string const &nom):ModuleOrientation(nom)
{
	ign::matis::orientation::ModeleProjection::InitAllIO();
	ori = ign::matis::orientation::ModeleProjection::ReadFile(nom);
	std::string systemGeodesie = ori->GetSystemGeodesie();
	projection=systemGeodesie;
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
	double xLocal,yLocal,z;
	ori->ImageAndZToLocal(c,l,aPx[0],xLocal,yLocal);
	ori->LocalToWorld(xLocal,yLocal,aPx[0],projection.c_str(),x,y,z);
	//std::cout << "OrientationCon::ImageAndPx2Obj "<<c<<" "<<l<<" "<<aPx[0]<<" "<<projection<<" -> "<<x<<" "<<y<<std::endl;
}

void OrientationCon::Objet2ImageInit(double x, double y, const double *aPx,
							 double &c, double &l)const
{
	ori->WorldToImage(projection.c_str(),x,y,aPx[0],c,l);
	//std::cout << "OrientationCon::Objet2ImageInit "<<projection<<" "<<x<<" "<<y<<" "<<aPx[0]<<" -> "<<c<<" "<<l<<std::endl;
}

#endif
