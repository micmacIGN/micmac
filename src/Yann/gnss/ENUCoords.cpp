#include <cmath>

#include "ENUCoords.h"


double ENUCoords::distanceTo(ENUCoords enu){

	double de = (this->E - enu.E);
	double dn = (this->N - enu.N);
	double du = (this->U - enu.U);

	return sqrt(de*de + dn*dn + du*du);

}


// ---------------------------------------------------------------
// ENU Cartesian to ECEF coordinates
// ---------------------------------------------------------------
ECEFCoords ENUCoords::toECEFCoords(ECEFCoords base){

	ECEFCoords xyz;

	double slon, clon, slat, clat, blon, blat;
	double e, u, n;

	e = this->E;
	n = this->N;
	u = this->U;

	GeoCoords base_geo = base.toGeoCoords();

	blon = Utils::deg2rad(base_geo.longitude);
	blat = Utils::deg2rad(base_geo.latitude);

	slon = sin(blon);
	slat = sin(blat);
	clon = cos(blon);
	clat = cos(blat);

	xyz.X = -e*slon       - n*clon*slat  +  u*clon*clat    +  base.X;
	xyz.Y =  e*clon       - n*slon*slat  +  u*slon*clat    +  base.Y;
	xyz.Z =                 n*clat       +  u*slat         +  base.Z;

	return xyz;

}


double ENUCoords::elevationTo(ENUCoords p){
	ENUCoords visee(p.E-this->E, p.N-this->N, p.U-this->U);
	return atan2(visee.U, visee.norm2D());
}

double ENUCoords::azimuthTo(ENUCoords p){
	ENUCoords visee(p.E-this->E, p.N-this->N, p.U-this->U);
	return atan2(visee.E, visee.N);
}

double ENUCoords::dot(ENUCoords p){
		return this->E*p.E + this->N*p.N + this->U*p.U;
}

double ENUCoords::norm2D(){
		return sqrt(this->E*this->E + this->N*this->N);
}

double ENUCoords::norm3D(){
	return sqrt(this->dot(*this));
}
