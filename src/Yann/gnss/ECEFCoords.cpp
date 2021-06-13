#include <cmath>

#include "ECEFCoords.h"


double ECEFCoords::distanceTo(ECEFCoords ecef){

	double dx = (this->X - ecef.X);
	double dy = (this->Y - ecef.Y);
	double dz = (this->Z - ecef.Z);

	return sqrt(dx*dx + dy*dy + dz*dz);

}


// ---------------------------------------------------------------
// ECEF Cartesian to Geodetic coordinates
// ---------------------------------------------------------------
GeoCoords ECEFCoords::toGeoCoords(){

	double h, p, t, n, X, Y, Z;

	GeoCoords geo;

	double f = Utils::Fe;
	double a = Utils::Re;
	double b = a*(1-f);
	double e = sqrt(f*(2-f));

	X = this->X;
	Y = this->Y;
	Z = this->Z;

	h = a*a-b*b;
	p = sqrt(X*X + Y*Y);
	t = atan2(Z*a, p*b);

	geo.longitude = atan2(Y,X);
	geo.latitude = atan2(Z+h/b*pow(sin(t),3), p-h/a*pow(cos(t),3));
	n = a/sqrt(1-pow(e*sin(geo.latitude),2));
	geo.height = (p/cos(geo.latitude))-n;

	// Radian to decimal degree conversion
	geo.longitude = Utils::rad2deg(geo.longitude);
	geo.latitude = Utils::rad2deg(geo.latitude);

	return geo;

}

// ---------------------------------------------------------------
// ECEF Cartesian to ENU coordinates
// ---------------------------------------------------------------
ENUCoords ECEFCoords::toENUCoords(ECEFCoords base){

	ENUCoords enu;

	double slon, clon, slat, clat, blon, blat;
	double x, y, z;

	GeoCoords base_geo = base.toGeoCoords();

	blon = Utils::deg2rad(base_geo.longitude);
	blat = Utils::deg2rad(base_geo.latitude);

	x = this->X - base.X;
	y = this->Y - base.Y;
	z = this->Z - base.Z;

	slon = sin(blon);
	slat = sin(blat);
	clon = cos(blon);
	clat = cos(blat);

	enu.E = -x*slon       + y*clon;
	enu.N = -x*clon*slat  - y*slon*slat  +  z*clat;
	enu.U =  x*clon*clat  + y*slon*clat  +  z*slat;

	return enu;

}

double ECEFCoords::elevationTo(ECEFCoords p){
	ENUCoords objectif = p.toENUCoords(*this);
	return atan2(objectif.U, objectif.norm2D());
}

double ECEFCoords::azimuthTo(ECEFCoords p){
	ENUCoords objectif = p.toENUCoords(*this);
	return atan2(objectif.E, objectif.N);
}

double ECEFCoords::dot(ECEFCoords p){
	return this->X*p.X + this->Y*p.Y + this->Z*p.Z;
}

double ECEFCoords::norm(){
	return sqrt(this->dot(*this));
}

void ECEFCoords::scalar(double factor){
    this->X *= factor;
    this->Y *= factor;
    this->Z *= factor;
}

// ---------------------------------------------------------
// Rotation du repère ECEF d'un angle theta (en radian)
// ---------------------------------------------------------
void ECEFCoords::rotate(double theta) {

    double c = cos(theta);
    double s = sin(theta);

    double x = this->X;
    double y = this->Y;

    this->X = +c*x + s*y;
    this->Y = -s*x + c*y;

}

// ---------------------------------------------------------
// Conversion de coordonnées ECEF -> ECI (= ECEF fixé au
// niveau de l'utilisateur à l'instant de la réception)
// ---------------------------------------------------------
void ECEFCoords::shiftECEFRef(double propagation_time) {
    this->rotate(Utils::dOMEGAe*propagation_time);
}

// ---------------------------------------------------------
// Normalisation d'un vecteur (norme L2)
// ---------------------------------------------------------
void ECEFCoords::normalize(){
    this->scalar(1/this->norm());
}

// ---------------------------------------------------------------
// Transformation d'une vitesse des coordonnées ECEF vers les
// les coordonnées planes ENU (par rapport au point ref)
// ---------------------------------------------------------------
ENUCoords ECEFCoords::toENUSpeed(ECEFCoords ref){
    return (ref+*this).toENUCoords(ref);
}
