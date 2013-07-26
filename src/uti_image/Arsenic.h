#ifndef ARSENIC_H
#define ARSENIC_H
//#include "StdAfx.h"

class DiaphFoc
{
public:
	DiaphFoc(){}
	~DiaphFoc(){}
	double diaph,foc;
	bool isComputed;

};

class ArsenicImage
{
public:
	ArsenicImage(){}
	~ArsenicImage(){}
	Im2D_REAL4 RChan;
	Im2D_REAL4 GChan;
	Im2D_REAL4 BChan;
	Im2D_INT1 Mask;
	cElNuage3DMaille* info3D;
	Pt2di SZ;
};

class Param3Chan
{
public:
	Param3Chan(){}
	~Param3Chan(){}
    vector<double> parRed, parBlue, parGreen;
	int size(){return this->parRed.size();};
private:
  
};

class PtsRadioTie
{
	public:
	PtsRadioTie(){}
	~PtsRadioTie(){}
	vector<double> kR;
	vector<double> kG;
	vector<double> kB;
	vector<Pt2dr> Pos;
	int size(){	return this->Pos.size();};
private:
  
};

class PtsHom
{
	public:
	PtsHom(){this->NbPtsCouple=0;}
	~PtsHom(){}
    vector<double> Gr1,Gr2,R1,G1,B1,R2,G2,B2,Dist1,Dist2;
	vector<Pt2dr> Pt1,Pt2;
	int NbPtsCouple;
	Pt2di SZ;
	int size(){	return this->Gr1.size();};
private:
  
};


#endif // ARSENIC_H
