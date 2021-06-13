#ifndef ARSENIC_H
#define ARSENIC_H
//#include "StdAfx.h"

class GrpVodka
{
public:
	GrpVodka(double diaph, double foc, bool isComputed){this->foc=foc;this->diaph=diaph;this->isComputed=isComputed;}
	~GrpVodka(){}
	double diaph,foc;
	bool isComputed;
	vector<string> aListIm;
	vector<double> ExpTime,ISO; 
	int size(){return (int)this->aListIm.size();}
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
	int size(){return (int)this->parRed.size();}
private:
  
};

class PtsHom
{
	public:
	PtsHom(){}
	~PtsHom(){}
    vector<double> Gr1,Gr2,Dist1,Dist2;
	Pt2di SZ;
    int size(){	return (int)Gr1.size();}
private:
  
};

class cl_PtsRadio
{
	public:
	cl_PtsRadio(){}
	~cl_PtsRadio(){}
    vector<double> kR,kG,kB;
	vector<Pt2dr> Pts;
	vector<int> OtherIm;
	Pt2di SZ;
    int size(){return (int)Pts.size();}

};

class cl_MatPtsHom
{
	public:
		cl_MatPtsHom(){}
		~cl_MatPtsHom(){}
		vector<cl_PtsRadio> aMat;
		int nbTotalPts(){
			int sum=0;
            for(int i=0;i<int(aMat.size());i++)
			{
                sum += aMat[i].size();
			}
			return sum;

		}
};
#endif // ARSENIC_H
