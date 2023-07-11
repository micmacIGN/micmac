#include "V1VII.h"
#include "MMVII_Geom2D.h"

/** \file  FluxPtsMMV1.cpp
    \brief file for using MMV1 flux; converting them a vect of point

    MMV1 has a very powerfull and elegant image processing toolbox (maybe I am not 100% objective ;-)
    MMVII will probably not have this kind of library (at leats in first version) as it  quite
    complicated to maintain and understand. By the way,  for many filter as long as I do not know
    exactly what I want, it's much faster to implement them with MMV1. 
*/

namespace MMVII
{

Neighbourhood DiscNeich(const tREAL8 & aRay)
{
    std::vector<Pt2di> aVPts;
    int aNb = round_up(aRay);

    for (int aKx=-aNb ; aKx<=aNb ; aKx++)
    {
        for (int aKy=-aNb ; aKy<=aNb ; aKy++)
        {
            if ((aKx*aKx+aKy*aKy)<= aRay*aRay)
               aVPts.push_back(Pt2di(aKx,aKy));
        }
    }

    return Neighbourhood(&(aVPts[0]),int(aVPts.size()));
}

// typedef std::vector<cPt2di> tResFlux;


void  FluxToV2Points(tResFlux & aRes,Flux_Pts aFlux)
{
    aRes.clear();
    Liste_Pts<int,int> aLV1(2);

    ELISE_COPY (aFlux, 1, aLV1);
    Im2D_INT4 aImV1 = aLV1.image();

    int  aNbPts   = aImV1.tx();
    int * aDataX  = aImV1.data()[0];
    int * aDataY  = aImV1.data()[1];

    for (int aKp=0 ; aKp<aNbPts ; aKp++)
    {
         aRes.push_back(cPt2di(aDataX[aKp],aDataY[aKp]));
    }
}

void  GetPts_Circle(tResFlux & aRes,const cPt2dr & aC,double aRay,bool with8Neigh)
{
     FluxToV2Points(aRes,circle(ToMMV1(aC),aRay,with8Neigh));
}

tResFlux  GetPts_Circle(const cPt2dr & aC,double aRay,bool with8Neigh)
{
     tResFlux aRes;
     GetPts_Circle(aRes,aC,aRay,with8Neigh);

     return aRes;
}

// extern Flux_Pts ellipse(Pt2dr c,REAL A,REAL B,REAL teta,bool v8 = true);
void  GetPts_Ellipse(tResFlux & aRes,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh,tREAL8 aDil)
{
     Flux_Pts aFlux = ellipse(ToMMV1(aC),aRayA,aRayB,aTeta,with8Neigh);
     if (aDil>0)
        aFlux = dilate(aFlux,DiscNeich(aDil));

     FluxToV2Points(aRes,aFlux);
}

void  GetPts_Ellipse(tResFlux & aRes,const cPt2dr & aC,double aRayA,double aRayB, double aTeta,bool with8Neigh)
{
	GetPts_Ellipse(aRes,aC,aRayA,aRayB,aTeta,with8Neigh,-1);
}



void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2,tREAL8 aDil)
{
     Flux_Pts aFlux = line(ToMMV1(ToI(aP1)),ToMMV1(ToI(aP2)));
     if (aDil>0)
        aFlux = dilate(aFlux,DiscNeich(aDil));
     FluxToV2Points(aRes,aFlux);
}

void  GetPts_Line(tResFlux & aRes,const cPt2dr & aP1,const cPt2dr &aP2)
{
	GetPts_Line(aRes,aP1,aP2,-1.0);
}


/*
void Txxxxxxxxx()
{
	Flux_Pts disc(Pt2dr,REAL,bool front = true);

	Neighbourhood
}
*/


};




