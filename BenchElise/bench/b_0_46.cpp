/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



#include "StdAfx.h"
#include "bench.h"
#include <algorithm>

static Pt3dr PProjRand(REAL ProfMin = 1)
{
    return
        Pt3dr
        (
              NRrandom3()-0.5,
              NRrandom3()-0.5,
              NRrandom3() + ProfMin
        );
}

static REAL ProfRand(REAL VMin = 0.1)
{
     return NRrandom3() + VMin;
}                           


void bench_deriv_formal
     (
         Fonc_Num f,
         INT dim,
         REAL delta 
     )
{
    Im2D<REAL,REAL> I(1,dim);
    PtsKD pt(dim);
    for (INT d=0 ; d< dim ; d++)
    {
        REAL v = NRrandom3();
        pt(d) = v;
        I.data()[d][0] = v;
    }
    REAL v;
    ELISE_COPY(to_flux(I),Rconv(f),sigma(v));
    BENCH_ASSERT(ElAbs(f.ValFonc(pt)-v)<epsilon);

	{
    for (INT d =0; d< dim ; d++)
    {
       REAL aDer1 =  f.deriv(d).ValFonc(pt);
       REAL aDer2 =  f.ValDeriv(pt,d);
       BENCH_ASSERT(ElAbs(aDer1-aDer2)<epsilon);
    }
	}

	{
    for (INT d =0; d< dim ; d++)
    {
       PtsKD p1(dim);
       for (INT d1 =0; d1< dim ; d1++)
           p1(d1) = pt(d1);
       p1(d) = pt(d) + delta;
       REAL v1 = f.ValFonc(p1);

       REAL v2 = v;
       REAL fact = 1.0;
       Fonc_Num dk = f;
       REAL puis = 1.0;
       for (INT N = 1; N < 4 ; N++)
       {
           fact /= (REAL) N;
           dk = dk.deriv(d);
           puis *= delta;
           v2 += dk.ValFonc(pt) * puis * fact;
       }
       BENCH_ASSERT(ElAbs(v1-v2)<ElSquare(delta));
    }
	}
}

void bench_deriv_formal()
{
    bench_deriv_formal(FX+0,1,0.1);
    bench_deriv_formal(0+FX+0+FY*0+cos(FZ)*1.0,3,0.1);

    bench_deriv_formal((cos(FX)[FX*FX]),2,0.1);

    bench_deriv_formal(tan(1/(1+FX*FX)),2,0.1);
    bench_deriv_formal(atan(1/(1+FX*FX)),2,0.1);
    bench_deriv_formal(sqrt(1+FX*FX),2,0.1);
    bench_deriv_formal(log(1+FX*FX),2,0.1);
    bench_deriv_formal(exp(FX+cos(FX)),2,0.1);
    bench_deriv_formal
    (
        1/(1+FX)+1/(2+FY) + 1/(4+sin(FZ)+cos(FY)),
        3,
        0.1
    );

    Fonc_Num fElem = cos(FX+sin(FY));
    Fonc_Num FP1 = 1;
    for (INT aK = 0 ; aK < 7 ; aK++)
    {
       double som[3];
       Fonc_Num FP2 = PowI(fElem,aK);
       ELISE_COPY
       (
            rectangle(Pt2di(-10,-10),Pt2di(10,10)),
            Virgule
            (
                Abs(FP1-FP2),
                Abs(FP1.deriv(0)-FP2.deriv(0)),
                Abs(FP1.deriv(1)-FP2.deriv(1))
            ),
            VMax(som,3)
       );
      
       BENCH_ASSERT(som[0]<epsilon);
       BENCH_ASSERT(som[1]<epsilon);
       BENCH_ASSERT(som[2]<epsilon);

       bench_deriv_formal ( FP2, 2, 0.1);
       bench_deriv_formal ( FP1, 2, 0.1);

       FP1 = FP1 * fElem;
    }

    cout << "end formal deriv \n";
}

template <class Type> ElMatrix<Type> RanMat(INT tx,INT ty,Type *,bool Sym=false)
{
     ElMatrix<Type> R(tx,ty);
     for (INT x=0; x<tx; x++)
         for (INT y=0; y<ty; y++)
	 {
             R(x,y) = NRrandom3()-0.5;
	     if (Sym) R(y,x) = R(x,y);
	 }
     return R;
}

void matrix_is_Id(const ElMatrix<REAL>  & M)
{
     BENCH_ASSERT(M.tx() == M.ty());
     INT n = M.tx();
     for (INT k1=0 ;k1<n ; k1++)
         for (INT k2=0 ;k2<n ; k2++)
         {
             REAL vth = (k1==k2);
             BENCH_ASSERT(ElAbs(vth-M(k1,k2))<epsilon);
         }
}
void bench_matrix_Id(INT n)
{
     ElMatrix<REAL>  M(n);
     matrix_is_Id(M);
}

static Pt3dr RanPt3d(REAL scale = 100.9)
{
    return Pt3dr
           ( 
                scale*(NRrandom3()-0.5),
                scale*(NRrandom3()-0.5),
                scale*(NRrandom3()-0.5)
           );
}

void bench_matrix_Rot()
{
     ElMatrix<REAL> M =  ElMatrix<REAL>::Rotation
                         (
                             100 * NRrandom3(),
                             100 * NRrandom3(),
                             100 * NRrandom3()
                         );

     ElMatrix<REAL> I3 = ElMatrix<REAL>(3);

      for (INT k=0 ; k< 10; k++)
      {
          Pt3dr p1 = RanPt3d();
          Pt3dr p2 = RanPt3d();
          Pt3dr q1 = M*p1;
          Pt3dr q2 = M*p2;
          REAL v1 = scal(p1,p2);
          REAL v2 = scal(q1,q2);
          
          BENCH_ASSERT(ElAbs(v1-v2)<epsilon);

          ElMatrix<REAL> M2 = M*4.0+ElMatrix<REAL>(3);
          ElMatrix<REAL> M3 = I3 + M*4.0;
          Pt3dr r1 =  q1*4+p1;
          Pt3dr r2 = M2 * p1;
          Pt3dr r3 = M3 * p1;
          BENCH_ASSERT((euclid(r1-r2)+ euclid(r1-r3))<epsilon);
      }

      matrix_is_Id(M * M.transpose());
      matrix_is_Id( M.transpose()*M);


     ElMatrix<REAL> M2 = M.ColSchmidtOrthog();
     BENCH_ASSERT(M2.L2(M)<epsilon);

     ElMatrix<REAL> M3 = M* 2  +  RanMat(3,3,(REAL *)0)*1e-1;
     M3 = M3.ColSchmidtOrthog();
     M3 = M3 * M3.transpose();

     BENCH_ASSERT(M3.L2(I3)<epsilon);


     ElMatrix<REAL> I5 = ElMatrix<REAL>(5);
     ElMatrix<REAL> M6 = ElMatrix<REAL>(5)*10 + RanMat(5,5,(REAL *)0);
     ElMatrix<REAL> M7 = gaussj(M6);
     ElMatrix<REAL> M8 = M6*M7;
     BENCH_ASSERT(M8.L2(I5)<epsilon);


     {
         Pt3dr u = RanPt3d(1) + Pt3dr(4,0,0);
         Pt3dr v = RanPt3d(1) + Pt3dr(0,4,0);
         Pt3dr w = RanPt3d(1) + Pt3dr(0,0,0);

         Pt3dr ImU = RanPt3d(1);
         Pt3dr ImV = RanPt3d(1);
         Pt3dr ImW = RanPt3d(1);
     
         ElMatrix<REAL>  MIB = MatFromImageBase (u,v,w,ImU,ImV,ImW);
         BENCH_ASSERT(euclid(ImU-MIB*u)<epsilon);
         BENCH_ASSERT(euclid(ImV-MIB*v)<epsilon);
         BENCH_ASSERT(euclid(ImW-MIB*w)<epsilon);
     }
}

void bench_AngFromRot(REAL a,REAL b,REAL c)
{

   ElMatrix<REAL> M =  ElMatrix<REAL>::Rotation(a,b,c);
   REAL a2,b2,c2;
   AngleFromRot(M,a2,b2,c2);
   ElMatrix<REAL> M2 = ElMatrix<REAL>::Rotation(a2,b2,c2);
   
   REAL d = M.L2(M2);
   BENCH_ASSERT(d<1e-4);
   if (ElAbs(sin(b))< 0.7)
   {
       BENCH_ASSERT(d<ElSquare(epsilon));
   }
}

void bench_AngFromRot()
{
    for (INT k=0; k< 2000 ; k++)
    {
       REAL a = 100*(NRrandom3()-0.5);
       REAL b = 100*(NRrandom3()-0.5);
       REAL c = 100*(NRrandom3()-0.5);
       REAL eps = (NRrandom3()-0.5) * 10e-5 * 4;
       bench_AngFromRot(a,b,c);
       bench_AngFromRot(a,PI/2.0+eps,c);
       bench_AngFromRot(a,-PI/2.0+eps,c);
    }
}

static void find_racine(REAL x,REAL y,REAL X[],REAL Y[],INT nb,INT multip = 1)
{
    for (INT k=0; k<nb ; k++)
    {
       if (
                  (ElAbs(X[k]-x) < BIG_epsilon)
              &&  (ElAbs(Y[k]-y) < BIG_epsilon)
          )
          multip --;
       if (multip == 0)
          return;
   }
   BENCH_ASSERT(false);
}


class PolyBench
{
     public :

         PolyBench(INT NbRoot,INT NbNoRoot) :
             mPol(1.0) 
         {
             for (INT k=0; k<NbRoot ; k++)
             {
                 REAL r = 1+2*k+NRrandom3();
                 if (NRrandom3() > 0.5)
                    r = -r;
                 AddRoot(r);
             }
			 {
             for (INT k=0; k<NbNoRoot ; k++)
             {
                  AddNoRoot((NRrandom3()-0.5)*10,0.1+NRrandom3());
             }
			 }
         }
         void AddRoot(REAL x) 
         {
            mRoots.push_back(x);
            ElSTDNS sort(mRoots.begin(),mRoots.end());
            mPol = ElPolynome<REAL>(-x,1) *mPol;
         }

         void AddNoRoot(REAL x,REAL eps) 
         {
           ElPolynome<REAL> px(-x,1);
           ElPolynome<REAL> pe(eps);
            mPol = (px*px+pe) *mPol;
         }
         void ShowVect(const char * name,const ElSTDNS vector<REAL> & v) const
         {
             cout << "[" << name << "] ";
             for (INT k=0; k<(INT)v.size() ; k++)
                 cout << " " << v ;
             cout << "\n";
         }

         void Verif() const
         {
             ElSTDNS vector<REAL> Sol2;
             RealRootsOfRealPolynome(Sol2,mPol,1e-9,100);
             
             BENCH_ASSERT(Sol2.size()==mRoots.size());
             for (INT k=0; k<(INT) Sol2.size() ; k++)
             {
                  REAL dif = ElAbs(Sol2[k]-mRoots[k]);
                  BENCH_ASSERT(dif  < 1e-7);
             }
         }
       
     private :
        ElSTDNS vector<REAL>     mRoots;
        ElPolynome<REAL> mPol;
};


void bench_polynome()
{
     for  (INT k=0 ; k < 10000; k++)
     {
          ElPolynome<REAL> P1(1,2,3);
          BENCH_ASSERT(ElAbs(P1(1)-6)<epsilon);
          BENCH_ASSERT(ElAbs(P1(0)-1)<epsilon);
          BENCH_ASSERT(ElAbs(P1(-1)-2)<epsilon);

          ElPolynome<REAL> P2(3,2,1);
          REAL a = 100 * (NRrandom3()-0.5);
          REAL b = 100 * (NRrandom3()-0.5);
          REAL c = 100 * (NRrandom3()-0.5);
          ElPolynome<REAL> P3(a,b,c);
          ElPolynome<REAL> P4(b+1,c,a);
          ElPolynome<REAL> P5(b+a,c,a-b);

          REAL x = 3*(NRrandom3()-0.5);
          REAL v1 = P1(x)*P2(x)*P3(x)+P4(x)*2-3*P5(x);
          REAL v2 = (P1*P2*P3+P4*2.0-P5*3.0)(x);
          BENCH_ASSERT(ElAbs(v1-v2)<epsilon);
     }


	 {
     for  (INT k=0 ; k < 1000; k++)
     {
        REAL r1 = NRrandom3();
        REAL r2 = 2+NRrandom3();
        REAL r3 = 4+NRrandom3();
        REAL r4 = 6+NRrandom3();

        INT degre = 4; // 4 ou 7
        ElPolynome<REAL>  P =    ElPolynome<REAL> (-r1,1)
                               * ElPolynome<REAL> (-r2,1)
                               * ElPolynome<REAL> (1,0,1) ; // Pas de racine

         if (degre == 7)
            P =    P
                 * ElPolynome<REAL> (-r2,1)
                 * ElPolynome<REAL> (-r3,1)
                 * ElPolynome<REAL> (-r4,1);
        REAL coeff[20];
        for (INT d=0; d <= degre ; d++) 
            coeff[d] = P[d];

        REAL X[20];
        REAL Y[20];
        mueller(degre,coeff,X,Y);

        find_racine(r1,0,X,Y,degre);
        if (degre == 7)
            find_racine(r2,0,X,Y,degre,2);
        else
            find_racine(r2,0,X,Y,degre,1);
        find_racine(0,1,X,Y,degre,1);
        find_racine(0,-1,X,Y,degre,1);
     }
	 }

	 {
     for  (INT k=0 ; k < 200; k++)
     {
        for (INT i=0; i< 6 ; i++)
            for (INT j=0; j< 3 ; j++)
               PolyBench(i,j).Verif();
     }
	 }

}
void ElPhotogram::bench_photogram_0()
{
    Pt3dr A = PProjRand();
    Pt3dr B = PProjRand();
    Pt3dr C = PProjRand();

    REAL La = ProfRand();
    REAL Lb = ProfRand();
    REAL Lc = ProfRand();


    Pt3dr A3 = A * La;
    Pt3dr B3 = B * Lb;
    Pt3dr C3 = C * Lc;


    REAL dA3B3 =  euclid(A3-B3);
    REAL dA3C3 =  euclid(A3-C3);
    REAL dB3C3 =  euclid(B3-C3);
    ElSTDNS list<Pt3dr> L3;
    ProfChampsFromDist ( L3,A,B,C,dA3B3,dA3C3,dB3C3);


    Pt3dr Labc(La,Lb,Lc);
    REAL  EcartL = 1e2;
    REAL  EcartD = 0;                     

    for ( ElSTDNS list<Pt3dr>::iterator it =L3.begin(); it!=L3.end(); it++)
    {
       EcartL = ElMin(EcartL,euclid(Labc-*it));

       Pt3dr A4 = A * it->x;
       Pt3dr B4 = B * it->y;
       Pt3dr C4 = C * it->z;

       REAL EcD =    ElAbs(euclid(A4-B4)-dA3B3)
                  +  ElAbs(euclid(A4-C4)-dA3C3)
                  +  ElAbs(euclid(B4-C4)-dB3C3);
       ElSetMax(EcartD,EcD);
    }


    ELISE_ASSERT(EcartL<1e-3,"ElPhotogram::bench_photogram_0");
    ELISE_ASSERT(EcartD<1e-2,"ElPhotogram::bench_photogram_0");
}
                                  

Pt3dr PrandTeta(REAL teta0)
{
       Pt3dr aRes
       (
            Pt2dr::FromPolar
            (
                 1+3*NRrandom3(),
                 teta0+(NRrandom3()-0.5) *0.2
            ),
            (NRrandom3()-0.5)* 2
       );
	
     if (ElAbs(aRes.z) < 0.001)
	aRes.z = 0.001;
     return  aRes;
}


static REAL RandExp(REAL Fdist,REAL Ray,INT exp)
{
     REAL res =   Fdist * NRrandom3() * pow(1.0/Ray,exp);
     return  res;
}

void BenchOrFromPtsApp()
{
     Pt3dr Tr(NRrandom3()*2,NRrandom3()*2,10+NRrandom3()*2);
     ElMatrix<REAL>  M = ElMatrix<REAL>::Rotation 
                        ((NRrandom3()-0.5)*10, NRrandom3()-0.5, NRrandom3()-0.5);

     //ElRotation3D   Orient ( Tr,M);
     ElRotation3D   Orient ( Tr,M, true/*isDirect*/); // __NEW
     REAL Focale = 1+NRrandom3()*10;
     Pt2dr centre(NRrandom3(),NRrandom3());

     Pt3dr p1 = PrandTeta(0);
     Pt3dr p2 = PrandTeta(PI*2.0/3.0);
     Pt3dr p3 = PrandTeta(PI*4.0/3.0);
     Pt3dr p4 = PrandTeta(PI);


     REAL Fdist = 1e-2;
     REAL Rdist = 5.0;
     Pt2dr CDist(NRrandom3(),NRrandom3());
     REAL c3 = RandExp(Fdist,Rdist,3);
     REAL c5 = RandExp(Fdist,Rdist,5);
     REAL c7 = RandExp(Fdist,Rdist,7);
     ElDistRadiale_Pol357 Dist (1e5, CDist,c3,c5,c7);
  
     //cCamStenopeDistRadPol       Cam0(Focale,centre,Dist);
     //cCamStenopeDistRadPol       CamOr(Focale,centre,Dist);
     cCamStenopeDistRadPol       Cam0( false/*isDistC2M*/, Focale, centre, Dist, vector<double>() );
     cCamStenopeDistRadPol       CamOr( false, Focale, centre, Dist, vector<double>() );
     CamOr.SetOrientation(Orient);

     Pt2dr q1 =  CamOr.R3toF2(p1);
     Pt2dr q2 =  CamOr.R3toF2(p2);
     Pt2dr q3 =  CamOr.R3toF2(p3);
     Pt2dr q4 =  CamOr.R3toF2(p4);


     ElSTDNS list<ElRotation3D>  Lor;
     Cam0.OrientFromPtsAppui(Lor,p1,p2,p3,q1,q2,q3);


     REAL Ecart = 10;
     for (ElSTDNS list<ElRotation3D>::iterator it=Lor.begin(); it!=Lor.end() ; it++)
     {
         CamStenope        Cam2(Cam0,*it);

         REAL d1 = euclid(Cam2.R3toF2(p1),q1); 
         REAL d2 = euclid(Cam2.R3toF2(p2),q2); 
         REAL d3 = euclid(Cam2.R3toF2(p3),q3); 
	 /*
         REAL d4 = euclid(Cam2.R3toF2(p4),q4); 
	 cout << "D4  = " << d4 << "\n";
	 */


         BENCH_ASSERT(d1<1e-2);
         BENCH_ASSERT(d2<1e-2);
         BENCH_ASSERT(d3<1e-2);
         ElSetMin
         (
             Ecart,
                euclid(CamOr.Orient().tr()-it->tr())
             +   CamOr.Orient().Mat().L2(it->Mat())
         );
     }

     ElSTDNS list<Pt3dr> L3; 
     L3.push_back(p1);
     L3.push_back(p2);
     L3.push_back(p3);
     L3.push_back(p4);

     ElSTDNS list<Pt2dr> L2; 
     L2.push_back(q1);
     L2.push_back(q2);
     L2.push_back(q3);
     L2.push_back(q4);

     if (Ecart > epsilon)
     {
         ElSTDNS list<Pt3dr> L3Rob(L3); 
         ElSTDNS list<Pt2dr> L2Rob(L2); 
	 for (INT k=0 ;k<10 ; k++)
	 {
	    REAL rho = 1 + 0.5 * cos(double(k));
	    REAL teta = k* sin(k*k+cos(double(k)));
            Pt3dr aP (Pt2dr::FromPolar(rho,teta),1+cos(1.2678*k));
            aP = PrandTeta(k+0.1*NRrandC());
	    Pt2dr aQ = CamOr.R3toF2(aP);

	    L3Rob.push_back(aP);
	    L2Rob.push_back(aQ);
	 }
	 REAL aDMin;

	 //ElRotation3D aRotRob = Cam0.CombinatoireOFPA(6,L3Rob, L2Rob,&aDMin);
	 ElRotation3D aRotRob = Cam0.CombinatoireOFPA( false/*tousDevant*/, 6, L3Rob, L2Rob, &aDMin ); // __NEW

	 REAL aDROT = 
               euclid(aRotRob.tr()-CamOr.Orient().tr())
           +   aRotRob.Mat().L2(CamOr.Orient().Mat()) ;

	 BENCH_ASSERT(aDMin<epsilon);
	 BENCH_ASSERT(aDROT<epsilon);
     }
     else
     {

        //ElRotation3D Or= Cam0.OrientFromPtsAppui(L3,L2);
        ElRotation3D Or= Cam0.OrientFromPtsAppui( false/*tousDevant*/, L3, L2 ); // __NEW
	REAL aDROT =     euclid(Or.tr()-CamOr.Orient().tr())
                     +   Or.Mat().L2(CamOr.Orient().Mat()) ;

        BENCH_ASSERT ( aDROT < BIG_epsilon);
     }
    
}

// Classe pour verifier que si on merdoie sur l'orientation
// des espaces (terrain ou camera) on ne trouve pas de solution
// au pb des 4 points d'appuis


class TestOrient
{
      public :

         REAL t0();

         virtual Pt2dr  DirD2(Pt2dr) = 0;
         // virtual Pt2dr  InvD2(Pt2dr) = 0;
};

REAL TestOrient::t0()
{
     Pt3dr Tr(NRrandom3()*2,NRrandom3()*2,10+NRrandom3()*2);
     ElMatrix<REAL>  M = ElMatrix<REAL>::Rotation 
                        ((NRrandom3()-0.5)*10, NRrandom3()-0.5, NRrandom3()-0.5);

     //ElRotation3D   Orient ( Tr,M);
     ElRotation3D   Orient ( Tr,M, true/*isDirect*/); // __NEW
     REAL Focale = 1+NRrandom3()*10;
     Pt2dr centre(0,0);

     Pt3dr p1 = PrandTeta(0);
     Pt3dr p2 = PrandTeta(PI*2.0/3.0);
     Pt3dr p3 = PrandTeta(PI*4.0/3.0);
     Pt3dr p4 = PrandTeta(PI);
  
     //CamStenopeIdeale  Cam0(Focale,centre);
     //CamStenopeIdeale  CamSimul(Focale,centre);
     CamStenopeIdeale  Cam0(false/*isC2M*/, Focale, centre, vector<double>()/*ParamAF*/ ); // __NEW
     CamStenopeIdeale  CamSimul(false/*isC2M*/, Focale, centre, vector<double>()/*ParamAF*/ ); // __NEW
     CamSimul.SetOrientation(Orient);

     Pt2dr q1 =  CamSimul.R3toF2(p1);
     Pt2dr q2 =  CamSimul.R3toF2(p2);
     Pt2dr q3 =  CamSimul.R3toF2(p3);
     Pt2dr q4 =  CamSimul.R3toF2(p4);


     ElSTDNS list<Pt3dr> L3; 
     L3.push_back(p1);
     L3.push_back(p2);
     L3.push_back(p3);
     L3.push_back(p4);

     ElSTDNS list<Pt2dr> L2; 
     L2.push_back(DirD2(q1));
     L2.push_back(DirD2(q2));
     L2.push_back(DirD2(q3));
     L2.push_back(DirD2(q4));

     //ElRotation3D Or= Cam0.OrientFromPtsAppui(L3,L2);
     ElRotation3D Or= Cam0.OrientFromPtsAppui( false/*tousDevant*/, L3, L2 ); // __NEW


     CamStenopeIdeale   Cam2(Cam0,Or);

/*
     REAL d1 = euclid(Cam2.R3toF2(p1),DirD2(q1)); 
     REAL d2 = euclid(Cam2.R3toF2(p2),DirD2(q2)); 
     REAL d3 = euclid(Cam2.R3toF2(p3),DirD2(q3)); 
*/
     REAL d4 = euclid(Cam2.R3toF2(p4),DirD2(q4)); 


     return d4;
}

class TestOrientSymD2 : public TestOrient
{
      public :


         Pt2dr  DirD2(Pt2dr p) {return p.conj();}
         // Pt2dr  InvD2(Pt2dr p) = 0;
};


class TestOrientRotD2 : public TestOrient
{
      public :

         TestOrientRotD2 (REAL teta) :
           _rot (Pt2dr::FromPolar(1.0,teta))
         {}

         Pt2dr _rot;

         Pt2dr  DirD2(Pt2dr p) {return p * _rot;}
};


class TestOrientScaleD2 : public TestOrient
{
      public :

         TestOrientScaleD2 (REAL scale) :
           _scale (scale)
         {}

         REAL _scale;

         Pt2dr  DirD2(Pt2dr p) {return p * _scale;}
};


void bench_dist_radiale(Video_Win * aW)
{
    REAL Fact = 10 + 100 * NRrandom3();

    Pt2dr centre(NRrandom3()*Fact,NRrandom3()*Fact);

    REAL aRMax = Fact;
    ElDistRadiale_Pol357 Dist
    (
             aRMax,
             centre,
             NRrandom3()*0.05/3 * pow(Fact,-2),
             NRrandom3()*0.05/5 * pow(Fact,-4),
             NRrandom3()*0.05/7 * pow(Fact,-6)
    );

    ElDistRadiale_PolynImpair anInv = Dist.DistRadialeInverse(aRMax,2);
    ElDistRadiale_PolynImpair anInv2 = Dist.DistRadialeInverse(aRMax,4);
                       

    if (aW)
    {
        aW->clear();
        INT aNb = 40;
        REAL CoefX = aW->sz().x / (2*aRMax);
        REAL CoefY = aW->sz().y / (3 * Dist.DistDirecte(aRMax));
        for (INT aK = 0 ; aK < aNb ; aK++)
        {
              REAL R1 = (2*aRMax) * aK / REAL( aNb);
              REAL R2 = (2*aRMax) * (aK+1) / REAL (aNb);

              aW->draw_seg
              (
                     Pt2dr(R1*CoefX,Dist.DistDirecte(R1)*CoefY),
                     Pt2dr(R2*CoefX,Dist.DistDirecte(R2)*CoefY),
                     aW->pdisc()(P8COL::red)
              );
        }
    }


    REAL Mul = 0.1 + NRrandom3() * 10;
    ElDistRadiale_PolynImpair aDM = Dist.MapingChScale(Mul);

    for (INT k=0 ; k<100 ; k++)
    {
         Pt2dr p0 =  centre + Pt2dr::FromPolar(Fact*NRrandom3() * 0.90,NRrandom3()*300);

         Pt2dr Peps(NRrandom3()-0.5,NRrandom3()-0.5);
         Peps =  Peps * 1e-4 * Fact;

         Pt2dr p1 = p0+ Peps;

         Pt2dr q0 = Dist.Direct(p0);
         Pt2dr q1 = Dist.Direct(p1);

	 Pt2dr uP0 = Dist.Inverse(q0);
	 // Pt2dr vP0 = anInv.Direct(q0);
	 Pt2dr wP0 = anInv2.Direct(q0);

	 // cout << euclid(wP0,p0)   << " " << euclid(uP0,p0)<< "\n";
	 BENCH_ASSERT(euclid(uP0,p0) / Fact <BIG_epsilon);
	 BENCH_ASSERT(euclid(wP0,p0)/ Fact <BIG_epsilon);

	 Pt2dr aPM = aDM.Direct(p0);
	 Pt2dr aQM =  Dist.Direct(p0/Mul) * Mul;

	 // cout << euclid(aPM,aQM) << "\n";
	 BENCH_ASSERT(euclid(aPM,aQM)/Fact <epsilon);

         ElMatrix<REAL> DM(2,2);
         Dist.Diff(DM, (p0+p1)/2.0);

         Pt2dr q2 = q0 + DM * Peps;
   
         BENCH_ASSERT((euclid(q2,q1) / euclid(Peps))<epsilon);
    }
}


void BenchDiffCam()
{
     Pt3dr Tr(NRrandom3()*2,NRrandom3()*2,10+NRrandom3()*2);
     REAL A01 = (NRrandom3()-0.5)*10;
     REAL A02 = (NRrandom3()-0.5);
     REAL A12 = (NRrandom3()-0.5);
     ElMatrix<REAL>  M = ElMatrix<REAL>::Rotation (A01,A02,A12);

     //ElRotation3D   Orient ( Tr,M);
     ElRotation3D   Orient ( Tr,M, true/*isDirect*/); // __NEW

     // Parcequ'il existe une ambiguite sur les angles :
     A01 = Orient.teta01();
     A02 = Orient.teta02();
     A12 = Orient.teta12();


     REAL Focale = 1+NRrandom3()*1e-1;
     Pt2dr CFoc(NRrandom3(),NRrandom3());

    Pt2dr CDist(NRrandom3(),NRrandom3());

    ElDistRadiale_Pol357 Dist
    (
             1e5,
             CDist,
             NRrandom3()*0.1,
             NRrandom3()*0.1,
             NRrandom3()*0.1
    );

    //cCamStenopeDistRadPol  Cam0(Focale,CFoc,Dist);
    cCamStenopeDistRadPol Cam0( false /*isDistC2M*/, Focale, CFoc, Dist, vector<double>() ); // __NEW
    
    
    Cam0.SetOrientation(Orient);



    for (INT k=0; k< 5; k++)
    {
        Pt3dr  p0   (NRrandom3(),NRrandom3(),NRrandom3());
        Pt3dr  Peps (NRrandom3(),NRrandom3(),NRrandom3());
        Peps = Peps * 1e-5;

        Pt3dr p1 = p0 + Peps;
        ElMatrix<REAL>  MD = Cam0.DiffR3F2((p0+p1)/2.0);

        Pt2dr q0 = Cam0.R3toF2(p0);
        Pt2dr q1 = Cam0.R3toF2(p1);
        Pt2dr q2 = q0 + mul32(MD,Peps);

        BENCH_ASSERT((euclid(q1-q2)/euclid(Peps))<epsilon);

        
         REAL epsAng = 1e-5;
         REAL a01 =   NRrandom3()*epsAng ;
         REAL a02 =   NRrandom3()*epsAng ;
         REAL a12 =   NRrandom3()*epsAng ;
         REAL epsTr = 1e-5;
         Pt3dr dtr = Pt3dr(NRrandom3(),NRrandom3(),NRrandom3())*epsTr;

         ElMatrix<REAL>  Ma = ElMatrix<REAL>::Rotation (A01+a01,A02+a02,A12+a12);
         //ElRotation3D   O2 (Tr+dtr,Ma);
         ElRotation3D   O2 (Tr+dtr,Ma, true/*isDirect*/); // __NEW
         CamStenope  Cam2(Cam0,O2);

         ElMatrix<REAL>  MaD = ElMatrix<REAL>::Rotation 
                               (A01+a01/2.0,A02+a02/2.0,A12+a12/2.0);

         //ElRotation3D   O2D (Tr+dtr/2.0,MaD);
         ElRotation3D   O2D (Tr+dtr/2.0,MaD, true/*isDirect*/); // __NEW
         CamStenope     Cam2D(Cam0,O2D);
         ElMatrix<REAL> MDP = Cam2D.DiffR3F2Param(p0);
         ElMatrix<REAL> Param(1,6);
         Param(0,0) = dtr.x;
         Param(0,1) = dtr.y;
         Param(0,2) = dtr.z;
         Param(0,3) = a01;
         Param(0,4) = a02;
         Param(0,5) = a12;
         ElMatrix<REAL>  EstDP = MDP * Param;
         
         Pt2dr r0 = Cam0.R3toF2(p0);
         Pt2dr r1 = Cam2.R3toF2(p0);
         Pt2dr r2 = r0 + Pt2dr(EstDP(0,0),EstDP(0,1));
         REAL delta = (euclid(r1-r2)/epsAng);
         BENCH_ASSERT(delta<epsilon);
    }
}


void bench_FoncMeanSquare()
{
    ElSTDNS list<Fonc_Num> l;

    Fonc_Num Fa = 1.0;
    Fonc_Num Fb = FX;
    Fonc_Num Fc = FY;
    Fonc_Num Fd = Square(FX);
    Fonc_Num Fe = cos(FY);

    l.push_back(Fa);
    l.push_back(Fb);
    l.push_back(Fc);
    l.push_back(Fd);
    l.push_back(Fe);

    REAL a = 100.0;
    REAL b = 2.0;
    REAL c = 4.0;
    REAL d = 5.0;
    REAL e = 150.0;

    Fonc_Num Obs = a*Fa+b*Fb+c*Fc+d*Fd+e*Fe;
    Flux_Pts flx = disc(Pt2dr(0,0),200);

    ElMatrix<REAL8> sols = MatrFoncMeanSquare    
                           (
                               flx,
                               l,
                               Obs+(frandr()-0.5)*0.1,
                               1.0
                           );

    BENCH_ASSERT(ElAbs(a-sols(0,0))<1e-3);
    BENCH_ASSERT(ElAbs(b-sols(0,1))<1e-3);
    BENCH_ASSERT(ElAbs(c-sols(0,2))<1e-3);
    BENCH_ASSERT(ElAbs(d-sols(0,3))<1e-3);
    BENCH_ASSERT(ElAbs(e-sols(0,4))<1e-3);


    Fonc_Num appr = ApproxFoncMeanSquare    
                    (
                         flx,
                         l,
                         Obs+(frandr()-0.5)*0.1,
                         1.0
                    );

    REAL dif;
    ELISE_COPY(flx,Abs(appr-Obs),VMax(dif));
    BENCH_ASSERT(dif<1e-2);
}


INT aRanDegre()
{
    return round_ni(1.1+5*NRrandom3());
}



static Polynome2dReal    Rand_Polynome2dReal(REAL anAmpl,REAL attenNonLin,bool DistY)
{
   INT aDegre =  aRanDegre();

   Polynome2dReal aRes (aDegre,anAmpl);

   for (INT kMon=0 ; kMon<aRes.NbMonome() ; kMon++)
   {
        REAL aCoeff = (0.5+0.5 * NRrandom3()) * anAmpl;
        if (DistY && (kMon == 1))
           aCoeff *= -1;  // Pour Avoir Tjs une mat inversible
        INT aDtot = aRes.KthMonome(kMon).DegreTot();
        if (aDtot ==0) 
             aCoeff *= 0.1;
        if ((aDtot>1) && (attenNonLin < 0.5))
            aCoeff *= attenNonLin/(aRes.NbMonome()*aDtot);
        aRes.SetCoeff(kMon,aCoeff);
   }
   return aRes;
}


void    bench_Polynome2dReal()
{
    for (INT NbTest=0 ; NbTest<1000 ; NbTest++)
    {

        REAL anAmpl = 1.0 + 100 * NRrandom3();
        Polynome2dReal aPol = Rand_Polynome2dReal(anAmpl,1.0,false);

        ElDistortionPolynomiale aDistId (anAmpl,1e-7);


        for (INT NbVal = 0; NbVal < 10 ; NbVal++)
        {
            
            Pt2dr aVal = Pt2dr(anAmpl*NRrandom3(),anAmpl*NRrandom3()) * 0.3;


             // Verifie operations sur les polynomes
            {
                Polynome2dReal    aPol1 = Rand_Polynome2dReal(1.0+100*NRrandom3(),1.0,false);
                Polynome2dReal    aPol2 = Rand_Polynome2dReal(1.0+100*NRrandom3(),1.0,false);
                REAL aScal1 = 100 * (NRrandom3()+1);
                REAL aScal2 = 100 * (NRrandom3()+1);

                Polynome2dReal aPolRes = (aPol1/aScal1 + aPol2*aScal2) -aPol1;

                REAL aV1 = (aPol1(aVal)/aScal1 + aPol2(aVal)*aScal2) -aPol1(aVal);
                REAL aV2 = aPolRes(aVal);
                REAL aDif = ElAbs(aV1-aV2);
                REAL ampl = ElMax(1.0,ElMax(ElAbs(aV1),ElAbs(aV2)));
                aDif /= ampl;

                 BENCH_ASSERT(aDif<epsilon);
            }

            {
                 REAL aDId = euclid(aVal,aDistId.Direct(aVal));
                 BENCH_ASSERT(aDId<epsilon);
            }


            REAL anEps = 1e-5;
            Pt2dr aDeltaX = Pt2dr (anAmpl,0) * anEps;
            Pt2dr aDeltaY = Pt2dr (0,anAmpl) * anEps;

            Pt2dr aGrad = aPol.grad(aVal);
            Pt2dr  aDif = Pt2dr
                          (
                            aPol(aVal+aDeltaX)-aPol(aVal-aDeltaX),
                            aPol(aVal+aDeltaY)-aPol(aVal-aDeltaY)
                          ) / (2*anAmpl*anEps);
            BENCH_ASSERT(euclid(aGrad-aDif) < epsilon);


            Polynome2dReal aPolX = Rand_Polynome2dReal(anAmpl,1e-2,false);
            Polynome2dReal aPolY= Rand_Polynome2dReal(anAmpl,1e-2,true);

            REAL anEpsDist = 1e-10;


           // verifie l'inversion par differenciation

            ElDistortionPolynomiale aDist(aPolX,aPolY,anEpsDist);
            Pt2dr anIm = aDist.Direct(aVal);
            Pt2dr anImInv = aDist.Inverse(anIm);


            REAL anEcart =  euclid(aVal,anImInv);
            BENCH_ASSERT(anEcart <epsilon);

            REAL aChScale = (0.1 + 10 * NRrandom3());
            ElDistortionPolynomiale aDistCHS = aDist.MapingChScale(aChScale);

            Pt2dr v1 = aDistCHS.Direct(aVal*aChScale);
            Pt2dr v2 = aDist.Direct(aVal)*aChScale;
            BENCH_ASSERT(euclid(v1,v2)<epsilon);

           // verifie l'inversion polynome

           if ((NbTest < 100) && (NbVal==0))
           {
               ElDistortionPolynomiale aDInv = aDist.NewPolynLeastSquareInverse
                                       (
                                            anAmpl,
                                            aDist.DistX().DMax()+2
                                       );


               Pt2dr aV1 = aDInv.Direct(aDist.Direct(aVal));
               Pt2dr aV2 = aDist.Direct(aDInv.Direct(aVal));

               BENCH_ASSERT( euclid(aVal-aV1) < 1e-3*anAmpl);
               BENCH_ASSERT( euclid(aVal-aV2) < 1e-3*anAmpl);
            }

        }
    }
}




void    bench_DistPoly()
{
     bench_Polynome2dReal();
}


void bench_matrix_creuse()
{
     for (INT kF =0; kF < 1000 ; kF++)
     {
          INT aNLign = round_ni(1 + NRrandom3() * 20);
          INT aNCol = round_ni(1 + NRrandom3() * 20);

          cElMatCreuseGen * aMCreuse =
		   cElMatCreuseGen::StdNewOne(aNCol,aNLign,false);

	  ElMatrix<REAL> aMat(aNCol,aNLign,0.0);

	  INT aNbElNN = round_ni(aNLign * aNCol * NRrandom3());

	  for (INT kEl =0 ; kEl <aNbElNN ; kEl++)
	  {
	     INT x =  round_ni(aNCol* NRrandom3())%aNCol;
	     INT y =  round_ni(aNLign* NRrandom3())%aNLign;
	     REAL aP = NRrandC() * 10;
	     aMCreuse->AddElem(x,y,aP);
             aMat(x,y) += aP;
	  }
	  Im1D_REAL8 aV(aNCol);
	  ELISE_COPY(aV.all_pts(),frandr()*10,aV.out());

	  ElMatrix<REAL> aM(1,aNCol);
	  for (INT aY =0; aY<aNCol ; aY++)
              aM(0,aY) = aV.data()[aY];

	  Im1D_REAL8 aR1 = aMCreuse->MulVect(aV);
          ElMatrix<REAL> aM2 = aMat * aM;

	  BENCH_ASSERT(aM2.tx()==1);
	  BENCH_ASSERT(aM2.ty()==aR1.tx());

	  for (INT Y=0 ; Y<aM2.ty() ; Y++)
	  {
		REAL aDif = ElAbs(aM2(0,Y)-aR1.data()[Y]);
		// cout << "aDif = " << aDif << "\n";
		BENCH_ASSERT(aDif<epsilon);
	  }
     }
}

void bench_jacobi()
{
    INT N = round_ni(1+20*NRrandom3());
    ElMatrix<REAL> aMatSym = RanMat(N,N,(double *)0,true);

    ElMatrix<REAL> aValP(N,N);
    ElMatrix<REAL> aVecP(N,N);

    jacobi_diag(aMatSym,aValP,aVecP);

    ElMatrix<REAL> aTVec = aVecP.transpose();
    ElMatrix<REAL> anId = aTVec * aVecP;
    matrix_is_Id(anId);

    ElMatrix<REAL> aMVer =  aVecP * aValP * aTVec;
    BENCH_ASSERT(aMVer.L2(aMatSym)<epsilon);

}

void  bench_svdcmp() 
{
    INT N = round_ni(1+20*NRrandom3());
    ElMatrix<REAL> aMat = RanMat(N,N,(double *)0,false);


    ElMatrix<REAL> aU(N,N);
    ElMatrix<REAL> aD(N,N);
    ElMatrix<REAL> aV(N,N);

    bool SetToDirect = (N<4) && (NRrandC() < 0);

    svdcmp_diag   (aMat,aU,aD,aV,SetToDirect);

    ElMatrix<REAL> aTU = aU.transpose();
    ElMatrix<REAL> aTV = aV.transpose();

     matrix_is_Id(aU*aTU);
     matrix_is_Id(aV*aTV);

    ElMatrix<REAL> aMVer =  aU * aD * aV;

    BENCH_ASSERT(aMVer.L2(aMat)<epsilon);
}


static cElHomographie HRAND()
{
    cElComposHomographie aCHX(NRrandC(),NRrandC(),NRrandC());
    cElComposHomographie aCHY(NRrandC(),NRrandC(),NRrandC());
    cElComposHomographie aCH1(NRrandC()/5,NRrandC()/5,1.0);

    return cElHomographie(aCHX,aCHY,aCH1);
}

void BenchHomographie()
{
    cElHomographie aHom1 = HRAND();
    cElHomographie aHomB = HRAND();

    cElHomographie aH1B = aHom1 * aHomB;

    ElPackHomologue aPack;
    INT aNbPts = round_ni(NRrandom3() * 20);
    for (INT K=0 ; K< aNbPts ; K++)
    {
         Pt2dr aP (NRrandC(),NRrandC());

         //aPack.add(ElCplePtsHomologues(aP,aHom1.Direct(aP),1.0));
         aPack.Cple_Add(ElCplePtsHomologues(aP,aHom1.Direct(aP),1.0)); // __NEW
    }
    cElHomographie aHom2(aPack,NRrandC() >0);

    cElHomographie aHI = aHom2.Inverse();

    cDistHomographie aDist(aPack,NRrandC() >0);

    REAL aScale = 0.1 + 10*NRrandom3();
    cDistHomographie aHS = aDist.MapingChScale(aScale);

    if (aNbPts < 4)
       return;

    for 
    (
        ElPackHomologue::const_iterator anIt=aPack.begin();
        anIt != aPack.end() ;
        anIt++
    )
    {
        REAL aDist = euclid(aHom2.Direct(anIt->P1()),anIt->P2());
        BENCH_ASSERT(aDist<BIG_epsilon);
    }



    for (INT K=0 ; K< 10 ; K++)
    {
         Pt2dr aP1 (NRrandC(),NRrandC());
         Pt2dr aQ1 = aHom1.Direct(aP1);
         Pt2dr aQ2 = aHom2.Direct(aP1);
         Pt2dr aQ3 = aDist.Direct(aP1);

         Pt2dr aP2 = aHI.Direct(aQ1);
         Pt2dr aP3 = aDist.Inverse(aQ1);

	 Pt2dr aQS1 =  aHS.Direct(aP1);
	 Pt2dr aQS2 =  aDist.Direct(aP1/aScale) * aScale;
	 Pt2dr aP4 =   aHS.Inverse(aQS1);

         BENCH_ASSERT(euclid(aQ1,aQ2)<BIG_epsilon);
         BENCH_ASSERT(euclid(aP1,aP2)<BIG_epsilon);

         BENCH_ASSERT(euclid(aQ1,aQ3)<BIG_epsilon);
         BENCH_ASSERT(euclid(aP1,aP3)<BIG_epsilon);


         BENCH_ASSERT(euclid(aQS1,aQS2)<BIG_epsilon);
         BENCH_ASSERT(euclid(aP1,aP4)<BIG_epsilon);

          Pt2dr P1B = aH1B.Direct(aP1);
          Pt2dr Q1B = aHom1.Direct(aHomB.Direct(aP1));
          REAL D1B = euclid(P1B,Q1B);

          BENCH_ASSERT(D1B<BIG_epsilon);

    }
}

void bench_trace_det(INT aN)
{
   ElMatrix<REAL> M1 = RanMat(aN,aN,(REAL *)0,false);
   ElMatrix<REAL> M2 = RanMat(aN,aN,(REAL *)0,false);

   ElMatrix<REAL> M1p2= M1*M2;
   ElMatrix<REAL> M1s2= M1+M2;
   ElMatrix<REAL> M1B2 = gaussj(M2)*M1*M2;

   REAL D1   = M1.Det();
   REAL D2   = M2.Det();
   REAL D1p2 = M1p2.Det();
   
   REAL T1  = M1.Trace();
   REAL T2  = M2.Trace();
   REAL T1s2  = M1s2.Trace();
   REAL T1B2  = M1B2.Trace();

   BENCH_ASSERT(ElAbs(D1p2-D1*D2)<epsilon);
   BENCH_ASSERT(ElAbs(T1s2-(T1+T2))<epsilon);
   BENCH_ASSERT(ElAbs(T1B2-(T1))<BIG_epsilon);
}

void bench_matrix()
{


    {
          // Video_Win aW = Video_Win::WStd(Pt2di(500,500),1.0);
          for (INT k=0; k < 200 ; k++)
          {
              bench_dist_radiale(0); // &aW);
          }
   }


    for (INT n=1 ; n<1000 ; n++)
    {
       BenchHomographie();
    }

    for (INT n=1 ; n<1000 ; n++)
    {
         for (INT aD=1 ; aD<4 ; aD++)
             bench_trace_det(aD);
    }

    for (INT n=1 ; n<1000 ; n++)
    {
        bench_svdcmp();
        bench_jacobi();
    }

    for (INT n=1 ; n<1000 ; n++)
    {
        BenchOrFromPtsApp();
    }


    bench_matrix_creuse();

    bench_DistPoly();
    {
        All_Memo_counter MC_INIT;
        stow_memory_counter(MC_INIT);       

        bench_FoncMeanSquare();

        verif_memory_state(MC_INIT);       
    }


	{
     for (INT k=0; k < 500 ; k++)
         BenchDiffCam();
	}


    INT NbTest = 100;

	{
    for (INT k=0; k < NbTest ; k++)
    {
        TestOrientScaleD2 TScaleD2(0.5 + 8*NRrandom3()); 
        TScaleD2.t0();
    }
	}


	{
    for (INT k=0; k < NbTest ; k++)
    {
        TestOrientRotD2 TOrD2(8*NRrandom3()); 
        TOrD2.t0();
    }
	}


	{
    for (INT k=0; k < NbTest ; k++)
    {
        TestOrientSymD2 TSymD2; 
        TSymD2.t0();
    }
	}




	{
    for (INT n=1 ; n<2000 ; n++)
    {
         ElPhotogram::bench_photogram_0();
    }
	}

    {
        for (INT n=1 ; n<20 ; n++)
            bench_matrix_Id(n);
    }
    
	{
    for (INT n=1 ; n<200 ; n++)
        bench_matrix_Rot();
    }
	bench_AngFromRot();


    bench_polynome();
    cout << "end matrix \n";
    
}









