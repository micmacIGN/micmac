
#include "StdAfx.h"
#include "bench.h"


void bench_sqrt_cmplx(Pt2dr aP)
{
     Pt2dr R1,R2;
     RacineCarreesComplexe(aP,R1,R2);
     REAL aD = euclid(aP-R1*R1) +  euclid(aP-R2*R2) ;

     BENCH_ASSERT(aD<epsilon);
}


void bench_eq2(REAL a0,REAL a1,REAL a2)
{
     INT CAS;
     Pt2dr X[2];
     RacinesPolyneDegre2Reel(a0,a1,a2,CAS,X[0],X[1]);

     for (INT k=0 ; k<2 ; k++)
     {
        Pt2dr aRes = (X[k]*Pt2dr(a0,0) + Pt2dr(a1,0))*X[k] + Pt2dr(a2,0);
	REAL aD =euclid(aRes);
	BENCH_ASSERT(aD<epsilon);
     }
}

void bench_eq3(REAL a0,REAL a1,REAL a2,REAL a3)
{
     INT CAS;
     Pt2dr X[3];
     RacinesPolyneDegre3Reel(a0,a1,a2,a3,CAS,X[0],X[1],X[2]);

     for (INT k=0 ; k<3 ; k++)
     {
        Pt2dr aRes = ((X[k]*Pt2dr(a0,0) + Pt2dr(a1,0))*X[k] + Pt2dr(a2,0))*X[k]+Pt2dr(a3,0);
        REAL aD =euclid(aRes);
	// cout << "P3, aD = " << aD << "\n";
        BENCH_ASSERT(aD<epsilon);
     }
}


void bench_eq4(REAL a0,REAL a1,REAL a2,REAL a3,REAL a4)
{
     INT CAS;
     Pt2dr X[4];
     RacinesPolyneDegre4Reel(a0,a1,a2,a3,a4,CAS,X[0],X[1],X[2],X[3]);

     for (INT k=0 ; k<4 ; k++)
     {
        Pt2dr aRes = (((X[k]*Pt2dr(a0,0) + Pt2dr(a1,0))*X[k] + Pt2dr(a2,0))*X[k]+Pt2dr(a3,0))*X[k]+ Pt2dr(a4,0);
        REAL aD =euclid(aRes);
	// cout << "Ad4 = " << aD <<"\n";
        BENCH_ASSERT(aD<1e-4);
     }
}

static REAL Eps() { return (NRrandom3()-0.5) * 1e-5;}

void bench_eq3_from_roots(REAL R1,REAL R2,REAL R3)
{
     ElPolynome<REAL> aP =  ElPolynome<REAL>::FromRoots(R1,R2,R3);
     bench_eq3(aP.at(3),aP.at(2),aP.at(1),aP.at(0));

     aP =  ElPolynome<REAL>::FromRoots(R1) * ElPolynome<REAL>(R3*R3+R2*R2,2*R2,1.0);
     bench_eq3(aP.at(3),aP.at(2),aP.at(1),aP.at(0));

     aP =  ElPolynome<REAL>::FromRoots(R1) * ElPolynome<REAL>(Eps()+R2*R2,2*R2,1.0);
     bench_eq3(aP.at(3),aP.at(2),aP.at(1),aP.at(0));
}

void bench_eq4_from_roots(REAL R1,REAL R2,REAL R3,REAL R4)
{
     ElPolynome<REAL> aP =  ElPolynome<REAL>::FromRoots(R1,R2,R3,R4);
     bench_eq4(aP.at(4),aP.at(3),aP.at(2),aP.at(1),aP.at(0));


     aP =  ElPolynome<REAL>::FromRoots(R1,R2) * ElPolynome<REAL>(R4*R4+R3*R3,2*R3,1.0);
     bench_eq4(aP.at(4),aP.at(3),aP.at(2),aP.at(1),aP.at(0));

     aP =  ElPolynome<REAL>::FromRoots(R1,R2) * ElPolynome<REAL>(Eps()+R3*R3,2*R3,1.0);
      bench_eq4(aP.at(4),aP.at(3),aP.at(2),aP.at(1),aP.at(0));


     aP =  ElPolynome<REAL>(R2*R2+R1*R1,2*R1,1.0) * ElPolynome<REAL>(Eps()+R3*R3,2*R3,1.0);
     bench_eq4(aP.at(4),aP.at(3),aP.at(2),aP.at(1),aP.at(0));

}





void bench_pjeq234()
{
    for (INT aK= 0 ; aK< 10000000 ; aK++)
    {
      if (aK%100==0)
          cout << "--------------  Ak " <<  aK << "\n";

	 REAL a0 = 0.1 + NRrandom3();
	 REAL a1 = ( NRrandom3()-0.5) * 2;
	 REAL a2 = ( NRrandom3()-0.5) * 4;
	 REAL a3 = ( NRrandom3()-0.5) * 8;
	 REAL a4 = ( NRrandom3()-0.5) * 16;

         bench_eq2(a0,a1,a2);
         bench_eq3(a0,a1,a2,a3);
         bench_eq4(a0,a1,a2,a3,a4);
         bench_sqrt_cmplx(Pt2dr::FromPolar(a0*10,a3*100));

	 INT k1 = INT(( NRrandom3()-0.5) * 10);
	 INT k2 = INT(( NRrandom3()-0.5) * 10);
	 INT k3 = INT(( NRrandom3()-0.5) * 10);
	 INT k4 = INT(( NRrandom3()-0.5) * 10);

	 bench_eq3_from_roots(k1,k2,k3);  // Racines entieres
	 bench_eq3_from_roots(k1,k2,k2);  // Racine double
	 bench_eq3_from_roots(k2,k2,k2);  // Racine double

	 bench_eq3_from_roots(k1+Eps(),k2,k3);  // Racines entieres
	 bench_eq3_from_roots(k1+Eps(),k1+Eps(),k2+Eps());  // Racines quasi double
	 bench_eq3_from_roots(k3+Eps(),k3+Eps(),k3+Eps());  // Racines quasi triple


	 bench_eq4_from_roots(k1,k2,k3,k4);  // Racines entieres
	 bench_eq4_from_roots(k1,k2,k3,k3);  // 1 Racines double
	 bench_eq4_from_roots(k1,k1,k3,k3);  // 2 Racines double
	 bench_eq4_from_roots(k1,k3,k3,k3);  // 1 Racines triple

	 bench_eq4_from_roots(k1,k2,k3+Eps(),k3+Eps());  // 1 Racines double
	 bench_eq4_from_roots(k1+Eps(),k1+Eps(),k3,k3);  // 2 Racines double
	 bench_eq4_from_roots(k1,k3+Eps(),k3+Eps(),k3+Eps());  // 1 Racines triple

         bench_eq4(a0,Eps(),a2,Eps(),a4);  // Equation quasi bi-carree
    }
}


