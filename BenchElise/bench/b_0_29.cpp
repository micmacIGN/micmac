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



#define DDIR ELISE_NAME_DATA_DIR "BENCH_ORILIB" DIRECT_NAME

#if (USE_NOYAU)
void bench_orilib()
{
}
#else

class cOptimBenchPaX : public Optim2DParam
{
      public :
            cOptimBenchPaX(CpleEpipolaireCoord & aCple,Pt2dr aPIm1,Pt2dr aPIm2) :
                Optim2DParam(1e-9,1e10,1e-9,false),
                mCple (aCple),
                mPIm1 (aPIm1),
                mPIm2 (aPIm2)
            {
            }

      private :

              REAL Op2DParam_ComputeScore(REAL aDx,REAL aDy)
              {
                    return euclid(mPIm2,mCple.Hom12(mPIm1,Pt2dr(aDx,aDy)));
              }

              CpleEpipolaireCoord & mCple;
              Pt2dr mPIm1;
              Pt2dr mPIm2;
};


void bench_petp_to_terrain
     (
            const std::string&  aNameA,Pt2dr aPA,
            const std::string&  aNameB,Pt2dr aPB
     )
{
     //Data_Ori3D_Std  anOriA(aNameA.c_str(),false,false);
     //Data_Ori3D_Std  anOriB(aNameB.c_str(),false,false);
     Data_Ori3D_Std  anOriA(aNameA.c_str(),false,false,false/*QuickGrid*/); // __NEW
     Data_Ori3D_Std  anOriB(aNameB.c_str(),false,false,false); // __NEW

     CpleEpipolaireCoord * aCple = CpleEpipolaireCoord::OriEpipolaire
                                   (aNameA,aPA,aNameB,aPB,1.0);



     INT aNb = 20;
 
     Im1D_REAL8  X1(aNb);
     Im1D_REAL8  Y1(aNb);
     ELISE_COPY
     (
          X1.all_pts(),
          Virgule(aPA.x+20 *frandr(),aPA.y+20 *frandr()),
          Virgule(X1.out(),Y1.out())
     );


     Im1D_REAL8  ZParX(aNb);
     Im1D_REAL8  TParX(aNb);
     ELISE_COPY
     (
          X1.all_pts(),
          Virgule(frandr()*10,frandr()*1),
          Virgule(ZParX.out(),TParX.out())
     );


     Pt2d<Fonc_Num>  fP1(X1.in(),Y1.in());
     Pt2d<Fonc_Num>  fPParx(ZParX.in(),TParX.in());
     Im1D_REAL8  X2(aNb);
     Im1D_REAL8  Y2(aNb);
     Pt2d<Fonc_Num>  fHom12 = aCple->Hom12(fP1,fPParx);
     ELISE_COPY
     (
          X1.all_pts(),
          Virgule(fHom12.x,fHom12.y),
          Virgule(X2.out(),Y2.out())
     );
     Pt2d<Fonc_Num>  fP2(X2.in(),Y2.in());



     Im1D_REAL8  aXT(aNb);
     Im1D_REAL8  aYT(aNb);
     Im1D_REAL8  aZT(aNb);
     Im1D_REAL8  aD(aNb);


     ELISE_COPY
     (
            X1.all_pts(),
            anOriA.petp_to_carto(fP1,&anOriB,fP2),
            Virgule(aXT.out(),aYT.out(),aZT.out(),aD.out())
     );

     for (INT aK = 0 ; aK<aNb ; aK++)
     {
         Pt3dr aCarto0 (aXT.data()[aK],aYT.data()[aK],aZT.data()[aK]);
         REAL aDist;

         Pt2dr aP1(X1.data()[aK],Y1.data()[aK]);
         Pt2dr aP2(X2.data()[aK],Y2.data()[aK]);
         Pt2dr aParX(ZParX.data()[aK],TParX.data()[aK]);

         Pt3dr aCarto1 = anOriA.to_terrain ( aP1, anOriB, aP2, aDist);
         aCarto1 = anOriA.terr_to_carte(aCarto1);
         
         REAL aD = euclid(aCarto0-aCarto1);
         BENCH_ASSERT(aD<epsilon);

         Pt2dr aQ2 = aCple->Hom12(aP1,aParX);
         aD = euclid(aP2,aQ2);
         BENCH_ASSERT(aD<epsilon);
     }

     delete aCple;
}


void bench_epipole_orilib()
{
     std::string aName317(DDIR "4_317.ori");
     std::string aName318(DDIR "4_318.ori");

     //Data_Ori3D_Std  Ori317(aName317.c_str(),false,false);
     //Data_Ori3D_Std  Ori318(aName318.c_str(),false,false);
     Data_Ori3D_Std  Ori317(aName317.c_str(),false,false,false/*QuickGrid*/); // __NEW
     Data_Ori3D_Std  Ori318(aName318.c_str(),false,false,false); // __NEW



   // Deux Homologues approx
     Pt2dr HomApp317(2026,2430);
     Pt2dr HomApp318(1944,1166);

     bench_petp_to_terrain(aName317,HomApp317,aName318,HomApp318);

     REAL aDist;
     Pt3dr aPter = Ori317.to_terrain(HomApp317,Ori318,HomApp318,aDist);

     Data_Ori3D_Std * Epi317,* Epi318;
     Data_Ori3D_Std::ororient_epipolaires
     (
           Epi317,Ori317,HomApp317,
           Epi318,Ori318,HomApp318
     );



     cOri3D_OneEpip aS317(&Ori317,Epi317);
     cOri3D_OneEpip aS318(&Ori318,Epi318);

     for (INT aK = 0 ; aK < 100; aK++)
     {
          Pt3dr aP = aPter+ Pt3dr(NRrandC(),NRrandC(),NRrandC()) * aK;

          Pt2dr aE317 = Epi317->to_photo(aP);
          Pt2dr aE318 = Epi318->to_photo(aP);
          REAL aDy = ElAbs(aE317.y-aE318.y);

         // Ce sont des systemes epipolaire si les points terrains se
         // projettent au memes y
          BENCH_ASSERT(aDy < epsilon);

          Pt2dr aPh317 = Ori317.to_photo(aP);

          // Verifie le passage des epip a la photo
          REAL aDist317 = euclid(aPh317,aS317.Inverse(aE317));
          BENCH_ASSERT(aDist317 < epsilon);


          // Verifie le passage de la photo aux epip
          aDist317 = euclid(aS317.Direct(aPh317),aE317);
          BENCH_ASSERT(aDist317 < GIGANTESQUE_epsilon);

          REAL aZ = 0.1 + NRrandom3() * 10;

          cOri3D_OneEpip aSZ317 = aS317.MapingChScale(aZ);
          cOri3D_OneEpip aSZ318 = aS318.MapingChScale(aZ);

          Pt2dr aZE317 = aSZ317.Epip()->to_photo(aP);
          Pt2dr aZE318 = aSZ318.Epip()->to_photo(aP);


          // le systeme homotetique, doit etre lui aussi epipolaire
          aDy = ElAbs(aZE317.y-aZE318.y);
          BENCH_ASSERT(aDy < epsilon);

          // est -ce que le systeme est homotetique
          Pt2dr aZPh318 = aSZ318.Phot()->to_photo(aP);
          Pt2dr aPh318 = Ori318.to_photo(aP);
          REAL aDist318 = euclid(aZPh318,aPh318 * aZ);
          BENCH_ASSERT(aDist318 < GIGANTESQUE_epsilon);

          aDist318 = euclid(aZE318,aE318 * aZ);
          BENCH_ASSERT(aDist318 < epsilon);


          aDist318 = euclid(aSZ318.Direct(aZPh318),aZE318)/aZ;
          BENCH_ASSERT(aDist318 < 2*GIGANTESQUE_epsilon);


          aDist318 = euclid(aZPh318,aSZ318.Inverse(aZE318))/aZ;
          BENCH_ASSERT(aDist318 < epsilon);

          if (aK%10==0)
          {
              CpleEpipolaireCoord * aCple = CpleEpipolaireCoord::OriEpipolaire
                                            (
                                               aName317,HomApp317,
                                               aName318,HomApp318,
                                               aZ
                                            );

              Pt2dr aQ1 = aCple->EPI1().Direct(HomApp317*aZ);
              Pt2dr aQ2 = aCple->EPI2().Direct(HomApp318*aZ);

              REAL aDx = ElAbs(aQ1.x-aQ2.x);

              aQ1 = aCple->EPI1().Direct(aPh317*aZ);
              aQ2 = aCple->EPI2().Direct(aPh318*aZ);

             
              aDy = ElAbs(aQ1.y-aQ2.y);
              BENCH_ASSERT(aDx < epsilon);
              BENCH_ASSERT(aDy < GIGANTESQUE_epsilon);


              cOptimBenchPaX  anOpt(*aCple,aPh317*aZ,aPh318*aZ);
              anOpt.optim();
              Pt2dr aPax = anOpt.param();


               Pt2dr aHom = aCple->Hom12(aPh317*aZ,aPax);
               REAL aDist = euclid(aHom,aPh318*aZ);

               BENCH_ASSERT(aDist<epsilon);
               BENCH_ASSERT(ElAbs(aPax.y)<GIGANTESQUE_epsilon);

               delete aCple;
          }
     }




     delete Epi317;
     delete Epi318;

     aS317.SetEpi0();
     aS317.SetPhot0();
     aS318.SetEpi0();
     aS318.SetPhot0();
}



void bench_orilib()
{
     bench_epipole_orilib();

     if (BUG_CPP_Fclose)
     {
        cout << "Pas de becnh Orilib avec ope,close ... \n";
        return;
     }


     Ori3D_Std O3a(DDIR "mep1.ori");
     Ori3D_Std O3b(DDIR "mep2.ori");


     

     Liste_Pts<REAL,REAL> l3(3);
     Liste_Pts<REAL,REAL> l2(2);

     Fonc_Num xyz(Virgule(frandr(),frandr(),frandr())*100);
     ELISE_COPY(rectangle(0,40).chc(xyz),1,l3);
     ELISE_COPY(l3.all_pts().chc(O3a.to_photo(Virgule(FX,FY,FZ))),1,l2);

     Im2D_REAL8 i3 = l3.image();
     Im2D_REAL8 i2 = l2.image();

     REAL * xt = i3.data()[0];
     REAL * yt = i3.data()[1];
     REAL * zt = i3.data()[2];

     REAL * xp = i2.data()[0];
     REAL * yp = i2.data()[1];

     for (INT i = 0; i<i3.tx(); i++)
     {
          Pt3dr ter(xt[i],yt[i],zt[i]);
          Pt2dr ph1(xp[i],yp[i]);
          Pt2dr ph2= O3a.to_photo(ter);

          BENCH_ASSERT( euclid(ph1-ph2)<epsilon);
     }

     {

         // BENCH SUR LES ECRITURES
     
          O3a.write_txt(ELISE_BFI_DATA_DIR "o3a.txt");
          Ori3D_Std O3a_1(ELISE_BFI_DATA_DIR "o3a.txt");

          O3a.write_bin(ELISE_BFI_DATA_DIR "o3a.bin");
          Ori3D_Std O3a_2(ELISE_BFI_DATA_DIR "o3a.bin",false,true);

          Pt2dr qa = O3a.to_photo(Pt3dr(0,0,0));
          Pt2dr qb = O3b.to_photo(Pt3dr(0,0,0));

          REAL d0,d1,d2;
          Pt3dr p0 = O3a.to_terrain(qa,O3b,qb,d0);
          Pt3dr p1 = O3a_1.to_terrain(qa,O3b,qb,d1);
          Pt3dr p2 = O3a_2.to_terrain(qa,O3b,qb,d2);

          BENCH_ASSERT
          (
                   (ElAbs(d0-d1)    < BIG_epsilon)
                && (ElAbs(d1-d2)    < BIG_epsilon)
                && (euclid(p0-p1) < BIG_epsilon)
                && (euclid(p1-p2) < BIG_epsilon)
          );
     }

     {
         // BENCH SUR LES HOMOTHETIE

           REAL zoom = 0.5;

           Pt3dr p1(0,0,0);
           Pt3dr p2(200,100,50);

           Pt2dr q1 = O3a.to_photo(p1);
           Pt2dr q2 = O3a.to_photo(p2);

          Ori3D_Std  O3a2 (O3a,zoom);

          Pt2dr  r1 = O3a2.to_photo(p1);
          Pt2dr  r2 = O3a2.to_photo(p2);

          BENCH_ASSERT
          (
                ( euclid(q1*zoom-r1) < GIGANTESQUE_epsilon)
            &&  ( euclid(q2*zoom-r2) < GIGANTESQUE_epsilon)
          );
     }

     {
         Fonc_Num f = Virgule(FX*10.0,FY*10.0,FX+FY);
         REAL dif[3];
         ELISE_COPY
         (
             // rectangle(Pt2di(-200,-200),Pt2di(200,200)),
             rectangle(Pt2di(2,2),Pt2di(200,200)),
             Abs(f-Virgule(O3a.to_photo(O3a.photo_et_z_to_terrain(f)),f.v2())),
             VMax(dif,3)
         );

         BENCH_ASSERT
         (
                 (dif[0] <1.0)
            &&   (dif[1] < 1.0)
            &&   (dif[2] < epsilon)
         );
     }
     printf("OK coherence Interface Orilib \n");
}
#endif



