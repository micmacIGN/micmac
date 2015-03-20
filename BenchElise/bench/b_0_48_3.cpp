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


class cBenchNappeReal
{
     public :
         cBenchNappeReal() {mV=0;mCpt++;}
         cBenchNappeReal(REAL aV) : mV (aV) {mCpt++;}
         ~cBenchNappeReal() {mCpt--;}

          REAL mV;

         static INT mCpt;
     private :
};

INT cBenchNappeReal::mCpt = 0;

class cBenchNappes
{
      public :


              cBenchNappes(Pt3di aSz);

              void OneBench
                   (
                       Fonc_Num f,
                       Fonc_Num aFZMin,
                       Fonc_Num aFZMax,
                       Pt2di    aSz
                   );

      private :
	      REAL & VCub(INT x,INT y,INT z)
	      {
		    return mDCub[z-mZMin][y][x];
	      }

	      Pt2di            mSzXY;
	      Im2D_INT2        mZSup;
	      INT2 **          mDZSup;
	      Im2D_INT2        mZInf;
	      INT2 **          mDZInf;
	      INT              mZMax;
	      INT              mZMin;
	      Im3D<REAL,REAL>  mCub;
	      REAL ***         mDCub;
	      cTplNape3D<cBenchNappeReal> mNap;
	      cTplNape2D<cBenchNappeReal> mNap2;
              INT              mDef;
};

cBenchNappes::cBenchNappes(Pt3di aSz) :
    mSzXY      (aSz.x,aSz.y),
    mZSup      (aSz.x,aSz.y),
    mDZSup     (mZSup.data()),
    mZInf      (aSz.x,aSz.y),
    mDZInf     (mZInf.data()),
    mZMax      (aSz.z+1),
    mZMin      (-aSz.z),
    mCub       (aSz.x,aSz.y,mZMax-mZMin),
    mDCub      (mCub.data()),
    mNap       (Pt2di(1,1),-1,1),
    mNap2      (1,-1,1),
    mDef       (-1000000)
{
}

void cBenchNappes::OneBench
(
     Fonc_Num f,
     Fonc_Num aFZMin,
     Fonc_Num aFZMax,
     Pt2di    aSz
)
{

    aFZMin = Max(round_ni(aFZMin),mZMin);
    aFZMax = Min(round_ni(aFZMax),mZMax);



     mNap.Resize(aSz,aFZMin,aFZMax);
     mNap2.Resize(aSz.x,aFZMin[Virgule(FX,0)],aFZMax[Virgule(FX,0)]);



    ELISE_COPY ( mCub.all_pts(),mDef,mCub.out());
    ELISE_COPY
    (
        rectangle(Pt2di(0,0),aSz),
        Virgule(aFZMin,aFZMax),
        Virgule(mZInf.out(),mZSup.out())
    ); 


    Fonc_Num ZLoc  = FZ+mZMin;
    Fonc_Num FLoc =  trans(f,Pt3di(0,0,mZMin));
    REAL aSCube,aSCarre;

    Symb_FNum fSom(FLoc * (cos(ZLoc+sin(FX)) + 1.0/(1+FY*FY)));


    ELISE_COPY 
    (
           select
           (
               select(mCub.all_pts(),(aFZMin<=ZLoc)&&(ZLoc<aFZMax)),
               (FX<aSz.x) && (FY<aSz.y)
           ),
           Virgule
           (
               FLoc,
               fSom,
               fSom* (FY==0)
           ),
           Virgule(mCub.out(),sigma(aSCube),sigma(aSCarre))
    );


    for (INT anX =0 ; anX<aSz.x ; anX++)
        for (INT anY =0 ; anY<aSz.y ; anY++)
        {
             INT aZ0 = mDZInf[anY][anX];
             INT aZ1 = mDZSup[anY][anX];
             for (INT aZ=aZ0 ; aZ<aZ1 ; aZ++)
             {
                  REAL aV = VCub(anX,anY,aZ);
                  mNap.El(Pt3di(anX,anY,aZ)) = aV;

                  if (anY==0)
                     mNap2.El(Pt2di(anX,aZ)) = aV;
             }
        }



    REAL aSNap = 0.0;
    REAL aSNap2 = 0.0;
    for (INT anX =-2 ; anX<aSz.x +4; anX++)
        for (INT anY =-3 ; anY<aSz.y +5; anY++)
        {
             for (INT aZ =mZMin-2 ; aZ< mZMax+4; aZ++)
             {
                 Pt3di aP(anX,anY,aZ);
                 if (mNap.Inside(aP))
                 {
                     REAL aV = mNap.El(aP).mV;
                     aSNap += aV * (cos(aZ+sin(double(anX))) + 1.0/(1+ElSquare(anY)));
                 }
                 if (anY==0)
                 {
                     Pt2di aP2(anX,aZ);
                     if (mNap2.Inside(aP2))
                     {
                         REAL aV = mNap2.El(aP2).mV;
                         aSNap2 += aV * (cos(aZ+sin(double(anX))) + 1.0/(1+ElSquare(anY)));
                     }
                 }
             }
        }

     REAL aDif = ElAbs(aSCube-aSNap);
     BENCH_ASSERT(aDif<epsilon);

     aDif = ElAbs(aSCarre-aSNap2);
     BENCH_ASSERT(aDif<epsilon);
}


static Pt2di PRand(INT x0,INT y0,INT szX,INT szY)
{
   return    Pt2di 
             (
                  round_ni(x0+szX*NRrandom3()),
                  round_ni(y0+szY*NRrandom3())
             );

}

static Pt2di PRandNN(INT x0,INT y0,INT szX,INT szY)
{
    Pt2di u = PRand(x0,y0,szX,szY);
    return (((u.x==0) && (u.y==0)) ? PRandNN(x0,y0,szX,szY) : u);
}


void bench_nappes()
{
     for (INT k=0 ; k< 50 ; k++)
     {
          {
             cBenchNappes aB(Pt3di(30,30,20));

             for (INT aT = 0 ; aT < 10 ; aT++)
             {
                 INT aNum = round_down(NRrandom3() * 6);
                 Pt2di aSz = PRand(1,1,20,20);
                 Fonc_Num Zmin1 = - (2 + 10 * (1+cos(FX)));
                 Fonc_Num Zmax1 = Zmin1 + 2 + 5 * (1+sin(FY));
                 switch( aNum)
                 {
                     case 0 :
                         aB.OneBench(FZ,-2-FX/6,2+FY/3,Pt2di(25,10));
                     break;

                     case 1 :
                         aB.OneBench(FX,-2-FX/6,2+FY/3,Pt2di(10,20));
                     break;

                     case 2 :
                         aB.OneBench(FY,-2-FX/6,2+FY/3,aSz);
                     break;


                     case 3 :
                         aB.OneBench(cos(FY)+FX,Zmin1,2+FY/3,aSz);
                     break;

                     case 4 :
                          aB.OneBench(cos(FY)+FX,Zmin1,Zmax1,aSz);
                     break;

                     default :
                         aB.OneBench(FY,-2,2,Pt2di(4,4));
                     break;
                 }
                 aSz.x = aSz.x;
             }
          }
          BENCH_ASSERT(cBenchNappeReal::mCpt ==0);
     }
}



void bench_cLineMapRect()
{
    for (INT k=0 ; k < 100 ; k++)
    {
         cLineMapRect aLMR(PRand(2000,2000,500,500));

              aLMR.Init(Pt2di(1,1),Pt2di(0,0),Pt2di(10,10));

         INT aNbT = (k<5) ? 5 : 20;
         for (INT aT=0 ; aT < aNbT ; aT++)
         {
              Pt2di aP0 = PRand(0,0,5,5);
              Pt2di aP1 = PRand(50,50,50,50);

               if (k < 5) 
                  aP1 = PRand(1000,1000,50,50);

              Pt2di anU = PRandNN(0,0,5,5);

              aLMR.Init(anU,aP0,aP1);

              Pt2dr aUN = Pt2dr(anU) / euclid(anU);

              INT aNb = 0;

              INT Som[4];
              ELISE_COPY
              (
                  rectangle(aP0,aP1),
                  Virgule(1,FX,FY,FX*FY),
                  sigma(Som,4)
              );

              while (const cLineMapRect::tContPts *  aV = aLMR.Next())
              {
                     aNb++;
                     REAL y0=0,y1=0;
                     Pt2di aPrec;
		     // Taille previsible des buffers
                     BENCH_ASSERT((INT)aV->size()<=dist8(aP1-aP0));
                     for (INT aK =0 ; aK<(INT) aV->size() ; aK++)
                     {
                         //Pt2dr aP = (*aV)[aK];
                         Pt2dr aP( (*aV)[aK] ); // __NEW
                         Pt2di aPi = round_ni(aP);
                         REAL y = aP ^ aUN;
                         if (aK == 0)
                         {
                             y0 = y1 = y;
                         }
                         else
                         {
                             y0 = ElMin(y0,y);
                             y1 = ElMax(y1,y);
                             // Prouve que les points sont Freeman chaine
                             BENCH_ASSERT(dist8(aPi-aPrec)==1);
                         }
                         Som[0] -= 1;
                         Som[1] -= aPi.x;
                         Som[2] -= aPi.y;
                         Som[3] -= aPi.x*aPi.y;
                         aPrec  = aPi;
                     }

                  // prouve que les points sont bien a peu pres alignes
                     BENCH_ASSERT(ElAbs(y0-y1) < 1.0);
              }
              // Prouve que les points forment un recouvrt exact du rect
              for (INT k=0 ; k<4 ; k++)
                  BENCH_ASSERT(Som[k] == 0);

              // Prouve que les segment ne sont pas + disperse que necessaire
                  BENCH_ASSERT(aNb<=dist4(aP1-aP0));
         }
    }
}







