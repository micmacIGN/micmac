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

static bool ImCox = false;

static Pt2di SzI(156,166);

static Im2D_U_INT1 Im1()
{
    if (ImCox)
        return Im2D_U_INT1::FromFile("/home/pierrot/Data/Cox/pm-14.pgm");

     Im2D_U_INT1 I(SzI.x,SzI.y);

     static const INT NbV = 4;
     INT  SzV[NbV] =  {1,2,4,8};
     REAL PdsV[NbV] = {1,1,1,1};
     ELISE_COPY
     (
         I.all_pts(),
	 10.0+ unif_noise_4(PdsV,SzV,NbV) * 235.0,
	 I.out()
     );

     return I;

}

static Im2D_REAL4 XParalaxe()
{
       Im2D_REAL4 aRes(SzI.x,SzI.y);

       if (ImCox) return aRes;

       INT NbLab = 5;
       Fonc_Num FLab = Min(NbLab-1,Iconv(frandr() * NbLab));
       for (INT k=0; k< 6 ; k++)
           FLab = label_maj(FLab,NbLab,Box2di(8));

       ELISE_COPY
       (
         aRes.all_pts(),
           (unif_noise_4(8) -0.5) * 18.0
	 + (FLab -(NbLab-1)/2.0) * 0,
	 aRes.out()
       );

       return aRes;
}

static Fonc_Num AUC(Fonc_Num f) {return Max(0,Min(255,f));}
static Im2D_U_INT1 Im2(Im2D_U_INT1 Im1,Im2D_REAL4 XPar)
{
    if (ImCox)
        return Im2D_U_INT1::FromFile("/home/pierrot/Data/Cox/pm-2.pgm");

    Im2D_U_INT1 aRes(SzI.x,SzI.y);
    ELISE_COPY
    (
         aRes.all_pts(),
	 AUC(
	      Im1.in_proj()[Virgule(FX+XPar.in(),FY)]
	      * (0.9 + 0.2 *unif_noise_4(6))
	    + (unif_noise_4(2)-0.5)*20
	 ),
	 aRes.out()
    );
    return aRes;
}

void VisuParalx(Video_Win aW,Fonc_Num f,REAL & aV0,REAL & aV1,bool CalcV)
{
   if (CalcV)	
       ELISE_COPY ( aW.all_pts(),Rconv(f),VMin(aV0) | VMax(aV1));

   ELISE_COPY
   (
        aW.all_pts(),
        AUC(round_ni((f-aV0) * (255.0/(aV1-aV0)))),
       aW.ogray()
   );
}


void Revert(Im2D_U_INT1  anIm)
{ 

     for (INT y=0; y<anIm.ty() ; y++)
         for (INT x1=0,x2 =anIm.tx()-1; x1<x2 ;x1++,x2-- )
	     ElSwap(anIm.data()[y][x1],anIm.data()[y][x2]);
}

// 122397 MPD
// 122444
// SinkFlow=122455    
#define Visu true

void BenchRoy()
{
   REAL aV0,aV1;

   INT ParMin = -25;
   INT ParMax = 25;


   Pt2di aP0 =ImCox ? Pt2di(350,300) :Pt2di(20,20);
   Pt2di aP1 =ImCox ? Pt2di(470,450) :Pt2di(SzI.x-20,SzI.y-20);

   Im2D_U_INT1 aI2 = Im1();
   Im2D_REAL4 XPar = XParalaxe();
   Im2D_U_INT1 aI1 = Im2(aI2,XPar);
   // Revert(aI2); Revert(aI1);

   if (Visu)
   {
      Video_Win  aWParX = Video_Win::WStd(aI1.sz(),1);
      aWParX.set_title("ParX Vraie");
      VisuParalx(aWParX,XPar.in(),aV0,aV1,true);
   }




   Im2D_INT2 aRes =
                    TestCoxRoy
                    (
                          2,
                          aI1,
                          aI2,
                          aP0,
                          aP1,
                          ParMin,
                          ParMax
                    );



   if (Visu)
   {
      Fonc_Num FRes = trans(aRes.in(0),-aP0);
      Video_Win  aWParX = Video_Win::WStd(aI1.sz(),1);
      aWParX.set_title("ParX Calc");
      VisuParalx(aWParX,FRes,aV0,aV1,false);
   }

}


// typedef cV4CoxRoyAlgo cCoxRoyAlgo;

static EliseCorrel2D * TheCorr;


static Pt2di TheP0;

static const REAL TheFact =  100.0;
static const REAL TheCsteRegul = 0.01;
static const REAL TheCoeffRegul = 0.02;


Pt2di   P2(INT x,INT y,INT z,bool im1)
{
   return Pt2di
          (
              x + TheP0.x + (im1 ? 0 : (z)),
              y + TheP0.y
          );
}

INT MakeCapa(REAL aStCapa)
{
    return round_ni(ElMin(1.0,aStCapa)*TheFact);
}



void InitCostCoxRoy ( cInterfaceCoxRoyAlgo &    aCRA)
{

    for (INT anX=aCRA.X0(); anX<aCRA.X1() ; anX++)
        for (INT anY=aCRA.Y0(); anY<aCRA.Y1() ; anY++)
             for (INT aZ = aCRA.ZMin(anX,anY); aZ< aCRA.ZMax(anX,anY) ; aZ++)
             {
                 // cRoyPt aPU (anX,anY,aZ);
                 Pt2di  U1 = P2(anX,anY,aZ,true);
                 Pt2di  U2 = P2(anX,anY,aZ,false);
                 REAL c  = 1.0-TheCorr->Correl(U1,U2);
                 aCRA.SetCostVert(anX,anY,aZ ,MakeCapa(c));
             }
   aCRA.SetStdCostRegul(0.02,1);
}




Im2D_INT2  TestCoxRoy
           (
                INT         aSzV,
                Im2D_U_INT1 imG1,
                Im2D_U_INT1 imG2,
                Pt2di       aP0,
                Pt2di       aP1,
                INT         aParMin,
                INT         aParMax
           )
{
   ElTimer  aRoyTimer;
   TheP0 = aP0;
   Pt2di aSz = aP1 -aP0;
   Im2D_INT2  aRes (aSz.x,aSz.y);

   TheCorr = new EliseCorrel2D
                 (
                     imG1.in(0),
                     imG2.in(0),
                     imG1.sz(),
                     aSzV,
                     false,
                     -1,
		     true,
		     true
                 );

   Im2D_INT2 aIZMin(aSz.x,aSz.y,(INT2)(aParMin));
   Im2D_INT2 aIZMax(aSz.x,aSz.y,(INT2)(aParMax));

   cInterfaceCoxRoyAlgo * TheCRA = cInterfaceCoxRoyAlgo::NewOne
                (
                      aSz.x,
                      aSz.y,
                      aIZMin.data(),
                      aIZMax.data(),
                      false,
                      false
                );
    InitCostCoxRoy(*TheCRA);





   INT MpdFlow=0;
   if (false)
   {
      cInterfaceCoxAlgo * pSCA =     cInterfaceCoxAlgo::StdNewOne
		                     (
			                 aSz,
			                 0,
			                 aParMax-aParMin,
			                 2,
			                 false
			              );
      for (INT anX=aP0.x ; anX<aP1.x ; anX++)
          for (INT anY=aP0.y ; anY<aP1.y ; anY++)
              for (INT aZ=0 ; aZ<(aParMax-aParMin) ; aZ++)
	      {
		  Pt2di aP(anX,anY);
                  REAL aC = TheCorr->Correl(aP,aP+Pt2di(aZ+aParMin,0));
		  pSCA->SetCost(Pt3di(anX-aP0.x,anY-aP0.y,aZ),MakeCapa(1-aC));
	      }
      MpdFlow =  pSCA->PccMaxFlow();
      Im2D_INT2 aSol = pSCA->Sol(0);   
      Video_Win aW =  Video_Win::WStd(aSol.sz(),1.0);
      aW.set_title("MPD");
      INT aV0,aV1;
      ELISE_COPY(aSol.all_pts(),aSol.in(),VMin(aV0)|VMax(aV1));
      ELISE_COPY(aSol.all_pts(),(aSol.in()-aV0)*(255.0/(aV1-aV0)),aW.ogray());
      delete pSCA;
   }
   cout << "MPD Flow " << MpdFlow << "\n";





    aRoyTimer.reinit();
    INT aFl1 = TheCRA->TopMaxFlowStd(aRes.data());
    cout << "Roy Time = " << aRoyTimer.uval() << "\n";


    ELISE_COPY(aRes.all_pts(),aRes.in(),aRes.out());


    {
        INT aV0,aV1;
        ELISE_COPY(aRes.all_pts(),aRes.in(),VMin(aV0)|VMax(aV1));
        cout << "Cox Roy Interv = " << aV0  << " --- " << aV1 << "\n";
    }

    delete TheCRA;

    INT ecartPlani = 2;
    INT ecartAlti = 5;

    ELISE_COPY
    (
          aIZMin.all_pts(),
          rect_min(aRes.in(aParMax),ecartPlani) -ecartAlti,
          aIZMin.out()
    );
    ELISE_COPY
    (
          aIZMax.all_pts(),
          rect_max(aRes.in(aParMin),ecartPlani) +ecartAlti,
          aIZMax.out()
    );


   Im2D_INT2  aRes2 (aSz.x,aSz.y);
   TheCRA = cInterfaceCoxRoyAlgo::NewOne
                (
                      aSz.x,
                      aSz.y,
                      aIZMin.data(),
                      aIZMax.data(),
                      false,
                      false
                );
    InitCostCoxRoy(*TheCRA);

    aRoyTimer.reinit();
    INT aFl2 = TheCRA->TopMaxFlowStd(aRes2.data());
    cout << "Roy Time 2 = " << aRoyTimer.uval() << "\n";

    INT aNbDif;
    ELISE_COPY
    (
         aRes.all_pts(), 
         aRes.in()!=aRes2.in(),
         sigma(aNbDif)
    );
    BENCH_ASSERT(aNbDif==0);

    cout << "FLOWS : " << aFl1 << " " << aFl2 << "\n";
    cout << "Nb Dif = " << aNbDif << "\n";




    delete TheCorr;
    return aRes;
}

