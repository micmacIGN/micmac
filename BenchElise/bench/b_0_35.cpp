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


//===================================================
//===================================================
//===================================================


class Canny_Deriche_Param
{
    public :
       Canny_Deriche_Param(REAL alpha);
       REAL a,a1,a2,a3,a4;
       REAL b1,b2;

       REAL mAmpl;
};

Canny_Deriche_Param::Canny_Deriche_Param(REAL alpha)
{
   REAL exp_alpha = (float) exp( (double) - alpha);
   REAL exp_2alpha = exp_alpha * exp_alpha;

   REAL kd = - (1-exp_alpha)*(1-exp_alpha)/exp_alpha;
   REAL ks = (1-exp_alpha)*(1-exp_alpha)/(1+ 2*alpha*exp_alpha - exp_2alpha); 
   
   a = kd * exp_alpha;
   b1 = 2 * exp_alpha;
   b2 = exp_2alpha;

   a1 = ks;  
   a2= ks * exp_alpha * (alpha-1) ;
   a3 = ks * exp_alpha * (alpha+1); 
   a4 = ks * exp_2alpha;


    mAmpl = (2*(b1-a-2*b2))/(1-b1+b2);

}

/* Merci a Canny, Rachid Deriche et Tuan Dang */

void deriche_uc 
(
      REAL4  *     IGX,
      REAL4  *     IGY,
      U_INT1 *     IE,
      INT          DX,
      INT          DY,
      REAL4        alpha
)

{
   float         *YP, *YM;    
   int           i, j, size;

   
   size = DX * DY ;          

   YP = (float *) malloc (size*sizeof(float));
   YM = (float *) malloc (size*sizeof(float));

   const Canny_Deriche_Param p (alpha);
   
   /***********************************************/
   /***** Derivation selon LIGNE : on fixe y  *****/       
   /***********************************************/
   /**** Filtrage de Gauche a Droite ****/
   for (i=0; i< DY; i++) {     
      YM[i*DX+0] = 0;  
      YM[i*DX+1] = 0;    
      for (j=2; j< DX; j++) {        
        YM[i*DX+j] =  p.a * (float)IE[i*DX+j-1] 
                      + p.b1 * YM[i*DX + j-1] 
                      - p.b2 * YM[i*DX + j-2] ;   
      }
   }                      
   /**** Filtrage de Droite a Gauche ****/
   for (i=0; i< DY; i++) {     
      YP[i*DX+DX-1] = 0;  
      YP[i*DX+DX-2] = 0;    
      for (j= DX-3; j >= 0; j--) { 
        YP[i*DX+j] = -p.a *(float)IE[i*DX+j+1] 
                     + p.b1 * YP[i*DX + j+1] 
                     - p.b2 * YP[i*DX + j+2] ;        
      }
   }
   /**** Derivee selon LIGNE ****/
   for (i=0; i< DY; i++) {
     for (j=0; j< DX; j++) {
          IGX[i*DX + j] = YP[i*DX + j] + YM[i*DX + j] ;

     }
   }
   /*******************************************************************/
   /***** Lissage suivant COLONNE --> composante en x du gradient *****/       
   /*******************************************************************/    
   /**** Filtrage de Gauche a Droite ****/         
   for (j=0; j< DX; j++) {
        YM[0*DX+j]= 0; 
        YM[1*DX+j]= 0;             
        for (i= 2; i< DY; i++) {             
           YM[i*DX+j] = p.a1 * IGX[i*DX+j] + p.a2 * IGX[(i-1)*DX + j]
                      + p.b1 * YM[(i-1)*DX + j] - p.b2 * YM[(i-2)*DX + j];
        }
   } 
   /**** Filtrage de Droite a Gauche ****/          
   for (j=0; j< DX; j++) {        
        YP[(DY-1)*DX+j] =0; 
        YP[(DY-2)*DX+j] =0;
        for (i= DY-3; i>= 0; i--) {       
           YP[i*DX+j] = p.a3 * IGX[(i+1)*DX+j] - p.a4 * IGX[(i+2)*DX+j]
                      + p.b1 * YP[(i+1)*DX+j] - p.b2 * YP[(i+2)*DX+j];
        }
   }    
   /**** Lissage suivant COLONNE ****/   
   for (i= 0; i< DY; i++) {
     for (j= 0; j< DX; j++) {
          IGX[i*DX + j] = YP[i*DX + j] + YM[i*DX + j];
     }
   }
    
   /************************************************/
   /***** Derivation selon COLONNE : on fixe x *****/       
   /************************************************/          
   /**** Filtrage de Gauche a Droite ****/   
   for (j=0; j< DX; j++) {     
      YM[0*DX+j] = 0; 
      YM[1*DX+j] = 0;    
      for (i=2; i< DY; i++) {          
        YM[i*DX+j] = p.a * (float)IE[(i-1)*DX+j] 
                     + p.b1 * YM[(i-1)*DX +j] 
                     - p.b2 * YM[(i-2)*DX + j] ;
      }
   }                 
   /**** Filtrage de Droite a Gauche ****/
   for (j=0; j< DX; j++) {     
      YP[(DY-1)*DX+j] = 0; 
      YP[(DY-2)*DX+j] = 0;    
      for (i= DY-3; i >= 0; i--) {
        YP[i*DX+j] = -p.a * (float)IE[(i+1)*DX+j] 
                      + p.b1 * YP[(i+1)*DX + j] 
                      - p.b2 * YP[(i+2)*DX+j] ;
      }
   }
   /**** Derivee selon COLONNE ****/ 
   for (i=0; i< DY; i++) {
     for (j=0; j< DX; j++) {
          IGY[i*DX + j] = YP[i*DX + j] + YM[i*DX + j];
     }
   }
   /******************************************************************/
   /* Lissage suivant LIGNE --> composante en y du gradient  *****/       
   /******************************************************************/    
   /**** Filtrage de Gauche a Droite ****/   

   for (i=0; i< DY; i++) {
        YM[i*DX+0]= 0;  
        YM[i*DX+1]= 0;             
        for (j= 2; j< DX; j++) {             
           YM[i*DX+j] = p.a1 * IGY[i*DX+j] + p.a2 * IGY[i*DX + j-1]
                      + p.b1 * YM[i*DX + j-1] - p.b2 * YM[i*DX + j-2];
        }
   } 

   /**** Filtrage de Droite a Gauche ****/          
   for (i=0; i< DY; i++) {        
        YP[i*DX+ DX-1] =0;  
        YP[i*DX+ DX-2] =0;
        for (j= DX-3; j>= 0; j--) {       
           YP[i*DX+j] = p.a3 * IGY[i*DX+j+1] - p.a4 * IGY[i*DX+j+2]
                      + p.b1 * YP[i*DX+j+1] - p.b2 * YP[i*DX+j+2];
        }
   }
   /**** Lissage suivant LIGNE ****/       
   for (i= 0; i< DY; i++) {
     for (j= 0; j< DX; j++) {
          IGY[i*DX + j] = YP[i*DX + j] + YM[i*DX + j];
     }
   }

   free(YP);
   free(YM);
}


void bench_deriche_gen
     (
          Fonc_Num aF,
          INT szx,
          INT szy,
          REAL alpha,
	  bool  isCste,
	  Pt2dr aRef
     ) 
{
     Im2D_U_INT1 i(szx,szy);
     ELISE_COPY(i.all_pts(),aF,i.out());
     Canny_Deriche_Param aCDP(alpha);

     Im2D_REAL4 gxl(szx,szy,1);
     Im2D_REAL4 gyl(szx,szy,2);

     deriche_uc
     (
          gxl.data_lin(),
          gyl.data_lin(),
          i.data_lin(),
          i.tx(),
          i.ty(),
          alpha
     );

     REAL dx,dy;
     ELISE_COPY
     (
         gxl.interior(20),
         Abs
         (
             Virgule(gxl.in(),gyl.in()) / aCDP.mAmpl
           - deriche(i.in(0),alpha,20)
         ),
         Virgule(VMax(dx),VMax(dy))
     );

     if (isCste)
     {
        std::cout << "GX = " 
		   <<  " " << gxl.data()[50][50] / aCDP.mAmpl
		   <<  " " << gyl.data()[50][10] / aCDP.mAmpl
		   << "\n";
         getchar();
     }

     BENCH_ASSERT((dx<1.0) && (dy< 1.0));
}

void bench_deriche_rand
     (
          INT szx,
          INT szy,
          REAL alpha
     ) 
{
    bench_deriche_gen(Iconv(255*frandr()),szx,szy,alpha,false,Pt2dr(0,0));
}


void bench_deriche() 
{

     bench_deriche_gen(FX,100,100,0.5,true,Pt2dr(1,0));
     bench_deriche_gen(FY,100,100,1.0,true,Pt2dr(1,0));
     // bench_deriche_gen(1.2*FX+2.3*FY,100,100,2.0,true,Pt2dr(1,0));

     bench_deriche_rand(300,400,0.5);
     bench_deriche_rand(300,400,1.0);
     bench_deriche_rand(300,400,2.0);
     printf("OK deriche \n");
}
