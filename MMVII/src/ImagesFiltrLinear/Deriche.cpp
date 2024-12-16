#include "MMVII_Linear2DFiltering.h"

//===================================================
//===================================================
//===================================================

namespace MMVII
{


class Canny_Deriche_Param
{
    public :
       Canny_Deriche_Param(tREAL8 alpha);
       tREAL8 a,a1,a2,a3,a4;
       tREAL8 b1,b2;

       tREAL8 mAmpl;
};

Canny_Deriche_Param::Canny_Deriche_Param(tREAL8 alpha)
{
   tREAL8 exp_alpha = (tREAL8) exp( (double) - alpha);
   tREAL8 exp_2alpha = exp_alpha * exp_alpha;

   tREAL8 kd = - (1-exp_alpha)*(1-exp_alpha)/exp_alpha;
   tREAL8 ks = (1-exp_alpha)*(1-exp_alpha)/(1+ 2*alpha*exp_alpha - exp_2alpha); 
   
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

template <class TypeOut,class TypeIn>
   void deriche_ll                          // Deriche Low Level interface
   (
      TypeOut  *     IGX,
      TypeOut  *     IGY,
      TypeIn *       IE,
      tINT8          DX,
      tINT8          DY,
      tREAL8         alpha
  )
{
   // TypeOut         *YP, *YM;    
   //int           i, j;

   cPt2di aSz(DX,DY);

   cIm2D<TypeOut>  aImYP(aSz);
   cIm2D<TypeOut>  aImYM(aSz);

   TypeOut * YP = aImYP.DIm().RawDataLin();
   TypeOut * YM = aImYM.DIm().RawDataLin();


   const Canny_Deriche_Param p (alpha);
   
   /***********************************************/
   /***** Derivation selon LIGNE : on fixe y  *****/       
   /***********************************************/
   /**** Filtrage de Gauche a Droite ****/
   for (int i=0; i< DY; i++) {     
      YM[i*DX+0] = 0;  
      YM[i*DX+1] = 0;    
      for (int j=2; j< DX; j++) {        
        YM[i*DX+j] =  p.a * (tREAL8)IE[i*DX+j-1] 
                      + p.b1 * YM[i*DX + j-1] 
                      - p.b2 * YM[i*DX + j-2] ;   
      }
   }                      
   /**** Filtrage de Droite a Gauche ****/
   for (int i=0; i< DY; i++) {     
      YP[i*DX+DX-1] = 0;  
      YP[i*DX+DX-2] = 0;    
      for (int j= DX-3; j >= 0; j--) { 
        YP[i*DX+j] = -p.a *(tREAL8)IE[i*DX+j+1] 
                     + p.b1 * YP[i*DX + j+1] 
                     - p.b2 * YP[i*DX + j+2] ;        
      }
   }
   /**** Derivee selon LIGNE ****/
   for (int i=0; i< DY; i++) {
     for (int j=0; j< DX; j++) {
          IGX[i*DX + j] = YP[i*DX + j] + YM[i*DX + j] ;

     }
   }
   /*******************************************************************/
   /***** Lissage suivant COLONNE --> composante en x du gradient *****/       
   /*******************************************************************/    
   /**** Filtrage de Gauche a Droite ****/         
   for (int j=0; j< DX; j++) {
        YM[0*DX+j]= 0; 
        YM[1*DX+j]= 0;             
        for (int i= 2; i< DY; i++) {             
           YM[i*DX+j] = p.a1 * IGX[i*DX+j] + p.a2 * IGX[(i-1)*DX + j]
                      + p.b1 * YM[(i-1)*DX + j] - p.b2 * YM[(i-2)*DX + j];
        }
   } 
   /**** Filtrage de Droite a Gauche ****/          
   for (int j=0; j< DX; j++) {        
        YP[(DY-1)*DX+j] =0; 
        YP[(DY-2)*DX+j] =0;
        for (int i= DY-3; i>= 0; i--) {       
           YP[i*DX+j] = p.a3 * IGX[(i+1)*DX+j] - p.a4 * IGX[(i+2)*DX+j]
                      + p.b1 * YP[(i+1)*DX+j] - p.b2 * YP[(i+2)*DX+j];
        }
   }    
   /**** Lissage suivant COLONNE ****/   
   for (int i= 0; i< DY; i++) {
     for (int j= 0; j< DX; j++) {
          IGX[i*DX + j] = YP[i*DX + j] + YM[i*DX + j];
     }
   }
    
   /************************************************/
   /***** Derivation selon COLONNE : on fixe x *****/       
   /************************************************/          
   /**** Filtrage de Gauche a Droite ****/   
   for (int j=0; j< DX; j++) {     
      YM[0*DX+j] = 0; 
      YM[1*DX+j] = 0;    
      for (int i=2; i< DY; i++) {          
        YM[i*DX+j] = p.a * (tREAL8)IE[(i-1)*DX+j] 
                     + p.b1 * YM[(i-1)*DX +j] 
                     - p.b2 * YM[(i-2)*DX + j] ;
      }
   }                 
   /**** Filtrage de Droite a Gauche ****/
   for (int j=0; j< DX; j++) {     
      YP[(DY-1)*DX+j] = 0; 
      YP[(DY-2)*DX+j] = 0;    
      for (int i= DY-3; i >= 0; i--) {
        YP[i*DX+j] = -p.a * (tREAL8)IE[(i+1)*DX+j] 
                      + p.b1 * YP[(i+1)*DX + j] 
                      - p.b2 * YP[(i+2)*DX+j] ;
      }
   }
   /**** Derivee selon COLONNE ****/ 
   for (int i=0; i< DY; i++) {
     for (int j=0; j< DX; j++) {
          IGY[i*DX + j] = YP[i*DX + j] + YM[i*DX + j];
     }
   }
   /******************************************************************/
   /* Lissage suivant LIGNE --> composante en y du gradient  *****/       
   /******************************************************************/    
   /**** Filtrage de Gauche a Droite ****/   

   for (int i=0; i< DY; i++) {
        YM[i*DX+0]= 0;  
        YM[i*DX+1]= 0;             
        for (int j= 2; j< DX; j++) {             
           YM[i*DX+j] = p.a1 * IGY[i*DX+j] + p.a2 * IGY[i*DX + j-1]
                      + p.b1 * YM[i*DX + j-1] - p.b2 * YM[i*DX + j-2];
        }
   } 

   /**** Filtrage de Droite a Gauche ****/          
   for (int i=0; i< DY; i++) {        
        YP[i*DX+ DX-1] =0;  
        YP[i*DX+ DX-2] =0;
        for (int j= DX-3; j>= 0; j--) {       
           YP[i*DX+j] = p.a3 * IGY[i*DX+j+1] - p.a4 * IGY[i*DX+j+2]
                      + p.b1 * YP[i*DX+j+1] - p.b2 * YP[i*DX+j+2];
        }
   }
   /**** Lissage suivant LIGNE ****/       
   for (int i= 0; i< DY; i++) {
     for (int j= 0; j< DX; j++) {
          IGY[i*DX + j] = YP[i*DX + j] + YM[i*DX + j];
     }
   }
   // Not in initial code , but in mmv1

   for (int i= 0; i< DY; i++) {
     for (int j= 0; j< DX; j++) {
         IGX[i*DX + j] /= p.mAmpl;
         IGY[i*DX + j] /= p.mAmpl;
     }
   }

}

template<class Type> void ComputeDeriche(cImGrad<Type> & aResGrad,const cDataIm2D<Type> & aImIn,double aAlpha)
{
   aImIn.AssertSameArea(aResGrad.mGx.DIm());
   aImIn.AssertSameArea(aResGrad.mGy.DIm());

   deriche_ll    
   (
      aResGrad.mGx.DIm().RawDataLin(),
      aResGrad.mGy.DIm().RawDataLin(),
      aImIn.RawDataLin(),
      aImIn.Sz().x(),
      aImIn.Sz().y(),
      aAlpha
  );

/*
    auto  aV1In  = cMMV1_Conv<Type>::ImToMMV1(aImIn);
    auto  aV1Gx  = cMMV1_Conv<Type>::ImToMMV1(aResGrad.mGx.DIm());
    auto  aV1Gy  = cMMV1_Conv<Type>::ImToMMV1(aResGrad.mGy.DIm());

    ELISE_COPY
    (
          aV1In.all_pts(),
          deriche(aV1In.in_proj(),aAlpha,10),
          Virgule(aV1Gx.out(),aV1Gy.out())
    );
*/
}


template<class Type> cImGrad<Type> Deriche(const cDataIm2D<Type> & aImIn,double aAlpha)
{
    cImGrad<Type> aResGrad(aImIn.Sz());
    ComputeDeriche(aResGrad,aImIn,aAlpha);
    return aResGrad;
}


template void deriche_ll(tREAL4 *IGX,tREAL4  * IGY,tREAL4 * IE,tINT8 DX,tINT8 DY,tREAL8 alpha);
template cImGrad<tREAL4> Deriche(const cDataIm2D<tREAL4> & aImIn,double aAlpha);
template void ComputeDeriche(cImGrad<tREAL4> & aResGrad,const cDataIm2D<tREAL4> & aImIn,double aAlpha);

};


