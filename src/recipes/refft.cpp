/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"

// #define DATA(IND) data[IND]
#define DATA(IND) (((IND) & 1) ? Re[(IND)>>1] : Im[((IND)-1)>>1])



void NR_fourn (
		REAL8 * Re,
		REAL8 * Im,
                INT *   nn,
                INT ndim,
                INT isign
              )
{
	for (INT aD =1 ; aD<= ndim ; aD++)
            ELISE_ASSERT
            (
               is_pow_of_2(nn[aD]),
               "Not Pow of 2 Dim in  NR_fourn"
            );

        int i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
        int ibit,idim,k1,k2,n,nprev,nrem,ntot;
        REAL8 tempi,tempr;
        REAL8 theta,wi,wpi,wpr,wr,wtemp;

        ntot=1;
        for (idim=1;idim<=ndim;idim++)
                ntot *= nn[idim];
        nprev=1;
        for (idim=ndim;idim>=1;idim--) {
                n=nn[idim];
                nrem=ntot/(n*nprev);
                ip1=nprev << 1;
                ip2=ip1*n;
                ip3=ip2*nrem;
                i2rev=1;
                for (i2=1;i2<=ip2;i2+=ip1) {
                        if (i2 < i2rev) {
                                for (i1=i2;i1<=i2+ip1-2;i1+=2) {
                                        for (i3=i1;i3<=ip3;i3+=ip2) {
                                                i3rev=i2rev+i3-i2;
                                                ElSwap(DATA(i3),DATA(i3rev));
                                                ElSwap(DATA(i3+1),DATA(i3rev+1));
                                        }
                                }
                        }
                        ibit=ip2 >> 1;
                        while (ibit >= ip1 && i2rev > ibit) {
                                i2rev -= ibit;
                                ibit >>= 1;
                        }
                        i2rev += ibit;
                }
                ifp1=ip1;
                while (ifp1 < ip2) {
                        ifp2=ifp1 << 1;
                        theta=isign*6.28318530717959/(ifp2/ip1);
                        wtemp=sin(0.5*theta);
                        wpr = -2.0*wtemp*wtemp;
                        wpi=sin(theta);
                        wr=1.0;
                        wi=0.0;
                        for (i3=1;i3<=ifp1;i3+=ip1) {
                                for (i1=i3;i1<=i3+ip1-2;i1+=2) {
                                        for (i2=i1;i2<=ip3;i2+=ifp2) {
                                                k1=i2;
                                                k2=k1+ifp1;
                                                tempr=wr*DATA(k2)-wi*DATA(k2+1);
                                                tempi=wr*DATA(k2+1)+wi*DATA(k2);
                                                DATA(k2)=DATA(k1)-tempr;
                                                DATA(k2+1)=DATA(k1+1)-tempi;
                                                DATA(k1) += tempr;
                                                DATA(k1+1) += tempi;
                                        }
                                }
                                wr=(wtemp=wr)*wpr-wi*wpi+wr;
                                wi=wi*wpr+wtemp*wpi+wi;
                        }
                        ifp1=ifp2;
                }
                nprev *= n;
        }
}

void NR_fourn (
	        REAL8 * Re,
		REAL8 * Im,
		Pt2di aSz,
		bool  aDirect
              )

{
	INT nn[3];
	nn[2] = aSz.x;
	nn[1] = aSz.y;
        NR_fourn (Re,Im,nn,2,(aDirect ? 1 : -1));
}





void  ElFFT
      (
            Im2D_REAL8 aReIm,
            Im2D_REAL8 aImIm,
	    bool       aDirect
      )
{
    ELISE_ASSERT(aReIm.sz()==aImIm.sz(), "Dif Dim In El FFT");
    NR_fourn
    (
        aReIm.data_lin(),
	aImIm.data_lin(),
	aReIm.sz(),
	aDirect
    );
}

void  ElFFTCorrelCirc
      (
            Im2D_REAL8 aReIm1,
            Im2D_REAL8 aReIm2
      )
{
    Pt2di aSz = aReIm1.sz();
    ELISE_ASSERT(aSz==aReIm2.sz(),"Diff size in ElFFTCorrel");

    Im2D_REAL8 aImIm1(aSz.x,aSz.y,0.0);
    Im2D_REAL8 aImIm2(aSz.x,aSz.y,0.0);

    ElFFT(aReIm1,aImIm1,true);
    ElFFT(aReIm2,aImIm2,true);

    ELISE_COPY
    (
         aReIm1.all_pts(),
	 mulc
	 (
	      Virgule(aReIm1.in(), aImIm1.in()),
	      Virgule(aReIm2.in(),-aImIm2.in())
	 )/(aSz.x*aSz.y),
	 Virgule(aReIm1.out(),aImIm1.out())
    );

    ElFFT(aReIm1,aImIm1,false);
}

Im2D_REAL8   ElFFTCorrelPadded
             (
                Fonc_Num f1,
                Fonc_Num f2,
                Pt2di    aSzInit
             )
{
      Pt2di aSz 
            (
                   2* Pow_of_2_sup(aSzInit.x),
                   2* Pow_of_2_sup(aSzInit.y)
	    );

    Im2D_REAL8 aPadIm1(aSz.x,aSz.y,0.0);
    Im2D_REAL8 aPadIm2(aSz.x,aSz.y,0.0);

    ELISE_COPY(rectangle(Pt2di(0,0),aSzInit),f1,aPadIm1.out());
    ELISE_COPY(rectangle(Pt2di(0,0),aSzInit),f2,aPadIm2.out());


    ElFFTCorrelCirc(aPadIm1,aPadIm2);
    return aPadIm1;
}

Im2D_REAL8   ElFFTCorrelPadded
             (
                Im2D_REAL8 aReIm1,
                Im2D_REAL8 aReIm2
             )
{
   return ElFFTCorrelPadded(aReIm1.in(),aReIm2.in(),aReIm1.sz());
}


Pt2di DecIm2DecFFT(Im2D_REAL8 anIm,Pt2di aDecIm1)
{
    return Pt2di
           (
               mod(aDecIm1.x,anIm.tx()),
               mod(aDecIm1.y,anIm.ty())
           );
}

Pt2di DecFFT2DecIm(Im2D_REAL8 anIm,Pt2di aP)
{
    Pt2di aSzC  = anIm.sz()/2;

    if (aP.x >= aSzC.x)
        aP.x -=  anIm.tx();
    if (aP.y >= aSzC.y)
        aP.y -=  anIm.ty();

    return aP;
}

Pt2d<Fonc_Num> FN_DecFFT2DecIm(Im2D_REAL8 anIm)
{
    Pt2di aSzC  = anIm.sz()/2;
    
    return Pt2d<Fonc_Num>
	   (
                FX -anIm.tx()*(FX>=aSzC.x),
                FY -anIm.ty()*(FY>=aSzC.y)
	   );
}

REAL ImCorrFromNrFFT(Im2D_REAL8 anIm,Pt2di aDecIm1)
{
   aDecIm1 = DecIm2DecFFT(anIm,aDecIm1);
   return anIm.data()[aDecIm1.y][aDecIm1.x];
}


template <class Type> class ImIntegrale
{
     public :
         ImIntegrale(Fonc_Num aFonc,Pt2di aSz);

         Type SomRect(Pt2di aP0,Pt2di aP1);

     private :
         INT               mTx;
         INT               mTy;
         Im2D<Type,Type>   mIm;
         Type **          mD;
};


template <class Type>  Type ImIntegrale<Type>::SomRect(Pt2di aP0,Pt2di aP1)
{
    pt_set_min_max(aP0,aP1);
    aP0.SetSup(Pt2di(0,0));
    aP1.SetInf(Pt2di(mTx,mTy));

    return   mD[aP1.y][aP1.x]
           + mD[aP0.y][aP0.x]
           - mD[aP1.y][aP0.x]
           - mD[aP0.y][aP1.x];
}




template <class Type> ImIntegrale<Type>::ImIntegrale(Fonc_Num aFonc,Pt2di aSz) :
     mTx    (aSz.x),
     mTy    (aSz.y),
     mIm    (mTx+1,mTy+1,0.0),
     mD     (mIm.data())
{
    ELISE_COPY
    (
        rectangle(Pt2di(1,1),Pt2di(mTx+1,mTy+1)),
        trans(aFonc,Pt2di(-1,-1)),
        mIm.out()
    );

    for (INT y= 0 ; y<=mTy ; y++)
    {
         for (INT x= 1 ; x<=mTx ; x++)
             mD[y][x] += mD[y][x-1];
         if (y)
         {
            for (INT x= 0 ; x<=mTx ; x++)
               mD[y][x] += mD[y-1][x];
         }
    }
}


static REAL CorrectSurfMin(REAL aCoef,REAL aSurf,REAL aSurfMin)
{
    if ((aSurfMin<= 0) ||  (aSurf>= aSurfMin))
       return aCoef;

    REAL aPCOeff = aSurf / aSurfMin;
    REAL aPM1    = 1.0 - aPCOeff;

   return aPCOeff * aCoef + aPM1 * -1.0;
}

Im2D_REAL8   ElFFTCorrelNCPadded
             (
                Im2D_REAL8 aReIm1,
                Im2D_REAL8 aReIm2,
                REAL       anEps,
                REAL       aSurfMin
             )
{
   Pt2di aSz = aReIm1.sz();
   Im2D_REAL8 aCor = ElFFTCorrelPadded(aReIm1,aReIm2);

   ImIntegrale<REAL> aSom(1.0,aSz);
   ImIntegrale<REAL> aSom1(aReIm1.in(),aSz);
   ImIntegrale<REAL> aSom2(aReIm2.in(),aSz);
   ImIntegrale<REAL> aSom11(aReIm1.in()*aReIm1.in(),aSz);
   ImIntegrale<REAL> aSom22(aReIm2.in()*aReIm2.in(),aSz);

    for (INT x=  -(aSz.x-1) ; x<= aSz.x ; x++)
    {
        for (INT y=  -(aSz.y-1) ; y<= aSz.y ; y++)
        {


              Pt2di aP(x,y);

              REAL aS  = aSom.SomRect(aP,aSz+aP);
              REAL aS1 = aSom1.SomRect(aP,aSz+aP);
              REAL aS11 = aSom11.SomRect(aP,aSz+aP);
              REAL aS2 = aSom2.SomRect(-aP,aSz-aP);
              REAL aS22 = aSom22.SomRect(-aP,aSz-aP);

              REAL aS12 = ImCorrFromNrFFT(aCor,aP);

              aS = ElMax(aS,anEps);

              aS1 /= aS;
              aS2/= aS;

              aS11 =  aS11/aS - aS1*aS1;
              aS12 =  aS12/aS - aS1*aS2;
              aS22 =  aS22/aS - aS2*aS2;

              REAL aRes = aS12/sqrt(ElMax(anEps,aS11*aS22));

              aRes = CorrectSurfMin(aRes,aS,aSurfMin);

              Pt2di aDecF  = DecIm2DecFFT(aCor,aP);
              aCor.data()[aDecF.y][aDecF.x] = aRes;
        }
    }

   return aCor;
}


Im2D_REAL8   ElFFTPonderedCorrelNCPadded
             (
                Fonc_Num   aF1,
                Fonc_Num   aF2,
                Pt2di      aSz,
                Fonc_Num   aPds1,
                Fonc_Num   aPds2,
                REAL       anEps,
                REAL       aSurfMin
             )
{

   Im2D_REAL8 anImS12 = ElFFTCorrelPadded(      aF1*aPds1,      aF2*aPds2,   aSz);
   Im2D_REAL8 anImS11 = ElFFTCorrelPadded(  aF1*aF1*aPds1,          aPds2,   aSz);
   Im2D_REAL8 anImS22 = ElFFTCorrelPadded(          aPds1,  aF2*aF2*aPds2,   aSz);
   Im2D_REAL8 anImS1  = ElFFTCorrelPadded(      aF1*aPds1,          aPds2,   aSz);
   Im2D_REAL8 anImS2  = ElFFTCorrelPadded(          aPds1,      aF2*aPds2,   aSz);
   Im2D_REAL8 anImS   = ElFFTCorrelPadded(          aPds1,          aPds2,   aSz);

   Im2D_REAL8 anImCorrel(anImS12.tx(),anImS12.ty(),-10);


    for (INT x=  -(aSz.x-1) ; x<= aSz.x ; x++)
    {
        for (INT y=  -(aSz.y-1) ; y<= aSz.y ; y++)
        {

              Pt2di aP(x,y);

              REAL aS12 = ImCorrFromNrFFT(anImS12,aP);
              REAL aS11 = ImCorrFromNrFFT(anImS11,aP);
              REAL aS22 = ImCorrFromNrFFT(anImS22,aP);
              REAL aS1  = ImCorrFromNrFFT(anImS1,aP);
              REAL aS2  = ImCorrFromNrFFT(anImS2,aP);
              REAL aS   = ImCorrFromNrFFT(anImS,aP);


              aS = ElMax(aS,anEps);

              aS1 /= aS;
              aS2/= aS;

              aS11 =  aS11/aS - aS1*aS1;
              aS12 =  aS12/aS - aS1*aS2;
              aS22 =  aS22/aS - aS2*aS2;

              REAL aCoefCor = aS12/sqrt(ElMax(anEps,aS11*aS22));

              REAL aRes = CorrectSurfMin(aCoefCor,aS,aSurfMin);

              Pt2di aDecF  = DecIm2DecFFT(anImS12,aP);
              anImCorrel.data()[aDecF.y][aDecF.x] = aRes;
        }
    }
    return anImCorrel;
}








/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
