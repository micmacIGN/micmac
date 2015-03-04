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

// NROptF1vND : Num Recipes Optimisation de Fonction d'1 var, Non Derivable

/******************************************************/
/*                                                    */
/*                NROptF1vND                         */
/*                                                    */
/******************************************************/


NROptF1vND::~NROptF1vND() {}

#define GOLD 1.618034
#define GLIMIT 100.0
#define TINY 1.0e-20
#define NRMAX(a,b) ((a) > (b) ? (a) : (b))
#define SIGN(a,b) ((b) > 0.0 ? fabs(a) : -fabs(a))
#define SHFT(a,b,c,d) (a)=(b);(b)=(c);(c)=(d);     

#define R 0.61803399
#define C (1.0-R)  
#define CGOLD 0.3819660
#define ZEPS 1.0e-10
#define MOV3(a,b,c, d,e,f) (a)=(d);(b)=(e);(c)=(f);


     //++++++++++++++++++++++++++++++++
     //     Bracketting
     //++++++++++++++++++++++++++++++++

NROptF1vND::NROptF1vND(int aNbIterMax) :
    mNbIterMax (aNbIterMax),
    TOL        (1e-6)
{
}

void NROptF1vND::mnbrack
     ( 
           REAL *ax,REAL *bx,REAL *cx,
           REAL *fa,REAL * fb,REAL *fc
     )        
{
        REAL ulim,u,r,q,fu,dum;

        *fa=NRF1v(*ax);
        *fb=NRF1v(*bx);
        if (*fb > *fa) {
                SHFT(dum,*ax,*bx,dum)
                SHFT(dum,*fb,*fa,dum)
        }
        *cx=(*bx)+GOLD*(*bx-*ax);
        *fc=NRF1v(*cx);
        while (*fb > *fc) {
                r=(*bx-*ax)*(*fb-*fc);
                q=(*bx-*cx)*(*fb-*fa);
                u=(*bx)-((*bx-*cx)*q-(*bx-*ax)*r)/
                        (2.0*SIGN(NRMAX(fabs(q-r),TINY),q-r));
                ulim=(*bx)+GLIMIT*(*cx-*bx);
                if ((*bx-u)*(u-*cx) > 0.0) {
                        fu=NRF1v(u);
                        if (fu < *fc) {
                                *ax=(*bx);
                                *bx=u;
                                *fa=(*fb);
                                *fb=fu;
                                return;
                        } else if (fu > *fb) {
                                *cx=u;
                                *fc=fu;
                                return;
                        }
                        u=(*cx)+GOLD*(*cx-*bx);
                        fu=NRF1v(u);                       
                } else if ((*cx-u)*(u-ulim) > 0.0) {
                        fu=NRF1v(u);
                        if (fu < *fc) {
                                SHFT(*bx,*cx,u,*cx+GOLD*(*cx-*bx))
                                SHFT(*fb,*fc,fu,NRF1v(u))
                        }
                } else if ((u-ulim)*(ulim-*cx) >= 0.0) {
                        u=ulim;
                        fu=NRF1v(u);
                } else {
                        u=(*cx)+GOLD*(*cx-*bx);
                        fu=NRF1v(u);
                }
                SHFT(*ax,*bx,*cx,u)
                SHFT(*fa,*fb,*fc,fu)
        }
}

bool  NROptF1vND::NROptF1vContinue() const
{
    return   (fabs(x3-x0) > mTolGolden*(fabs(x1)+fabs(x2)))
            && ((mNbIterMax<=0) || (mNbIter<mNbIterMax))
           ;
}

REAL NROptF1vND::golden(REAL ax,REAL bx,REAL cx,REAL tol,REAL * xmin)
{
        REAL f0,f1,f2,f3;  // x0,x1,x2,x3; 
        GccUse(f0); GccUse(f3);  // Vaut pas chercher a comprende code NR

        mTolGolden = tol;


        x0=ax;
        x3=cx;
        if (fabs(cx-bx) > fabs(bx-ax)) {
                x1=bx;
                x2=bx+C*(cx-bx);
        } else {
                x2=bx;
                x1=bx-C*(bx-ax);
        }
        f1=(NRF1v)(x1);
        f2=(NRF1v)(x2);
        mNbIter = 0;
        while (NROptF1vContinue() ) {
// std::cout << "nb iter " << mNbIter << "\n";
                mNbIter++;
                if (f2 < f1) {
                        SHFT(x0,x1,x2,R*x1+C*x3)
                        SHFT(f0,f1,f2,(NRF1v)(x2))
                } else {
                        SHFT(x3,x2,x1,R*x2+C*x0)
                        SHFT(f3,f2,f1,(NRF1v)(x1))
                }
        }
        if (f1 < f2) {
                *xmin=x1;
                return f1;
        } else {
                *xmin=x2;
                return f2;
        }
}                    

                                                  


REAL NROptF1vND::PrivBrent
     (
         REAL ax,REAL bx,REAL cx,
         REAL tol,REAL * xmin,INT ITMAX
     )
{
        int iter;
        REAL a,b,d,etemp,fu,fv,fw,fx,p,q,r,tol1,tol2,u,v,w,x,xm;
        REAL e=0.0;

        d = 1e20; // becuase Unitialized, (in fact e=0 => OK)
        a=((ax < cx) ? ax : cx);
        b=((ax > cx) ? ax : cx);
        x=w=v=bx;
        fw=fv=fx=(NRF1v)(x);
        for (iter=1;iter<=ITMAX;iter++) {
                xm=0.5*(a+b);
                tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
                if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
                        *xmin=x;
                        return fx;
                }
                if (fabs(e) > tol1) {
                        r=(x-w)*(fx-fv);
                        q=(x-v)*(fx-fw);
                        p=(x-v)*q-(x-w)*r;
                        q=2.0*(q-r);
                        if (q > 0.0) p = -p;
                        q=fabs(q);
                        etemp=e;
                        e=d;
                        if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
                                d=CGOLD*(e=(x >= xm ? a-x : b-x));
                        else {
                                d=p/q;
                                u=x+d;
                                if (u-a < tol2 || b-u < tol2)
                                        d=SIGN(tol1,xm-x);
                        }
                } else {
                        d=CGOLD*(e=(x >= xm ? a-x : b-x));
                }                                      
                u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
                fu=(NRF1v)(u);
                if (fu <= fx) {
                        if (u >= x) a=x; else b=x;
                        SHFT(v,w,x,u)
                        SHFT(fv,fw,fx,fu)
                } else {
                        if (u < x) a=u; else b=u;
                        if (fu <= fw || w == x) {
                                v=w;
                                w=u;
                                fv=fw;
                                fw=fu;
                        } else if (fu <= fv || v == x || v == w) {
                                v=u;
                                fv=fu;
                        }
                }
        }
        *xmin=x;
        return fx;
}                                          


Pt2dr  NROptF1vND::brent (bool ForBench)
{
    Pt2dr aP;
    REAL xx,fx,fb,fa,bx,ax;

    ax=0.0;
    xx=1.0;
    bx=2.0;
    mnbrack(&ax,&xx,&bx,&fa,&fx,&fb);
    aP.y =PrivBrent
          (
          ax,xx,bx,
          ForBench ? 1e-8 : TOL,
          &aP.x,
          200
          );

    return aP;
}


/******************************************************/
/*                                                    */
/*                NROptF1vDer                         */
/*                                                    */
/******************************************************/


REAL NROptF1vDer::PrivBrent
     (
            REAL ax,REAL bx,REAL cx,
            REAL tol,REAL * xmin,INT ITMAX
     )
{
	int iter,ok1,ok2;
	REAL a,b,d=1e5,d1,d2,du,dv,dw,dx,e=0.0;
	REAL fu,fv,fw,fx,olde,tol1,tol2,u,u1,u2,v,w,x,xm;

	a=(ax < cx ? ax : cx);
	b=(ax > cx ? ax : cx);
	x=w=v=bx;
	fw=fv=fx=(NRF1v)(x);
	dw=dv=dx=(DerNRF1v)(x);
	for (iter=1;iter<=ITMAX;iter++) {
		xm=0.5*(a+b);
		tol1=tol*fabs(x)+ZEPS;
		tol2=2.0*tol1;
		if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
			*xmin=x;
			return fx;
		}
		if (fabs(e) > tol1) {
			d1=2.0*(b-a);
			d2=d1;
			if (dw != dx)  d1=(w-x)*dx/(dx-dw);
			if (dv != dx)  d2=(v-x)*dx/(dx-dv);
			u1=x+d1;
			u2=x+d2;
			ok1 = (a-u1)*(u1-b) > 0.0 && dx*d1 <= 0.0;
			ok2 = (a-u2)*(u2-b) > 0.0 && dx*d2 <= 0.0;
			olde=e;
			e=d;
			if (ok1 || ok2) {
				if (ok1 && ok2)
					d=(fabs(d1) < fabs(d2) ? d1 : d2);
				else if (ok1)
					d=d1;
				else
					d=d2;
				if (fabs(d) <= fabs(0.5*olde)) {
					u=x+d;
					if (u-a < tol2 || b-u < tol2)
						d=SIGN(tol1,xm-x);
				} else {
					d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
				}
			} else {
				d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
			}
		} else {
			d=0.5*(e=(dx >= 0.0 ? a-x : b-x));
		}
		if (fabs(d) >= tol1) {
			u=x+d;
			fu=(NRF1v)(u);
		} else {
			u=x+SIGN(tol1,d);
			fu=(NRF1v)(u);
			if (fu > fx) {
				*xmin=x;
				return fx;
			}
		}
		du=(DerNRF1v)(u);
		if (fu <= fx) {
			if (u >= x) a=x; else b=x;
			MOV3(v,fv,dv, w,fw,dw)
			MOV3(w,fw,dw, x,fx,dx)
			MOV3(x,fx,dx, u,fu,du)
		} else {
			if (u < x) a=u; else b=u;
			if (fu <= fw || w == x) {
				MOV3(v,fv,dv, w,fw,dw)
				MOV3(w,fw,dw, u,fu,du)
			} else if (fu < fv || v == x || v == w) {
				MOV3(v,fv,dv, u,fu,du)
			}
		}
	}
        *xmin=x;
        return fx;
}


REAL NROptF1vDer::rtsafe(REAL x1,REAL x2,REAL xacc,INT MAXIT)
{
	int j;
	REAL df,dx,dxold,f,fh,fl;
	REAL swap,temp,xh,xl,rts;

        fl = NRF1v(x1); //  (*funcd)(x1,&fl,&df);
	fh = NRF1v(x2); // (*funcd)(x2,&fh,&df);
        ELISE_ASSERT((fl*fh)<0,"Root must be bracketed in RTSAFE");
	if (fl < 0.0) {
		xl=x1;
		xh=x2;
	} else {
		xh=x1;
		xl=x2;
		swap=fl;
		fl=fh;
		fh=swap;
	}
	rts=0.5*(x1+x2);
	dxold=fabs(x2-x1);
	dx=dxold;
	f = NRF1v(rts); df = DerNRF1v(rts); // (*funcd)(rts,&f,&df);
	for (j=1;j<=MAXIT;j++) {
		if ((((rts-xh)*df-f)*((rts-xl)*df-f) >= 0.0)
			|| (fabs(2.0*f) > fabs(dxold*df))) {
			dxold=dx;
			dx=0.5*(xh-xl);
			rts=xl+dx;
			if (xl == rts) return rts;
		} else {
			dxold=dx;
			dx=f/df;
			temp=rts;
			rts -= dx;
			if (temp == rts) return rts;
		}
		if (fabs(dx) < xacc) return rts;
	        f = NRF1v(rts); df = DerNRF1v(rts); //  (*funcd)(rts,&f,&df);
		if (f < 0.0) {
			xl=rts;
			fl=f;
		} else {
			xh=rts;
			fh=f;
		}
	}
	return rts;
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
