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
                                



/* Merci Numerical recipes (Press,Flannery,Teukolsky,Vetterling ...).
*/

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
	a[k][l]=h+s*(g-h*tau);


/*************************************************************/
/*************************************************************/
/*************************************************************/
/*************************************************************/



template<class Type>  INT NR_jacobi (Type ** A ,INT n ,Type *diag ,Type ** V)
{
	INT j,iq,ip,i;
	Type tresh,theta,tau,t,sm,s,h,g,c;

	diag--;

	static ElFilo<Type *> Fa;  NR_InitNrMat(Fa,A,n); Type ** a= Fa.tab();
	static ElFilo<Type *> Fv;  NR_InitNrMat(Fv,V,n); Type ** v = Fv.tab();

	static ElFilo<Type> Fb;  NR_InitVect(Fb,n); Type * b = Fb.tab();
	static ElFilo<Type> Fz;  NR_InitVect(Fz,n); Type * z = Fz.tab();

	for (ip=1;ip<=n;ip++) {
		for (iq=1;iq<=n;iq++) v[ip][iq]=0.0;
		v[ip][ip]=1.0;
	}
	for (ip=1;ip<=n;ip++) {
		b[ip]=diag[ip]=a[ip][ip];
		z[ip]=0.0;
	}
	for (i=1;i<=50;i++) {



		sm=0.0;
		for (ip=1;ip<=n-1;ip++) {
			for (iq=ip+1;iq<=n;iq++)
				sm += ElAbs(a[ip][iq]);
		}
		if (sm == 0.0) {
			return(1);
		}
		if (i < 4)
			tresh=0.2*sm/(n*n);
		else
			tresh=0.0;
		for (ip=1;ip<=n-1;ip++) {
			for (iq=ip+1;iq<=n;iq++) {
				g=100.0*ElAbs(a[ip][iq]);
				if (i > 4 && ElAbs(diag[ip])+g == ElAbs(diag[ip])
					&& ElAbs(diag[iq])+g == ElAbs(diag[iq]))
					a[ip][iq]=0.0;
				else if (ElAbs(a[ip][iq]) > tresh) {
					h=diag[iq]-diag[ip];
					if (ElAbs(h)+g == ElAbs(h))
						t=(a[ip][iq])/h;
					else {
						theta=0.5*h/(a[ip][iq]);
						t=1.0/(ElAbs(theta)+sqrt(1.0+theta*theta));
						if (theta < 0.0) t = -t;
					}
					c=1.0/sqrt(1+t*t);
					s=t*c;
					tau=s/(1.0+c);
					h=t*a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					diag[ip] -= h;
					diag[iq] += h;
					a[ip][iq]=0.0;
					for (j=1;j<=ip-1;j++) {
						ROTATE(a,j,ip,j,iq)
					}
					for (j=ip+1;j<=iq-1;j++) {
						ROTATE(a,ip,j,j,iq)
					}
					for (j=iq+1;j<=n;j++) {
						ROTATE(a,ip,j,iq,j)
					}
					for (j=1;j<=n;j++) {
						ROTATE(v,j,ip,j,iq)
					}
				}
			}
		}
		for (ip=1;ip<=n;ip++) {
			b[ip] += z[ip];
			diag[ip]=b[ip];
			z[ip]=0.0;
		}
	}
	return(0);
}

#undef ROTATE


template <class Type> class cCmpTHEVP
{
    public :
       bool operator ()(const int & i1,const int & i2)
       {
            return (*THEVP)(i1,0) < (*THEVP)(i2,0);
       }
       cCmpTHEVP( ElMatrix<Type> * aMat) :
            THEVP (aMat)
       {
       }
 
       ElMatrix<Type> * THEVP;
};


template<class Type> std::vector<int> Tpl_jacobi
     (
          const ElMatrix<Type> &  aMat0,
          ElMatrix<Type>  & aValP,
          ElMatrix<Type> &  aVecP
     )
{
    ElMatrix<Type>   aMatSym = aMat0;
    INT n = aMatSym.tx();
    ELISE_ASSERT(n==aMatSym.ty(),"Not Squre in jacobi");

    aVecP.set_to_size(n,n);
    aValP.set_to_size(n,1);
    NR_jacobi(aMatSym.data(),n,aValP.data()[0],aVecP.data());

    std::vector<int> aRes;
    for (int aK=0 ; aK<n ; aK++)
        aRes.push_back(aK);

    cCmpTHEVP<Type> the_cCmpTHEVP(&aValP);
    std::sort(aRes.begin(),aRes.end(),the_cCmpTHEVP);

   return aRes;
}


std::vector<int> jacobi
     (
          const ElMatrix<REAL> &  aMat0,
          ElMatrix<REAL>  & aValP,
          ElMatrix<REAL> &  aVecP
     )
{
   return Tpl_jacobi(aMat0,aValP,aVecP);
}

std::vector<int> jacobi
     (
          const ElMatrix<REAL16> &  aMat0,
          ElMatrix<REAL16>  & aValP,
          ElMatrix<REAL16> &  aVecP
     )
{
   return Tpl_jacobi(aMat0,aValP,aVecP);
}




void MatLigneToDiag
     (
         const  ElMatrix<REAL> & aLine,
	 ElMatrix<REAL> &        aDiag
     )
{
   ELISE_ASSERT(aLine.ty()==1,"Not a Line Matrix in MatLigneToDiag");
   INT n =  aLine.tx();
   aDiag.set_to_size(n,n);

   for (INT anX=0; anX<n ; anX++)
       for (INT anY=0; anY<n ; anY++)
           aDiag(anX,anY) = (anX==anY) ? aLine(anX,0) : 0;
}

std::vector<int> jacobi_diag
     (
          const ElMatrix<REAL> &  aMatSym,
          ElMatrix<REAL>  & aValP,
          ElMatrix<REAL> &  aVecP
     )
{
   ElMatrix<REAL> aLineVP(aMatSym.tx(),1);

   std::vector<int>  aRes = jacobi(aMatSym,aLineVP,aVecP);
    
   MatLigneToDiag(aLineVP,aValP);
   return aRes;
}

        


/*  #define PYTHAG(a,b) ((at=ElAbs(a)) > (bt=ElAbs(b)) ? \
  (ct=bt/at,at*sqrt(1.0+ct*ct)) : (bt ? (ct=at/bt,bt*sqrt(1.0+ct*ct)): 0.0))
*/

static inline double  PYTHAG(const double & a,const double & b)
{
   double at=ElAbs(a);
   double bt=ElAbs(b);

   // BUG NR => genere div/0 si a et b null
   if ((at<1e-30) && (bt<1e-30))
      return sqrt(a*a+b*b);
   
   if (at>bt)
   {
       double ct = bt/at;
       return at*sqrt(1.0+ElSquare(ct));
   }
   else
   {
      double ct = at/bt;
      return bt*sqrt(1.0+ElSquare(ct));
   }
   
}
// #define SIGN(a,b) ((b) >= 0.0 ? ElAbs(a) : -ElAbs(a))
static double  SIGN(const double & a,const double & b) 
{
	return ((b)>=0.0 ? ElAbs(a) : -ElAbs(a));
}


void NR_svdcmp(double ** A,int m,int n,double * w,double ** V)
{
	int flag,i,its,j,jj,k,l=-1,nm=-1;
	double c,f,h,s,x,y,z;
	double anorm=0.0,g=0.0,scale=0.0;

	ELISE_ASSERT( (m == n),"SVDCMP: You must augment A with extra zero rows");
	w--;
	static ElFilo<REAL *> Fa; NR_InitNrMat(Fa,A,n);  REAL ** a = Fa.tab();
	static ElFilo<REAL *> Fv; NR_InitNrMat(Fv,V,n);   REAL ** v = Fv.tab();
	static ElFilo<REAL>  Frv1;  NR_InitVect(Frv1,n); REAL * rv1 = Frv1.tab();


	//  Dans NR, la condition etait, (m >= n)
	for (i=1;i<=n;i++) {
		l=i+1;
		rv1[i]=scale*g;
		g=s=scale=0.0;
		if (i <= m) {
			for (k=i;k<=m;k++) scale += ElAbs(a[k][i]);
			if (scale) {
				for (k=i;k<=m;k++) {
					a[k][i] /= scale;
					s += a[k][i]*a[k][i];
				}
				f=a[i][i];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				a[i][i]=f-g;
				if (i != n) {
					for (j=l;j<=n;j++) {
						for (s=0.0,k=i;k<=m;k++) s += a[k][i]*a[k][j];
						f=s/h;
						for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
					}
				}
				for (k=i;k<=m;k++) a[k][i] *= scale;
			}
		}
		w[i]=scale*g;
		g=s=scale=0.0;
		if (i <= m && i != n) {
			for (k=l;k<=n;k++) scale += ElAbs(a[i][k]);
			if (scale) {
				for (k=l;k<=n;k++) {
					a[i][k] /= scale;
					s += a[i][k]*a[i][k];
				}
				f=a[i][l];
				g = -SIGN(sqrt(s),f);
				h=f*g-s;
				a[i][l]=f-g;
				for (k=l;k<=n;k++) rv1[k]=a[i][k]/h;
				if (i != m) {
					for (j=l;j<=m;j++) {
						for (s=0.0,k=l;k<=n;k++) s += a[j][k]*a[i][k];
						for (k=l;k<=n;k++) a[j][k] += s*rv1[k];
					}
				}
				for (k=l;k<=n;k++) a[i][k] *= scale;
			}
		}
		anorm=ElMax(anorm,(ElAbs(w[i])+ElAbs(rv1[i])));
	}
	for (i=n;i>=1;i--) {
		if (i < n) {
			if (g) {
				for (j=l;j<=n;j++)
					v[j][i]=(a[i][j]/a[i][l])/g;
				for (j=l;j<=n;j++) {
					for (s=0.0,k=l;k<=n;k++) s += a[i][k]*v[k][j];
					for (k=l;k<=n;k++) v[k][j] += s*v[k][i];
				}
			}
			for (j=l;j<=n;j++) v[i][j]=v[j][i]=0.0;
		}
		v[i][i]=1.0;
		g=rv1[i];
		l=i;
	}
	for (i=n;i>=1;i--) {
		l=i+1;
		g=w[i];
		if (i < n)
			for (j=l;j<=n;j++) a[i][j]=0.0;
		if (g) {
			g=1.0/g;
			if (i != n) {
				for (j=l;j<=n;j++) {
					for (s=0.0,k=l;k<=m;k++) s += a[k][i]*a[k][j];
					f=(s/a[i][i])*g;
					for (k=i;k<=m;k++) a[k][j] += f*a[k][i];
				}
			}
			for (j=i;j<=m;j++) a[j][i] *= g;
		} else {
			for (j=i;j<=m;j++) a[j][i]=0.0;
		}
		++a[i][i];
	}
	for (k=n;k>=1;k--) 
        {
		for (its=1;its<=30;its++) 
                {

			flag=1;
			for (l=k;l>=1;l--) 
                        {
				nm=l-1;
				if (ElAbs(rv1[l])+anorm == anorm) {
					flag=0;
					break;
				}
				if (ElAbs(w[nm])+anorm == anorm) break;
			}
			if (flag) 
                        {
				c=0.0;
				s=1.0;
				for (i=l;i<=k;i++) {
					f=s*rv1[i];
					if (ElAbs(f)+anorm != anorm) {
						g=w[i];
						h=PYTHAG(f,g);
						w[i]=h;
						h=1.0/h;
						c=g*h;
						s=(-f*h);
						for (j=1;j<=m;j++) {
							y=a[j][nm];
							z=a[j][i];
							a[j][nm]=y*c+z*s;
							a[j][i]=z*c-y*s;
						}
					}
				}
			}
			z=w[k];
			if (l == k) 
                        {
				if (z < 0.0) {
					w[k] = -z;
					for (j=1;j<=n;j++) v[j][k]=(-v[j][k]);
				}
				break;
			}
			ELISE_ASSERT
                        (
				 its <= 30,
				 "No convergence in 30 SVDCMP iterations"
                        );
			x=w[l];
			nm=k-1;
			y=w[nm];
			g=rv1[nm];
			h=rv1[k];
			f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g=PYTHAG(f,1.0);
			f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
			c=s=1.0;
			for (j=l;j<=nm;j++) 
                        {
				i=j+1;
				g=rv1[i];
				y=w[i];
				h=s*g;
				g=c*g;
				z=PYTHAG(f,h);
				rv1[j]=z;
				c=f/z;
				s=h/z;
				f=x*c+g*s;
				g=g*c-x*s;
				h=y*s;
				y=y*c;
				for (jj=1;jj<=n;jj++) {
					x=v[jj][j];
					z=v[jj][i];
					v[jj][j]=x*c+z*s;
					v[jj][i]=z*c-x*s;
				}
				z=PYTHAG(f,h);
				w[j]=z;
				if (z) {
					z=1.0/z;
					c=f*z;
					s=h*z;
				}
				f=(c*g)+(s*y);
				x=(c*y)-(s*g);
				for (jj=1;jj<=m;jj++) {
					y=a[jj][j];
					z=a[jj][i];
					a[jj][j]=y*c+z*s;
					a[jj][i]=z*c-y*s;
				}
			}
			rv1[l]=0.0;
			rv1[k]=f;
			w[k]=x;
		}
	}
}

static REAL SetToMatDirecte(ElMatrix<REAL> & aMat)
{
   if (aMat.Det() >0)
     return 1;

   for (int y=0; y<aMat.ty() ; y++)
       aMat(0,y) *= -1;
   return -1;
}


ElRotation3D AverRotation(const std::vector<ElRotation3D> & aVRot,const std::vector<double> & aVWeights)
{
    ELISE_ASSERT(aVRot.size()==aVWeights.size(),"AverRotation dif size");
    Pt3dr aC(0,0,0);
    ElMatrix<double> aMat(3,3,0.0);
    double aSomW = 0.0;

    for (int aK=0 ; aK<int(aVRot.size()) ; aK++)
    {
        double aW = aVWeights[aK];
        aC = aC + aVRot[aK].tr() * aW;
        aSomW += aW;
        aMat = aMat + aVRot[aK].Mat() * aW;
    }
    aC = aC / aSomW;
    aMat = NearestRotation(aMat*(1.0/aSomW));

    return ElRotation3D(aC,aMat,true);
}



ElMatrix<REAL> NearestRotation ( const ElMatrix<REAL> & aMat)
{
     INT n = aMat.tx();
     ElMatrix<REAL> anU(n),aV(n),aDiag(n);

     svdcmp_diag(aMat,anU,aDiag,aV,true);
     

     ElMatrix<REAL> aMul (n);

     for (int aK=0 ; aK < n ; aK++)
       aMul(aK,aK) = (aDiag(aK,aK) > 0) ? 1 : -1;


     return anU *  aMul * aV;
}


ElMatrix<REAL> VecKern ( const ElMatrix<REAL> & aMat)
{
     INT n = aMat.tx();
     ElMatrix<REAL> anU(n),aV(n),aDiag(n);

     svdcmp_diag(aMat,anU,aDiag,aV,true);
     int aKMin = 0;
     double aVPMin = 1e100;
     for (int aK=0 ; aK < n ; aK++)
     {
         double aVP = ElAbs(aDiag(aK,aK));
         if (aVP < aVPMin)
         {
            aVPMin = aVP;
            aKMin = aK;
         }
     }

     ElMatrix<REAL> aVecP(1,n);
     for (int aK=0 ; aK < n ; aK++)
        aVecP(0,aK) = (aKMin==aK);

     return aV.transpose() * aVecP;
}

ElMatrix<REAL> VecOfValP(const ElMatrix<REAL> & aMat,REAL aVP)
{
    return VecKern(aMat -  ElMatrix<REAL>(aMat.tx())*aVP);
}


Pt3dr AxeRot(const ElMatrix<REAL> & aMat)
{
    ElMatrix<REAL> aVec =  VecOfValP(aMat,1.0);

    return Pt3dr(aVec(0,0),aVec(0,1),aVec(0,2));
}


double TetaOfAxeRot(const ElMatrix<REAL> & aMat, Pt3dr & aP1)
{ 
    Pt3dr aP2,aP3;

    MakeRONWith1Vect(aP1,aP2,aP3);
    Pt3dr aQ2 = aMat * aP2;
    
    double aC = scal(aP2,aQ2);
    double aS = scal(aP3,aQ2);

    return atan2(aS,aC);
}

double LongBase(const ElRotation3D &aR )
{
   return euclid(aR.tr());
}
ElRotation3D ScaleBase(const ElRotation3D & aR,const double & aScale)
{
    return ElRotation3D(aR.tr()*aScale,aR.Mat(),true);
}





/*
void MakeRONConservByPrio(Pt3dr & aV1,Pt3dr & aV2,Pt3dr & aV3)
{
    aV1 = vunit(aV1);
    aV2 = vunit(aV2);
    aV3 = aV1 ^ aV2;
    aV2 = aV3 ^aV1;

    aV2 = vunit(aV2);
}
*/

void MakeRONWith1Vect(Pt3dr & aV1,Pt3dr & aV2,Pt3dr & aV3)
{
    aV1 = vunit(aV1);
    if (ElAbs(aV1.x) < ElAbs(aV1.y))
       aV2 = Pt3dr(0,aV1.z,-aV1.y);
    else
       aV2 = Pt3dr(aV1.z,0,-aV1.x);

   aV2 = vunit(aV2);
   aV3 = aV1 ^ aV2;
}

Pt3dr MakeOrthon(Pt3dr & aV1,Pt3dr & aV2)
{
   Pt3dr aU1 = vunit(aV1);
   Pt3dr aU2 = vunit(aV2);

   aV1 = vunit(aU1+aU2);
   aV2 = vunit(aU1-aU2);
   return aV1 ^ aV2;
}

Pt3dr SchmitComplMakeOrthon(Pt3dr & aV1,Pt3dr & aV2)
{
   aV1 = vunit(aV1);
   aV2 = vunit(aV2 - aV1 * scal(aV1,aV2));

   return aV1 ^ aV2;
}




ElMatrix<double>  MakeMatON(Pt3dr aV1,Pt3dr aV2)
{
   Pt3dr aV3 = MakeOrthon(aV1,aV2);

   return MatFromCol(aV1,aV2,aV3);
}


ElMatrix<REAL> ComplemRotation
               (
                    const Pt3dr & anAnt1,
                    const Pt3dr & anAnt2,
                    const Pt3dr & anIm1,
                    const Pt3dr & anIm2
               )
{
  return    MakeMatON(anIm1,anIm2)
         *  MakeMatON(anAnt1,anAnt2).transpose();
}


void svdcmp
     (
          const ElMatrix<REAL> & aMat,
	  ElMatrix<REAL> & anU,
	  ElMatrix<REAL> & aDiag,
	  ElMatrix<REAL> & aV,
          bool             direct
     )
{
     anU  = aMat;
     INT n = anU.tx();
     ELISE_ASSERT(n==anU.ty(),"Not Squre in jacobi");
     aDiag.set_to_size(n,1);
     aV.set_to_size(n,n);

     NR_svdcmp(anU.data(),n,n,aDiag.data()[0],aV.data());

     if (direct)
     {
         REAL sign = SetToMatDirecte(anU) * SetToMatDirecte(aV);
         if (sign < 0)
            aDiag(0,0) *= -1;
     }
     aV.self_transpose();
}

void svdcmp_diag
     (
          const ElMatrix<REAL> & aMat,
	  ElMatrix<REAL> & anU,
	  ElMatrix<REAL> & aDiag,
	  ElMatrix<REAL> & aV,
          bool             direct
     )
{
   ElMatrix<REAL> aLineVP(aMat.tx(),1);
   svdcmp(aMat,anU,aLineVP,aV,direct);
   MatLigneToDiag(aLineVP,aDiag);
}



void NR_QRDecomp(double **A, int n, double *c, double *d, int *sing)
{
   // Mise au convention NR
   c--; d--;
   static ElFilo<REAL *> Fa; NR_InitNrMat(Fa,A,n);  REAL ** a = Fa.tab();



   int i,j,k;
   double scale,sigma,sum,tau;
   *sing=0;
   for (k=1;k<n;k++)
   {
      scale=0.0;
      for (i=k;i<=n;i++)
      {
          scale=ElMax(scale,ElAbs(a[i][k]));
      }
      if (scale == 0.0)
      {
         *sing=1;
         c[k]=d[k]=0.0;
      }
      else {
         for (i=k;i<=n;i++)
         {
             a[i][k] /= scale;
         }
         for (sum=0.0,i=k;i<=n;i++)
         {
              sum += ElSquare(a[i][k]);
         }
         // sigma=SIGN(sqrt(sum),a[k][k]);
         sigma=SIGN(sqrt(sum),a[k][k]);
         a[k][k] += sigma;
         c[k]=sigma*a[k][k];
         d[k] = -scale*sigma;
         for (j=k+1;j<=n;j++)
         {
            for (sum=0.0,i=k;i<=n;i++)
            {
                 sum += a[i][k]*a[i][j];
            }
            tau=sum/c[k];
            for (i=k;i<=n;i++)
            {
                a[i][j] -= tau*a[i][k];
            }
         }
      }
   }
   d[n]=a[n][n];
   if (d[n] == 0.0) *sing=1;
}

//void QRCorrectSign()

/*
   (a b) (-1 0)  =  (-a b)
   (c d) (0  1)     (-c d)

   Si S est une matrice de signe et QR une decomposition QR alors  (QS) (SR) aussi,
   on peut donc changer le signe des colonne de Q et des lignes de R de manière a 
   avoir une diagonale > 0
*/

std::pair<ElMatrix<double>, ElMatrix<double> > QRDecomp(const ElMatrix<double> & aM0)
{
   ElMatrix<double> aMat(aM0);
   int aN = aMat.tx();
   ELISE_ASSERT(aMat.ty()==aN,"Non Square Mat in QR-Dcmp");
   std::vector<double> d(aN+1);
   std::vector<double> c(aN+1);
   int aSign;

   NR_QRDecomp(aMat.data(),aN,VData(c),VData(d),&aSign);

   ElMatrix<double> aR(aN,aN);
  
   for (int anX=0 ; anX <aN ; anX++)
   {
       for (int anY=0 ; anY <aN ; anY++)
       {
           if (anX>anY)  aR(anX,anY) = aMat(anX,anY) ;
           else if (anX==anY)  aR(anX,anY) = d[anX] ;
           else aR(anX,anY) = 0;
       }
   }


   ElMatrix<double> aQ = aM0 * gaussj(aR);

   for (int aDiag=0 ; aDiag <aN ; aDiag++)
   {
       if (aR(aDiag,aDiag) < 0)
       {
           for (int aK=0 ; aK <aN ; aK++)
           {
               aR(aK,aDiag) *= -1;
               aQ(aDiag,aK) *= -1;
           }
       }
   }


   return std::pair<ElMatrix<double>, ElMatrix<double> >(aQ,aR); 
}

ElMatrix<double> InvertLine(const ElMatrix<double> & aM0)
{
   int aSzX = aM0.tx();
   int aSzY = aM0.ty();
   ElMatrix<double> aRes(aSzX,aSzY);

   for (int anY=0 ; anY <aSzY ; anY++)
   {
      for (int anX=0 ; anX <aSzX ; anX++)
      {
           aRes(anX,anY) = aM0(anX,aSzY-anY-1);
      }
  }
  return aRes;
}

ElMatrix<double> InvertCol(const ElMatrix<double> & aM0)
{
   int aSzX = aM0.tx();
   int aSzY = aM0.ty();
   ElMatrix<double> aRes(aSzX,aSzY);

   for (int anY=0 ; anY <aSzY ; anY++)
   {
      for (int anX=0 ; anX <aSzX ; anX++)
      {
           aRes(anX,anY) = aM0(aSzX-anX-1,anY);
      }
  }
  return aRes;
}



/* Soit S la matrice (0 0 1)
                     (0 1 0)
                     (1 0 0)
   On a InvertLine(A)  = S A
   On a InvertCol(A)  = A S   
         S S  = Id

   On calcule la QRDec de t(SA) = QR

   Donc  A = S tR  tQ = (StRS) SQ
   StRS est diag sup, SQ est tjrs Orthog
*/



 
std::pair<ElMatrix<double>, ElMatrix<double> > RQDecomp(const ElMatrix<double> & aM0)
{
   int aN = aM0.tx();
   ELISE_ASSERT(aM0.ty()==aN,"Non Square Mat in QR-Dcmp");

   std::pair<ElMatrix<double>, ElMatrix<double> >  aQR = QRDecomp(InvertLine(aM0).transpose());

   const ElMatrix<double> & aQ = aQR.first;
   const ElMatrix<double> & aR = aQR.second;

    // ShowMatr("AAAA",aR);
   //  ShowMatr("BBBB",aR.transpose());
   // ShowMatr("CCCC",InvertLine(InvertCol(aR.transpose())));

  ElMatrix<double> aR2 = InvertLine(InvertCol(aR.transpose()));
  ElMatrix<double> aQ2 = InvertLine(aQ.transpose());

  // std::cout <<  "RRrrRRr " << aR2.Det() << " " << aQ2.Det() << "\n";
  if ( aN<=3)
  {
       if (aQ2.Det() < 0)
       {
            aR2 =  aR2 * -1;
            aQ2 =  aQ2 * -1;
       }
  }

  return std::pair<ElMatrix<double>, ElMatrix<double> > (aR2,aQ2);
   // return std::pair<ElMatrix<double>, ElMatrix<double> > (InvertLine(InvertCol(aR.transpose())),InvertLine(aQ.transpose()));
}



void TestQR(int aN)
{
   ElMatrix<double> aM(aN,aN);

   for (int anX=0 ; anX <aN ; anX++)
   {
       for (int anY=0 ; anY <aN ; anY++)
       {
            aM(anX,anY) = 100 * NRrandC();
       }
   }

   if (0)
   {
        std::pair<ElMatrix<double>, ElMatrix<double> > aQR =  QRDecomp(aM);

        ElMatrix<double> aQ = aQR.first;
        ElMatrix<double> aR = aQR.second;


        ElMatrix<double> anId =  aQ * aQ.transpose();
        ElMatrix<double> anId2(aN,true);
        ElMatrix<double> aT2 = aM - aQ*aR;

        std::cout << "QR-Test OrthoNo " << anId.L2(anId2) << " " << aT2.L2() << "\n";
        ShowMatr("QQq-RRr",aR);

   }

   {
        std::pair<ElMatrix<double>, ElMatrix<double> > aRQ =  RQDecomp(aM);

        ElMatrix<double> aR = aRQ.first;
        ElMatrix<double> aQ = aRQ.second;


        ElMatrix<double> anId =  aQ * aQ.transpose();
        ElMatrix<double> anId2(aN,true);
        ElMatrix<double> aT2 = aM - aR * aQ;

        int aNbNeg=0;
        for (int aK=0 ;  aK< aN ; aK++)
            aNbNeg += (aR(aK,aK) <0);

        double aSomInf = 0;
        for (int anX=0 ;  anX< aN ; anX++)
           for (int anY=0 ;  anY< aN ; anY++)
               if (anY>anX) 
                  aSomInf += ElAbs(aR(anX,anY));

        std::cout << "RQ-Test OrthoNo " << anId.L2(anId2) << " " << aT2.L2() 
                  << " NEG=" << aNbNeg  << " SomInf=" << aSomInf << "\n";
 
        ShowMatr("Rrr-QQq",aR);

   }
}

#undef SIGN
#undef MAX
#undef PYTHAG

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
