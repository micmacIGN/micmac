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


/*
     ELISE'S INTERFACE to NUMERICAL RECIPES.
*/



template <class Type> void NR_InitNrMat(ElFilo<Type *> & Mat,Type ** a,INT n)
{
    Mat.clear();
    Mat.pushlast((Type *)0);
    for (INT k=0;k<n;k++)
        Mat.pushlast(a[k]-1);
}

template void NR_InitNrMat(ElFilo<REAL *> & Mat,REAL ** a,INT n);
template void NR_InitNrMat(ElFilo<REAL16 *> & Mat,REAL16 ** a,INT n);

template <class Type> void  NR_InitVect(ElFilo<Type> & V,INT n)
{
    V.clear();
    for (INT k=0; k<=n; k++)
        V.pushlast(0);
}
template void  NR_InitVect(ElFilo<REAL> & V,INT n);
template void  NR_InitVect(ElFilo<REAL16> & V,INT n);

#define SWAP(a,b) {REAL16 temp=(a);(a)=(b);(b)=temp;}


template <class Type> bool gaussj_svp(Type **A,INT n)
{


	int i,icol=0,irow=0,j,k=0,l,ll;
	Type big,dum,pivinv;
        
        static ElFilo<Type *> a;  
        NR_InitNrMat(a,A,n);

        static ElFilo<INT> indxc(n+1);      NR_InitVect(indxc,n);
        static ElFilo<INT> indxr(n+1); NR_InitVect(indxr,n);
        static ElFilo<INT> ipiv(n+1);  NR_InitVect(ipiv,n);


        k=0;
	for (j=1;j<=n;j++) ipiv[j]=0;
	for (i=1;i<=n;i++) {
		big=0.0;
		for (j=1;j<=n;j++)
                {
			if (ipiv[j] != 1)
                        {
				for (k=1;k<=n;k++) 
                                {
					if (ipiv[k] == 0) {
						if (ElAbs(a[j][k]) >= big) {
							big=ElAbs(a[j][k]);
							irow=j;
							icol=k;
						}
					} 
                                        else 
                                        {
                                           if (ipiv[k] > 1) 
                                              return false;
                                        }
				}
                          }
                }
		++(ipiv[icol]);
		if (irow != icol) {
			for (l=1;l<=n;l++) SWAP(a[irow][l],a[icol][l])
		}
                if ((irow==0) || (icol==0)) 
                {
                    ELISE_ASSERT(false,"very singular matrix in Gausj");
                }
		indxr[i]=irow;
		indxc[i]=icol;
                if (a[icol][icol] == 0.0) 
                   return false;
		pivinv=1.0/a[icol][icol];
		a[icol][icol]=1.0;
		for (l=1;l<=n;l++) a[icol][l] *= pivinv;
		for (ll=1;ll<=n;ll++)
                {
			if (ll != icol) {
				dum=a[ll][icol];
				a[ll][icol]=0.0;
				for (l=1;l<=n;l++) a[ll][l] -= a[icol][l]*dum;
			}
                }
	}
	for (l=n;l>=1;l--) {
		if (indxr[l] != indxc[l])
			for (k=1;k<=n;k++)
				SWAP(a[k][indxr[l]],a[k][indxc[l]]);
	}

/*
if (MPD_MM())
{
    if ( n > 10)
    {    
         std::cout << "GaussjSZ=" << n << "\n";
         for (int aK=0 ; aK<n  ; aK++)
             std::cout << "kk gaussj_svp  " << A[aK][aK] << "\n";
    }
}
*/

        return true;
}

template <class Type> void gaussj(Type **A,INT n)
{
    if (! gaussj_svp(A,n))
    {
       ELISE_ASSERT(false,"Singular Matrix");
    }
}


template bool gaussj_svp(REAL **A,INT n);
template void gaussj(REAL **A,INT n);

template bool gaussj_svp(REAL16 **A,INT n);
template void gaussj(REAL16 **A,INT n);


/**********************************************/
/*                                            */
/*         GaussjPrec                         */
/*                                            */
/**********************************************/

GaussjPrec::GaussjPrec(INT n,INT m) :
   _n    (n),
   _m    (m),
   _M    (n,n),
   _Minv (n,n),
   _b    (m,n),
   _x    (m,n),
   _eps  (m,n),
   _ec   (m,n)
{
}


void GaussjPrec::SelfSetMatrixInverse(ElMatrix<REAL> & aM,INT aNbIter)
{
   ELISE_ASSERT(aM.tx()==aM.ty(),"NoSquareMat in GaussjPrec::SelfSetMatrixInverse");

   set_size_nm(aM.tx(),aM.tx());
   _M = aM;
   for (int x=0; x<_n ; x++)
       for (int y=0; y<_n ; y++)
           _b(x,y) = (x==y);

   ElTimer Chrono;
   init_rec();

   for (; aNbIter>0 ; aNbIter--)
      amelior_sol();


   aM = _x;
}


void GaussjPrec::set_size_nm(INT n,INT m)
{
  _n = n;
  _m = m;
   _M.set_to_size(n,n);
   _Minv.set_to_size(n,n);
   _b.set_to_size(n,m);
   _x.set_to_size(n,m);
   _eps.set_to_size(n,m);
   _ec.set_to_size(n,m);
}

void GaussjPrec::set_size_m(INT m)
{
    set_size_nm(_n,m);
}

void GaussjPrec::set_ecart()
{
    //  _ec = _b-_M * _x;
    _ec.mul(_M,_x);
    _ec.sub(_b,_ec);
}

bool GaussjPrec::init_rec()
{
    _Minv = _M;
    if (self_gaussj_svp(_Minv))
    {
       _x.mul(_Minv,_b); // _x = _Minv * _b;
       set_ecart();
       return true;
    }
    else
    {
       return false;
    }
}
REAL GaussjPrec::ecart() const
{
    return _ec.L2();
}

void GaussjPrec::amelior_sol()
{
    _eps.mul(_Minv,_ec); // _eps = _Minv * _ec;
    _x.add(_x,_eps);     // _x  = _x + _eps;
    set_ecart();
}


REAL GaussjPrec::ecart_inv() const
{
    return EcartInv(_M,_Minv);
}

#undef SWAP






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
