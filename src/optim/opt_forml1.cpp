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



bool Optim_L1FormLin::AbscD1::operator < (const AbscD1 & A2) const
{
   return _x0 < A2._x0;
}

Optim_L1FormLin::AbscD1::AbscD1(ElMatrix<REAL> & sc,INT k)
{
    _k = k;
    REAL v0 = sc(0,k);
    REAL v1 = sc(1,k);

    _pds = v1-v0;
    if (_pds)
       _x0 = - v0/_pds;
    else
       _x0 = 0;
    _pds = ElAbs(_pds);
}


Optim_L1FormLin::Optim_L1FormLin (const ElMatrix<REAL> & Flin) :
   _NbVar            (Flin.tx()-1),
   _NbForm           (Flin.ty()),
   _Flin             (Flin),
   _GP               (_NbVar,2),
   _MGauss           (_GP.M()),
   _MVGauss          (_GP.b()),
   _Sol              (_GP.x()),
   _SolDirRech       (2,_NbVar+1,-1.0),
   _Scal1D           (2,_NbForm,0.0),
   _BestSol          (1,_NbVar,0.0),
   _bench_comb_made  (false)
{
    ELISE_ASSERT
    (
        _NbForm>=_NbVar,
        "Number Of Constraint > Number of Variable (in Optim_L1FormLin)"
    );
    for (INT k=0;k<_NbForm ; k++)
        _Flin (_NbVar,k) = - _Flin (_NbVar,k) ;
}

REAL Optim_L1FormLin::Kth_score(const ElMatrix<REAL> & M,INT KLig)
{
    REAL res = 0.0;
    for (INT kc=0; kc<_NbVar ; kc++)
        res += M(0,kc)*_Flin(kc,KLig);

    return ElAbs(res-_Flin(_NbVar,KLig));
}

REAL Optim_L1FormLin::score(const ElMatrix<REAL> & M)
{
    REAL res = 0.0;
    for (INT KLig=0 ; KLig<_NbForm ; KLig++)
        res += Kth_score(M,KLig);
    return res;
}

bool  Optim_L1FormLin::Sol ( const ElFilo<INT> & SubSet)
{
    ELISE_ASSERT(SubSet.nb() == _NbForm,"Size in Sol");
    INT jF = 0;
    for (INT kf=0; kf <SubSet.nb() ; kf++)
    {
        if (SubSet[kf])
        {
           for (INT kc =0; kc<_NbVar; kc++)
               _MGauss(kc,jF) = _Flin(kc,kf);
           _MVGauss(1,jF) = _MVGauss(0,jF) = _Flin(_NbVar,kf);
           jF++;
        }
    }


    if (! _GP.init_rec())
       return false;
    for (INT k=0; k<6; k++)
        _GP.amelior_sol();

    return true;
}

REAL Optim_L1FormLin::score(ElFilo<INT> & SubSet)
{
    if (Sol(SubSet))
    {
       return score(_Sol);
    }
    else
    {
       return 1e60;
    }
}




REAL Optim_L1FormLin::EcartVar(INT v)
{
    for (INT kc =0; kc<_NbVar; kc++)
        _MGauss(kc,_NbVar-1) = (kc==v);
    _MVGauss(0,_NbVar-1) = 0;
    _MVGauss(1,_NbVar-1) = 1;

    if (_GP.init_rec())
    {
       for (INT k=0 ; k<3; k++)
          _GP.amelior_sol();
       return _GP.ecart();
    }
    else
       return 1e70;
}


bool Optim_L1FormLin::ExploreChVAR
     (
        ElFilo<INT> & SubSet,
        REAL	    & sc_min,
        INT           kv
     )
{

   _NbStep ++;

    SubSet[kv] = 0;
    INT jF = 0;
    for (INT kf=0; kf<SubSet.nb() ; kf++)
    {
        if (SubSet[kf])
        {
           for (INT kc =0; kc<_NbVar; kc++)
               _MGauss(kc,jF) = _Flin(kc,kf);
           _MVGauss(1,jF) = _MVGauss(0,jF) = _Flin(_NbVar,kf);
           jF ++;
        }
    }

    REAL ecmin = 1e60;
    for (INT v =0; (v<_NbVar) && (ecmin > 1e-5) ; v++)
    {
        REAL ecart = EcartVar(v);
        if (ecart < ecmin)
        {
            for (INT v2 =0; v2<_NbVar ; v2++)
            {
               _SolDirRech(0,v2) = _GP.x()(0,v2);
               _SolDirRech(1,v2) = _GP.x()(1,v2);
            } 
             ecmin = ecart;
        }
    }
    if (ecmin > 1e-2) 
       return false;

    _Scal1D.mul(_Flin,_SolDirRech);

    _vad1.clear();

	{
    for (INT kf =0; kf<_NbForm; kf++)
        if (! SubSet[kf])
           _vad1.push_back(AbscD1(_Scal1D,kf));
	}

    STDSORT(_vad1.begin(),_vad1.end());

    REAL ptot = 0;
    for (INT k=0; k<(INT)_vad1.size() ; k++)
        ptot +=  _vad1[k]._pds;

    INT kgot = -1;

    REAL pds_av = -ptot;
	{
    for (INT k=0; k<(INT)_vad1.size() ; k++)
    {
        REAL pds_apres = pds_av + 2* _vad1[k]._pds;
        if ((pds_av <0) && ( pds_apres >=0))
           kgot = _vad1[k]._k;
        pds_av =  pds_apres;
    }
	}
    

    ELISE_ASSERT(kgot != -1,"Inc in  Optim_L1FormLin");

/*
    if (kgot == kv)
       return false;
*/

    SubSet[kgot] = 1;


    REAL sc = score(SubSet);
    if ((sc < sc_min) && (kgot != kv))
    {
       sc_min = sc;
       _BestSol = _Sol;
       return true;
    }
    else
       return false;
}



INT Optim_L1FormLin::RandF()
{
    return (INT) (_NbForm * NRrandom3());
}


bool Optim_L1FormLin::get_sol_adm(ElFilo<INT> & SubSet)
{
    SubSet.clear();
    for (INT k=0; k<_NbVar; k++)
        SubSet.pushlast(1);

    {
	for (INT k=_NbVar; k<_NbForm; k++)
        SubSet.pushlast(0);

	{
          for (INT NTest = 0 ; NTest < 100000 ; NTest++)
          {
                INT jF = 0;
                for (INT kf=0; kf <SubSet.nb() ; kf++)
                {
                    if (SubSet[kf])
                    {
                       for (INT kc =0; kc<_NbVar; kc++)
                           _MGauss(kc,jF) = _Flin(kc,kf);
                       jF++;
                    }
                }
                if (_GP.init_rec() && (_GP.ecart_inv() < 1e-5))
                {
                   return true;
                }
                INT k1 = RandF();
                while (SubSet[k1]) k1=  RandF();
                INT k2 = RandF();
                while (! SubSet[k2]) k2=  RandF();
                SubSet[k1] = 1;
                SubSet[k2] = 0;
          }
	}
    }

    return false;
}


ElMatrix<REAL> Optim_L1FormLin::MpdSolve()
{
    ElFilo<INT> SubSet;
    get_sol_adm(SubSet);

    REAL sc_min = score(SubSet);
    _BestSol = _Sol;

    INT nb_test = 0;

    _NbStep = 0;

    while (nb_test <_NbVar )
    {
        for (INT kv =0; kv <_NbForm ; kv++)
        {
             if (SubSet[kv])
             {
                if  (ExploreChVAR(SubSet,sc_min,kv))
                    nb_test =0;
                else
                    nb_test++;
             }
        }
    }
    return _BestSol;
}



ElMatrix<REAL> Optim_L1FormLin::Solve()
{
    return BarrodaleSolve();
}




/*********************************************************/
/*********************************************************/
/*********************************************************/
/*********            CRAIG                       ********/
/*********************************************************/
/*********************************************************/
/*********************************************************/

/*
  Big must be set equal to any very large double precision constant.
  It's value here is appropriate for the Sun.
*/
#define BIG 1.0e37

	/*
	  stdlib.h - used to define the return value of malloc() and free().
	*/

#include <cstdlib>

void Elise_Craig_Barrodale_Roberts_l1(INT m,INT  n,REAL * a,REAL *b,REAL toler,REAL * x,REAL * e)
{
	/*
	  This algorithm uses a modification of the simplex method
	  to calculate an L1 solution to an over-determined system
	  of linear equations.

	  input:  n = number of unknowns.
		  m = number of equations (m >= n).
		  a = two dimensional double precision array of size
			m+2 by n+2.  On entry, the coefficients of the
			matrix must be stored in the first m rows and
			n columns of a.
		  b = one dimensional double precision array of length
			m.  On entry, b must contain the right hand
			side of the equations.
		  toler = a small positive tolerance.  Empirical
				evidence suggests toler=10**(-d*2/3)
				where d represents the number of
				decimal digits of accuracy available.
				Essentially, the routine regards any
				quantity as zero unless its magnitude
				exceeds toler.  In particular, the
				routine will not pivot on any number
				whose magnitude is less than toler.

	  output:  a = the first m rows and n columns are garbage on
			output.
		   a[m][n] = the minimum sum of the absolute values of
				the residuals.
		   a[m][n+1] = the rank of the matrix of coefficients.
		   a[m+1][n] = e-xit code with values:
				0 - optimal solution which is probably
					non-unique.
				1 - unique optimal solution.
				2 - calculations terminated prematurely
					due to rounding errors.
		   a[m+1][n+1] = number of simplex iterations performed.
		   b = garbage on output.
		   x = one dimensional double precision array of length
			n.  On e-xit, this array contains a solution to
			the L1 problem.
		   e = one dimensional double precision array of size n.
			On e-xit, this array contains the residuals in
			the equations.

	  Reference:  I. Barrodale and F. D. Roberts, "Algorithm 478:
		Solution of an Overdetermined System of Equations in the
		L1 norm," CACM 17, 1974, pp. 319-320.

	  Programmer:  R. J. Craig
		       kat3@ihgp.ih.lucent.com
		       (630) 979-1822
		       Lucent Technologies
		       2000 Naperville Rd.
		       Room:  1A-365
		       Napeville, IL. 60566-7033
	*/

	double min, max, *ptr, *ptr1, *ptr2, *ptr3;
	double  d=0, pivot;
	long double sum;
	int out, *s, m1, n1, m2, n2, i, j, kr, kl, kount, in, k, l;
	bool stage, test;

    out = in = -123456; test = false;  // Warb initialize

	/* initialization */

	m1 = m + 1;
	n1 = n + 1;
	m2 = m + 2;
	n2 = n + 2;
	s = (int *)malloc(m*sizeof(int));
	for (j=0, ptr=a+m1*n2; j<n; j++, ptr++) {
		*ptr = (double)(j+1);
		x[j] = 0.0;
	}
	for (i=0, ptr=a+n1, ptr1=a+n, ptr2=a; i<m; i++, ptr += n2,
		ptr1 += n2, ptr2 += n2) {
		*ptr = n + i + 1;
		*ptr1 = b[i];
		if (b[i] <= 0.0)
			for (j=0, ptr3=ptr2; j<n2; j++, ptr3++)
				*ptr3 *= -1.0;
		e[i] = 0.0;
	}

	/* compute marginal costs */

	for (j=0, ptr=a; j<n1; j++, ptr++) {
		sum = (long double)0.0;
		for (i=0, ptr1=ptr; i<m; i++, ptr1 += n2) {
			sum += (long double)(*ptr1);
		}
		*(a+m*n2+j) = (double)sum;
	}

	stage = true;
	kount = -1;
	kr = kl = 0;
loop1:	if (stage) {

		/*
		   Stage I:
			Determine the vector to enter the basis.
		*/
	
		max = -1.0;
		for (j=kr, ptr=a+m1*n2+kr, ptr1=a+m*n2+kr; j<n; j++, ptr++,
			ptr1++) {
			if (fabs(*ptr)<=(double)n && (d=fabs(*ptr1))>max) {
				max = d;
				in = j;
			}
		}
		if (*(a+m*n2+in) < 0.0)
			for (i=0, ptr=a+in; i<m2; i++, ptr += n2)
				*ptr *= -1.0;
	} else {

		/*
		   Stage II
			Determine the vector to enter the basis.
		*/

		max = -BIG;
		for (j=kr, ptr=a+m*n2+kr; j<n; j++, ptr++) {
			d = *ptr;
			if (d >= 0.0 && d > max) {
				max = d;
				in = j;
			} else if (d <= -2.0) {
				d = -d - 2.0;
				if (d > max) {
					max = d;
					in = j;
				}
			}
		}
		if (max <= toler) {
			l = kl - 1;
			for (i=0, ptr=a+n, ptr1=a+kr; i<=l; i++, ptr += n2,
				ptr1 += n2)
				if (*ptr < 0.0)
					for (j=kr, ptr2=ptr1; j<n2; j++, ptr2++)
						*ptr2 *= -1.0;
			*(a+m1*n2+n) = 0.0;
			if (kr == 0) {
				for (j=0, ptr=a+m*n2; j<n; j++, ptr++) {
					d = fabs(*ptr);
					if (d <= toler || 2.0-d <= toler)
						goto end;
				}
				*(a+m1*n2+n) = 1.0;
			}
			goto end;
		} else if (*(a+m*n2+in) <= 0.0) {
			for (i=0, ptr=a+in; i<m2; i++, ptr += n2)
				*ptr *= -1.0;
			*(a+m*n2+in) -= 2.0;
		}
	}

	/* Determine the vector to leave the basis */

	for (i=kl, k = -1, ptr=a+kl*n2+in, ptr1=a+kl*n2+n; i<m; i++,
		ptr += n2, ptr1 += n2) {
		if (*ptr > toler) {
			k++;
			b[k] = (*ptr1)/(*ptr);
			s[k] = i;
			test = true;
		}
	}
loop2:	if (k <= -1)
		test = false;
	else {
		min = BIG;
		for (i=0; i<=k; i++)
			if (b[i] < min) {
				j = i;
				min = b[i];
				out = s[i];
			}
		b[j] = b[k];
		s[j] = s[k];
		k--;
	}

	/* check for linear dependence in stage I */

	if (!test && stage) {
		for (i=0, ptr=a+kr, ptr1=a+in; i<m2; i++, ptr += n2,
			ptr1 += n2) {
			d = *ptr;
			*ptr = *ptr1;
			*ptr1 = d;
		}
		kr++;
	} else if (!test) {
		*(a+m1*n2+n) = 2.0;
		goto end;
	} else {
		pivot = *(a+out*n2+in);
		if (*(a+m*n2+in) - pivot - pivot > toler) {
			for (j=kr, ptr=a+out*n2+kr, ptr1=a+m*n2+kr; j<n1; j++,
				ptr++, ptr1++) {
				d = *ptr;
				*ptr1 -= 2.0*d;
				*ptr = -d;
			}
			*(a+out*n2+n1) *= -1.0;
			goto loop2;
		} else {

			/* pivot on a[out][in]. */

			for (j=kr, ptr=a+out*n2+kr; j<n1; j++, ptr++)
				if (j != in)
					*ptr /= pivot;
			for (i=0, ptr=a+in, ptr1=a; i<m1; i++,
				ptr += n2, ptr1 += n2)
				if (i != out) {
					d = *ptr;
					for (j=kr, ptr2=ptr1+kr,
						ptr3=a+out*n2+kr; j<n1; j++,
						ptr2++, ptr3++)
						if (j != in)
							*ptr2 -= d*(*ptr3);
				}
			for (i=0, ptr=a+in; i<m1; i++, ptr += n2)
				if (i != out)
					*ptr /= -pivot;
			*(a+out*n2+in) = 1.0/pivot;
			d = *(a+out*n2+n1);
			*(a+out*n2+n1) = *(a+m1*n2+in);
			*(a+m1*n2+in) = d;
			kount++;
			if (stage) {

				/* interchange rows in stage I */

				kl++;
				for (j=kr,ptr=a+out*n2+kr,ptr1=a+kount*n2+kr;
					j<n2; j++, ptr++, ptr1++) {
					d = *ptr;
					*ptr = *ptr1;
					*ptr1 = d;
				}
			}
		}
	}
	if (kount + kr == n-1)
		stage = false;
	goto loop1;

	/* prepare for final return */

end:	for (i=0, ptr=a+n1, ptr1=a+n; i<m; i++, ptr += n2, ptr1 += n2) {
		k = (int)(*ptr);
		d = *ptr1;
		if (k < 0) {
			k = -k;
			d = -d;
		}
		k--;
		if (i < kl)
			x[k] = d;
		else {
			k -= n;
			e[k] = d;
		}
	}
	*(a+m1*n2+n1) = (double)(kount+1);
	*(a+m*n2+n1) = (double)(n1-kr -1);
	for (i=kl, ptr=a+kl*n2+n, sum=(long double)0.0; i<m; i++, ptr += n2)
		sum += (long double)(*ptr);
	*(a+m*n2+n) = (long double)sum;
	free((char *)s);
}


void Craig_etal_L1
     (
             Im2D_REAL8  A,
             Im1D_REAL8  B,
             REAL        TOLER,
             Im1D_REAL8  SOL,
             Im1D_REAL8  RESIDU
     )
{
    INT n = SOL.tx();
    INT m = B.tx();

    BENCH_ASSERT
    (
           (A.tx() == n+2)
        && (A.ty() == m+2)
        && (B.tx() == m)
        && (SOL.tx() == n)
        && (RESIDU.tx() == m)
    );
    Elise_Craig_Barrodale_Roberts_l1
    (
        m,n,
        A.data_lin(),
        B.data(),
        TOLER,
        SOL.data(),
        RESIDU.data()
    );
}


ElMatrix<REAL>  Optim_L1FormLin::BarrodaleSolve()
{
   Im2D_REAL8  A(_NbVar+2,_NbForm+2);
   Im1D_REAL8  B(_NbForm);
   Im1D_REAL8  SOL(_NbVar);
   Im1D_REAL8  RESIDU(_NbForm);

   REAL ** a = A.data();
   REAL *  b = B.data();

   for (INT x =0; x<_NbVar ; x++)
        for (INT y =0; y<_NbForm ; y++)
        {
             a[y][x] = _Flin(x,y);
             b[y] = _Flin(_NbVar,y);
        }

  Craig_etal_L1(A,B,1e-8,SOL,RESIDU);


  for (INT k=0; k<_NbVar ; k++)
      _BestSol(0,k) = SOL.data()[k];

   return _BestSol;

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
