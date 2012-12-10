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



/******************************************************/
/******************************************************/
/**                                                  **/
/**               FoncNVarND                         **/
/**               NRLineMinimND                      **/
/**                                                  **/
/******************************************************/
/******************************************************/

     /*----------------------------------------------------*/
     /*                NRLineMinimND                       */
     /*----------------------------------------------------*/

template <class Type> class  NRLineMinimDer;

template <class Type> class  NRLineMinimND : public NROptF1vND
{
     public :
         friend class NRLineMinimDer<Type>;

         NRLineMinimND(FoncNVarND<Type>  &);
         void linmin(Type * p,Type *x,REAL * fret);

     private :
         void SetScal(REAL);
         void NRSetLine(Type * p ,Type * x);
         void OutLine(REAL x);

         REAL NRF1v(REAL);

         FoncNVarND<Type> &     mFNV;
         INT                    mNbVar;
         Im1D<Type,REAL>        mTmpVal;
         Type *                 mDataTmpV;
         Type  *                mP0;
         Type  *                mDir;
};

#define TOL 2.0e-6

template <class Type>  NRLineMinimND<Type>::NRLineMinimND(FoncNVarND<Type> & fnv) :
    mFNV       (fnv),
    mNbVar     (fnv.NbVar()),
    mTmpVal    (fnv.NbVar()),
    mDataTmpV  (mTmpVal.data()),
    mP0        (0),
    mDir       (0)
{
}


template <class Type> void NRLineMinimND<Type>::SetScal(REAL x)
{
    for (INT k=0; k<mNbVar ; k++)
       mDataTmpV[k] = (Type) (mP0[k] + x * mDir[k]);
}

template <class Type> void NRLineMinimND<Type>::NRSetLine(Type * p,Type * x)
{
    mP0 = p+1;
    mDir = x+1;
}

template <class Type> void NRLineMinimND<Type>::OutLine(REAL x)
{
    for (INT j=0;j<mNbVar;j++)
    {
        mDir[j] *= (Type) x;
        mP0[j]  += mDir[j];
    }
}


template <class Type> REAL NRLineMinimND<Type>::NRF1v(REAL x)
{
    SetScal(x);
    return mFNV.ValFNV(mDataTmpV);
}


template <class Type> void NRLineMinimND<Type>::linmin(Type * p,Type *x,REAL * fret)
{
    NRSetLine(p,x);
    Pt2dr  aP = brent();
    *fret = aP.y;
    OutLine(aP.x);
}



     /*----------------------------------------------------*/
     /*                FoncNVarND                          */
     /*----------------------------------------------------*/



template <class Type> FoncNVarND<Type>::~FoncNVarND() {}

template <class Type> FoncNVarND<Type>::FoncNVarND(INT NBVAR) :
    _NbVar   (NBVAR)
{
}

template <class Type> REAL FoncNVarND<Type>::NRValFNV(const Type * p)
{
    return ValFNV(p+1);
}

template <class Type> INT FoncNVarND<Type>::NbVar() const
{
    return _NbVar;
}


template <class Type> void FoncNVarND<Type>::powel
                           (
                                 Type *p,
                                 REAL ftol,
                                 int *iter,
                                 REAL * fret,
                                 INT ITMAX
                           )
{
     Im2D_REAL8 XI(_NbVar+1,_NbVar+1);
     ELISE_COPY(XI.all_pts(),FX==FY,XI.out());
     REAL ** xi = XI.data();

     INT n = _NbVar;

     Im1D<Type,REAL> aPT(_NbVar+1) ,PPT(_NbVar+1),XIT(_NbVar+1);

     int i,ibig,j;
     REAL t,fptt,fp,del;
     Type *pt = aPT.data(),*ptt = PPT.data(),*xit = XIT.data();

     NRLineMinimND<Type>  LM(*this);

	*fret=(NRValFNV)(p);
	for (j=1;j<=n;j++) pt[j]=p[j];
	for (*iter=1;;(*iter)++) {
		fp=(*fret);
		ibig=0;
		del=0.0;
		for (i=1;i<=n;i++) {
			for (j=1;j<=n;j++) xit[j]=(Type) (xi[j][i]);
			fptt=(*fret);
			LM.linmin(p,xit,fret);
			if (fabs(fptt-(*fret)) > del) {
				del=fabs(fptt-(*fret));
				ibig=i;
			}
		}
		if (2.0*fabs(fp-(*fret)) <= ftol*(fabs(fp)+fabs(*fret))) {
			return;
		}
		for (j=1;j<=n;j++) {
			ptt[j]=(Type) (2.0*p[j]-pt[j]);
			xit[j]=p[j]-pt[j];
			pt[j]=p[j];
		}
		fptt=(NRValFNV)(ptt);
		if (fptt < fp) {
			t=2.0*(fp-2.0*(*fret)+fptt)*ElSquare(fp-(*fret)-del)-del*ElSquare(fp-fptt);
			if (t < 0.0) {
			        LM.linmin(p,xit,fret);
				for (j=1;j<=n;j++) xi[j][ibig]=xit[j];
			}
		}
	}
}



template <class Type> INT FoncNVarND<Type>::powel
                          (
                               Type *           aM,
                               REAL             ftol,
                               INT              ITMAX
                          )
{
     INT iter;
     REAL fret;
     powel(aM-1,ftol,&iter,&fret,ITMAX);
     return iter;
}

template class FoncNVarND<REAL>;
template class NRLineMinimND<REAL>;

template class FoncNVarND<REAL4>;
template class NRLineMinimND<REAL4>;




/******************************************************/
/******************************************************/
/**                                                  **/
/**               FoncNVarDer                        **/
/**               NRLineMinimDer                      **/
/**                                                  **/
/******************************************************/
/******************************************************/

     /*----------------------------------------------------*/
     /*                NRLineMinimDer                      */
     /*----------------------------------------------------*/
template <class Type> class  NRLineMinimDer : public NROptF1vDer
{
     public :

         NRLineMinimDer(FoncNVarDer<Type> &);
         void dlinmin(Type * p,Type *x,REAL * fret);

     private :

         REAL NRF1v(REAL);
         REAL DerNRF1v(REAL);

         NRLineMinimND<Type>    mLMND;
         FoncNVarDer<Type> &    mFNV;
         Im1D<Type,REAL>        mTmpGrad;
         Type *                 mDataGrad;

};


template <class Type> NRLineMinimDer<Type>::NRLineMinimDer(FoncNVarDer<Type> & fnv) :
    mLMND         (fnv),
    mFNV          (fnv),
    mTmpGrad      (fnv.NbVar()),
    mDataGrad     (mTmpGrad.data())
{
}



template <class Type> REAL NRLineMinimDer<Type>::NRF1v(REAL x)
{
    return mLMND.NRF1v(x);
}




template <class Type> REAL NRLineMinimDer<Type>::DerNRF1v(REAL x)
{
    mLMND.SetScal(x);

    mFNV.GradFNV(mDataGrad,mLMND.mDataTmpV);

    REAL res = 0.0;
    for (INT k=0; k<mLMND.mNbVar ; k++)
       res += mDataGrad[k]  * mLMND.mDir[k];

    return res;
}




template <class Type> void NRLineMinimDer<Type>::dlinmin(Type * p,Type *x,REAL * fret)
{
    mLMND.NRSetLine(p,x);

    Pt2dr aP = brent();
    *fret = aP.y;
    mLMND.OutLine(aP.x);
}



     /*----------------------------------------------------*/
     /*                FoncNVarDer                         */
     /*----------------------------------------------------*/



template <class Type> FoncNVarDer<Type>::FoncNVarDer(INT NBVAR) :
   FoncNVarND<Type>(NBVAR)
{
}

template <class Type> void FoncNVarDer<Type>::NRGradFNV(const Type * v,Type *x)
{
    GradFNV(x+1,v+1);
}



#define EPS 1.0e-10


template <class Type> void FoncNVarDer<Type>::GradConj
                      (
                          Type *p,
                          REAL ftol,
                          INT *iter,
                          REAL *fret,
                          INT ITMAX
                       )
{
        INT n = this->_NbVar;
	int j,its;
	REAL  gg,gam,fp,dgg;

        NRLineMinimDer<Type>  LM(*this);

        Im1D<Type,REAL> aG(this->_NbVar+1) ,aH(this->_NbVar+1),aXI(this->_NbVar+1);
	Type *g=aG.data(),*h=aH.data(),*xi =aXI.data();


	fp= this->NRValFNV(p);
	NRGradFNV(p,xi);
	for (j=1;j<=n;j++) {
		g[j] = -xi[j];
		xi[j]=h[j]=g[j];
	}
	for (its=1;its<=ITMAX;its++) {
		*iter=its;
		LM.dlinmin(p,xi,fret);
		if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
			return;
		}
		fp=(this->NRValFNV)(p);
		NRGradFNV(p,xi);
		dgg=gg=0.0;
		for (j=1;j<=n;j++) {
			gg += g[j]*g[j];
/*		  dgg += xi[j]*xi[j];	*/
			dgg += (xi[j]+g[j])*xi[j];
		}
		if (gg == 0.0) {
			return;
		}
		gam=dgg/gg;
		for (j=1;j<=n;j++) {
			g[j] = -xi[j];
			xi[j]=h[j]=(Type) (g[j]+gam*h[j]);
		}
	}
}

template <class Type> INT FoncNVarDer<Type>::GradConj
                          (
                               Type *           aM,
                               REAL             ftol,
                               INT              ITMAX
                          )
{
     INT iter;
     REAL fret;
     GradConj(aM-1,ftol,&iter,&fret,ITMAX);
     return iter;
}


template class NRLineMinimDer<REAL>;
template class NRLineMinimDer<REAL4>;
template class FoncNVarDer<REAL>;
template class FoncNVarDer<REAL4>;

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
