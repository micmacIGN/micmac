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

#define NoTemplateOperatorVirgule


#include "StdAfx.h"
#include <iterator>



/*
void f()
{
   std::list<int> l;
   std::vector<int> v(l.begin(),l.end());
}
*/

template <class Type> const ElSTDNS vector<Type>  StdTmpVector (const ElSTDNS list<Type> & l)
{
    ElSTDNS vector<Type> v;
    v.clear();

    ElSTDNS copy(l.begin(),l.end(),ElSTDNS back_inserter(v));
    return v;
}



class ClassFoncMeanSquare
{
    public :
        ClassFoncMeanSquare
        (
            ElSTDNS list<Fonc_Num> Lfonc,
            Fonc_Num       Obs,
            Fonc_Num       Pds
        ) ;
        void             Cumul(Flux_Pts);
        ElMatrix<REAL8>  Sols();
        Fonc_Num         Approx();
        
    private :

        Im1D_REAL8        mSoms;
        REAL8 *           mDataS;
        INT               mNbVar;
        ElSTDNS vector<Fonc_Num>  mVars;
        ElSTDNS vector<Symb_FNum> mSVar;
        Fonc_Num          mObs;
        Symb_FNum         mSObs;
        Fonc_Num          mPds;
        Symb_FNum         mSPds;
        ElMatrix<REAL8>   mMatr;
        ElMatrix<REAL8>   mVect;
        Fonc_Num          mFGlob;
};


ClassFoncMeanSquare::ClassFoncMeanSquare
( 
    ElSTDNS list<Fonc_Num> Lfonc,
    Fonc_Num       Obs,
    Fonc_Num       Pds
) :
    mSoms   (1),
    mNbVar  ((int) Lfonc.size()),
    mVars   (StdTmpVector(Lfonc)), // Lfonc.begin(),Lfonc.end()),
    mSVar   (), // Lfonc.begin(),Lfonc.end()),
    mObs    (Obs),
    mSObs   (Obs),
    mPds    (Pds),
    mSPds   (Pds),
    mMatr   (mNbVar,mNbVar,0.0),
    mVect   (1,mNbVar,0.0),
    // mFGlob  (mSPds*mSVar[0]*mSObs)
    mFGlob  (0)
{
    std::copy(Lfonc.begin(),Lfonc.end(),std::back_inserter(mSVar));
    mFGlob  =mSPds*mSVar[0]*mSObs;

   INT SzVec =1;
   for (INT k=1; k<mNbVar; k++)
   {
       SzVec++;
       mFGlob = Virgule(mFGlob,mSPds*mSVar[k]*mSObs);
   }

   for (INT k1= 0; k1<mNbVar ; k1++)
       for (INT k2= k1; k2<mNbVar ; k2++)
       {
           SzVec++;
           mFGlob = Virgule(mFGlob,mSPds*mSVar[k1]*mSVar[k2]);
       }
   mSoms  = Im1D_REAL8(SzVec);
   mDataS = mSoms.data();
}


void ClassFoncMeanSquare::Cumul(Flux_Pts flux)
{
   ELISE_COPY(flux,mFGlob,sigma(mDataS,mSoms.tx()));

   INT SzVec =0;
   for (INT k=0; k<mNbVar; k++)
   {
       mVect(0,k) += mDataS[SzVec];
       SzVec++;
   }

   for (INT k1= 0; k1<mNbVar ; k1++)
       for (INT k2= k1; k2<mNbVar ; k2++)
       {
            mMatr(k1,k2) += mDataS[SzVec];
            if (k1 != k2)
               mMatr(k2,k1) += mDataS[SzVec];
           SzVec++;
       }
}

ElMatrix<REAL8>  ClassFoncMeanSquare::Sols()
{
   return gaussj(mMatr) * mVect;
}

Fonc_Num ClassFoncMeanSquare::Approx()
{
   ElMatrix<REAL8> s = Sols();
   Fonc_Num f =  s(0,0) * mVars[0];
   for (INT k=1 ; k<mNbVar ; k++)
       f = f +  s(0,k) * mVars[k];
   return f;
}


Fonc_Num SomPondFoncNum
         (
              ElSTDNS list<Fonc_Num> Lfonc,
              ElMatrix<REAL8>  s
         )      
{
   std::vector<Fonc_Num> Vars (StdTmpVector(Lfonc));
   ELISE_ASSERT((s.ty()==(INT)Vars.size())&&(s.tx()==1),"SomPondFoncNum, size diffs");
   if (! s.ty()) 
      return 0;

   Fonc_Num f =  s(0,0) * Vars[0];
   for (INT k=1 ; k<s.ty() ; k++)
       f = f +  s(0,k) * Vars[k];
   return f;
}

ElMatrix<REAL8> MatrFoncMeanSquare
                (
                     Flux_Pts       flux,
                     ElSTDNS list<Fonc_Num> Lfonc,
                     Fonc_Num       Obs,
                     Fonc_Num       Pds
                ) 
{
   ClassFoncMeanSquare  CFMS(Lfonc,Obs,Pds);
   CFMS.Cumul(flux);


   return CFMS.Sols();
}


Fonc_Num ApproxFoncMeanSquare
         (
            Flux_Pts       flux,
            ElSTDNS list<Fonc_Num> Lfonc,
            Fonc_Num       Obs,
            Fonc_Num       Pds
         ) 
{
   ClassFoncMeanSquare  CFMS(Lfonc,Obs,Pds);
   CFMS.Cumul(flux);

   return CFMS.Approx();
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
