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

#include "Digeo.h"



/****************************************/
/*                                      */
/*           cConvolSpec                */
/*                                      */
/****************************************/

template <class Type> class cSomFiltreSep :
                                   public Simple_OPBuf1<Type,Type>
{
    public :
        cSomFiltreSep( cConvolSpec<Type> * );
 
         void ShowLine(Type * aLine, Video_Win * aW)
         {
             int aSzX = this->x1() - this->x0();
             Im2D<Type,Type> anIm(aSzX,1);
             for (int anX = this->x0(); anX<this->x1() ; anX++)
             {
                  anIm.data()[0][anX-this->x0()] = aLine[anX];
             }
             int anY = this->ycur();
             ELISE_COPY
             (
                 rectangle(Pt2di(this->x0(),anY),Pt2di(this->x1(),anY+1)),
                 trans(anIm.in(),-Pt2di(this->x0(),anY)),
                 aW->ogray()
             );
         }

    private :
       void  calc_buf (Type  ** output,Type  *** input);

        
        void ConvolLine(int y);

        cConvolSpec<Type> * mConvol;
        ~cSomFiltreSep();
        int                 mDeb;
        int                 mFin;
        Type  **            mLineFiltered;
        Type  **            mInPut;

        Simple_OPBuf1<Type,Type> * dup_comp();
};

template <class Type> cSomFiltreSep<Type>::cSomFiltreSep(cConvolSpec<Type> * aCS ):
    mConvol  (aCS),
    mDeb     (aCS->Deb()),
    mFin     (aCS->Fin()),
    mLineFiltered (0),
    mInPut          (0)
{
    
}

template <class Type> cSomFiltreSep<Type>::~cSomFiltreSep()
{
   if (mLineFiltered)
   {
       DELETE_MATRICE(mLineFiltered,Pt2di(this->x0Buf(),this->y0Buf()),Pt2di(this->x1Buf(),this->y1Buf()));
   }
}

template <class Type> Simple_OPBuf1<Type,Type> * cSomFiltreSep<Type>::dup_comp()
{
   cSomFiltreSep<Type> * aRes = new cSomFiltreSep<Type>(mConvol);

   Pt2di aP0(this->x0Buf(),this->y0Buf());
   Pt2di aP1(this->x1Buf(),this->y1Buf());

   aRes->mLineFiltered =  NEW_MATRICE(aP0,aP1,Type);

   return aRes;
}

template <class Type> void cSomFiltreSep<Type>::ConvolLine(int y)
{

   mConvol->Convol(mLineFiltered[y],mInPut[y],this->x0(),this->x1());
}

template <class Type> void cSomFiltreSep<Type>::calc_buf (Type  ** output,Type  *** AllInput)
{
   mInPut = AllInput[0];
   // ShowLine(mInPut[0],aW2Digeo);
   if (this->first_line())
   {
      for (INT y=this->y0Buf(); y<this->y1Buf()-1 ; y++)
           ConvolLine(y);
   }
   ConvolLine(this->y1Buf()-1);

   // ShowLine(mLineFiltered[0],aW3Digeo);

  
  mConvol->ConvolCol(output[0],mLineFiltered,this->x0(),this->x1(),0);
  rotate_plus_data(mLineFiltered,this->y0Buf(),this->y1Buf());

}



Fonc_Num LinearSepFilter
         (
             Fonc_Num                aFonc,
             cConvolSpec<INT> *    aFiltrI,
             cConvolSpec<double> * aFiltrD
         )
{
  ELISE_ASSERT(aFiltrI->Sym() && aFiltrD->Sym(),"LinearSepFilter handle only symetric filter");

  bool IntF = aFonc.integral_fonc(true);
  int aD = IntF ? aFiltrI->Fin() :  aFiltrD->Fin() ;

  
/*
  if (aD != aFiltrD->Fin())
  {
      std::cout << "FILTRE, SzI " << aFiltrI->Fin() << " " << aFiltrD->Fin() << "\n";
      ELISE_ASSERT(false,"Incoh INT/DOUBLE in LinearSepFilter");
  }
*/

  return create_op_buf_simple_tpl
            (

                IntF ? new cSomFiltreSep<INT>(aFiltrI) : 0,
                IntF ? 0 : new cSomFiltreSep<double>(aFiltrD),
                aFonc,
                1,
                Box2di(aD)
            );

}

Fonc_Num GaussSepFilter(Fonc_Num   aFonc,double aSigma,double anEpsilon)
{
   if (aSigma==0) return aFonc;
   return LinearSepFilter
          (
              aFonc,
              IGausCS(aSigma,anEpsilon),
              RGausCS(aSigma,anEpsilon)
          );
}

/****************************************/
/*                                      */
/*           cConvolSpec                */
/*                                      */
/****************************************/

template <class Type> 
cConvolSpec<Type>::cConvolSpec(tBase* aFilter,int aDeb,int aFin,int aNbShit,bool ForGC) :
    mNbShift (aNbShit),
    mDeb      (aDeb),
    mFin      (aFin),
    mForGC    (ForGC),
    mDataCoeff (0),
    mSym       (true),
    mFirstGet  (true)
{
    for (int aK=aDeb; aK<= aFin ; aK++)
    {
       mCoeffs.push_back(aFilter[aK]);
    }

    mDataCoeff = (&(mCoeffs[0])) -aDeb;

    if (aFin+aDeb != 0)
    {
        mSym = false;
    }
    else
    {
        for (int aK=1 ; aK<= aDeb; aK++) // aK<=aFin ou le test est desactivé ?
        {
            if (ElAbs(mDataCoeff[aK]-mDataCoeff[-aK])>1e-6)
               mSym = false;
        }
    }
    theVec.push_back(this);
}

template <class Type>  typename El_CTypeTraits<Type>::tBase  *  cConvolSpec<Type>::DataCoeff()
{
   return mDataCoeff;
}


template <class Type>
bool cConvolSpec<Type>::Match(tBase *  aDFilter,int aDeb,int aFin,int  aNbShit,bool ForGC)
{
    if (   
           (aNbShit!=mNbShift)
        || (mDeb!=aDeb)
        || (mFin!=aFin)
        || (mForGC!=ForGC)
       )
       return false;

  for (int aK=aDeb; aK<=aFin ; aK++)
     if (ElAbs(mCoeffs[aK-aDeb]-aDFilter[aK]) >1e-4)
        return false;

  return true;
}

template <class Type> bool cConvolSpec<Type>::IsCompiled() const
{
   return false;
}

template <class Type>
cConvolSpec<Type> * cConvolSpec<Type>::GetExisting(tBase* aFilter,int aDeb,int aFin,int aNbShift,bool ForGC)
{
   for (int aK=0 ; aK<int(theVec.size()) ; aK++)
      if (theVec[aK]->Match(aFilter,aDeb,aFin,aNbShift,ForGC))
         return theVec[aK];

   return 0;
}

template <class Type>
cConvolSpec<Type> * cConvolSpec<Type>::GetOrCreate(tBase* aFilter,int aDeb,int aFin,int aNbShift,bool ForGC)
{
  cConvolSpec<Type> * aRes = GetExisting(aFilter,aDeb,aFin,aNbShift,ForGC);
  if (! aRes) 
  {
      aRes =    new  cConvolSpec<Type>(aFilter,aDeb,aFin,aNbShift,ForGC);
  }
  if (aRes->mFirstGet)
  {
     aRes->mFirstGet = false;
     // std::cout << "CodeIssCompiled " << aRes->IsCompiled() << "\n";
  }
  return   aRes;
}

inline void SelfShift(double &,const int &)  {}
inline void SelfShift(U_INT1 & aV,const int & aShft)
{
   aV >>= aShft;
}
inline void SelfShift(U_INT2 & aV,const int & aShft)
{
   aV >>= aShft;
}
inline void SelfShift(INT & aV,const int & aShft)
{
   aV >>= aShft;
}

template <class Type>
          void cConvolSpec<Type>::ConvolCol(Type * Out,Type **In,int aX0,int aX1,int anYIn)
{
    Type aV0 = mDataCoeff[0];
    Type * aL0 = In[anYIn];
    for (int anX = aX0; anX<aX1 ; anX++)
    {
          Out[anX] = aL0[anX] * aV0;
    }


    for (int aDY= (mSym?1:mDeb)  ; aDY<=mFin ; aDY++)
    {
        if (aDY)
        {
            Type *aLP = In[anYIn+aDY];
            Type  aVP = mDataCoeff[aDY];
            if (mSym)
            {
                Type *aLM = In[anYIn-aDY];
                for (int anX = aX0; anX<aX1 ; anX++)
                {
                   Out[anX] += aVP * (aLP[anX] + aLM[anX]) ;
                }
            }
            else
            {
                for (int anX = aX0; anX<aX1 ; anX++)
                {
                   Out[anX] += aVP * aLP[anX];
                }
            }
        }
    }
    for (int anX = aX0; anX<aX1 ; anX++)
    {
          Out[anX] = ShiftDr(Out[anX],mNbShift);
    }
}


template <class Type>
void cConvolSpec<Type>::Convol(Type * Out,Type * In,int aK0,int aK1)
{
   In += aK0;
   if (mSym)
   {
      for (int aK= aK0 ; aK<aK1 ; aK++)
      {
          tBase aRes = In[0] * mDataCoeff[0];
          for (int aDelta=1 ; aDelta <= mFin ; aDelta++)
          {
              aRes +=  (In[aDelta]+In[-aDelta]) * mDataCoeff[aDelta];
          }
          Out[aK] = ShiftDr(aRes,mNbShift);
          In++;
      }
   }
   else
   {
      for (int aK= aK0 ; aK<aK1 ; aK++)
      {
          tBase aRes = 0;
          for (int aDelta=mDeb ; aDelta <= mFin ; aDelta++)
          {
              aRes +=  In[aDelta] * mDataCoeff[aDelta];
          }
          Out[aK] = ShiftDr(aRes,mNbShift);
          In++;
      }
   }
}



template <class Type> int cConvolSpec<Type>::Deb() const {return mDeb;}
template <class Type> int cConvolSpec<Type>::Fin() const {return mFin;}
template <class Type> bool cConvolSpec<Type>::Sym() const {return mSym;}




template <> std::vector<cConvolSpec<U_INT1> *>  cConvolSpec<U_INT1>::theVec(0);
template <> std::vector<cConvolSpec<U_INT2> *>  cConvolSpec<U_INT2>::theVec(0);
template <> std::vector<cConvolSpec<REAL4> *>  cConvolSpec<REAL4>::theVec(0);
template <> std::vector<cConvolSpec<INT> *>  cConvolSpec<INT>::theVec(0);
template <> std::vector<cConvolSpec<REAL8> *>  cConvolSpec<REAL8>::theVec(0);



InstantiateClassTplDigeo(cConvolSpec)

//InstantiateClassTplDigeo(cTplImInMem)

/************************************/

// Nb = 4 ;; 1 24 248 990 1570 990 248 24 1 

#if (0)
class cConvolSpec_U_INT2_Sig_0_5_12 : public cConvolSpec<U_INT2>
{
   public :
      void Convol(U_INT2 * Out,U_INT2 * In,int aK0,int aK1)
      {
           In+=aK0;
           Out+=aK0;
           for (int aK=aK0; aK<aK1 ; aK++)
           {
                *(Out++) =  (
                                2048
                              +    1*(In[-4]+In[4])
                              +   24*(In[-3]+In[3])
                              +  248*(In[-2]+In[2])
                              +  990*(In[-1]+In[1])
                              + 1570*(In[0])
                          ) >> 12;

                 In++;
           }
      }


      cConvolSpec_U_INT2_Sig_0_5_12(int * aFilter) :
            cConvolSpec<U_INT2>(aFilter+4,-4,4,12,false)
      {
      }
};


void cAppliDigeo::InitConvolSpec()
{
    static bool theFirst = true;
    if (! theFirst) return;
    theFirst = false;

    {
       int theCoeff[9] = {1,24,248,990,1570,990,248,24,1};
       new cConvolSpec_U_INT2_Sig_0_5_12(theCoeff);
    }
}
#endif




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
