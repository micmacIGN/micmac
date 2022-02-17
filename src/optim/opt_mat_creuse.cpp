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

/****************************************************/
/*                                                  */
/*             cElMatCreuseGen                      */
/*                                                  */
/****************************************************/



cElMatCreuseGen::cElMatCreuseGen(bool OptSym,INT aNbCol,INT aNbLig) :
   mOptSym (OptSym),
   mNbCol (aNbCol),
   mNbLig (aNbLig),
   mDiagPreCond (1),
   mDDPrec      (0)
{
}

void cElMatCreuseGen::Test()
{
   // ELISE_ASSERT(false,"cElMatCreuseGen::Test");
}

void cElMatCreuseGen::PrepPreCond()
{
   ELISE_ASSERT(mNbCol==mNbLig,"cElMatCreuseGen::PrepPreCond");
   mDiagPreCond.Resize(mNbCol);
   mDDPrec   =  mDiagPreCond.data();
   for (int aK=0 ; aK<mNbCol ; aK++)
          mDDPrec[aK] = LowGetElem(aK,aK);
}

void cElMatCreuseGen::VPCDo(double * aDOut,double * aDIn)
{
   ELISE_ASSERT(mNbCol==mNbLig,"cElMatCreuseGen::PrepPreCond");
   for (int aK=0 ; aK<mNbCol; aK++)
   {
      aDOut[aK] = aDIn[aK] / mDDPrec[aK];
   }
}

bool cElMatCreuseGen::OptSym() const
{
   return mOptSym;
}


Im1D<tSysCho,tSysCho>  Im16(const double * aD8,int aNb)
{
   Im1D<tSysCho,tSysCho> aRes(aNb);
   tSysCho * aD16 = aRes.data();
   for (int aK=0 ; aK<aNb ; aK++)
      aD16[aK] = aD8[aK];
   return aRes;
}

void  cElMatCreuseGen::MulVect8(double * out,const double * in) 
{
    Im1D<tSysCho,tSysCho> aIOut = Im16(out,mNbLig);
    Im1D<tSysCho,tSysCho> aIIn = Im16(in,mNbLig);
    MulVect(aIOut.data(),aIIn.data());

    tSysCho * aDOut16 = aIOut.data();
    for (int aK=0 ; aK<mNbLig ; aK++)
      out[aK] = aDOut16[aK];
}

Im1D<tSysCho,tSysCho> cElMatCreuseGen::MulVect(Im1D<tSysCho,tSysCho> aV) const
{
   ELISE_ASSERT(aV.tx() == mNbCol,"Bad Size in cElMatCreuseGen::MulVect");
   Im1D<tSysCho,tSysCho> aRes(mNbLig,0.0);
   // aRes.raz();
   MulVect(aRes.data(),aV.data());

   return aRes;
}

cElMatCreuseGen::~cElMatCreuseGen()
{
}

void cElMatCreuseGen::Verif(const std::string & aMes) 
{
    ELISE_ASSERT(false,"cElMatCreuseGen::Verif");
}


void cElMatCreuseGen::AddElem(INT aX,INT aY,REAL aVal)
{
   ELISE_ASSERT
   (
       (aX>=0) && (aX<mNbCol) && (aY>=0) && (aY<mNbLig),
       "Bad Indexe in  cElMatCreuseMap::AddElem"
   );

   if (aVal != 0.0)
      LowAddElem(aX,aY,aVal);
}

void cElMatCreuseGen::SetOffsets(const std::vector<INT> &)
{
}


void cElMatCreuseGen::AddLineInd
                      (
		           INT aKY,
		           INT aY,
			   REAL aCyP,
			   const std::vector<INT> & aVInd,
			   REAL * aCoeff
		       )
{
     INT aNbInd = (INT) aVInd.size();
     for (INT aK2=0 ; aK2<aNbInd ; aK2++)
     {
             INT aX = aVInd[aK2];
             REAL aCX = aCoeff[aK2];
             if (aCX)
                 AddElem(aX,aY,aCX*aCyP);
      }
}

void  cElMatCreuseGen::EqMatIndexee
      (
                          const std::vector<INT> & aVInd,
                          REAL aPds,REAL ** aMat
      ) 
{
    INT aNbInd = (INT) aVInd.size();
    for (INT aKY=0 ; aKY<aNbInd ; aKY++)
        for (INT aKX=0 ; aKX<aNbInd ; aKX++)
            AddElem(aVInd[aKY],aVInd[aKX],aMat[aKY][aKX] * aPds);

}

void cElMatCreuseGen::LowAddElem(INT aX,INT aY,REAL aVal) 
{
    if ((!mOptSym) || (aX>=aY))
       LowSetElem(aX,aY,LowGetElem(aX,aY)+aVal);
}

bool cElMatCreuseGen::IsOptForEcrireInMatr() const
{
   return false;
}

bool cElMatCreuseGen::IsOptForSousP3x3() const
{
   return false;
}

bool cElMatCreuseGen::IsOptForQuadSet0() const
{
   return false;
}

bool cElMatCreuseGen::IsOptForAddEqIndexee() const
{
   return false;
}

void cElMatCreuseGen::Indexee_EcrireDansMatrWithQuad
              (   ElMatrix<tSysCho> &aMatr,
                  const std::vector<cSsBloc> &  aVx,
                  const std::vector<cSsBloc> &  aVy
              )   const
{
    ELISE_ASSERT(false,"cElMatCreuseGen::Indexee_EcrireDansMatrWithQuad");
}

void cElMatCreuseGen::SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     )
{
    ELISE_ASSERT(false,"cElMatCreuseGen::SoutraitProduc3x3");
}
void cElMatCreuseGen::Indexee_QuadSet0 (const std::vector<cSsBloc> & aVIndx,
                                        const std::vector<cSsBloc> & aVIndy)
{
    ELISE_ASSERT(false,"cElMatCreuseGen::Indexee_QuadSet0");
}

void cElMatCreuseGen::VMAT_GSSR_AddNewEquation_Indexe
                      (
                        const std::vector<cSsBloc> * aVSB,
                        double *  FullCoeff,
                        int aNbTot,
                        REAL aPds,tSysCho *,REAL aB)
{
    ELISE_ASSERT(false,"cElMatCreuseGen::V_GSSR_AddNewEquation_Indexe");
}




/****************************************************/
/*                                                  */
/*             cElMatCreuseMap                      */
/*                                                  */
/****************************************************/

#include <map>


class cElMatCreuseMap : public cElMatCreuseGen
{
	public :
            cElMatCreuseMap(INT aNbCol,INT aNbLig);

	    tSysCho   LowGetElem(INT I,INT J) const ;
            void    LowSetElem(INT aX,INT aY,const tSysCho &);
	    void Reset();
            void PrecCondQuad(double *); 
	private:
            typedef std::pair<INT,REAL> tPair;
            typedef std::map<INT,REAL>  tLigne;
            typedef std::vector<tLigne> tMat;

	    tMat  mMat;

	    void  MulVect(tSysCho * out,const tSysCho * in) const;
	    void  tMulVect(tSysCho * out,const tSysCho * in) const;


};

cElMatCreuseMap::cElMatCreuseMap(INT aNbCol,INT aNbLig) :
    cElMatCreuseGen (false,aNbCol,aNbLig),
    mMat(aNbLig)
{
}

tSysCho  cElMatCreuseMap::LowGetElem(INT aX,INT aY) const
{
    return const_cast<cElMatCreuseMap&>(*this).mMat[aY][aX];
}

void cElMatCreuseMap::LowSetElem(INT aX,INT aY,const tSysCho & aVal)
{
    mMat[aY][aX] = aVal;
}

void cElMatCreuseMap::MulVect(tSysCho * out,const tSysCho * in) const
{
    INT aK=0;
    for 
    (
        tMat::const_iterator itLigne = mMat.begin();    
        itLigne !=mMat.end(); 
        itLigne ++
    )
    {
	*out = 0;
        for 
        (
            tLigne::const_iterator itElem = itLigne->begin();    
            itElem!= itLigne->end(); 
            itElem++
        )
        {
	    *out += itElem->second*in[itElem->first];
        }
	out++;
	aK++;
    }
}

void cElMatCreuseMap:: tMulVect(tSysCho * out,const tSysCho * in) const
{

    for (INT aK=0 ; aK<mNbLig ; aK++)
    {
	out[aK] = 0;
    }


    for 
    (
        tMat::const_iterator itLigne = mMat.begin();    
        itLigne !=mMat.end(); 
        itLigne ++
    )
    {
        for 
        (
            tLigne::const_iterator itElem = itLigne->begin();    
            itElem!= itLigne->end(); 
            itElem++
        )
        {
	    out[itElem->first] += *in * itElem->second ;
        }
	in++;
    }
}

void cElMatCreuseMap::PrecCondQuad(double * aD)
{
    for ( int aK=0 ; aK<int(mMat.size()) ; aK++)
    {
        for 
        (
            tLigne::iterator itElem = mMat[aK].begin();    
            itElem!= mMat[aK].end(); 
            itElem++
        )
        {
	    itElem->second  /= aD[aK]*aD[itElem->first]; 
        }
    }
}

void cElMatCreuseMap::Reset()
{
    for 
    (
        tMat::iterator itLigne = mMat.begin();    
        itLigne !=mMat.end(); 
        itLigne ++
    )
    {
        for 
        (
            tLigne::iterator itElem = itLigne->begin();    
            itElem!= itLigne->end(); 
            itElem++
        )
        {
	    itElem->second = 0;
        }
    }
}




/****************************************************/
/*                                                  */
/*             cElMatCreuseStrFixe                  */
/*                                                  */
/****************************************************/

/*
   Matrices adaptees aux systemes dans lesquels les inconnues sont les pixels
d'une image( eventuellement multi canal) et un pixel n'est en relation qu'avec sont
pixel voisin

Pas tres adaptes au systeme multi canal parce que par exemple si on numerote
dans le sens Canal varie d'abord, ensuite X ensuite Y

      012   345  678    ...

    Soit Tx Ty NbC la taille, l'indexe de x y c est :
   
         c + x *  NbC + y * NbC * Tx

 Le pb est que, notamment pour la sur resolution, on aura pas forcement les meme
strcuture suivant les cannaux, par exemple :

   les pixels d'un meme canal peuvent etre differents suivant 


*/

class cElMatCreuseStrFixe : public cElMatCreuseGen
{
	public :
               cElMatCreuseStrFixe (INT aNbVar);
	private :
	       void Show() const; 
               tSysCho  LowGetElem(INT aX,INT aY) const ;
               void    LowSetElem(INT aX,INT aY,const tSysCho &);


               virtual void Reset(); // Remet tous les elements a 0
               virtual void  MulVect(tSysCho * out,const tSysCho * in) const;
               virtual void  tMulVect(tSysCho * out,const tSysCho * in) const;
               void AddLineInd
                    (
		           INT aKY,
		           INT aY,
			   REAL aCyP,
			   const std::vector<INT> & aVInd,
			   REAL * aCoeff
                    );

	       void SetOffsets (const std::vector<INT> & aVOfset);
	       // void VerifOffset(const std::vector<INT> & aVOfset);

	        INT mNbVar;
		struct  cDiag
		{
		       Im1D_REAL8 mIm;
		       REAL8 *    mD;
                       INT mOffset;


		       cDiag(INT aNbVar,INT anOffset) :
			       mIm      (aNbVar,0.0),
		               mD       (mIm.data()),
			       mOffset  (anOffset)
		       {
		       }
		};

		std::vector<cDiag *>    mVDiags;
		std::vector<std::vector<cDiag *> >  mCurDiags;

               void  MulVectLine(tSysCho * out,const tSysCho * in,bool Transp) const;
               void  EqMatIndexee
                     (
                          const std::vector<INT> & aVInd,
                          REAL aPds,REAL ** aMat
                     ) ;
};





void cElMatCreuseStrFixe::Show() const
{
	INT aNbD = (INT) mVDiags.size();
	cout << "++ NB DIAGS = " << aNbD <<  "\n";
	for (INT aK =0 ; aK<aNbD ; aK++)
	{
            const cDiag & aDiag =  *mVDiags[aK];
            cout << "  Offset = " << aDiag.mOffset 
		    << " @Data : " << aDiag.mD << "\n";
            for (INT aK=0 ; aK<mNbVar ; aK++)
                cout << "  -   " << aK << " : " << aDiag.mD[aK] << "\n";
	}
}

void cElMatCreuseStrFixe::Reset()
{
   for (INT aKD=0; aKD<INT(mVDiags.size()) ; aKD++)
   {
        REAL8 * aD = mVDiags[aKD]->mD;
	for (INT aKY=0 ; aKY< mNbVar ; aKY++)
            aD[aKY] = 0;
   }
}


tSysCho  cElMatCreuseStrFixe::LowGetElem(INT aX,INT aY)  const
{
   ELISE_ASSERT
   (
       false,
       "No cElMatCreuseStrFixe::LowAddElem"
   );
   //return *((double*)0);
   // return *((double*)&aX); // this is bad news anyway but avoid clang's warning
   return (tSysCho)0; // should avoid any warning ?
}
void cElMatCreuseStrFixe::LowSetElem(INT aX,INT aY,const tSysCho &)
{
   ELISE_ASSERT
   (
       false,
       "No cElMatCreuseStrFixe::LowASetElem"
   );
}

cElMatCreuseStrFixe::cElMatCreuseStrFixe(INT aNbVar) :
    cElMatCreuseGen(false,aNbVar,aNbVar),
    mNbVar (aNbVar)
{
}

void  cElMatCreuseStrFixe::EqMatIndexee
      (
           const std::vector<INT> & aVInd,
           REAL aPds,REAL ** aMat
      ) 
{
     INT aNbInd = (INT) aVInd.size();
     for (INT aKY = 0 ; aKY<aNbInd ; aKY++)
     {
	 INT aY = aVInd[aKY];
         AddLineInd(aKY,aY,aPds,aVInd,aMat[aKY]);
     }
}

void cElMatCreuseStrFixe::AddLineInd
     (
		           INT aKY,
		           INT aY,
			   REAL aCyP,
			   const std::vector<INT> & aVInd,
			   REAL * aCoeff
     )
{
    std::vector<cDiag *> & aDiag = mCurDiags[aKY];
    for (INT aK=0 ; aK< INT(aVInd.size()) ; aK++)
    {
           aDiag[aK]->mD[aY]  += aCyP * aCoeff[aK];
    }
}

void cElMatCreuseStrFixe::SetOffsets(const std::vector<INT> & aVInd)
{
     mCurDiags.clear();
     for (INT aKY=0 ;aKY<INT(aVInd.size()) ; aKY++)
     {
         std::vector<cDiag *> aV0;
         mCurDiags.push_back(aV0);

	 INT anY = aVInd[aKY];
         for (INT aKX=0 ; aKX<INT(aVInd.size()) ; aKX++)
         {
	     INT anX = aVInd[aKX];
	     cDiag * pGot = 0;
	     INT anOffset = anX-anY;
	     for (INT aKD=0 ; aKD<INT(mVDiags.size()) ; aKD++)
	     {
                  cDiag * pDiag = mVDiags[aKD];
                  if (pDiag->mOffset == anOffset)
                     pGot = pDiag;
	     }
	     if (pGot==0)
	     {
                  mVDiags.push_back(new cDiag(mNbVar,anOffset));
		  pGot = mVDiags.back();
	     }
	     mCurDiags.back().push_back(pGot);
         }
     }

}

void  cElMatCreuseStrFixe::MulVectLine(tSysCho * out,const tSysCho * in,bool isTransp) const
{
      for (INT aY = 0 ; aY<mNbVar ; aY++)
          out[aY] = 0;

      for (INT aD = 0 ; aD<INT(mVDiags.size()) ; aD++)
      {
          INT anOff  = mVDiags[aD]->mOffset;
          REAL * aData  = mVDiags[aD]->mD;
          INT Y0 = ElMax(0,-anOff);
          INT Y1 = ElMin(mNbVar,mNbVar-anOff);

	  if (isTransp)
	  {
             for (INT aY = Y0 ; aY<Y1 ; aY++)
                 out[aY+anOff] +=  aData[aY] * in[aY];
	  }
	  else
	  {
             for (INT aY = Y0 ; aY<Y1 ; aY++)
                 out[aY] +=  aData[aY] * in[aY+anOff];
	  }
      }
}

void  cElMatCreuseStrFixe::MulVect(tSysCho * out,const tSysCho * in) const
{
     MulVectLine(out,in,false);
}

void  cElMatCreuseStrFixe::tMulVect(tSysCho * out,const tSysCho * in) const
{
     MulVectLine(out,in,true);
}


/****************************************************/
/*                                                  */
/*             cElMatCreuseBlocSym                  */
/*                                                  */
/****************************************************/

/*
   Biblio sur les inversion exacte par cholesky creux :

    http://en.wikipedia.org/wiki/Minimum_degree_algorithm  : algo d'ordonnancement
    p 57 de triggs sur la decomposition de cholesky par profil

*/

//  Les blocs diagonaux sont stockee comme les autres (dont moitie
//  inferieure "perdue")
 


class cBlocMCBS
{
    public :
       static bool IsNull (const cBlocMCBS *);
       bool IsNull () const;

       cBlocMCBS(Pt2di mNumBl,Pt2di aP0,Pt2di aP1);
       tSysCho & DataRel(int aX,int aY)
       {
/*
ELISE_ASSERT
(
   (aX>=0)  && (aY>=0) && (aX<mNbX) && (aY<mNbY),
   "cBlocMCBS::DataRel"
);
*/
           return mDataL[aX+mNbX*aY];
       }

       // Suppose aCol <= aCol2 deja realise, n'inclut pas le carre diag
       tSysCho ScalCol(int aCol,const cBlocMCBS &  aBl2,int aCol2) const;

       inline tSysCho * PtrDataAbs(int aX,int aY) 
       { 
            return mData2[aY]+aX; 
       }
       inline const  tSysCho *  PtrDataAbs(int aX,int aY)  const
       { 
            return mData2[aY]+aX; 
       }



       inline tSysCho & DataAbs(int aX,int aY) 
       { 
/*
ELISE_ASSERT
(
   (aX>=mX0)  && (aY>=mY0) && (aX<mX0+mNbX) && (aY<mY0+mNbY),
   "cBlocMCBS::DataAbs"
);
*/
            return mData2[aY][aX]; 
       }
       cBlocMCBS * XNext() const {return mXNextBl;}
       cBlocMCBS * YNextUp() const {return mYNextUpBl;}
       cBlocMCBS * YNextDown() const {return mYNextDownBl;}
       int NumBlX() const {return mNumBl.x;}
       int NumBlY() const {return mNumBl.y;}

       tSysCho * DataL();
       void   Reset();
       int    NbX() const;
       int    NbY() const;
       Pt2di  P0()  const;

       void SetXNextBloc(cBlocMCBS *);
       void SetYNextUpBloc(cBlocMCBS *);
       void SetYNextDownBloc(cBlocMCBS *);

    private :

       cBlocMCBS * mXNextBl;
       cBlocMCBS * mYNextUpBl;
       cBlocMCBS * mYNextDownBl;
       Pt2di     mNumBl;
       int       mX0;
       int       mY0;
       int       mNbX;
       int       mNbY;
       tSysCho *  mDataL;
       tSysCho ** mData2;
};

tSysCho cBlocMCBS::ScalCol(int aCol,const cBlocMCBS &  aBl2,int aCol2) const
{
   //  std::cout << "SC " << aCol <<  aCol2 << "\n";
   tSysCho aRes = 0;

   // Toute la col, 
   int aNb =  mNbY;
   // Sauf si c'est un element diag
   if (mX0==mY0)
   {
        aNb = aCol+1 - mY0;
        // Si c'est les deux meme colone , on supprime le carre de la diag
        if (aCol==aCol2)
           aNb--;
   }

   //  std::cout << "SC N " << aNb << "\n";
   const tSysCho * aData1 =  PtrDataAbs(aCol,mY0) ;
   const tSysCho * aData2 =  aBl2.PtrDataAbs(aCol2,aBl2.mY0) ;

   for (int aK=0 ; aK<aNb ; aK++)
   {
      aRes += (*aData1) * (*aData2);
      aData1 += mNbX;
      aData2 += aBl2.mNbX;
   }

   //  std::cout << "SCR " << aRes << "\n";

   return aRes;
}

int  cBlocMCBS::NbX() const {return mNbX;}
int  cBlocMCBS::NbY() const {return mNbY;}

void cBlocMCBS::SetXNextBloc(cBlocMCBS * aNB)
{
   mXNextBl = aNB;
}
void cBlocMCBS::SetYNextUpBloc(cBlocMCBS * aNB)
{
   mYNextUpBl = aNB;
}
void cBlocMCBS::SetYNextDownBloc(cBlocMCBS * aNB)
{
   mYNextDownBl = aNB;
}


Pt2di  cBlocMCBS::P0()  const
{
   return Pt2di(mX0,mY0);
}

bool cBlocMCBS::IsNull () const
{
   int aNb = mNbX*mNbY;
   for (int aK=0 ; aK<aNb ; aK++)
       if (mDataL[aK] != 0.0)
          return false;
    return true;
}

bool cBlocMCBS::IsNull (const cBlocMCBS * aBl) 
{
   return (aBl==0) || (aBl->IsNull());
}

void cBlocMCBS::Reset()
{
   MEM_RAZ(mDataL,mNbX*mNbY);
}

cBlocMCBS::cBlocMCBS(Pt2di aNumBl,Pt2di aP0,Pt2di aP1) :
    mXNextBl (0),
    mYNextUpBl (0),
    mYNextDownBl (0),
    mNumBl  (aNumBl),
    mX0   (aP0.x),
    mY0   (aP0.y),
    mNbX  (aP1.x-aP0.x),
    mNbY  (aP1.y-aP0.y),
    mDataL (new tSysCho [mNbX*mNbY]),
    mData2 (new tSysCho * [mNbY])
{
   for (int aK=0 ; aK<mNbY ; aK++)
      mData2[aK] = mDataL + aK * mNbX - aP0.x;
   mData2 -= aP0.y;
   Reset();
}

tSysCho * cBlocMCBS::DataL()
{
   return mDataL;
}

//  Classe pour les matrice creuses symetrique ayant une structure de bloc
//  "naturelle".
//
//  Comme dab : mDataBloc[Y][X], et on doit avoir X>=Y
//
//  Plus tard, si on veut optimiser la mult (hyper majorite de bloc vide)
//  on pourra faire des liste chainee de blocs, a initialiser avant la mult
//  (avec un compteur de bloc pour savoir si a jour)

typedef enum
{
   ePrecCondJacobi,
   ePrecCondBlock,
   ePrecCondSSOR,
   eCholesNoPrec
} eModePreCond;

class cElMatCreuseBlocSym : public cElMatCreuseGen
{
	public :
               cElMatCreuseBlocSym 
               (
                    const  std::vector<cIncIntervale *> &  Blocs,
                    const  std::vector<int> &              ISolve2Bloc,
                    eModePreCond                           aMode
               );
               virtual void PrepPreCond();
               void ChainageBlocs();
               virtual void VPCDo(double * aDOut,double * aDIn);

               int NumBl0OfCol(int aCol) const;

               bool DirectInverse(const tSysCho *,tSysCho *);

	private :
               void TestCholesky() const;
               void CalculCholesky();
               void Verif(const std::string & aMes) ;
	       // void Show() const; 

               tSysCho   LowGetElem(INT aX,INT aY) const ;
               void      LowSetElem(INT aX,INT aY,const tSysCho &) ;


// Optimise
               tSysCho  ScaleCol(int aCol1,int aCol2);
// Inefficace, mais pour verif
               tSysCho  SimpleScaleCol(int aCol1,int aCol2);

               tSysCho  SimpleGet(INT aX,INT aY) const ;   // Test basique
               virtual void Reset(); // Remet tous les elements a 0
               void VerifScal(); // Remet tous les elements a 0
               void ShowStruct(bool Basic); // Remet tous les elements a 0


               // Pour tester les inversion partielles utilisees dans SSOR on a besoin 	de multiplie par des ss ensemble
               void TestMulVect(tSysCho * out,const tSysCho * in,tSysCho aPdsInf,tSysCho aPdsDiag,tSysCho aPdsSup) const;

               virtual void  MulVect(tSysCho * out,const tSysCho * in) const;


               virtual void  tMulVect(tSysCho * out,const tSysCho * in) const
               {
                   ELISE_ASSERT(false,"cElMatCreuseBlocSym::tMulVec");
               }


               cBlocMCBS * BlocOfKbxKby(int aKBx,int aKBy) const;

               const  std::vector<cIncIntervale *> &  mStrBlocs;
               const  std::vector<int> &              mISolve2Bloc;
               int                                    mNbEl;
               int                                    mNbBlocs;
               cBlocMCBS ***                          mDataBlocs;
               cBlocMCBS **                           mBloc0OfCol;
               eModePreCond                           mModePC;

           //  Matrice de Pre-Cond par Bloc
               cElMatCreuseBlocSym *                  mMatPCB;

           //  Matrice de Cholesky 
               cElMatCreuseBlocSym *                  mMChol;

               bool IsOptForSousP3x3() const 
               { 
                   return true; 
               };
               void Test();

               bool IsOptForAddEqIndexee() const
               { 
                   return true; 
               };
               bool IsOptForEcrireInMatr() const
               {
                  return true;
               }

               bool IsOptForQuadSet0() const
               {
                  return true;
               }
               void SoutraitProduc3x3
                    (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                    );


              void VMAT_GSSR_AddNewEquation_Indexe
                      (
                        const std::vector<cSsBloc> * aVSB,
                        double *  FullCoeff,
                        int aNbTot,
                        REAL aPds,tSysCho *,REAL aB);


              void  Indexee_EcrireDansMatrWithQuad
                    (
                           ElMatrix<tSysCho> &aMatr,
                           const std::vector<cSsBloc> &  aVx,
                           const std::vector<cSsBloc> &  aVy
                     ) const;

              void Indexee_QuadSet0 (const std::vector<cSsBloc> & aVIndx,
                                     const std::vector<cSsBloc> & aVIndy);


              void SolveUperSys(tSysCho * aDOut,const tSysCho * aDIn,tSysCho aW);
              void SolveLowerSys(tSysCho * aDOut,const tSysCho * aDIn,tSysCho aW);
              void SolveDiagSys(tSysCho * aDOut,const tSysCho * aDIn,tSysCho aW);
};
/*
*/


void cElMatCreuseBlocSym::Indexee_QuadSet0 (const std::vector<cSsBloc> & aVx,
                                            const std::vector<cSsBloc> & aVy)
{
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        const cSsBloc & aBlX = aVx[aKx];
        int aI0x = aBlX.I0AbsSolve();
        int aNbX = aBlX.Nb();
        const cIncIntervale * aIntX  = aBlX.Intervale();
        int aNumBlocIntervX    = aIntX->NumBlocSolve();

        for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
        {
            const cSsBloc & aBlY = aVy[aKy];
            int aI0y = aVy[aKy].I0AbsSolve();



            if ((aI0x>=aI0y) || (!mOptSym))
            {
                const cIncIntervale * aIntY  = aBlY.Intervale();
                int aNumBlocIntervY    = aIntY->NumBlocSolve();
                cBlocMCBS * aBlocOut = BlocOfKbxKby(aNumBlocIntervX, aNumBlocIntervY);
                int aI1y = aBlY.I1AbsSolve();
                for (int aYIn=aI0y ; aYIn<aI1y ; aYIn++)
                {
                    MEM_RAZ (aBlocOut->PtrDataAbs(aI0x,aYIn), aNbX);
                }
            }
        }
    }

}


void cElMatCreuseBlocSym::ChainageBlocs()
{
    
    for (int aKB=0 ; aKB<mNbBlocs ; aKB++)
    {
        cBlocMCBS * aB0 =  mDataBlocs[aKB][aKB];
        if (aB0==0)
        {
           std::cout << "Bloc Nul " << aKB 
                     << " I-Alloc " << mStrBlocs[aKB]->I0Alloc() 
                     <<         " " << mStrBlocs[aKB]->I1Alloc() << "\n";
        }
    }
 //========= CHAINAGE DES BLOCS 
    for (int aKB=0 ; aKB<mNbBlocs ; aKB++)
    {

        cBlocMCBS * aB0 =  mDataBlocs[aKB][aKB];
        {
           int aKBy = aKB;
           if (aB0==0)
           {
              std::cout << "Bloc " << aKB << " on " << mNbBlocs << "\n";
              ELISE_ASSERT(false,"Bloc Diag0 cElMatCreuseBlocSym::PrepPreCond Chainage");
           }
           for (int aKBx=aKBy+1 ; aKBx<mNbBlocs ; aKBx++)
           {
                cBlocMCBS * aNextBl = mDataBlocs[aKBy][aKBx];
                if (!cBlocMCBS::IsNull(aNextBl))
                {
                    aB0->SetXNextBloc(aNextBl);
                    aB0 = aNextBl;
                }
           }
           aB0->SetXNextBloc(0);
        }

        aB0 =  mDataBlocs[aKB][aKB];
        {
           aB0->SetYNextDownBloc(0);
           int aKBx = aKB;
           ELISE_ASSERT(aB0!=0,"cElMatCreuseBlocSym::PrepPreCond Chainage");
           for (int aKBy=aKBx-1 ; aKBy>=0 ; aKBy--)
           {
                cBlocMCBS * aNextBl = mDataBlocs[aKBy][aKBx];
                if (!cBlocMCBS::IsNull(aNextBl))
                {
                    aB0->SetYNextUpBloc(aNextBl);
                    aNextBl->SetYNextDownBloc(aB0);
                    aB0 = aNextBl;
                }
           }
           aB0->SetYNextUpBloc(0);
           mBloc0OfCol[aKBx] = aB0;
        }
    }

}

void cElMatCreuseBlocSym::PrepPreCond()
{

    ChainageBlocs();

 //======================================
    if (mModePC == ePrecCondBlock)
    {
        if (mMatPCB==0)
        {
            mMatPCB =  new cElMatCreuseBlocSym(mStrBlocs,mISolve2Bloc,ePrecCondBlock);
        }
        for (int aKB=0 ; aKB<mNbBlocs ; aKB++)
        {
              cBlocMCBS *  aBIn = BlocOfKbxKby(aKB,aKB);
              cBlocMCBS *  aBOut = mMatPCB->BlocOfKbxKby(aKB,aKB);
              // VUGCC4.6 Pt2di aP0 = aBIn->P0();
              int aSz = aBIn->NbX();
              ELISE_ASSERT
              (
                      (aBIn->NbY()==aSz)
                   && (aBOut->NbX()==aSz)
                   && (aBOut->NbY()==aSz),
                   "cElMatCreuseBlocSym::PrepPreCond"
              );
              ElMatrix<tSysCho> aMat(aSz,aSz);
              for (int aKy = 0 ; aKy < aSz; aKy++)
              {
                  for (int aKx =  aKy ; aKx < aSz; aKx++)
                  {
                       aMat(aKy,aKx) = aMat(aKx,aKy) = aBIn->DataRel(aKx,aKy);
                  }
              }
              self_gaussj(aMat);
              for (int aKy = 0 ; aKy < aSz; aKy++)
              {
                  for (int aKx = 0 ; aKx < aSz; aKx++)
                  {
                      aBOut->DataRel(aKx,aKy) = aMat(aKx,aKy);
                  }
              }
        } 
    }
    else if ((mModePC == ePrecCondJacobi) || (mModePC == ePrecCondSSOR))
    {
       cElMatCreuseGen::PrepPreCond();
    }

    //  VerifScal();
}

bool cElMatCreuseBlocSym::DirectInverse(const tSysCho * aDIn,tSysCho * aDOut)
{
    if (ShowCholesky)
    {
       ShowStruct(true);
    }
    ChainageBlocs();
    CalculCholesky();

    if (ShowCholesky)
    {
        std::cout << "MAT \n";
        std::cout << "CHOL \n";
        mMChol->ShowStruct(true);
    }

   
    for (int aK=0 ; aK<mNbEl ; aK++)
    {
        aDOut[aK] = 0;
    }

    for (int anIter = 0 ; anIter<2 ; anIter++)
    {
        Im1D<tSysCho,tSysCho>  aMOut(mNbEl);
        tSysCho * aDMO = aMOut.data();
        MulVect(aDMO,aDOut);

        Im1D<tSysCho,tSysCho>  aIEr(mNbEl);
        tSysCho * aDEr = aIEr.data();

        tSysCho aSomEr = 0;
        for (int aK=0 ; aK<mNbEl ; aK++)
        {
            aDEr[aK] = aDIn[aK] - aDMO[aK];
            aSomEr += ElAbs(aDEr[aK]);
        }
        // std::cout << "ER = " << aSomEr << "\n";

        Im1D<tSysCho,tSysCho> aImTmp(mNbEl);
        tSysCho * aDTmp = aImTmp.data();
        Im1D<tSysCho,tSysCho> aMInvEr(mNbEl);
        tSysCho * aDMInvEr = aMInvEr.data();

        mMChol->SolveLowerSys(aDTmp,aDEr,1.0);
        mMChol->SolveUperSys(aDMInvEr,aDTmp,1.0);

/*
   a 0 0   a b d      a*a 
   b c 0 * 0 c e          b2+c2
   d e f   0 0 f                 d2+e2+f2


   a b c   1/a   -b/ad
   0 d e   0     1/d    -e/fd
   0 0 f   0     0      1/f
*/
/*
if (MPD_MM())
{
    for (int aK=0 ; aK<mNbCol ; aK++)
    {
       double aSom=0;
       for (int aI=0 ; aI<mNbCol ; aI++)
           aSom += ElSquare(mMChol->SimpleGet(aK,aI));
       std::cout << "DIAGCHO " <<  aSom  << " " << mNbEl << " " << mNbCol << "\n";
    }
}
*/
        for (int aK=0 ; aK<mNbEl ; aK++)
        {
            aDOut[aK] += aDMInvEr[aK];
        }

    }


    return true;
}

/*
//   (D/W + L) D-1 (D/W + tL)   W/(2-W)
*/

// Inverse le systeme triangulaire superiereure (avec diagonale divise par W)
void cElMatCreuseBlocSym::SolveLowerSys(tSysCho * aDOut,const tSysCho * aDIn,tSysCho aW)
{
   for (int aKBy=0 ; aKBy<mNbBlocs   ; aKBy++)
   {
       int anY0 = mStrBlocs[aKBy]->I0Solve();
       int anY1 = mStrBlocs[aKBy]->I1Solve();
       for (int anY=anY0 ; anY<anY1 ; anY++)
       {
           tSysCho aSom = aDIn[anY];
           cBlocMCBS * aBl0 = mDataBlocs[aKBy][aKBy];
// std::cout << "TATA -----------------\n";
           for (cBlocMCBS * aBl = aBl0;  aBl ; aBl = aBl->YNextUp())
           {
                int aKx = aBl->NumBlY();

                int anX0 =  mStrBlocs[aKx]->I0Solve();
                int anX1 =  (aBl==aBl0) ? anY : mStrBlocs[aKx]->I1Solve();
                for (int anX = anX0 ; anX < anX1 ; anX++)
                {
                     aSom -= aDOut[anX] * aBl->DataAbs(anY,anX);
                }
           }
           aDOut[anY] = (aSom * aW) / mDDPrec[anY];
       }
   }
}



void cElMatCreuseBlocSym::SolveUperSys(tSysCho * aDOut,const tSysCho * aDIn,tSysCho aW)
{
   for (int aKBy=mNbBlocs-1; aKBy>=0  ; aKBy--)
   {
       int anY0 = mStrBlocs[aKBy]->I0Solve();
       int anY1 = mStrBlocs[aKBy]->I1Solve();
       for (int anY=anY1-1 ; anY>=anY0 ; anY--)
       {
           tSysCho aSom = aDIn[anY];
           cBlocMCBS * aBl0 = mDataBlocs[aKBy][aKBy];
           for (cBlocMCBS * aBl = aBl0;  aBl ; aBl = aBl->XNext())
           {
                int aKx = aBl->NumBlX();
                int anX0 = (aBl==aBl0) ? (anY+1) : mStrBlocs[aKx]->I0Solve();
                int anX1 = mStrBlocs[aKx]->I1Solve();
                for (int anX = anX0 ; anX < anX1 ; anX++)
                {
                     aSom -= aDOut[anX] * aBl->DataAbs(anX,anY);
                }
           }
           aDOut[anY] = (aSom * aW) / mDDPrec[anY];
       }
   }
}



void cElMatCreuseBlocSym::VPCDo(double * aDOut,double * aDIn)
{
    // A priori devenu inutile et incompatible avec  tSysCho
    ELISE_ASSERT(false,"No more cElMatCreuseBlocSym::VPCDo");
}
/*
void cElMatCreuseBlocSym::VPCDo(double * aDOut,double * aDIn)
{
    if (mModePC==ePrecCondBlock)
    {
          mMatPCB->MulVect(aDOut,aDIn);
    }
    else if (mModePC==ePrecCondJacobi)
    {
        cElMatCreuseGen::VPCDo(aDOut,aDIn);
    }
    else if (mModePC == ePrecCondSSOR)
    {
        double aW = 0.5;
        Im1D_REAL8 aImTmp(mNbCol);
        double * aDTmp = aImTmp.data();
        SolveUperSys(aDTmp,aDIn,aW);

        double aWCorr = (2-aW)/aW;
        for (int aK=0 ; aK<mNbCol ; aK++)
        {
            aDTmp[aK] = aDTmp[aK] * mDDPrec[aK] * aWCorr;
        }
        SolveLowerSys(aDOut,aDTmp,aW);
    }
    
}
*/



void  cElMatCreuseBlocSym::Indexee_EcrireDansMatrWithQuad
     (
            ElMatrix<tSysCho> &aMatr,
            const std::vector<cSsBloc> &  aVx,
            const std::vector<cSsBloc> &  aVy
     )  const
{
    if (mOptSym)
    {
       // Le fait de ne remplir que la matrice superieur n'a de sens
       // que si elle est carre ou de type "ligne par bloc" (auquel cas
       // c'est sans effet
        ELISE_ASSERT
        (
            (aVx.size()==1) || (&aVx==&aVy),
            "Incoherence in sym optim"
        );
    }
    tSysCho ** aDOut = aMatr.data();
    int xout=0;
    for (int aKx=0 ; aKx <int(aVx.size()) ; aKx++)
    {
        const cSsBloc & aBlX = aVx[aKx];
        int aI0x = aBlX.I0AbsSolve();
        int aNbX = aBlX.Nb();
        const cIncIntervale * aIntX  = aBlX.Intervale();
        int aNumBlocIntervX    = aIntX->NumBlocSolve();

        int yout=0;
        for (int aKy=0 ; aKy <int(aVy.size()) ; aKy++)
        {
            const cSsBloc & aBlY = aVy[aKy];
            int aI0y = aVy[aKy].I0AbsSolve();

            if ((aI0x>=aI0y) || (!mOptSym))
            {
                bool toTransp = (aKx<aKy) && mOptSym &&  (&aVx==&aVy);
                const cIncIntervale * aIntY  = aBlY.Intervale();
                int aNumBlocIntervY    = aIntY->NumBlocSolve();
                cBlocMCBS * aBlocOut = BlocOfKbxKby(aNumBlocIntervX, aNumBlocIntervY);
                int aI1y = aBlY.I1AbsSolve();
                // Le cas de la transposition est en pratique tres minoritaire, on ne l'optimise
                // donc pas (il s'est manifeste la 1ere fois sur le bug piazzabra)
                if (toTransp)
                {
                    int aNbY = aI1y - aI0y;
                    for (int aKX = 0 ; aKX<aNbX  ; aKX++)
                    {
                       for (int aKY = 0 ; aKY<aNbY  ; aKY++)
                       {
                            aMatr(yout+aKY,xout+aKX) =  aBlocOut->DataAbs(aI0x+aKX,aI0y+aKY);
                       }
                    }
                    yout += aNbY;
                }
                else
                {
                   for (int aYIn=aI0y ; aYIn<aI1y ; aYIn++)
                   {
                       memcpy
                       (
                           aDOut[yout]+xout,
                           aBlocOut->PtrDataAbs(aI0x,aYIn),
                           aNbX*sizeof(tSysCho) 
                       );
                       yout++;
                   }
                }
            }
            else
            {
                yout += aBlY.Nb();
            }
        }
        xout += aNbX;
    }
}


void cElMatCreuseBlocSym::VMAT_GSSR_AddNewEquation_Indexe
     (
         const std::vector<cSsBloc> * aVSB,
         double *  aFullCoeff,
         int aNbTot,
         REAL aPds,tSysCho * aDataLin,REAL aB
    )
{

   tSysCho a2P =  2 * aPds ;
   tSysCho a2PB =  a2P *aB;
   int aNbBl = (int)aVSB->size();

   int Y0InBloc =0;
   for (int aKBly=0 ; aKBly <aNbBl ; aKBly++)
   {
        const cSsBloc & aBlY = (*aVSB)[aKBly];
        const cIncIntervale * aIntY  = aBlY.Intervale();
        int aNumBlocIntervY    = aIntY->NumBlocSolve();

        int aI0y = aBlY.I0AbsSolve();
        int aI1y = aBlY.I1AbsSolve();


        int X0InBloc =0;
        for (int aKBlx=0 ; aKBlx <aNbBl ; aKBlx++)
        {
            const cSsBloc & aBlX = (*aVSB)[aKBlx];
            const cIncIntervale * aIntX  = aBlX.Intervale();
            int aNumBlocIntervX    = aIntX->NumBlocSolve();
            if ((!mOptSym) || (aNumBlocIntervX>=aNumBlocIntervY))
            {
               cBlocMCBS * aBlocOut=BlocOfKbxKby(aNumBlocIntervX,aNumBlocIntervY);

               int aI0x = aBlX.I0AbsSolve();
               int aI1x = aBlX.I1AbsSolve();
               int yin = Y0InBloc;

               for (int aYOut = aI0y; aYOut<aI1y ; aYOut++)
               {
                   tSysCho aPCV1 =  a2P * aFullCoeff[yin++];
                   int aDebX  = (mOptSym && (aNumBlocIntervX==aNumBlocIntervY)) ? aYOut : aI0x;

                   int xin = X0InBloc + aDebX-aI0x;
                   tSysCho * aLineOut = aBlocOut->PtrDataAbs(aDebX,aYOut);
                   for (int XOut = aDebX ; XOut<aI1x ; XOut++)
                   {
                       *(aLineOut++)  += aPCV1 * aFullCoeff[xin++];
                   }
               }

            }
            X0InBloc += aBlX.Nb();
        }

        int yin = Y0InBloc;

        for (int YOut=aI0y ; YOut<aI1y ; YOut++)
        {
            aDataLin[YOut] -=   aFullCoeff[yin++] * a2PB;
        }
        Y0InBloc += aBlY.Nb();
   }
}



void cElMatCreuseBlocSym::SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     )
{
   tSysCho ** aData1 = aM1.data();
   tSysCho ** aData2 = aM2.data();

   tSysCho * aL2A = aData2[0];
   tSysCho * aL2B = aData2[1];
   tSysCho * aL2C = aData2[2];
   int aNbBl = (int)aYVSB->size();
   int Y0InBloc = 0;

   for (int aKBly=0 ; aKBly <aNbBl ; aKBly++)
   {
        const cSsBloc & aBlY = (*aYVSB)[aKBly];
        const cIncIntervale * aIntY  = aBlY.Intervale();
        int aNumBlocIntervY    = aIntY->NumBlocSolve();

        int aI0y = aBlY.I0AbsSolve();
        int aI1y = aBlY.I1AbsSolve();


        int X0InBloc =0;
        for (int aKBlx=0 ; aKBlx <aNbBl ; aKBlx++)
        {
            const cSsBloc & aBlX = (*aYVSB)[aKBlx];
            const cIncIntervale * aIntX  = aBlX.Intervale();
            int aNumBlocIntervX    = aIntX->NumBlocSolve();
            if ((!mOptSym) || (aNumBlocIntervX>=aNumBlocIntervY))
            {
               cBlocMCBS * aBlocOut=BlocOfKbxKby(aNumBlocIntervX,aNumBlocIntervY);

               int aI0x = aBlX.I0AbsSolve();
               int aI1x = aBlX.I1AbsSolve();
               int yin = Y0InBloc;

               for (int aYOut = aI0y; aYOut<aI1y ; aYOut++,yin++)
               {
                   tSysCho A1 = aData1[yin][0];
                   tSysCho B1 = aData1[yin][1];
                   tSysCho C1 = aData1[yin][2];

                   int aDebX  = (mOptSym && (aNumBlocIntervX==aNumBlocIntervY)) ? aYOut : aI0x;

                   int xin = X0InBloc + aDebX-aI0x;
                   tSysCho * aLineOut = aBlocOut->PtrDataAbs(aDebX,aYOut);
                   for (int XOut = aDebX ; XOut<aI1x ; XOut++)
                   {
                       *aLineOut -= A1 * aL2A[xin] + B1 * aL2B[xin] + C1 *aL2C[xin];
                       xin++;
                       aLineOut++;
                   }
               }

            }
            X0InBloc += aBlX.Nb();
        }

        Y0InBloc += aBlY.Nb();
   }

}

void cElMatCreuseBlocSym::Verif(const std::string & aMes) 
{
    ELISE_ASSERT(false,"cElMatCreuseBlocSym::Verif::Verif");
}

typedef cBlocMCBS * tBlP;
typedef cBlocMCBS ** tBlPP;

               // eModePreCond                           mModePC; ePrecCondJacobi, ePrecCondBlock, ePrecCondSSOR




cElMatCreuseBlocSym::cElMatCreuseBlocSym 
(
      const  std::vector<cIncIntervale *> &  strBlocs,
      const  std::vector<int> &              ISolve2Bloc,
      eModePreCond                           aMode
) :
  cElMatCreuseGen  (true, (int)ISolve2Bloc.size(), (int)ISolve2Bloc.size()),
  mStrBlocs        (strBlocs),
  mISolve2Bloc          (ISolve2Bloc),
  mNbEl            ((int)ISolve2Bloc.size()),
  mNbBlocs         ((int)strBlocs.size()),
  mBloc0OfCol      (new tBlP [mNbBlocs]),
  mModePC          (aMode),
  mMatPCB          (0),
  mMChol           (0)
{   

   // Verifiction
   ELISE_ASSERT(strBlocs.size()!=0,"Er1 in cElMatCreuseBlocSym::cElMatCreuseBlocSym");
   ELISE_ASSERT(strBlocs[0]->I0Solve() ==0,"Er2 in cElMatCreuseBlocSym::cElMatCreuseBlocSym");
   ELISE_ASSERT(strBlocs[mNbBlocs-1]->I1Solve() ==int(mISolve2Bloc.size()),"Er3 in cElMatCreuseBlocSym::cElMatCreuseBlocSym");
   for (int aKb = 1 ; aKb< mNbBlocs  ; aKb++)
   {
       ELISE_ASSERT
       (
            strBlocs[aKb-1]->I1Solve() ==strBlocs[aKb]->I0Solve(),
            "Er4 in cElMatCreuseBlocSym::cElMatCreuseBlocSym"
       );
   }
   for (int aKb = 0 ; aKb< mNbBlocs  ; aKb++)
   {
       ELISE_ASSERT
       (
            strBlocs[aKb]->I0Solve() < strBlocs[aKb]->I1Solve(),
            "Er5 in cElMatCreuseBlocSym::cElMatCreuseBlocSym"
       );
   }
   for (int aKN=0 ; aKN<int(mISolve2Bloc.size()) ; aKN++)
   {
       int aN = mISolve2Bloc[aKN];
       ELISE_ASSERT((aN>=0)&&(aN<int(strBlocs.size())),"Er6 in cElMatCreuseBlocSym::cElMatCreuseBlocSym");
       cIncIntervale * anI =  mStrBlocs[aN];
       ELISE_ASSERT
       (
          (aKN>=anI->I0Solve())&& (aKN<anI->I1Solve()),
          "Er7 in cElMatCreuseBlocSym::cElMatCreuseBlocSym"
       );
   }
   

   //
   mDataBlocs  = new  tBlPP [mNbBlocs] ;

   for (int aKy = 0 ; aKy< mNbBlocs  ; aKy++)
   {
       mDataBlocs[aKy] =  new tBlP [mNbBlocs-aKy];
       mDataBlocs[aKy] -= aKy;
       for (int aKx = aKy ; aKx< mNbBlocs  ; aKx++)
       {
            mDataBlocs[aKy][aKx] = 0;
       }
   }

// Debug("END CREATE");
}

cBlocMCBS * cElMatCreuseBlocSym::BlocOfKbxKby(int aKBx,int aKBy) const
{
   cBlocMCBS * & aRes = mDataBlocs[aKBy][aKBx];
   if (aRes ==0)
   {
/*
       if (::ShowPermutVar)
       {
           std::cout  << "BLOC " << aKBx << " " << aKBy 
                      << " SOLVE: x ["  << mStrBlocs[aKBx]->I0Solve()  <<" " << mStrBlocs[aKBx]->I1Solve() << "]"
                      << " y ["  << mStrBlocs[aKBy]->I0Solve() << " " << mStrBlocs[aKBy]->I1Solve() << "]"
                      << "\n";
       }
*/
       aRes = new cBlocMCBS
                  (
                      Pt2di(aKBx,aKBy),
                      Pt2di(mStrBlocs[aKBx]->I0Solve(),mStrBlocs[aKBy]->I0Solve()),
                      Pt2di(mStrBlocs[aKBx]->I1Solve(),mStrBlocs[aKBy]->I1Solve())
                  );
   }
   return aRes;
}

void  cElMatCreuseBlocSym::VerifScal()
{
   // std::cout << "TT 00 \n";
   // std::cout << ScaleCol(0,30) << "\n";
   // std::cout << "OK TT 00 \n";
   for (int aC1 = 0 ; aC1 < mNbEl ; aC1++)
   {
       for (int aC2 = 0 ; aC2 < mNbEl ; aC2++)
       {
           // std::cout << "IND " << aC1 << " " << aC2 << " " << mISolve2Bloc.size() <<  "\n";
           tSysCho aScA = SimpleScaleCol(aC1,aC2);
           // std::cout << "   = OK A \n";
           tSysCho aScB = ScaleCol(aC1,aC2);
           tSysCho aEps = 1e-6;

           tSysCho aDif = ElAbs((aScA-aScB)/(aEps+ElAbs(aScA)+ElAbs(aScB)));
           if (ElAbs(aScA) > aEps)
               std::cout << aScA << " " << aScB <<  " " << aDif << "\n";
           ELISE_ASSERT(aDif<aEps,"cElMatCreuseBlocSym::ShowStruct");
       }
   }
}

void  cElMatCreuseBlocSym::ShowStruct(bool basic)
{
   // std::cout << "TT 00 \n";

   for (int aKx = 0 ; aKx< mNbBlocs  ; aKx++)
       std::cout << (aKx/10) ;
   std::cout << "\n";

   for (int aKx = 0 ; aKx< mNbBlocs  ; aKx++)
       std::cout << (aKx%10) ;
   std::cout << "\n";

   for (int aKy = 0 ; aKy< mNbBlocs  ; aKy++)
   {
       for (int aKx = 0 ; aKx< mNbBlocs  ; aKx++)
       {
          int aKMx = ElMax(aKx,aKy);
          int aKMy = ElMin(aKx,aKy);

          std::string aS = ".";
          cBlocMCBS * aBl = mDataBlocs[aKMy][aKMx];

          if (aBl)
          {
               if (cBlocMCBS::IsNull(aBl))
                  aS = "o";
               else
               {
                  aS = "*";
                  if (aKx==aKy)
                     aS = "#";
               }
          }
          if (basic)
             aS = cBlocMCBS::IsNull(aBl) ? "+" : "#";

          // bool  aChIS0 = (! mMChol)  || (! cBlocMCBS::IsNull(mMChol->mDataBlocs[aKMy][aKMx]));

          // ELISE_ASSERT(! ((!iS0) && (!aChIS0)),"Show Struct Incoher Chol");

          // std::cout <<  (iS0 ? (aChIS0 ?"." :"-") : ( "*")) ;
           std::cout <<  aS ;
       }
       std::cout <<  " " << aKy << "\n";
   }
   std::cout << mMChol << "\n";
}


void  cElMatCreuseBlocSym::Reset()
{
   for (int aKy = 0 ; aKy< mNbBlocs  ; aKy++)
   {
       for (int aKx = aKy ; aKx< mNbBlocs  ; aKx++)
       {
          cBlocMCBS * aBl = mDataBlocs[aKy][aKx];
          if (aBl)
          {
             aBl->Reset();
          }
       }
   }
}



tSysCho  cElMatCreuseBlocSym::LowGetElem(INT aX,INT aY)  const
{
   if (aX <aY) ElSwap(aX,aY);

   int aKBx = mISolve2Bloc[aX];
   int aKBy = mISolve2Bloc[aY];

   cBlocMCBS *  aBloc = mDataBlocs[aKBy][aKBx];
   if (aBloc)
      return aBloc->DataAbs (aX, aY);
   return 0;
}


void  cElMatCreuseBlocSym::LowSetElem(INT aX,INT aY,const tSysCho & aVal) 
{
   if (aX <aY) ElSwap(aX,aY);

   int aKbX = mISolve2Bloc[aX];
   int aKbY = mISolve2Bloc[aY];
   cBlocMCBS *  aBloc = BlocOfKbxKby(aKbX,aKbY);

   tSysCho & aRes =  aBloc->DataAbs (aX, aY);

   aRes = aVal;
}

tSysCho  cElMatCreuseBlocSym::SimpleScaleCol(int aCol1,int aCol2)
{
    if (aCol1> aCol2) ElSwap(aCol1,aCol2);
    int aNb = ElMin(aCol1,aCol2-1);
    tSysCho aRes = 0; 
    for (int aK=0 ; aK<= aNb; aK++)
        aRes += SimpleGet(aCol1,aK) * SimpleGet(aCol2,aK);

    return aRes;
}

int cElMatCreuseBlocSym::NumBl0OfCol(int aCol) const
{
   return mBloc0OfCol[mISolve2Bloc[aCol]]->NumBlY();
}

void cElMatCreuseBlocSym::TestCholesky() const
{
   std::cout << "TestCholesky \n";
   tSysCho aSomDif  = 0;

   for (int anX=0 ; anX <mNbEl ; anX++)
   {
       for (int anY=0 ; anY <mNbEl ; anY++)
       {
             tSysCho aV1 =  LowGetElem(anX,anY);
             tSysCho aV2 =  mMChol->ScaleCol(anX,anY);
             if (anX==anY)
                aV2 += ElSquare(mMChol->LowGetElem(anX,anY));

             tSysCho aDif = ElAbs(aV1-aV2) / (1+ElAbs(aV1));

             aSomDif += aDif;
             if (aDif > 1e-5)
             {
                  std::cout  << "X=" << anX << " Y=" << anY  << " V1=" << aV1 << " V2=" << aV2 << " Dif=" << aDif << "\n";
             }
       }
        // std::cout  <<  "\n";
   }
    std::cout << "SOM DIF = " << aSomDif << "\n";
    getchar();
}


void cElMatCreuseBlocSym::CalculCholesky()
{
   if (mMChol==0)
   {
       mMChol = new cElMatCreuseBlocSym(mStrBlocs,mISolve2Bloc,eCholesNoPrec);
   }
   mMChol->Reset();


   for (int aKBlx=0 ; aKBlx <mNbBlocs ; aKBlx++)
   {
      cBlocMCBS * aXBloc0 = mBloc0OfCol[aKBlx];
      int aX0 = mStrBlocs[aKBlx]->I0Solve();
      int aX1 = mStrBlocs[aKBlx]->I1Solve();

      int aKBlZ0x  = aXBloc0->NumBlY();

      for (int aKBly= aKBlZ0x ; aKBly<= aKBlx ; aKBly++)
      {
         int aY0 = mStrBlocs[aKBly]->I0Solve();
         int aY1 = mStrBlocs[aKBly]->I1Solve();
         int aKBlZ1 = aKBly;
         cBlocMCBS * aYBloc0 = mBloc0OfCol[aKBly];
         int aKBlZ0y  = aYBloc0->NumBlY();
         int aKblZ0 = ElMax(aKBlZ0x,aKBlZ0y);

         bool AllBlocNull = cBlocMCBS::IsNull(mDataBlocs[aKBly][aKBlx]);
         for (int aKblZ=aKblZ0 ; AllBlocNull &&(aKblZ<=aKBlZ1) ; aKblZ++)
         {
              if (
                        (mMChol->mDataBlocs[aKblZ][aKBlx]!=0) 
                     && (mMChol->mDataBlocs[aKblZ][aKBly]!=0)
                 )
              {
                 AllBlocNull  = false;
              }
         }


         if (!AllBlocNull)
         {
            for (int anX=aX0 ; anX<aX1 ; anX++)
            {
                 int aY1Dyn = ElMin(aY1,anX+1);
                 for (int anY=aY0 ; anY<aY1Dyn ; anY++)
                 {

                      tSysCho aScal = 0;
                      for (int aKblZ=aKblZ0 ; aKblZ<=aKBlZ1 ; aKblZ++)
                      {
                           cBlocMCBS * aBXZ = mMChol->mDataBlocs[aKblZ][aKBlx];
                           cBlocMCBS * aBXY = mMChol->mDataBlocs[aKblZ][aKBly];

                           if (aBXZ && aBXY) 
                           {
                              aScal += aBXZ->ScalCol(anX,*aBXY,anY);
                           }
                      }
                      aScal = LowGetElem(anX,anY) - aScal;
                      if (anX != anY)
                      {
                         // mMChol->LowGetElem(anX,anY) =  aScal / mMChol->LowGetElem(anY,anY);
                         mMChol->LowSetElem(anX,anY,aScal / mMChol->LowGetElem(anY,anY));
                      }
                      else
                      {
                           if (aScal <=0)
                           {
                               std::cout << "ValueNeg =  "  << aScal << "\n";
                               static bool First = true;
                               if (First)
                               {
                                     std::cout << " Warn tape enter to continue" << std::endl;
                                     if (TheExitOnWarn) // ( || MPD_MM())
                                         ElEXIT(1,"aScal<=0 in Cholesky");
                                     else
                                     {
                                         if (TheGoonOnWarn)
                                         {
                                              cElWarning::EigenValueInCholeski.AddWarn("Choleski",__LINE__,__FILE__);
                                         }
                                         else
                                         {
                                            getchar();
                                         }
                                     }
                               }
                               First = false;
                               // ELISE_ASSERT(false,"Cholesly Diag, Beg");
                               aScal = 1;
                           }
                           mMChol->LowSetElem(anX,anY,sqrt(aScal));

                      }
                 }
            }
         }
      }
   }
   mMChol->ChainageBlocs();
   mMChol->cElMatCreuseGen::PrepPreCond();


}



tSysCho  cElMatCreuseBlocSym::ScaleCol(int aCol1,int aCol2)
{
// std::cout << "AAAAAAAAAAAAAAAA \n";
    if (aCol1> aCol2) ElSwap(aCol1,aCol2);
    // int aNb = ElMin(aCol1,aCol2-1);

    int aKBl1 = mISolve2Bloc[aCol1];
    int aKBl2 = mISolve2Bloc[aCol2];

    tSysCho aRes = 0;

// std::cout << "BBBBBBBBBBBbb  " << aKBl1 << " " << aKBl2 <<  "  " << mNbBlocs << " \n";
// std::cout << "    b111   " << mBloc0OfCol[aKBl1]   << " \n";
// std::cout << "    b222   " << mBloc0OfCol[aKBl2]   << " \n";
    for 
    (
       cBlocMCBS * aBl2 =  mBloc0OfCol[aKBl2], * aBl1 =  mBloc0OfCol[aKBl1] ;
       aBl2  && (aBl2->NumBlY() <= aKBl1) ;
       aBl2 = aBl2->YNextDown()
    )
    {
//  std::cout << "TTTTTTTTTTTTttooo \n";
        ELISE_ASSERT(aBl2!=0,"Internal Er (Bl2=0) in cElMatCreuseBlocSym::ScaleCol");
        while (aBl1 && (aBl1->NumBlY() < aBl2->NumBlY()))
        {
            aBl1 = aBl1->YNextDown();
        }

        ELISE_ASSERT(aBl1!=0,"Internal Er (Bl1=0) in cElMatCreuseBlocSym::ScaleCol");
        // Seul cas "interessant", les deux memes blocs sont != 0
        if (aBl1->NumBlY() == aBl2->NumBlY())
        {
            aRes += aBl1->ScalCol(aCol1,*aBl2,aCol2);
        }
        
    }
    //  cBlocMCBS **                           mBloc0OfCol;

/// std::cout << "ZZZZZZZZZZZZZZZZZZz \n";
    return aRes;
}


tSysCho  cElMatCreuseBlocSym::SimpleGet(INT aX,INT aY)  const
{
   if (aX <aY) ElSwap(aX,aY);

   int aKbX = mISolve2Bloc[aX];
   int aKbY = mISolve2Bloc[aY];
   cBlocMCBS *  aBloc = mDataBlocs[aKbY][aKbX];

   return aBloc ? aBloc->DataAbs (aX, aY) : 0;
}



void cElMatCreuseBlocSym::TestMulVect(tSysCho * out,const tSysCho * in,tSysCho aPdsInf,tSysCho aPdsDiag,tSysCho aPdsSup) const
{
    for (int aKy=0 ; aKy<mNbCol ; aKy++)
    {
        out[aKy] = 0;
        for (int aKx=0 ; aKx<mNbCol ; aKx++)
        {
            if (aKx== aKy)
                out[aKy] += SimpleGet(aKy,aKx) * in[aKx] * aPdsDiag;
            if (aKx >aKy)
                out[aKy] += SimpleGet(aKy,aKx) * in[aKx] * aPdsSup;
            if (aKx <aKy)
                out[aKy] += SimpleGet(aKy,aKx) * in[aKx] * aPdsInf;
        }
    }
}

void VerifMats(const std::string & aMes,Im1D_REAL8  aM1,Im1D_REAL8 aM2)
{
   tSysCho aSom,aMax;
   ELISE_COPY
   (
        aM1.all_pts(),
        Abs(aM1.in()-aM2.in()),
        sigma(aSom) | VMax(aMax)
   );

   std::cout << "Verif " << aMes << " DMax " << aMax << " DMoy " << aSom/aM1.tx() << "\n";
   for (int aK=aM1.tx()-1 ; aK>= aM1.tx()-10 ; aK--)
   {
       std::cout  << "$$$$ " << aK << " " << aM1.data()[aK] << " " << aM2.data()[aK] << "\n";
   }
}


void cElMatCreuseBlocSym::Test()
{

   // Im1D_REAL8 aI0(mNbCol);
   // ELISE_COPY(aI0.all_pts(), frandr(),aI0.out());

   Im1D<tSysCho,tSysCho> aI0(mNbCol);
   for (int aK=0 ; aK<mNbCol ; aK++)
       aI0.data()[aK] = NRrandom3();

   {
      // On verifie la mult
      Im1D<tSysCho,tSysCho> aI1(mNbCol);
      Im1D<tSysCho,tSysCho> aI2(mNbCol);

      MulVect(aI1.data(),aI0.data());
      TestMulVect(aI2.data(),aI0.data(),1,1,1);
      // VerifMats("Mult",aI1,aI2);
   }

   {
      tSysCho aW = 1.4;

      Im1D<tSysCho,tSysCho> aI1(mNbCol);
      TestMulVect(aI1.data(),aI0.data(),0,1.0/aW,1);
      
      Im1D<tSysCho,tSysCho> aI2(mNbCol);
      SolveUperSys(aI2.data(),aI1.data(),aW);
      // VerifMats("UpSys",aI0,aI2);
   }

   {
      tSysCho aW = 0.7;

      Im1D<tSysCho,tSysCho> aI1(mNbCol);
      TestMulVect(aI1.data(),aI0.data(),1.0,1.0/aW,0);
      
      Im1D<tSysCho,tSysCho> aI2(mNbCol);
      SolveLowerSys(aI2.data(),aI1.data(),aW);
      // VerifMats("LowSys",aI0,aI2);
    }



   std::cout << "TTT  cElMatCreuseBlocSym::Test\n"; getchar();
}



void  cElMatCreuseBlocSym::MulVect(tSysCho * out,const tSysCho * in) const
{
int NbNN = 0;

    for (int aK=0 ; aK<mNbCol ; aK++)
        out[aK] = 0;

    for (int aKy = 0 ; aKy< mNbBlocs  ; aKy++)
    {
       int anY0 = mStrBlocs[aKy]->I0Solve();
       int anY1 = mStrBlocs[aKy]->I1Solve();
NbNN--;  // POUR la DIAG
       // CA MARCHE PARCE QUE LES BLOG DIAGONAUX INFERIEUR SONT NUL ....
       for (cBlocMCBS * aBl = mDataBlocs[aKy][aKy];  aBl ; aBl = aBl->XNext())
       {
NbNN+=2;
           int aKx = aBl->NumBlX();
           int anX0 = mStrBlocs[aKx]->I0Solve();
           int anX1 = mStrBlocs[aKx]->I1Solve();
           tSysCho * aData = aBl->DataL();
           for (int anY=anY0; anY<anY1 ; anY++)
           {
                 for (int anX=anX0; anX<anX1 ; anX++)
                 {
                     tSysCho aV = *(aData++);
                     
                     out[anY] +=   aV* in[anX];
                     if (anX != anY)
                     {
                         out[anX] +=   aV* in[anY];
                     }
                 }
           }
       }
    }
}

#if (0)
#endif

/****************************************************/
/*                                                  */
/*             cFormQuadCreuse                      */
/*                                                  */
/****************************************************/

cFormQuadCreuse::cFormQuadCreuse(INT aNbVar,cElMatCreuseGen * aMatCr) :
   FoncNVarDer<REAL> (aNbVar),
   cGenSysSurResol
   (
        false,
        aMatCr->OptSym(),
        !aMatCr->OptSym(),
        //false,
        //true,
        false
   ),
   mNbVar     (aNbVar),
   mV0        (0.0),
   mFLin      (aNbVar,0.0),
   mDataLin   (mFLin.data()),
   mVGrad     (aNbVar,0.0),
   mDataGrad  (mVGrad.data()),
   mMat       (aMatCr),
   mEGSI      (0),
   mP000      (new PtsKD(aNbVar)),
   mMatIsOptForEcrireInMatr  (aMatCr->IsOptForEcrireInMatr()),
   mMatIsOptForSousP3x3      (aMatCr->IsOptForSousP3x3()),
   mMatIsOptForQuadSet0      (aMatCr->IsOptForQuadSet0()),
   mMatIsOptForAddEqIndexee  (aMatCr->IsOptForAddEqIndexee())
{
}

void cFormQuadCreuse::VMMDo(Im1D_REAL8 in,Im1D_REAL8 out)
{
   ELISE_ASSERT(out.tx()==mNbVar,"SMFGC_Asub");
   ELISE_ASSERT(false,"Remettre cFormQuadCreuse::VMMDo");
   //mMat->MulVect(out.data(),in.data());
}

void cFormQuadCreuse::VPCDo(Im1D_REAL8 in,Im1D_REAL8 out)
{
   mMat->VPCDo(out.data(),in.data());
/*
   double * aDOut = out.data();
   double * aDIn  = in.data();
   for (int aK=0 ; aK<mNbVar; aK++)
   {
      aDOut[aK] = aDIn[aK] / mDDPrec[aK];
   }
*/
}





INT cFormQuadCreuse::NbVar() const 
{
   return mNbVar;
}
void cFormQuadCreuse::V_GSSR_AddNewEquation (REAL aPds,REAL * aCoeff,REAL aB) 
{
     mV0 += aPds * ElSquare(aB);
     for (INT aX=0 ; aX<mNbVar ; aX++)
     {
         if (aCoeff[aX])
         {
	     REAL aCXP = 2 * aCoeff[aX] * aPds;
             mDataLin[aX] -=  aCXP * aB;
             for (INT aY=0 ; aY<mNbVar ; aY++)
             {
                 REAL aCY = aCoeff[aY];
		 if (aCY)
                     mMat->LowAddElem(aX,aY,aCXP*aCY);
	     }
         }
     }
}

void cFormQuadCreuse::VerifGSS(const std::string & aMes) 
{
    ELISE_ASSERT(false,"FormQuadCreuse::VerifGSS");
}


bool cFormQuadCreuse::GSSR_UseEqMatIndexee()
{
   return true;
   // return true;
}

void cFormQuadCreuse::V_GSSR_EqMatIndexee
     (
          const std::vector<INT> & aVInd,
          REAL aPds,REAL ** aMat,
          REAL * aVect,REAL aCste
     )
{
     mV0 += aPds * aCste;
     INT aNbInd = (INT) aVInd.size();
     for (INT aKX=0 ; aKX<aNbInd ; aKX++)
         mDataLin[aVInd[aKX]] +=  aPds * aVect[aKX];
     mMat->EqMatIndexee(aVInd,aPds,aMat);
}

tSysCho  cFormQuadCreuse::GetElemLin(int y) const
{
  return mDataLin[y];
}
void  cFormQuadCreuse::SetElemLin(int y,const tSysCho& aV)
{
   mDataLin[y] = aV;
}

tSysCho   cFormQuadCreuse::GetElemQuad(int x,int y) const
{
   return mMat->LowGetElem(x,y);
}
void  cFormQuadCreuse::SetElemQuad(int x,int y,const tSysCho& aV )
{
   mMat->LowSetElem(x,y,aV);
}


void cFormQuadCreuse::V_GSSR_AddNewEquation_Indexe 
                      (
                         const std::vector<cSsBloc> * aVSB,
                         double * aFullCoeff,int aNbTot,
                         const std::vector<INT> & aVInd,
		         REAL aPds,REAL * aCoeff,REAL aB,
                         cParamCalcVarUnkEl *
                      ) 
{
  mV0 += aPds * ElSquare(aB);
  if ( mMatIsOptForAddEqIndexee  && aVSB)
  {
      mMat->VMAT_GSSR_AddNewEquation_Indexe (aVSB,aFullCoeff,aNbTot,aPds,mDataLin,aB);
  }
  else
  {
     INT aNbInd = (INT) aVInd.size();
     for (INT aK1 = 0 ; aK1<aNbInd ; aK1++)
     {
	 INT aY = aVInd[aK1];
	 REAL aCYP = 2 * aCoeff[aK1] * aPds;
         mDataLin[aY] -=  aCYP * aB;
         mMat->AddLineInd(aK1,aY,aCYP,aVInd,aCoeff);
	 /*
         for (INT aK2=0 ; aK2<aNbInd ; aK2++)
         {
             INT aY = aVInd[aK2];
             REAL aCY = aCoeff[aK2];
             if (aCY)
                 mMat->AddElem(aX,aY,aCXP*aCY);
                 //mMat->LowAddElem(aX,aY,aCXP*aCY);
         }
	 */
     }
  }
}

void cFormQuadCreuse::SoutraitProduc3x3
                     (
                          bool                   Sym,
                          ElMatrix<tSysCho> &aM1,
                          ElMatrix<tSysCho> &aM2,
                          const std::vector<cSsBloc> * aYVSB
                     )
{
   if (mMatIsOptForSousP3x3)
      mMat->SoutraitProduc3x3(Sym,aM1,aM2,aYVSB);
   else
      cGenSysSurResol::SoutraitProduc3x3(Sym,aM1,aM2,aYVSB);
}


void cFormQuadCreuse::Indexee_EcrireDansMatrWithQuad
                      (
                             ElMatrix<tSysCho> &aMatr,
                             const std::vector<cSsBloc> &  aVx,
                             const std::vector<cSsBloc> &  aVy
                      )  const
{
   if (mMatIsOptForEcrireInMatr)
      mMat->Indexee_EcrireDansMatrWithQuad(aMatr,aVx,aVy);
   else
      cGenSysSurResol::Indexee_EcrireDansMatrWithQuad(aMatr,aVx,aVy);
}


void  cFormQuadCreuse::Indexee_QuadSet0
      (
            const std::vector<cSsBloc> & aVx,
            const std::vector<cSsBloc> & aVy
      )
{
   if (mMatIsOptForQuadSet0)
      mMat->Indexee_QuadSet0(aVx,aVy);
   else
      cGenSysSurResol::Indexee_QuadSet0(aVx,aVy);
}
/*
*/



double cFormQuadCreuse::CoeffNorm() const
{
   return 2.0;
}


bool cFormQuadCreuse:: AcceptContrainteNonUniV() const
{
   return false;
}




Im1D_REAL8   cFormQuadCreuse::V_GSSR_Solve(bool * aResOk) 
{

    Im1D_REAL8 aR2(mNbVar,0.0);

    Im1D<tSysCho,tSysCho> aR2Cho(mNbVar,0.0);
    if (mMat->DirectInverse(mFLin.data(),aR2Cho.data()))
    {
         for (int aK=0 ; aK<mNbVar ; aK++)
             aR2.data()[aK] = aR2Cho.data()[aK];
/*
         if (aResOk) *aResOk=true;
         ELISE_COPY(aR2.all_pts(),-aR2.in(),aR2.out());
         return aR2; 
*/
    }
    else
    {

        mMat->PrepPreCond();


       cControleGC aContGC(3);
       ELISE_ASSERT(false,"Remmetre grad-conj");
       // GradConjPrecondSolve(*this,*this,mFLin,aR2,aContGC);
    }


    ELISE_COPY(aR2.all_pts(),-aR2.in(),aR2.out());
    if (aResOk) *aResOk=true;
  
    return aR2;
}

void cFormQuadCreuse::V_GSSR_Reset() 
{
	mV0 = 0;
	ELISE_COPY(mFLin.all_pts(),0.0,mFLin.out());
	mMat->Reset();
}


cFormQuadCreuse::~cFormQuadCreuse()
{
   delete mMat;
   delete mEGSI;
   delete mP000;
}

void cFormQuadCreuse::AddDiff(Fonc_Num aFonc)
{
   if (mEGSI == 0)
      mEGSI = new ElGrowingSetInd(mNbVar);
   aFonc.VarDerNN(*mEGSI);
   AddDiff(aFonc,*mEGSI);
   mEGSI->clear();
}



void cFormQuadCreuse::AddDiff
     (
         Fonc_Num aFonc,
	 const ElGrowingSetInd & aSet
     )
{
     mV0 += aFonc.ValFonc(*mP000);
     for
     (
        ElGrowingSetInd::const_iterator anIt1 =  aSet.begin();
        anIt1 != aSet.end();
        anIt1 ++
     )
     {
        ELISE_ASSERT
        (
            (*anIt1>=0) && (*anIt1<mNbVar),
            "Bad Num Var in cFormQuadCreuse::AddDiff"
        );
	Fonc_Num aDP = aFonc.deriv(*anIt1);
	mDataLin[*anIt1] += aDP.ValFonc(*mP000);

        for
        (
           ElGrowingSetInd::const_iterator anIt2 =  aSet.begin();
           anIt2 != aSet.end();
           anIt2 ++
        )
           mMat->AddElem(*anIt1,*anIt2,aDP.ValDeriv(*mP000,*anIt2));

     }
}
 
REAL  cFormQuadCreuse::ValFNV(const REAL *  aP)
{
   REAL aRes = mV0;
   mMat->MulVect8(mDataGrad,aP);
   for (INT k=0 ; k<mNbVar ; k++)
      aRes += aP[k]*(mDataLin[k] + mDataGrad[k]/2.0);

   return aRes;
}


double  cFormQuadCreuse::ResiduOfSol(const double * aP)
{
   double aResidu = ValFNV(aP);

   return aResidu;
}

void  cFormQuadCreuse::GradFNV(REAL *grad,const REAL *  aP)
{
   mMat->MulVect8(mDataGrad,aP);
   for (INT k=0 ; k<mNbVar ; k++)
      grad[k]  = mDataLin[k] + mDataGrad[k];
}

void cFormQuadCreuse::SetOffsets(const std::vector<INT> & aVIndexes)
{
   mMat->SetOffsets(aVIndexes);
}


//   cElMatCreuseGen cElMatCreuseGen cElMatCreuseGen cElMatCreuseGen
//   cElMatCreuseGen cElMatCreuseGen cElMatCreuseGen cElMatCreuseGen
//   cElMatCreuseGen cElMatCreuseGen cElMatCreuseGen cElMatCreuseGen

cElMatCreuseGen * cElMatCreuseGen::StdNewOne(int X, int Y,bool Fixe)
{
    if (Fixe)
    {
       ELISE_ASSERT(X==Y,"UnSquare Mat in cElMatCreuseStrFixe");
       return new cElMatCreuseStrFixe(X);
    }
    return new cElMatCreuseMap(X,Y);
}

cElMatCreuseGen * cElMatCreuseGen::StdBlocSym
                  (
                      const  std::vector<cIncIntervale *> &  Blocs,
                      const  std::vector<int> &              ISolve2Bloc
                  )
{
    return new cElMatCreuseBlocSym(Blocs,ISolve2Bloc,ePrecCondSSOR);
}


void cElMatCreuseGen::PrecCondQuad(double *)
{
    ELISE_ASSERT(false,"cElMatCreuseGen::PrecCondQuad");
}

bool cElMatCreuseGen::DirectInverse(const tSysCho *,tSysCho *)
{
    return false;
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
