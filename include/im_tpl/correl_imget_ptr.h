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



#ifndef _ELISE_IM_CORREL_IMGET_PTR_
#define _ELISE_IM_CORREL_IMGET_PTR_

// Correlateur optimise (supposed to be) 

#define NBB_TFCS 8

template <class tElem> class  cTplImageCorrelateurSubPix
{
       public :

           typedef typename El_CTypeTraits<tElem>::tBase    tBase;
           typedef Im2D<tElem,tBase>               tIm;
           typedef TIm2D<tElem,tBase>              tTIm;
           typedef TImGet<tElem,tBase,NBB_TFCS>    tGet;

           cTplImageCorrelateurSubPix
           (
                 tIm  anIm1,
                 tIm  anIm2,
                 REAL aSzVign,
                 REAL aStep,
                 REAL aDefOut
           )  :
              mIm1    (anIm1),
              mIm2    (anIm2),
              mDefOut (aDefOut),
              mBCKernel  (-0.5),
	      mModeBicub (false)
           {
               INT NbPts = round_up(aSzVign/aStep);
               for (INT iX=-NbPts; iX<=NbPts ; iX++)
                   for (INT iY=-NbPts; iY<=NbPts ; iY++)
                   {
                      mPts1.push_back(Pt2dr(iX*aStep,iY*aStep));
                      mPts2.push_back(mPts1.back());
                   }

               mP0_I1 = mPts1.front();
               mP1_I1 = mPts1.back();
               mP0_I2 = mP0_I1;
               mP1_I2 = mP1_I1;
               mNbPts = INT(mPts1.size());
           }

	    void SetModeBicub(bool ModeBic)
	    {
		mModeBicub  = ModeBic;
	    }

           void  SetDistDiffVgn2(Pt2dr aP1C,const ElDistortion22_Gen & aDist,REAL E)
           {
               // Pt2dr aP2C = aDist.Direct(aP1C);
               Pt2dr aGx = (aDist.Direct(aP1C+Pt2dr(E,0))-aDist.Direct(aP1C+Pt2dr(-E,0))) / (2*E);
               Pt2dr aGy = (aDist.Direct(aP1C+Pt2dr(0,E))-aDist.Direct(aP1C+Pt2dr(0,-E))) / (2*E);
               Pt2dr aP0(0,0);
               Pt2dr aP1(0,0);

               for (INT aK=0; aK<mNbPts ; aK++)
               {
                    Pt2dr  aD1 = mPts1[aK];
                    Pt2dr  aD2  = aGx *aD1.x + aGy*aD1.y;
                    mPts2[aK] = aD2;
                    aP0.SetInf(aD2);
                    aP1.SetSup(aD2);
               }
               mP0_I2= aP0;
               mP1_I2= aP1;
           }

           // Correlation ou les valeurs sont arrondie en reel
           REAL icorrel(Pt2dr aPr1,Pt2dr aPr2)
           {
                 Pt2dr aP1 (aPr1);
                 Pt2dr aP2 (aPr2);

                 if (! OkTr12(aP1,aP2))
                    return mDefOut;

                 IMat_Inertie aMat;

                 for ( INT aK=0; aK<mNbPts; aK++)
                 {
                     aMat.add_pt_en_place
                     (
                         tGet::geti(mIm1,aP1+mPts1[aK]),
                         tGet::geti(mIm2,aP2+mPts2[aK])
                     );
                 }

                 return aMat.correlation();

           }

           // Correlation ou les valeurs sont arrondie en reel
           REAL rcorrel(Pt2dr aPr1,Pt2dr aPr2)
           {

                 Pt2dr aP1 (aPr1);
                 Pt2dr aP2 (aPr2);

                 if (!  OkTr12(aP1,aP2))
                    return mDefOut;

                 RMat_Inertie aMat;

                 for ( INT aK=0; aK<mNbPts; aK++)
                 {
                     aMat.add_pt_en_place
                     (
                         tGet::geti(mIm1,aP1+mPts1[aK]),
                         tGet::geti(mIm2,aP2+mPts2[aK])
                     );
                 }
                 return aMat.correlation();
           }

           INT NbPts() const { return mPts1.size();}


       protected :

           bool OkTr12(Pt2dr aP1,Pt2dr aP2)
           {
	       if (! mModeBicub)
	       {
                  return      tGet::inside_bilin(mIm1,aP1+mP0_I1)
                          &&  tGet::inside_bilin(mIm1,aP1+mP1_I1)
                          &&  tGet::inside_bilin(mIm2,aP2+mP0_I2)
                          &&  tGet::inside_bilin(mIm2,aP2+mP1_I2) ;
	       }
               return      tGet::inside_bicub(mIm1,aP1+mP0_I1)
                       &&  tGet::inside_bicub(mIm1,aP1+mP1_I1)
                       &&  tGet::inside_bicub(mIm2,aP2+mP0_I2)
                       &&  tGet::inside_bicub(mIm2,aP2+mP1_I2) ;
           }

           tTIm               mIm1;
           tTIm               mIm2;
           std::vector<Pt2dr> mPts1;
           std::vector<Pt2dr> mPts2;
           REAL               mDefOut;
           Pt2dr              mP0_I1;
           Pt2dr              mP1_I1;
           Pt2dr              mP0_I2;
           Pt2dr              mP1_I2;
           INT                mNbPts;
           cCubicInterpKernel mBCKernel;
	   bool               mModeBicub;
};

// Moments pour la correlation par moinbre carres

struct cBufResMomentCor2DLSQ
{

       cBufResMomentCor2DLSQ(Pt3dr aPtOut) :
               mPtOut    (aPtOut),
	       mVar      (3,3),
	       mDVar     (mVar.data()),
	       mCov      (3),
	       mDCov     (mCov.data()),
	       mSol      (3),
	       mDSol     (mSol.data())
       {
       }

       Pt3dr          mPtOut;
       bool           mIsOk;
       Im2D_REAL8 mVar;
       REAL8 **   mDVar;
       Im1D_REAL8 mCov;
       REAL8*     mDCov;
       Im1D_REAL8 mSol;
       REAL8*     mDSol;
};


struct  cMomentCor2DLSQ
{
        enum {IndI2 =0,IndGx =1,IndGy =2};
        void Init()
        {
                 mNb = 0;
		 s2  = 0; s22 = 0; 
		 sX  = 0; sXX = 0; 
		 sY  = 0; sYY = 0;
		 s2X = 0; s2Y = 0; sXY = 0;
		 s1  = 0; s12 = 0; s1X = 0; s1Y = 0;
        }
        void Cumul(INT aS,const cMomentCor2DLSQ & aMom)
        {
           mNb += aS * aMom.mNb;

           s2  += aS * aMom.s2;
           s22 += aS * aMom.s22;
           sX  += aS * aMom.sX;
           sXX += aS * aMom.sXX;
           sY  += aS * aMom.sY;
           sYY += aS * aMom.sYY;

           s2X += aS * aMom.s2X;
           s2Y += aS * aMom.s2Y;
           sXY += aS * aMom.sXY;

           s1  += aS * aMom.s1;
           s12 += aS * aMom.s12;
           s1X += aS * aMom.s1X;
           s1Y += aS * aMom.s1Y;
        }

        void Add(INT aS,REAL aV1,Pt3dr aGr)
        {
             mNb += aS;
             s2  += aS* aGr.z;
             s22 += aS* ElSquare(aGr.z);
             sX  += aS* aGr.x;
             sXX += aS* ElSquare(aGr.x);
             sY  += aS* aGr.y;
             sYY += aS* ElSquare(aGr.y);

             s2X +=  aS* aGr.z * aGr.x;
             s2Y +=  aS* aGr.z * aGr.y;
             sXY +=  aS* aGr.x * aGr.y;

             s1  += aS* aV1;
             s12 += aS* aV1 * aGr.z;
             s1X += aS* aV1 * aGr.x;
             s1Y += aS* aV1 * aGr.y;
        }

        Pt3dr  GetSol(cBufResMomentCor2DLSQ & aBuf) const
        {
		 aBuf.mDVar[IndI2][IndI2] = s22 -s2*(s2/mNb);
		 aBuf.mDVar[IndGx][IndGx] = sXX -sX*(sX/mNb);
		 aBuf.mDVar[IndGy][IndGy] = sYY -sY*(sY/mNb);

		 aBuf.mDVar[IndGx][IndI2] = 
	              aBuf.mDVar[IndI2][IndGx] = s2X -s2*(sX/mNb);
		 aBuf.mDVar[IndGy][IndI2] = 
	              aBuf.mDVar[IndI2][IndGy] = s2Y -s2*(sY/mNb);
		 aBuf.mDVar[IndGy][IndGx] = 
	              aBuf.mDVar[IndGx][IndGy] = sXY -sX*(sY/mNb);


		 aBuf.mDCov[IndI2] = s12 - s1 * (s2/mNb);
		 aBuf.mDCov[IndGx] = s1X - s1 * (sX/mNb);
		 aBuf.mDCov[IndGy] = s1Y - s1 * (sY/mNb);

                 aBuf.mIsOk=  gaussj_svp(aBuf.mDVar,3);
                 if (! aBuf.mIsOk)
                    return aBuf.mPtOut;

		 aBuf.mVar.MulVect(aBuf.mSol,aBuf.mCov);
                 Pt2dr aPSol(aBuf.mDSol[IndGx],aBuf.mDSol[IndGy]);

                 Pt3dr aRes (aPSol.x,aPSol.y,0.0);
                 return aRes;
        }

        REAL mNb;
	REAL s2;  REAL s22;
	REAL sX;  REAL sXX;
	REAL sY;  REAL sYY;
	REAL s2X; REAL s2Y; REAL sXY;
	REAL s1 ; REAL s12; REAL s1X; REAL s1Y;
};








template <class tElem> class cTplDiffCorrelSubPix :
                                public cTplImageCorrelateurSubPix<tElem>
{
     public :
           cTplDiffCorrelSubPix
           (
                 typename cTplImageCorrelateurSubPix<tElem>::tIm  anIm1,
                 typename cTplImageCorrelateurSubPix<tElem>::tIm  anIm2,
                 REAL aSzVign,
                 REAL aStep,
                 Pt3dr aPtOut
           )  :
              cTplImageCorrelateurSubPix<tElem> 
	          (anIm1,anIm2,aSzVign,aStep,aPtOut.z),
              mBuf(aPtOut)
           {
           }


           Pt3dr  Optim(Pt2dr aP1,Pt2dr aP2,bool Debug=false)
           {

                 if (!  this->OkTr12(aP1,aP2))
                    return mBuf.mPtOut;

                 mMom.Init();

                 for ( INT aK =0; aK<this->mNbPts; aK++)
                 {
                    Pt2dr PLoc2 = aP2+this->mPts2[aK];
                    Pt3dr aGr;

		    if  ( this->mModeBicub )
                         aGr =  BicubicInterpol(this->mBCKernel,this->mIm2._d, PLoc2);
		    else
                         aGr = this->mIm2.getVandDer(PLoc2);
                     mMom.Add(1,this->mIm1.getr(aP1+this->mPts1[aK]),aGr);
// cTplImageCorrelateurSubPix<tElem>::tGet::geti(this->mIm1,aP1+this->mPts1[aK]);
                 }
                 Pt3dr aRes = mMom.GetSol(mBuf);

                 if (! mBuf.mIsOk)
                    return mBuf.mPtOut;


                 aRes.x+= aP2.x;
                 aRes.y+= aP2.y;
                 Pt2dr aPSol(aRes.x,aRes.y);
                 aRes.z = this->rcorrel(aP1,aPSol);
                 return aRes;
           }

          
           const cMomentCor2DLSQ & Mom() const {return mMom;}

     private :

           cMomentCor2DLSQ         mMom;
           cBufResMomentCor2DLSQ   mBuf;
};



/*
template <class tElem> class cTplDiffCorrelSubPix :
                                public cTplImageCorrelateurSubPix<tElem>
{
     public :
           cTplDiffCorrelSubPix
           (
                 typename cTplImageCorrelateurSubPix<tElem>::tIm  anIm1,
                 typename cTplImageCorrelateurSubPix<tElem>::tIm  anIm2,
                 REAL aSzVign,
                 REAL aStep,
                 Pt3dr aPtOut
           )  :
              cTplImageCorrelateurSubPix<tElem> 
	          (anIm1,anIm2,aSzVign,aStep,aPtOut.z),
               mPtOut    (aPtOut),
	       mVar      (3,3),
	       mDVar     (mVar.data()),
	       mCov      (3),
	       mDCov     (mCov.data()),
	       mSol      (3),
	       mDSol     (mSol.data())
           {
           }
	       

	   enum {IndI2 =0,IndGx =1,IndGy =2};

           Pt3dr  Optim(Pt2dr aP1,Pt2dr aP2,bool Debug=false)
           {

                 if (!  this->OkTr12(aP1,aP2))
                    return mPtOut;

		 s2  = 0; s22 = 0; 
		 sX  = 0; sXX = 0; 
		 sY  = 0; sYY = 0;
		 s2X = 0; s2Y = 0; sXY = 0;
		 s1  = 0; s12 = 0; s1X = 0; s1Y = 0;

                 for ( INT aK =0; aK<this->mNbPts; aK++)
                 {
                    Pt2dr PLoc2 = aP2+this->mPts2[aK];
                    Pt3dr aGr;

		    if  ( this->mModeBicub )
                         aGr =  BicubicInterpol(this->mBCKernel,this->mIm2._d, PLoc2);
		    else
                         aGr = this->mIm2.getVandDer(PLoc2);

		     s2  += aGr.z;
		     s22 += ElSquare(aGr.z);
		     sX  += aGr.x;
		     sXX += ElSquare(aGr.x);
		     sY  += aGr.y;
		     sYY += ElSquare(aGr.y);

		     s2X +=  aGr.z * aGr.x;
		     s2Y +=  aGr.z * aGr.y;
		     sXY +=  aGr.x * aGr.y;

                     REAL aV1   =  this->mIm1.getr(aP1+this->mPts1[aK]);
// cTplImageCorrelateurSubPix<tElem>::tGet::geti(this->mIm1,aP1+this->mPts1[aK]);
		     s1 += aV1;
		     s12 += aV1 * aGr.z;
		     s1X += aV1 * aGr.x;
		     s1Y += aV1 * aGr.y;
                 }
		 REAL rNbPts = this->mNbPts;

		 mDVar[IndI2][IndI2] = s22 -s2*(s2/rNbPts);
		 mDVar[IndGx][IndGx] = sXX -sX*(sX/rNbPts);
		 mDVar[IndGy][IndGy] = sYY -sY*(sY/rNbPts);

		 mDVar[IndGx][IndI2] = 
	              mDVar[IndI2][IndGx] = s2X -s2*(sX/rNbPts);
		 mDVar[IndGy][IndI2] = 
	              mDVar[IndI2][IndGy] = s2Y -s2*(sY/rNbPts);
		 mDVar[IndGy][IndGx] = 
	              mDVar[IndGx][IndGy] = sXY -sX*(sY/rNbPts);


		 mDCov[IndI2] = s12 - s1 * (s2/rNbPts);
		 mDCov[IndGx] = s1X - s1 * (sX/rNbPts);
		 mDCov[IndGy] = s1Y - s1 * (sY/rNbPts);

                 bool isOk=  gaussj_svp(mDVar,3);
                 if (! isOk)
                    return mPtOut;

		 mVar.MulVect(mSol,mCov);
                 Pt2dr aPSol(aP2.x+mDSol[IndGx],aP2.y+mDSol[IndGy]);

                 Pt3dr aRes (aPSol.x,aPSol.y,this->rcorrel(aP1,aPSol));
                 return aRes;
           }

          

     private :

	   REAL s2;  REAL s22;
	   REAL sX;  REAL sXX;
	   REAL sY;  REAL sYY;
	   REAL s2X; REAL s2Y; REAL sXY;
	   REAL s1 ; REAL s12; REAL s1X; REAL s1Y;

           Pt3dr          mPtOut;
	   Im2D_REAL8 mVar;
	   REAL8 **   mDVar;
	   Im1D_REAL8 mCov;
	   REAL8*     mDCov;
	   Im1D_REAL8 mSol;
	   REAL8*     mDSol;
};
*/


#endif // _ELISE_IM_CORREL_IMGET_PTR_






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
