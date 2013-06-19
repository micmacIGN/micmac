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



#ifndef _ELISE_IM_SPECIAL_HOUGH_H_
#define _ELISE_IM_SPECIAL_HOUGH_H_



/************************************************************/
/*                                                          */
/*                 HOUGH-TRANSF                             */
/*                                                          */
/************************************************************/

class ElHough : public Mcheck
{
	public :
		typedef enum
		{
			eModeValues,
			eModeGradient
		} tModeAccum;



		// Fonction relativement couteuses car font appel a des transfo de
		// hough

		virtual REAL DynamicModeValues() = 0;
		virtual REAL DynamicModeGradient() = 0;
		REAL Dynamic(tModeAccum) ;


		static void BenchPolygoneClipBandeVert();


		Pt2di SzXY() const {return Pt2di(mNbX,mNbY);}
		Pt2di SzTR() const {return Pt2di(mNbTeta,mNbRho);}
		Pt2di SzTR_Tot() const {return Pt2di(NbTetaTot(),mNbRho);}

		INT NbX() const {return mNbX;}
		INT NbY() const {return mNbY;}
		INT NbTeta() const {return mNbTeta;}
		INT NbRho() const {return mNbRho;}
		virtual INT NbRabTeta() const =0;
		virtual INT NbRabRho()  const =0;
		INT NbTetaTot()  const {return NbTeta()+2*NbRabTeta();}

                typedef enum
                {
                    ModeBasic,
                    ModeBasicSubPix,
                    ModeStepAdapt, // Basic + pas rho adaptatif
                    ModePixelExact
                } Mode;

            static ElHough * NewOne
                             (
                                  Pt2di Sz,     // taille en XY
                                  REAL StepRho,  // Pas en pixel
                                  REAL StepTeta, // Pas en equiv pixel pour la diag
                                  Mode,          // mode d'accum
                                  REAL RabRho,   // rab en pixel sur l'accum
                                  REAL RabTeta   // rab en radian sur l'accum
                              );
            static ElHough * NewOne(const ElSTDNS string &);

            virtual Im2D_INT4 PdsInit() = 0;
            virtual INT NbCel() const = 0;
            virtual void write_to_file(const std::string & name) const = 0 ;


            // ON PASSE L'IMAGE de POIDS CA RETOURNE L'ACCUMULATEUR
            virtual Im2D_INT4 Pds(Im2D_U_INT1) =0;

            // ON PASSE L'IMAGE CA RETOURNE L'ACCUMULATEUR, CA UTILISE l'ANGLE DU GRAD, il
            // doit etre reetale entre 0 et 256
            //
            virtual Im2D_INT4 PdsAng(Im2D_U_INT1 aImPds,Im2D_U_INT1 aImTeta,REAL Incert,bool IsGrad=true) =0;

            virtual Seg2d Grid_Hough2Euclid(Pt2dr) const = 0;
            virtual Pt2dr Grid_Euclid2Hough(Seg2d) const = 0;
            virtual  void CalcMaxLoc
                     (   
                         Im2D_INT4,
                         ElSTDNS vector<Pt2di> & Pts, 
                         REAL VoisRho, 
                         REAL VoisTeta, 
                         REAL Vmin
                     ) = 0;  

            virtual bool BandeConnectedVsup
                         (
                              Pt2di       p1,
                              Pt2di       p2,
                              Im2D_INT4   Im,
                              INT         VInf,
                              REAL        Tol
                         ) = 0;
             virtual void FiltrMaxLoc_BCVS
                          (
                                ElSTDNS vector<Pt2di> & Pts,
                                Im2D_INT4 Im,
                                REAL  FactInf,
                                REAL  TolGeom,
                                REAL   VoisRho,
                                REAL   VoisTeta
                          ) = 0;         

            virtual ~ElHough();


     protected :
		ElHough(Pt2di Sz,REAL StepTeta);
		void SetNbRho(INT);
        REAL LongEstimTeta() const;

		REAL       mStepTetaInit;
     private :
		const INT  mNbX;
		const INT  mNbY;
		const INT  mNbTeta;
		INT        mNbRho;
		Mode       mMode;

         // Non Implem
         ElHough(const ElHough &);
         void operator = (const ElHough &);
};               


void ElSegMerge(ElSTDNS vector<Seg2d> & VecInits,REAL dLong,REAL dLarg);



class EHFS_PrgDyn;

class ElHoughFiltSeg // : public ElHoughFiltSeg
{
      public :

         ElHoughFiltSeg(REAL Step,REAL Width,REAL LentghMax,Pt2di SzBox);
         virtual ~ElHoughFiltSeg();
         void SetSeg(const SegComp &);


         Pt2dr  Abs2Loc(const Pt2dr & p) const {return mP02Loc + mCS2Loc * p;}
         Pt2dr  Loc2Abs(const Pt2dr & p) const {return mP02Abs + mCS2Abs * p;}



                                                  
                                                  

          Pt2di SzMax() const;
          Pt2di SzCur() const;
          INT   YCentreCur() const ; 
   
          void GenPrgDynGet(std::vector<Seg2d> &,REAL dMin);



		  // Marche pour Im2D_U_INT1 et Im2D_INT1, car fait appel OptimSeg ...
          Seg2d ExtendSeg(SegComp s0,REAL DeltaMin,Im2DGen & Im);

          REAL Step() const {return mStep;}

          virtual REAL CostState(bool IsSeg,INT Abcisse) = 0;
          virtual REAL CostChange ( bool IsSeg, INT Abcisse, INT Abcisse2); 
          virtual REAL AverageCostChange () = 0; 
          virtual bool ExtrFree(); // return false
          virtual REAL CostNeutre() ;  // 0.5


		  void ExtendImage_proj(Im2D_U_INT1,INT Delta = (1<<30));
      protected  :

          Seg2d OneExtendSeg (bool& Ok,SegComp s0,Im2DGen & Im);
          void MakeIm(Im2D_U_INT1 Res,Im2D_U_INT1 InPut,U_INT1 def);
          void MakeIm(Im2D_INT1 Res,Im2D_INT1 InPut,INT1 def);
		  virtual void UpdateSeg() = 0;

      // private :

         void VerifSize(Im2D_U_INT1 anIm);
         void VerifSize(Im2D_INT1 anIm);
         void VerifSize(Im1D_U_INT1 anIm);

/*          static const INT  NbBits = 8; */
	 enum { NbBits = 8 };
         
         REAL                    mStep;
         REAL                    mWidthMax;
         REAL                    mLengthMax;
         INT                     mNbXMax;
         INT                     mNbYMax;

         REAL                    mLength;
         INT                     mNbX;

         Pt2dr                        mInvTgtSeg;
         Pt2dr                        mCS2Loc;
         Pt2dr                        mP02Loc;
         Pt2dr                        mCS2Abs;
         Pt2dr                        mP02Abs;
         EHFS_PrgDyn *                mPrgDyn;
         std::vector<std::pair<INT,INT> >  mVPairI;
         std::vector<Seg2d>           mVSegsExt;
		 Box2di                       mBox;

};



class EHFS_ScoreIm : public ElHoughFiltSeg
{
      public :

          EHFS_ScoreIm
		  (
		       REAL         Step,
			   REAL         Width,
			   REAL         LentghMax,
			   Im2D_U_INT1  ImGlob,
			   REAL         CostChg,
			   REAL         VMinRadiom = 255.0
		  );

          Im2D_U_INT1  ImLoc();
		  void ExtendImage_proj(INT Delta = (1<<30));

	  private :

          virtual REAL CostState(bool IsSeg,INT Abcisse);
          virtual REAL AverageCostChange ();

		  void UpdateSeg();

          Im2D_U_INT1             mImLoc;
		  U_INT1 *                mGainIfSeg;
          Im2D_U_INT1             mImGlob;
          REAL                    mCostChg;      
		  REAL                    mVminRadiom;
};





class EHFS_ScoreGrad : public ElHoughFiltSeg
{
      public :

          EHFS_ScoreGrad
		  (
		       REAL         Step,
			   REAL         Width,
			   REAL         LentghMax,
			   Im2D_INT1    ImGlobGX,
			   Im2D_INT1    ImGlobGY,
			   REAL         CostChg,
			   REAL         EcarTeta,
			   REAL         EcarMaxLoc,
			   REAL         SeuilGrad
		  );

          Im2D_U_INT1 ImLocGRho();
          Im2D_U_INT1 ImLocGTeta();
          Im1D_U_INT1 ImMaxLoc();

		  void TestGain(INT x);

      private  :

          void MakeImGradXY ( Im2D_INT1 OutGx, Im2D_INT1 OutGy,
                              Im2D_INT1 InPutGx, Im2D_INT1 InPutGy,
                              INT1 def
                            );

          void MakeImGradRhoTeta
                            ( Im2D_U_INT1 OutRho, Im2D_U_INT1 OutTeta,
                              Im2D_INT1 InPutGx, Im2D_INT1 InPutGy,
                              INT1 def
                            );

          void   MakeImMaxLoc(Im1D_U_INT1 ImMaxLoc,Im2D_U_INT1 InRho);

		  void MakeGainIfSeg();

         virtual REAL CostState(bool IsSeg,INT Abcisse);
         virtual REAL AverageCostChange();

		 void UpdateSeg();

         Im2D_INT1        mImLocGX;
         Im2D_INT1        mImLocGY;
         Im2D_U_INT1      mImLocGRho;
		 U_INT1 **        mDataLocRho;
         Im2D_U_INT1      mImLocGTeta;
		 U_INT1 **        mDataLocTeta;
         Im1D_U_INT1      mImYMaxLoc;
	     U_INT1 *         mDataYMaxLoc;

         Im2D_INT1        mImGlobGX;
         Im2D_INT1        mImGlobGY;

		 REAL             mCostChgt;

		 //REAL             mEcarTeta;
         Im1D_REAL8       mPdsTeta;
         REAL8 *          mDataPdsTeta;

		 //REAL8            mEcarMaxLoc;
         Im1D_REAL8       mPdsMaxLoc;
         REAL8 *          mDataPdsMaxLoc;

		 //REAL8            mSeuilGrad;
         Im1D_REAL8       mPdsRho;
         REAL8 *          mDataPdsRho;    

         Im1D_REAL8       mGainIfSeg;
		 REAL8 *          mDataGainIfSeg;
};



#endif // _ELISE_IM_SPECIAL_HOUGH_H_


















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
