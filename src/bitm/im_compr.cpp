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


template <class Type>  void ImPackBitsCompr1Block
     (
	  		ElSTDNS vector<U_INT2> &  	LInd,
	  		ElSTDNS vector<U_INT2> &	VInd,
	  		ElSTDNS vector<U_INT1> &  	Length,
	  		ElSTDNS vector<Type> &	Vals,
                        const Type *  	line,
                        INT 				nb_tot
     )
{

     for (INT i =0; i<nb_tot;)
     {
         INT i0 = i;
         INT lim = ElMin(nb_tot,i+128);

         /*
             Si au moins 2 element et 2 premier egaux : RLE
         */
         if ((i+1<lim) && (line[i]==line[i+1]))
         {
			Type v0 = line[i];
			while((i<lim) && (line[i] == v0))    i++;
			// Length.push_back(2*(i-i0-1)+1);
			Length.push_back
                        (
                            Gen_PackB_IM::CodeOfLengthRLE(i-i0)
                        );
			Vals.push_back(v0);

         }
         /*
              Sinon "run" litteralle
         */
         else
         {
             i++;
             bool cont = true;
             while(cont && (i<lim))
             {
                 // 2 elet dif : on continue le run
                 if (line[i]!=line[i-1])
                    i++;
                 // 3 elt egaux au -, on arete le run; le prochain sera rle
                 else if ((i+1 <lim) && (line[i] == line[i+1]))
                 {
                    cont = false;
                    i--;
                 }
                 // si run de 2, entre deux run litteraux, on le saute
                 else if
                 (
                         (i+2<lim)
                      && (line[i]!=line[i+1])
                      && (line[i+1]!=line[i+2])
                 )
                      i+=3;
                 // si run de 2, juste avant 1 dernier element, on le saute
                 else if ( i+2 == lim)
                      i+=2;
                 // sinon, run de 2= a coder en RLE
                 else
                 {
                    cont = false;
                    i--;
                 }
             }
             // Length.push_back(2*(i-i0-1));
             Length.push_back(Gen_PackB_IM::CodeOfLengthLIT(i-i0));
             for (INT k=i0;k<i; k++)
                 Vals.push_back(line[k]);
         }
     }

     LInd.push_back((unsigned short) Length.size());
     VInd.push_back((unsigned short) Vals.size());
}

template <class Type> void ImPackBitsCompr
     (
	  		ElSTDNS vector<U_INT2> &  	LInd,
	  		ElSTDNS vector<U_INT2> &	VInd,
	  		ElSTDNS vector<U_INT1> &  	Length,
	  		ElSTDNS vector<Type> &		Vals,
            const Type *  		line,
          	INT 				nb_tot,
          	INT 				per
     )
{
	LInd.clear();
	LInd.push_back(0);
	VInd.clear();
	VInd.push_back(0);
	Length.clear();
	Vals.clear();

	for (INT n=0; n<nb_tot; n+=per)
	{
		ImPackBitsCompr1Block
		(
			LInd,
			VInd,
			Length,
			Vals,
			line+n,
			ElMin(nb_tot-n,per)
		);
	}
}



/********************************************************************/
/**************       Line_PackB_IM    ******************************/
/********************************************************************/


template <class Type> Line_PackB_IM<Type>::Line_PackB_IM() :
	mRuns	(0),
	mNbRuns	(0)
{
}


template <class Type> void Line_PackB_IM<Type>::init
(
    INT                         BlockInit,
	ElSTDNS vector<U_INT2> &    LInd,
	ElSTDNS vector<U_INT2> &    VInd,
	ElSTDNS vector<U_INT1> &    Length,
	ElSTDNS vector<Type> &		Vals,
	const Type *      	line,
	INT                 nb_tot,
	INT                 per
)                
{
	if (line)
		ImPackBitsCompr(LInd,VInd,Length,Vals,line,nb_tot,per);

    if (! mRuns)
        mRuns = new RunsOfPer [LInd.size()-1];

    for (INT k=0 ; k<(INT)(LInd.size()-1) ; k++)
    {
        INT indL0 = LInd[k];
        INT indL1 = LInd[k+1];
        INT indV0 = VInd[k];
        INT indV1 = VInd[k+1];

        ElSTDNS vector <U_INT1> & vL =  mRuns[k+BlockInit].mLRun;
        ElSTDNS vector <Type> &   vV =  mRuns[k+BlockInit].mVRun;

        vL.clear();
        vV.clear();
        vL.reserve(indL1-indL0);
        vV.reserve(indV1-indV0);

        for (INT iL = indL0 ; iL<indL1 ; iL++)
            vL.push_back(Length[iL]);

        for (INT iV = indV0 ; iV<indV1 ; iV++)
            vV.push_back(Vals[iV]);
    }
}





template <class Type> Line_PackB_IM<Type>::~Line_PackB_IM()
{
    delete [] mRuns;
}






/********************************************************************/
/**************       Data_PackB_IM    ******************************/
/********************************************************************/


		//++++++++++++++++++++++
		// Init_Data_PackB_IM
		//++++++++++++++++++++++

template <class Type> class Init_Data_PackB_IM :   public Simple_OPBuf1<INT,Type>
{
	public :
		Init_Data_PackB_IM(Data_PackB_IM<Type> & DPIM) :
			dpim(DPIM)
		{
		}

		 void  calc_buf (INT ** output,Type *** );
	private :
		Data_PackB_IM<Type> &	dpim;
  		ElSTDNS vector<U_INT2>	LInd;
        ElSTDNS vector<U_INT2>	VInd;
        ElSTDNS vector<U_INT1>	Length;
        ElSTDNS vector<Type>	Vals;
};


template <class Type> void Init_Data_PackB_IM<Type>::calc_buf (INT ** output,Type *** input)
{
	dpim._LINES[this->ycur()].init
	(
        0,
		LInd,
		VInd,
		Length,
		Vals,
		input[0][0],
		this->tx(),
		dpim._per
	);
}


		//++++++++++++++++++++++
		// DPIM_In_Comp
		//++++++++++++++++++++++

template <class Type> class DPIM_Im_Comp : public Fonc_Num_Comp_TPL<INT>
{
	public :
		DPIM_Im_Comp 
		(
            const Arg_Fonc_Num_Comp & arg,
            Data_PackB_IM<Type> &     DPIM,
            bool      with_def_value
         ) ;                    


		inline void RunRLE(INT x0,INT x1,Type v)
		{
			for (INT x=x0; x<x1; x++)
				_im[x] = v;
		}

		inline void RunLIT(INT x0,INT x1,const Type * v)
		{
			for (INT x=x0; x<x1; x++)
				_im[x] = *(v++);
		}

	private :

		 const Pack_Of_Pts * values(const Pack_Of_Pts * GPts) ;

		Data_PackB_IM<Type>	& 	_dpim;
		bool					_wdef;
		INT 					* _im;
};

template <class Type> DPIM_Im_Comp<Type>::DPIM_Im_Comp
(
	const Arg_Fonc_Num_Comp & arg,
	Data_PackB_IM<Type> &     DPIM,
	bool      with_def_value
)	:
	Fonc_Num_Comp_TPL<INT>(arg,1,arg.flux()),
	_dpim	(DPIM),
	_wdef	(with_def_value)
{
}

template <class Type> 
const Pack_Of_Pts * DPIM_Im_Comp<Type>::values(const Pack_Of_Pts * GPts) 
{
	 const RLE_Pack_Of_Pts * rle_pts = GPts->rle_cast();
	INT nb = rle_pts->nb();

	if (!_wdef)
	{
		if (!rle_pts->inside(_dpim._p0,_dpim._p1))
           El_User_Dyn.ElAssert
           (
               false,
               EEM0
                 << "   Compressed :  out of domain while reading (RLE mode)\n"
                 << "  pts : "
                 << ElEM(rle_pts, rle_pts->ind_outside(_dpim._p0,_dpim._p1))
                 << ", Bitmap limits : "
                 << ElEM(_dpim._p1,rle_pts->dim())
                 << "\n"
           );                            
	}
	 _pack_out->set_nb(nb);
	_im =  _pack_out->_pts[0]-rle_pts->vx0();

    DeCompr
	(
	       _dpim._LINES[rle_pts->y()],
		*this,
		rle_pts->vx0(),
		rle_pts->x1(),
		_dpim._per
	);

	return _pack_out;
}


		//++++++++++++++++++++++
		// DPIM_In_Not_Comp
		//++++++++++++++++++++++

template <class Type>  class DPIM_In_Not_Comp : public Fonc_Num_Not_Comp
{
	public :


		DPIM_In_Not_Comp(Data_PackB_IM<Type> * DPIM,PackB_IM<Type> PIM,INT DV,bool WDV) :
			_dpim 	(DPIM),
			_pim  	(PIM),
			_DefVal (DV),
			_WithDV (WDV)
		{
		}

	private :
	  	Fonc_Num_Computed * compute(const Arg_Fonc_Num_Comp &);

        bool integral_fonc(bool integral_flux) const 
		{
			return true;
		}

        INT dimf_out() const 
		{
			return 1;
		}
                void VarDerNN(ElGrowingSetInd &) const{ELISE_ASSERT(false,"No VarDerNN");}

		Data_PackB_IM<Type> * _dpim;
		PackB_IM<Type>		_pim;
		INT				_DefVal;
		bool			_WithDV;
};

template <class Type>  Fonc_Num_Computed * DPIM_In_Not_Comp<Type>::compute(const Arg_Fonc_Num_Comp & arg)
{
	Tjs_El_User.ElAssert
	(
		arg.flux()->dim() == 2,
		EEM0	<< "Dim shoud equal 2 in PackB_IM readind"
				<< " (got " <<  arg.flux()->dim() << ")"
	);

	Tjs_El_User.ElAssert
	(
		arg.flux()->type() == Pack_Of_Pts::rle,
		EEM0 << "Should use RLE flux for reading PackB_IM"
	);

	Fonc_Num_Computed * res = new DPIM_Im_Comp<Type> (arg,*_dpim,_WithDV);

	if	(_WithDV)
		res =  clip_fonc_num_def_val
               (
                  arg,
                  res,
                  arg.flux(),
                  _dpim->_p0,
                  _dpim->_p1,
                  _DefVal
               );                   

	return res;
}

             /*************************************/
             /*                                   */
             /*         DPIM_Out_Comp             */
             /*                                   */
             /*************************************/
typedef enum 
        {
             Mode_DPIM_STD,
             Mode_DPIM_Lut
        }
        Mode_DPIM_Out;

#define DPIM_INLINE  inline

template <class Type> class   DPIM_Out_Comp : public Output_Computed
{
	public :

	DPIM_Out_Comp
        (
            const Arg_Output_Comp &   arg,
            Data_PackB_IM<Type> &     DPIM,
            Mode_DPIM_Out             aMode,
            Fonc_Num                  aLut
        );                    
		~DPIM_Out_Comp();


		inline void RunRLE(INT x0,INT x1,Type v)
        {
            for (INT x=x0; x<x1 ; x++)
                *(mCurDec++) = v;
        }
		inline void RunLIT(INT x0,INT x1,const Type * v)
        {
            for (INT x=x0; x<x1 ; x++)
               *(mCurDec++) = *(v++);
        }

	private :


        void  update(const Pack_Of_Pts * pts,const Pack_Of_Pts * vals_gen);

		Data_PackB_IM<Type>	& 	mDPIM;
        INT                     mPer;
        Type *                  mBufDec;
        Type *                  mCurDec;

	    ElSTDNS vector<U_INT2>      mVLIND;
	    ElSTDNS vector<U_INT2>      mVVIND;
	    ElSTDNS vector<U_INT1>      mVLength;
	    ElSTDNS vector<Type>  	mVVals;
            Mode_DPIM_Out               mMode;
            bool                        mIsCste;
            INT                         mVCste;
            INT                         mNBB;
            Im1D<Type,INT>              mLut;
            Type *                      mDataLut;
            Im1D<Type,INT>              mLutId;
            Type *                      mDataLutId;



      // Var et methodes pour le mode special : 

        DPIM_INLINE  void  ModeSpecUpdate(INT x0Loc,INT x1Loc);

        typename Line_PackB_IM<Type>::RunsOfPer* mLPB;

        std::vector<INT> mSpecLRun;
        std::vector<Type> mSpecVRun;

        INT mSpecIndL;
        INT mSpecIndVal;
        INT mSpecX0Cur;
        INT mSpecX1Cur;
        INT mSpecMaxIndL;

        INT mLastVal;
        INT mLengthCum;
        bool mFisrt;


        void ShowState()
        {
             cout << " Ind L " << mSpecIndL
                  << " Ind V " << mSpecIndVal
                  << " Cur X0  " << mSpecX0Cur
                  << " Cur X1  " << mSpecX1Cur
                  << "\n";
        }

        inline INT SpecCalcX1()
        {
            return mSpecX0Cur + mLPB->run_length_pixel(mSpecIndL);
        }

        inline void SpecNextIndexes()
        {
              mSpecX0Cur = mSpecX1Cur;
              mSpecIndVal += mLPB->run_length_compr(mSpecIndL);
              mSpecIndL++;
              mSpecX1Cur = SpecCalcX1();
        }

        inline void SpecPushCurRun()
        {
               if (mLengthCum)
               {
                 mSpecVRun.push_back(mLastVal);
                 mSpecLRun.push_back(mLengthCum);
               }
        }

        inline void SpecPush1Run(Type aVal,INT aLength)
        {
             if ((!mFisrt) && (aVal == mLastVal))
             {
                  mLengthCum += aLength;
             }
             else
             {
                 SpecPushCurRun();
                 mLastVal = aVal;
                 mLengthCum = aLength;
             }
             mFisrt = false;
        }


       inline void SpecUpdatGenerique(Type * aLut,INT x0Run,INT x1Run)
       {
             if (x0Run < x1Run)
             {
                 if (mLPB->run_rle(mSpecIndL))
                 {
                    SpecPush1Run
                    (
                           aLut[mLPB->mVRun[mSpecIndVal]],
                           x1Run-x0Run
                    );
                 }
                 else
                 {
                     for (INT k=x0Run ; k<x1Run ; k++)
                         SpecPush1Run
                         (
                              aLut[mLPB->mVRun[mSpecIndVal+k]],
                              1
                         );
                 }
             }
       }

       inline void SpecReplicatePartiel(INT x0Run,INT x1Run)
       {
           SpecUpdatGenerique(mDataLutId,x0Run,x1Run);
       }

       inline void SpecReplicateAll()
       {
           SpecUpdatGenerique(mDataLutId,0,mLPB->run_length_pixel(mSpecIndL));
       }

       inline void SpecUpdatIntervalePartiel(INT x0Run,INT x1Run)
       {
           SpecUpdatGenerique(mDataLut,x0Run,x1Run);
       }

       inline void SpecUpdatIntervaleAll()
       {
           SpecUpdatGenerique(mDataLut,0,mLPB->run_length_pixel(mSpecIndL));
       }

};


template <class Type>  DPIM_Out_Comp<Type>::~DPIM_Out_Comp()
{
    DELETE_VECTOR(mBufDec,0);
}

template <class Type>  DPIM_Out_Comp<Type>::DPIM_Out_Comp
                       (
                          const Arg_Output_Comp &   arg,
                          Data_PackB_IM<Type> &     DPIM,
                          Mode_DPIM_Out             aMode,
                          Fonc_Num                  aLut
                       ) :
                         Output_Computed(1),
                         mDPIM    (DPIM),
                         mPer     (DPIM.per()),
                         mBufDec  (NEW_VECTEUR(0,2*mPer+arg.flux()->sz_buf(),Type)),
                         mMode    (aMode),
                         mIsCste  (aLut.IsCsteRealDim1(mVCste)),
                         mNBB     (nbb_type_num(type_of_ptr((Type *)0))),
                         mLut     (1<<mNBB),
                         mDataLut (mLut.data()),
                         mLutId     (1<<mNBB),
                         mDataLutId (mLutId.data())
{
    switch (aMode)
    {
       case Mode_DPIM_STD:
       break;

       case Mode_DPIM_Lut:
            ELISE_COPY(mLut.all_pts(),aLut,mLut.out());
       break;

    }
    ELISE_COPY(mLutId.all_pts(),FX,mLutId.out());
}


template <class Type> void  DPIM_Out_Comp<Type>::ModeSpecUpdate
                            (
                                 INT x0Loc,
                                 INT x1Loc
                             )
{
    // Un cas particulier courant , qui fait gagner bcp de temps


    if (mIsCste  && (x0Loc==0) &&(x1Loc==mPer))
    {
          mLPB->Clear();
          mLPB->PushRleSafe128(mVCste,mPer);
          return;
    }
    mSpecMaxIndL = (int) mLPB->mLRun.size();
    mLPB->mLRun.push_back(Gen_PackB_IM::CodeOfLengthRLE(1));



    mSpecVRun.clear();
    mSpecLRun.clear();
   
    mSpecIndL   =0;
    mSpecIndVal =0;
    mSpecX0Cur  = 0;
    mSpecX1Cur  = SpecCalcX1();

    // mLastVal;
    mLengthCum = 0 ;
    mFisrt = true;


    while (mSpecX1Cur <= x0Loc)
    {
         SpecReplicateAll();
         SpecNextIndexes();
    }

    SpecReplicatePartiel(0,x0Loc-mSpecX0Cur);

    /* Si l'intervalle a modifier est compris dans une seule RUN */
    if (x1Loc<=mSpecX1Cur)
    {
         SpecUpdatIntervalePartiel(x0Loc-mSpecX0Cur,x1Loc-mSpecX0Cur);
         SpecReplicatePartiel(x1Loc-mSpecX0Cur,mSpecX1Cur-mSpecX0Cur);
         SpecNextIndexes();
    }
    else
    {
         SpecUpdatIntervalePartiel(x0Loc-mSpecX0Cur,mSpecX1Cur-mSpecX0Cur);
         SpecNextIndexes();

          while (mSpecX1Cur <x1Loc)
          {
             SpecUpdatIntervaleAll();
             SpecNextIndexes();
          }


         SpecUpdatIntervalePartiel(0,x1Loc-mSpecX0Cur);
         SpecReplicatePartiel(x1Loc-mSpecX0Cur,mSpecX1Cur-mSpecX0Cur);
         SpecNextIndexes();
    }


    while (mSpecIndL != mSpecMaxIndL)
    {
         SpecReplicateAll();
         SpecNextIndexes();
    }

    mLPB->Clear();
    SpecPushCurRun();
    INT aNbRun = (INT) mSpecVRun.size();

    for (INT k=0 ; k<aNbRun ; k++)
    {
        mLPB->PushRleSafe128(mSpecVRun[k],mSpecLRun[k]);
    }
    
}

template <class Type> void DPIM_Out_Comp<Type>::update
                      (
                            const Pack_Of_Pts * pts_gen,
                            const Pack_Of_Pts * vals_gen
                      )
{
   const RLE_Pack_Of_Pts *      Pts = pts_gen->rle_cast();
   const Std_Pack_Of_Pts<INT> * SVals = vals_gen->int_cast();

   INT nb = Pts->nb();
   if (!nb) return;

   INT x0 = Pts->vx0();
   INT y = Pts->y();
   INT x1 = x0+nb;


   INT iBlock0 = (x0/mPer);
   INT iBlock1 = ((x1-1)/mPer+1);


   if (mMode != Mode_DPIM_STD)
   {
        for (INT iBlock = iBlock0; iBlock<iBlock1 ; iBlock++)
        {
            INT xDebBlock  = iBlock * mPer;
            mLPB =  mDPIM._LINES[y].mRuns+iBlock;
            ModeSpecUpdate
            (
                  ElMax(0,x0-xDebBlock),
                  ElMin(mPer,x1-xDebBlock)
            ); 
        }
   }
   else
   {
        INT xDebBlock  = iBlock0 * mPer;
        INT xFinBlock  =  ElMin(iBlock1*mPer,mDPIM.tx());
        INT * Vals = SVals->_pts[0];

        mCurDec  = mBufDec;
        DeCompr
        (
            mDPIM._LINES[y],
            *this,
            xDebBlock,
            xFinBlock,
            mPer    
        );

        convert
        (
          mBufDec + (x0-iBlock0*mPer),
          Vals,
          nb
        );


        mDPIM._LINES[y].init
        (
              iBlock0,
              mVLIND,
              mVVIND,
              mVLength,
              mVVals,
              mBufDec,
              xFinBlock-xDebBlock,
              mPer
        );
   }

}

             /*************************************/
             /*                                   */
             /*         DPIM_Out_NotComp          */
             /*                                   */
             /*************************************/

template <class Type> class DPIM_Out_NotComp  : public Output_Not_Comp  
{
    public :

       Output_Computed * compute(const Arg_Output_Comp & arg);

        DPIM_Out_NotComp
        (
              Data_PackB_IM<Type> *  dpim,
              PackB_IM<Type>         pim,
              Mode_DPIM_Out          aMode,
              Fonc_Num               aLut
        ) :
              _dpim  (dpim),
              _pim   (pim),
              mMode  (aMode),
              mLut   (aLut)
        {
        }

    private  :
         
		Data_PackB_IM<Type> *    _dpim;
		PackB_IM<Type>		 _pim;
                Mode_DPIM_Out            mMode;
                Fonc_Num                 mLut;

};

template <class Type> Output_Computed * DPIM_Out_NotComp<Type>::compute(const Arg_Output_Comp & arg)
{
	Tjs_El_User.ElAssert
	(
		arg.flux()->dim() == 2,
		EEM0	<< "Dim shoud equal 2 in PackB_IM write"
				<< " (got " <<  arg.flux()->dim() << ")"
	);

	Tjs_El_User.ElAssert
	(
		arg.flux()->type() == Pack_Of_Pts::rle,
		EEM0 << "Should use RLE flux for write PackB_IM"
	);



    Output_Computed * res = new DPIM_Out_Comp<Type>(arg,*_dpim,mMode,mLut);

    res =  out_adapt_type_fonc(arg,res,Pack_Of_Pts::integer);
    res = clip_out_put(res,arg,Pt2di(0,0),_dpim->sz());

    return res;
}




		//++++++++++++++++++++++
		// Data_PackB_IM
		//++++++++++++++++++++++


template <class Type> Data_PackB_IM<Type>::~Data_PackB_IM()
{
	DELETE_TAB(_LINES);
}

template <class Type> Data_PackB_IM<Type>::Data_PackB_IM
(
	INT 	tx,
	INT 	ty,
	Fonc_Num Finit,
	INT 	per
) :
	_LINES  (NEW_TAB(ty,Line_PackB_IM<Type>)), 
	_tx 	(tx),
	_ty 	(ty),
	_per	(ElAbs(per))
{
	_p0[0]= _p0[1] = 0;
	_p1[0]= _tx;
	_p1[1]= _ty;

	if (per > 0)
	{
		ELISE_COPY
		(
			rectangle(Pt2di(0,0),Pt2di(_tx,_ty)),
			create_op_buf_simple_tpl
			(
				new Init_Data_PackB_IM<Type> (*this),
				Finit,
				1,
				Box2di(Pt2di(0,0),Pt2di(0,0))
			),
			Output::onul()  
		);
	}

}


/********************************************************************/
/**************          PackB_IM      ******************************/
/********************************************************************/

template <class Type> PackB_IM<Type>::PackB_IM(Data_PackB_IM<Type> * dpim) :
	PRC0(dpim)
{
}


template <class Type> PackB_IM<Type>::PackB_IM(INT tx,INT ty,Fonc_Num F,INT per) :
	PRC0(new Data_PackB_IM<Type>(tx,ty,F,per))
{
}

template <class Type> Fonc_Num PackB_IM<Type>::in()
{
	return Fonc_Num(new DPIM_In_Not_Comp<Type>(dpim(),*this,-1234567,false));
}

template <class Type> Fonc_Num PackB_IM<Type>::in(INT val)
{
	return Fonc_Num(new DPIM_In_Not_Comp<Type>(dpim(),*this,val,true));
}

template <class Type> Elise_Rect PackB_IM<Type>::box() const
{
	return Elise_Rect(Pt2di(0,0),sz());
}

template <class Type> Data_PackB_IM<Type> * PackB_IM<Type>::dpim()
{
	return SAFE_DYNC(Data_PackB_IM<Type> *,_ptr);
}

template <class Type> const Data_PackB_IM<Type> * PackB_IM<Type>::dpim() const
{
	return SAFE_DYNC(const Data_PackB_IM<Type> *,_ptr);
}

template <class Type> Pt2di PackB_IM<Type>::sz() const 
{
	return dpim()->sz();
}

template <class Type> Output PackB_IM<Type>::out()
{
    return Output(new DPIM_Out_NotComp<Type>(dpim(),*this,Mode_DPIM_STD,-1));
}

template <class Type> Output PackB_IM<Type>::OutLut(Fonc_Num aLut)
{
    return Output(new DPIM_Out_NotComp<Type>(dpim(),*this,Mode_DPIM_Lut,aLut));
}
           

           

#define INIT_PACKB_IM(Type)\
template class Line_PackB_IM<Type>;\
template class Data_PackB_IM<Type>;\
template class PackB_IM<Type>;\
template class Init_Data_PackB_IM<Type>;\
template class DPIM_Im_Comp<Type>;\
template class DPIM_In_Not_Comp<Type>;


INIT_PACKB_IM(U_INT1);
INIT_PACKB_IM(U_INT2);








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
