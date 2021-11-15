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



#ifndef _ELISE_OPERATOR_H
#define _ELISE_OPERATOR_H

/*
    pour *, cas special de la dimension 2; + cas special ou l'un des operandes est de dimension 1
    rajouter la fonction scal, pour le cas general;

    +, max, min, *
    - (bin)


    &,&&,|,||,^,"^^",%, vrai % (tjs > 0) : binaire entiers.

    <,>,<=,>=,==,!= : comparaison, operande mixte, resultat entier.


     cos,sin,tang,atan,acos,asin,sinh,cosh,tanh,acosh,atanh
     exp,exp10  : reel, admetant eventuellement des operandes entiers mais converti en reel.

     pow, + function du numerical recipes.

     -,Abs,carre   : unaire preservant le type
     ~,!           : unaire entier

*/

/*
class OperBin  
{
   public :

     virtual void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const;
     virtual void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const;
     virtual void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const INT  *t2,INT nb) const;
     virtual void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const;

     virtual void t0_eg_t1_op_t2(INT  * t0,const REAL  * t1,const INT  *t2,INT nb) const;
     virtual void t0_eg_t1_op_t2(INT  * t0,const INT   * t1,const REAL  *t2,INT nb) const;
     virtual void t0_eg_t1_op_t2(INT  * t0,const REAL  * t1,const REAL *t2,INT nb) const;
};
*/

class OperBin  
{
};

    /*----------------------------------------*/
    /*                                        */
    /*    Binary  integer    Operators        */
    /*                                        */
    /*----------------------------------------*/

   // &, && ,||,^,|,xor

class OperBinInt : public OperBin
{
   public :
      virtual void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const = 0;
      virtual ~OperBinInt() {}
};

extern const OperBinInt & OpAndBB;          // &
extern const OperBinInt & OperAnd;          // &&
extern const OperBinInt & OpOrBB;           // |
extern const OperBinInt & OperOr;           // ||
extern const OperBinInt & OperXorBB;        //  ^
extern const OperBinInt & OperXor;    
extern const OperBinInt & OperStdMod;       // %
extern const OperBinInt & OperMod;          //  so that mod(-1,4) => 3 and not -1 !
extern const OperBinInt & OperRightShift;   //  >>
extern const OperBinInt & OperLeftShift;    //   <<

    /*----------------------------------------*/
    /*                                        */
    /*    comparison     Operators            */
    /*                                        */
    /*----------------------------------------*/

//  ==, != ,<= ,>=, >, <

class OperComp : public OperBin
{
   public :
     virtual void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const = 0;
     virtual void t0_eg_t1_op_t2(INT  * t0,const REAL  * t1,const INT  *t2,INT nb) const = 0;
     virtual void t0_eg_t1_op_t2(INT  * t0,const INT   * t1,const REAL  *t2,INT nb) const = 0;
     virtual void t0_eg_t1_op_t2(INT  * t0,const REAL  * t1,const REAL *t2,INT nb) const = 0;
     virtual ~OperComp() {}
};

extern const OperComp & OpEqual;
extern const OperComp & OpNotEq;
extern const OperComp & OpInfStr;
extern const OperComp & OpInfOrEq;
extern const OperComp & OpSupStr;
extern const OperComp & OpSupOrEq;


    /*----------------------------------------*/
    /*                                        */
    /*    Binary      Operators               */
    /*                                        */
    /*----------------------------------------*/

//  %


//      -, pow, /
class OperBinMixte : public OperBin 
{
   public :

       void t0_eg_t1_op_t2(REAL16 * t0,const REAL16 * t1,const REAL16 *t2,INT nb) const ;
       void t0_eg_t1_op_t2(_INT8 * t0,const _INT8 * t1,const _INT8 *t2,INT nb) const ;

       virtual void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const REAL *t2,INT nb) const = 0;
       virtual void t0_eg_t1_op_t2(REAL * t0,const REAL * t1,const INT  *t2,INT nb) const = 0;
       virtual void t0_eg_t1_op_t2(REAL * t0,const INT  * t1,const REAL *t2,INT nb) const = 0;
       virtual void t0_eg_t1_op_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const = 0;
       virtual ~OperBinMixte() {}
};


extern const OperBinMixte & OpDiv;
extern const OperBinMixte & OpPow2;
extern const OperBinMixte & OpMinus2;  // binary -
extern const OperBinMixte & OpF1OrF2IfBadNum;  // binary -


// +, *, max, min
class OperAssocMixte : public OperBinMixte 
{

   public :

     static OperAssocMixte * GetFromName(const std::string &,bool Svp=false);

     typedef enum
     {
         Sum,
         Max,
         Min,
         Mul
     } Id;
     inline Id id() const  { return _id;}

   protected :
     OperAssocMixte(Id);
   public :

     REAL16 opel(REAL16,REAL16) const;
     _INT8  opel(_INT8 ,_INT8 ) const;
     virtual REAL opel(REAL,REAL) const =0;
     virtual INT  opel(INT ,INT ) const =0;

     virtual Fonc_Num opf(Fonc_Num,Fonc_Num) const = 0;

     virtual INT  red_tab(const INT  *,INT nb,INT     v_init) const =0;
     virtual INT   ineutre(void) const =0;
     inline INT red_tab (const INT  * vals,INT nb) const
     {
             return red_tab(vals,nb,ineutre());
     }
     inline void  set_neutre(INT & v) const {v = ineutre();}
    
     virtual REAL red_tab(const REAL *,INT nb,REAL    v_init) const =0;
     virtual REAL  rneutre(void) const =0;
     inline REAL red_tab (const REAL  * vals,INT nb) const
     {
             return red_tab(vals,nb,rneutre());
     }


     long double red_tab (const long double  * /*vals*/,INT /*nb*/) const
     {
         ELISE_ASSERT(false,"No Red Tab Long Double");
         return 0;
     }


     inline void  set_neutre(REAL& v) const {v = rneutre();}

      virtual void integral(INT *out,const INT * in,INT nb) const = 0;
      virtual void integral(REAL *out,const REAL * in,INT nb) const = 0;

      virtual void reduce_seg
           ( INT * out, const INT * in, INT * buf_av, INT * buf_ar,
             INT   x_min, INT   x_max, INT  dx0, INT   dx1) const = 0;

      virtual void reduce_seg
           ( REAL * out, const REAL * in, REAL * buf_av, REAL * buf_ar,
             INT   x_min, INT   x_max, INT  dx0, INT   dx1) const = 0;

     virtual bool grp_oper() const;

     virtual void t0_eg_t1_opinv_t2(INT  * t0,const INT  * t1,const INT  *t2,INT nb) const;
     virtual void t0_eg_t1_opinv_t2(REAL  * t0,const REAL  * t1,const REAL  *t2,INT nb) const;


     virtual void t0_opinveg_t1(INT  * t0,const INT  * t1,INT nb) const;  // -=
     virtual void t0_opinveg_t1(REAL  * t0,const REAL  * t1,INT nb) const;

     virtual void t0_opeg_t1(INT  * t0,const INT  * t1,INT nb)  const;     // +=
     virtual void t0_opeg_t1(REAL  * t0,const REAL  * t1,INT nb)  const;

     virtual const char * name () const = 0;
     virtual ~OperAssocMixte() {}

   private :

     Id _id ;
};

inline double NeutreOpAss(const OperAssocMixte &  anOp,double *) {return anOp.rneutre();}
inline int    NeutreOpAss(const OperAssocMixte &  anOp,int    *) {return anOp.ineutre();}

typedef enum eComplFRA
{
    eCFR_None,
    eCFR_Per,
    eCFR_Neutre
} eComplFRA;

template <class Type> class cFastReducAssoc
{
     public :
         cFastReducAssoc(const OperAssocMixte & anOp,INT   x_min, INT   x_max, INT  dx0, INT   dx1) :
             mOp      (&anOp),
             mNeutre  (NeutreOpAss(*mOp,(Type*)0)),
             mXMin    (ElMin(x_min,x_max)),
             mXMax    (ElMax(x_min,x_max)),
             mDx0     (ElMin(dx0,dx1)),
             mDx1     (ElMax(dx0,dx1)),
             mPer     (mDx1-mDx0+1),
             mNbEl    (mXMax-mXMin + mPer),
             mOriX    (mXMin+mDx0),
             mBufIn   (mNbEl),
             mDIn     (mBufIn.data() - mOriX),
             mBufOut  (mNbEl),
             mDOut    (mBufOut.data() - mOriX),
             mBufAv   (mNbEl),
             mDAv     (mBufAv.data() - mOriX),
             mBufAr   (mNbEl),
             mDAr     (mBufAr.data() - mOriX)
         {
         }
         void Compute(int aDx0,int aDx1,eComplFRA aModeC)
         {
             if (aModeC!=eCFR_None)
             {
                ComplemInput(aModeC==eCFR_Per,aDx0,aDx1);
             }
             VerifDx(aDx0,aDx1);
             mOp->reduce_seg(mDOut,mDIn,mDAv,mDAr,mXMin,mXMax,aDx0,aDx1);
         }

         Type & In(int anX)
         {
            ELISE_ASSERT((anX>=mXMin+mDx0) && (anX<mXMax+mDx1),"cFastReducAssoc::SetIn");
            return mDIn[anX] ;
         }
         const Type & Out(int anX)
         {
            ELISE_ASSERT((anX>=mXMin) && (anX<mXMax),"cFastReducAssoc::SetIn");
            return mDOut[anX];
         }

     private :

        void ComplemInput(bool Periodik,int aDx0,int aDx1)
         {
             VerifDx(aDx0,aDx1);
             for (int aX=mXMin+aDx0; aX<mXMin ; aX++)
                mDIn[aX]  = Complem(aX,Periodik);
             for (int aX=mXMax; aX<mXMax+aDx1 ; aX++)
                mDIn[aX]  = Complem(aX,Periodik);
         }
         void SetOp(const OperAssocMixte & anOp) {mOp = & anOp;}


         const Type &  Complem(int aX,bool Per)
         {
             if (Per)
             {
                 int aXp = mXMin + mod((aX-mXMin),mXMax-mXMin);
                 return mDIn[aXp];
             }
             return mNeutre;
         }
         void VerifDx(int aDx0,int aDx1)
         {
              ELISE_ASSERT((aDx0>=mDx0) && (aDx1<mDx1),"cFastReducAssoc::Compute");
         }

         const OperAssocMixte * mOp;
         Type            mNeutre;
         int             mXMin;
         int             mXMax;
         int             mDx0;
         int             mDx1;
         int             mPer;
         int             mNbEl;
         int             mOriX;
         Im1D<Type,Type> mBufIn;
         Type *          mDIn;
         Im1D<Type,Type> mBufOut;
         Type *          mDOut;
         Im1D<Type,Type> mBufAv;
         Type *          mDAv;
         Im1D<Type,Type> mBufAr;
         Type *          mDAr;
};





        /***********************************************/
        /*         Unary operator                      */
        /***********************************************/


    /*----------------------------------------*/
    /*                                        */
    /*    Mixte operator                      */
    /*                                        */
    /*----------------------------------------*/

/*  
     Defined in general/util.h , should be something like :

     template <class Type> void tab_Abs (Type * out,const Type * in,INT nb);

     template <class Type> void tab_minus1 (Type * out,const Type * in,INT nb);

     template <class Type> void tab_square (Type * out,const Type * in,INT nb);

*/ 

    /*----------------------------------------*/
    /*                                        */
    /*    Mathematical operator               */
    /*                                        */
    /*----------------------------------------*/

extern void tab_sqrt(REAL * out, const REAL * in,INT nb);
extern void tab_cos(REAL * out, const REAL * in,INT nb);
extern void tab_sin(REAL * out, const REAL * in,INT nb);
extern void tab_tan(REAL * out, const REAL * in,INT nb);
extern void tab_atan(REAL * out, const REAL * in,INT nb);

extern void tab_exp(REAL * out, const REAL * in,INT nb);
extern void tab_log(REAL * out, const REAL * in,INT nb);
extern void tab_log2(REAL * out, const REAL * in,INT nb);

extern REAL erfcc (REAL x);
extern void tab_erfcc(REAL * out, const REAL * in,INT nb);


typedef Fonc_Num (*tOperFuncUnaire)(Fonc_Num f1);
tOperFuncUnaire  OperFuncUnaireFromName(const std::string & aName);

typedef Fonc_Num (*tOperFuncBin)(Fonc_Num f1,Fonc_Num f2);
tOperFuncBin  OperFuncBinaireFromName(const std::string & aName);




#endif  /* !  _ELISE_OPERATOR_H */


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
