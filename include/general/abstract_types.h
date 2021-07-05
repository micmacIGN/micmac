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


#define NoOperatorVirgule

#if (0) // (ELISE_unix)
// Pour les quelques fichiers (b_0_45.cpp par ex) qui generent des erreurs, a cause de "," dans la STL
#ifndef NoTemplateOperatorVirgule
template <class Type,class Type2>   Type operator , (Type,Type2)
{
      ToToTo();
}
#else
#ifndef NoSimpleTemplateOperatorVirgule
template <class Type>   Type operator , (Type,Type)
{
      ToToTo();
}
#endif
#endif
#endif


#ifndef _ELISE_ABSTRACT_TYPE_H
#define _ELISE_ABSTRACT_TYPE_H

#ifndef PI
// const REAL PI =  3.14159265358979323846;
// Je mets un define car j'ai un doute sur le fonctionnement quand
// on initialise d'autres constantes a partir de PI
#define PI  3.14159265358979323846
#endif

class OperAssocMixte ;
template <class Type> class   Std_Pack_Of_Pts;
class Fonc_Num;
class Neigh_Rel;




    /*----------------------------------------*/
    /*                                        */
    /*          Flux_Pts                      */
    /*                                        */
    /*----------------------------------------*/

// For commidity of end user, this class is named Flux_Pts;
// but, in fact, this essentially a class of pointer to 
// the class Flux_Pts_Not_Comp.



class Flux_Pts : public  PRC0
{
   // &Arg_Flux_Pts_Comp : a reference, just to allow
   // the definition of the class in another file.


   public :
     ~Flux_Pts(void);
     Flux_Pts(class Flux_Pts_Not_Comp *);
     class Flux_Pts_Computed * compute (const class  Arg_Flux_Pts_Comp &);

     Flux_Pts(Std_Pack_Of_Pts<INT> *);
     Flux_Pts(ElList<Pt2di>);
     Flux_Pts(Pt2di);

     Flux_Pts chc(class Fonc_Num);

     Flux_Pts operator || (Flux_Pts);
};

        //===============
        // GEOMETRIC
        //===============

            // 1-D
            // - - - 

extern Flux_Pts rectangle(INT,INT);

extern Flux_Pts border_rect(INT p1,INT p2,INT b1,INT b2);
extern Flux_Pts border_rect(INT p1,INT p2,INT b);

            // 2-D
            // - - - 

                   // surfacic

extern Flux_Pts rectangle(Pt2di p0,Pt2di p1);
extern Flux_Pts rectangle(const Box2di &);


                   // lineic

                         // lines 

extern Flux_Pts line(Pt2di p0,Pt2di p1);
extern Flux_Pts line(ElList<Pt2di>,bool ferm = false);
extern Flux_Pts line_4c(Pt2di p0,Pt2di p1);
extern Flux_Pts line_4c(ElList<Pt2di>,bool ferm = false);
extern Flux_Pts line_for_poly(ElList<Pt2di>);

extern Flux_Pts polygone(ElList<Pt2di>,bool front = true);
extern Flux_Pts quick_poly(ElList<Pt2di>);

/*  Interface pour convertir les container standard en "vielle"
    liste elise.
*/
template <class tContPts> ElList<Pt2di> ToListPt2di(const tContPts & aCont)
{
   ElList<Pt2di> aL;
   for 
   (
      typename tContPts::const_iterator itP=aCont.begin();
      itP!=aCont.end();
      itP++
   )
   {
       aL = aL+Pt2di(*itP);
   }
   return aL;
}

                       //  border of rectangles

extern Flux_Pts border_rect(Pt2di p1,Pt2di p2,Pt2di b1,Pt2di b2);
extern Flux_Pts border_rect(Pt2di p1,Pt2di p2,INT sz=1);

                         // cirle,ellipse, arc  

extern Flux_Pts circle(Pt2dr,REAL,bool v8 = true);
extern Flux_Pts disc(Pt2dr,REAL,bool front = true);
extern Flux_Pts arc_cir(Pt2dr,REAL r,REAL a0,REAL a1,bool v8 = true);

extern Flux_Pts fr_sector_ang(Pt2dr,REAL r,REAL a0,REAL a1,bool v8 = true);
extern Flux_Pts fr_chord_ang (Pt2dr,REAL r,REAL a0,REAL a1,bool v8 = true);
extern Flux_Pts sector_ang(Pt2dr,REAL r,REAL a0,REAL a1,bool front  = true);
extern Flux_Pts chord_ang  (Pt2dr,REAL r,REAL a0,REAL a1,bool front = true);

extern Flux_Pts ellipse(Pt2dr c,REAL A,REAL B,REAL teta,bool v8 = true);
extern Flux_Pts ell_fill(Pt2dr c,REAL A,REAL B,REAL teta,bool front = true);
extern Flux_Pts arc_ellipse
       (Pt2dr c,REAL A,REAL B,REAL teta,REAL a0,REAL a1,bool v8 = true);

extern Flux_Pts sector_ell
                (Pt2dr c,REAL A,REAL B,
                 REAL teta,REAL a0,
                 REAL a1,bool front = true);
extern Flux_Pts chord_ell
                (Pt2dr c,REAL A,REAL B,
                 REAL teta,REAL a0,
                 REAL a1,bool front = true);
extern Flux_Pts fr_sector_ell
                (Pt2dr c,REAL A,REAL B,
                 REAL teta,REAL a0,
                 REAL a1,bool v8 = true);
extern Flux_Pts fr_chord_ell
                (Pt2dr c,REAL A,REAL B,
                 REAL teta,REAL a0,
                 REAL a1,bool v8 = true);


                   // special 

                         // map the rectangle p1,p2 with digital line of direction u
Flux_Pts line_map_rect(Pt2di u,Pt2di p0,Pt2di p1);
                        // idem for 1-D rectangles 
Flux_Pts line_map_rect(INT  dx,INT  x0,INT  x1);

                // see "include/general/bitm.h" for lpts

            // 3-D
extern Flux_Pts rectangle(Pt3di p0,Pt3di p1);

            // K-D
            // - - - 

extern Flux_Pts rectangle(const INT*,const INT*,INT);
extern Flux_Pts border_rect(  const  INT * p1,
                              const  INT * p2,
                              const  INT * b1,
                              const  INT * b2,
                              INT     dim
                     );

extern Flux_Pts border_rect(  const  INT * p1,
                              const  INT * p2,
                                     INT   b ,
                                     INT   dim
                     );


        //==================
        // operator
        //==================


Flux_Pts select(Flux_Pts flx,Fonc_Num fonc); 
extern Flux_Pts dilate(Flux_Pts flx,Neigh_Rel fonc); 
extern Flux_Pts conc(Flux_Pts flx,Neigh_Rel Rel,INT nb_step_max = 1000000000); 


    /*----------------------------------------*/
    /*                                        */
    /*          Fonc_Num                      */
    /*                                        */
    /*----------------------------------------*/
class ElGrowingSetInd;
class cElCompileFN;

class Fonc_Num : public  PRC0
{
   friend class Arg_FNOPB;

   public :
     REAL ValFoncGen(Pt2di) const;  
     REAL ValFoncGenR(Pt2dr) const;  
          // valable qqsoit la fonction, assez long
         // REAL ValFonc(const class PtsKD &) const; pour les fonction analytique

     std::string NameCpp();
     bool        HasNameCpp();
     void        SetNameCpp(const std::string &);

     ~Fonc_Num();
     Fonc_Num(INT);
     Fonc_Num(INT,INT);
     Fonc_Num(Pt2di);
     Fonc_Num(INT,INT,INT);
     Fonc_Num(INT,INT,INT,INT);
     Fonc_Num(REAL);
     Fonc_Num(REAL,REAL);
     Fonc_Num(Pt2dr);
     Fonc_Num(REAL,REAL,REAL);
     Fonc_Num(REAL,REAL,REAL,REAL);
     Fonc_Num(class Fonc_Num_Not_Comp *);

     void compile(cElCompileFN &);
     bool integral_fonc (bool integral_flux) const ;
     INT dimf_out() const;
     REAL ValFonc(const class PtsKD &) const;
     Fonc_Num deriv(INT kth) const;
     Fonc_Num deriv(class cVarSpec ) const;
     INT DegrePoly() const; // retourne  conventionnellement -1 si pas polynome

     void VarDerNN(ElGrowingSetInd &) const; // Remplit l'indexe des variable a derivee non nulle
	 REAL ValDeriv(const  PtsKD &  pts,INT k) const;
	 INT  NumCoord() const; // En qq sorte la fonc inverse de kth_coord, 
	                        // => Erreur si pas une fonc coord
     void show(std::ostream & os) const;
     Fonc_Num Simplify() ;
     void inspect() const;  // debug
     bool is0() const;
     bool is1() const;
     bool IsCsteRealDim1(REAL & aVal) const;
     bool IsCsteRealDim1(INT  & aVal) const;

     Fonc_Num operator [] (Fonc_Num);
     
     Fonc_Num v0();     // = kth_proj(*this,0);
     Fonc_Num v1();     // = kth_proj(*this,1);
     Fonc_Num v2();     // = kth_proj(*this,2);

     Fonc_Num permut(const std::vector<int>&  kth) const;
     Fonc_Num kth_proj(INT kth) const;
     Fonc_Num shift_coord(int shift) const;


     class Fonc_Num_Computed * compute (const class  Arg_Fonc_Num_Comp &);

     Fonc_Num();

  //  private :

     static const Fonc_Num No_Fonc; // used as defualt values for some foncs,
                                    // really point to 0 !
     bool really_a_fonc() const;  // does it point to non 0


   //  Permet de comparer formellement deux expression formelle 
   //  utilise pour gerer des dictionnaire d'expression
   

       typedef enum {
                   eIsICste,
                   eIsRCste,
                   eIsFCoord,
                   eIsOpBin,
                   eIsOpUn,
                   eIsVarSpec,
		   eIsSurfIER,
                   eIsUnknown
                }  tKindOfExpr;

       INT CmpFormel(const Fonc_Num &) const;
       tKindOfExpr KindOfExpr() const;
       bool IsVarSpec() const;
       bool IsAtom() const;
       bool IsOpbin() const;
       bool IsOpUn() const;


   protected :
     class cTagCsteDer {};
     Fonc_Num (cTagCsteDer,REAL aVal,INT anInd,const std::string & );
};


class cVarSpec : public  Fonc_Num
{
     public :
         cVarSpec(REAL aVal,const std::string & = "");
         void Set(REAL aVal);
         double * AdrVal() const;

         const std::string & Name() const;

         cVarSpec(class Fonc_Num_Not_Comp *);
     private :


         friend class Fonc_Num;
         INT IndexeDeriv();
         static INT TheCptIndexeNumerotation;
};

template <> inline Fonc_Num ElMax (Fonc_Num ,Fonc_Num ) {ELISE_ASSERT(false,"ElMax(Fonc_Num)");return 0;}
template <> inline Fonc_Num ElMin (Fonc_Num ,Fonc_Num ) {ELISE_ASSERT(false,"ElMax(Fonc_Num)");return 0;}
template <> inline Fonc_Num ElAbs   (Fonc_Num ) {ELISE_ASSERT(false,"ElMax(Fonc_Num)");return 0;}

     // operator with Flux


      // rebuf specifies explictitely the ``rebufferisation'', the default is true
      // for rle Flux and false elsewhere.
// Flux_Pts select(Flux_Pts flx,Fonc_Num fonc,bool rebuf);  


     // coordinates functions 

extern Fonc_Num kth_coord(INT,bool HasAlwaysInitialValue=false,double InitialValue=0);
extern const Fonc_Num FX;  // FX = kth_coord(0)
extern const Fonc_Num FY;  // FY = kth_coord(1)
extern const Fonc_Num FZ;  // FY = kth_coord(2)

extern Fonc_Num Identite(INT dim);  // Identite = (FX,FY, .... ,kth_coord(dim-1)

extern Fonc_Num  inside(const INT * p0,const INT * p1,INT dim);
extern Fonc_Num  inside(Pt2di,Pt2di);

extern Fonc_Num CsteNDim(double aVal,INT dim);  // 
extern Fonc_Num CsteNDim(int    aVal,INT dim);  // 
     // random values

Fonc_Num frandr(); 


Fonc_Num gauss_noise_1(INT sz);
Fonc_Num gauss_noise_2(INT sz);
Fonc_Num gauss_noise_3(INT sz);
Fonc_Num gauss_noise_4(INT sz);
Fonc_Num gauss_noise(INT sz,int aK);

Fonc_Num unif_noise_1(INT sz);
Fonc_Num unif_noise_2(INT sz);
Fonc_Num unif_noise_3(INT sz);
Fonc_Num unif_noise_4(INT sz);
Fonc_Num unif_noise(INT sz,int aK);

Fonc_Num fonc_noise(INT sz,int aK,bool unif);

Fonc_Num gauss_noise_4(REAL * pds,INT * sz , INT nb);
Fonc_Num unif_noise_4(REAL * pds,INT * sz , INT nb);

Fonc_Num CorrelSquare
         (
             Fonc_Num aF1,
             Fonc_Num aF2,
             Fonc_Num aPond,
             INT aSquare,
             REAL anEpsilon
         );

     // concatenation 

#ifndef NoOperatorVirgule
extern   Fonc_Num operator , (Fonc_Num,Fonc_Num);
#endif
extern   Fonc_Num Virgule(Fonc_Num,Fonc_Num);
extern   Fonc_Num Virgule(Fonc_Num,Fonc_Num,Fonc_Num);
extern   Fonc_Num Virgule(Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num);
extern   Fonc_Num Virgule(Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num);
extern   Fonc_Num Virgule(Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num);

    /*  -  -  -  -  -  -  -  -  -  -  */
    /*            Liner Filter        */
    /*  -  -  -  -  -  -  -  -  -  -  */


Fonc_Num linear_red(const OperAssocMixte &,Fonc_Num f,INT x0,INT x1);

Fonc_Num linear_som(Fonc_Num f,INT x0,INT x1);
Fonc_Num linear_max(Fonc_Num f,INT x0,INT x1);
Fonc_Num linear_min(Fonc_Num f,INT x0,INT x1);

Fonc_Num lin_cumul_ass(const OperAssocMixte & op,Fonc_Num f,const OperAssocMixte * = 0);
Fonc_Num lin_cumul_sum(Fonc_Num f);
Fonc_Num lin_cumul_max(Fonc_Num f);
Fonc_Num lin_cumul_min(Fonc_Num f);
Fonc_Num linear_integral(Fonc_Num f);


Fonc_Num min_linear_cum2sens_max(Fonc_Num f);
Fonc_Num max_linear_cum2sens_min(Fonc_Num f);

Fonc_Num linear_Kth(Fonc_Num f,INT x0,INT x1,INT kth,INT max_vals=256);
Fonc_Num linear_median(Fonc_Num f,INT sz,INT max_vals = 256);


       // related to shading

Fonc_Num binary_shading(Fonc_Num f,REAL steep);
Fonc_Num gray_level_shading(Fonc_Num f);


       // related to stereo-vision

Fonc_Num proj_cav(Fonc_Num f,REAL zobs,REAL steep);
Fonc_Num phasis_auto_stereogramme(Fonc_Num f,REAL zobs,REAL steep);


    /*  -  -  -  -  -  -  -  -  -  -  -  - */
    /*            Buffered Filter          */
    /*  -  -  -  -  -  -  -  -  -  -  -  - */


//==========================================
//       SEE ALSO FILE : users_op_buf.h
//       SEE ALSO FILE : morpho.h
//==========================================

         //   exponential filter

Fonc_Num canny_exp_filt(Fonc_Num f,REAL fx,REAL fy,INT nb = -1); // nb<0= let ELISE compuetd it
Fonc_Num semi_cef(Fonc_Num f,REAL fx,REAL fy);
Fonc_Num  deriche(Fonc_Num f,REAL alpha,INT d0 = 5);

// Le filtre de exponentiel admet un inverse a support borne 
Fonc_Num inv_canny_exp_filt(Fonc_Num f,REAL fx,REAL fy);
Fonc_Num inv_canny_exp_filt(Fonc_Num f,REAL fxy);
Fonc_Num inv_semi_cef(Fonc_Num f,REAL fx,REAL fy);
Fonc_Num inv_semi_cef(Fonc_Num f,REAL fxy);



           // "flaging familly"

Fonc_Num  flag_vois (Fonc_Num,Box2di);
Fonc_Num  flag_front8 (Fonc_Num  f);
Fonc_Num  flag_front4 (Fonc_Num  f);
Fonc_Num  erod_8_hom (Fonc_Num);

Fonc_Num  flag_min8 (Fonc_Num  f);
Fonc_Num  flag_visu4front (Fonc_Num  f);

Fonc_Num nflag_sym(Fonc_Num);
Fonc_Num nflag_close_sym(Fonc_Num);
Fonc_Num nflag_open_sym(Fonc_Num);


           //  morphological filters

                  // extinction-function 

Fonc_Num extinc(Fonc_Num f,const class Chamfer &,INT max_d);
Fonc_Num extinc_32(Fonc_Num f,INT max_d = 256);
Fonc_Num extinc_d8(Fonc_Num f,INT max_d = 256);
Fonc_Num extinc_d4(Fonc_Num f,INT max_d = 256);
Fonc_Num extinc_5711(Fonc_Num f,INT max_d = 256);

                  // erosion

Fonc_Num erod(Fonc_Num f,const class Chamfer &,INT max_d);
Fonc_Num erod_32(Fonc_Num f,INT max_d);
Fonc_Num erod_d8(Fonc_Num f,INT max_d);
Fonc_Num erod_d4(Fonc_Num f,INT max_d);
Fonc_Num erod_5711(Fonc_Num f,INT max_d);

                  // dilataion

Fonc_Num dilat(Fonc_Num f,const class Chamfer &,INT max_d);
Fonc_Num dilat_32(Fonc_Num f,INT max_d);
Fonc_Num dilat_d8(Fonc_Num f,INT max_d);
Fonc_Num dilat_d4(Fonc_Num f,INT max_d);
Fonc_Num dilat_5711(Fonc_Num f,INT max_d);


                  // closing-opening

Fonc_Num close(Fonc_Num f,const Chamfer & chamf,INT max_d,INT delta = 0);
Fonc_Num close_32(Fonc_Num f,INT max_d,INT delta = 0);
Fonc_Num close_d8(Fonc_Num f,INT max_d,INT delta = 0);
Fonc_Num close_d4(Fonc_Num f,INT max_d,INT delta = 0);
Fonc_Num close_5711(Fonc_Num f,INT max_d,INT delta = 0);


Fonc_Num open(Fonc_Num f,const Chamfer & chamf,INT max_d,INT delta = 0);
Fonc_Num open_32(Fonc_Num f,INT max_d,INT delta = 0);
Fonc_Num open_d8(Fonc_Num f,INT max_d,INT delta = 0);
Fonc_Num open_d4(Fonc_Num f,INT max_d,INT delta = 0);
Fonc_Num open_5711(Fonc_Num f,INT max_d,INT delta = 0);

              // K-Lipschitz envelop (= distance algorithm
              // with no intial binarization)

Fonc_Num EnvKLipshcitz(Fonc_Num f,const Chamfer & chamf,INT max_dif);
Fonc_Num EnvKLipshcitz_5711(Fonc_Num f,INT max_dif);
Fonc_Num EnvKLipshcitz_32(Fonc_Num f,INT max_dif);
Fonc_Num EnvKLipshcitz_d8(Fonc_Num f,INT max_dif);
Fonc_Num EnvKLipshcitz_d4(Fonc_Num f,INT max_dif);


          //  filters relative to radiometries ordering

class Histo_Kieme : public Mcheck
{
        public :

                typedef enum
                                {
                                        bin_tree,
                                        last_rank,
                                        undef
                                } mode_h;

                typedef enum
                        {
                                KTH,
                                RANK
                        } mode_res;

                static Histo_Kieme * New_HK(mode_h,INT max_vals);
                // renvoit un pointeur sur une class derivee (BinTree_HK ou
                // LastRank_HK)
                static mode_h  Opt_HK(INT ty,INT max_vals);
                static mode_h  Opt_HK(mode_h aModePref,INT ty,INT max_vals);


                virtual void add(INT radiom) ;
                virtual void sub(INT radiom) ;
                virtual void AddPop(INT radiom,INT aPop) = 0;

                INT  RKthVal(REAL aProp,INT adef);


                virtual void raz() = 0;
               virtual INT kth_val(INT kth) = 0;
                virtual INT rank(INT radiom) = 0;

                virtual  ~Histo_Kieme();
                virtual void verif_vals(const INT *,INT nb);



        protected :
                INT   _max_vals;
                INT   mPopTot;
                Histo_Kieme(INT max_vals);

        private :
};


Fonc_Num rect_kth 
         (
                    Fonc_Num  f,
                    double kth,
                    Box2di  side,
                    INT   max_vals,
                    bool CatInit = false,
                    bool aModePond = false,
                    Histo_Kieme::mode_h=Histo_Kieme::undef
         );
Fonc_Num rect_kth 
         (
                    Fonc_Num  f,
                    double kth,
                    Pt2di  side,
                    INT   max_vals,
                    bool CatInit = false,
                    bool aModePond = false,
                    Histo_Kieme::mode_h=Histo_Kieme::undef
         );
Fonc_Num rect_kth 
         (
                    Fonc_Num  f,
                    double kth,
                    INT  side,
                    INT   max_vals,
                    bool CatInit = false,
                    bool aModePond = false,
                    Histo_Kieme::mode_h=Histo_Kieme::undef
                    );

Fonc_Num rect_median 
         (
                    Fonc_Num  f,
                    Box2di  side,
                    INT   max_vals,
                    bool CatInit = false,
                    bool aModePond = false,
                    Histo_Kieme::mode_h=Histo_Kieme::undef
         );
Fonc_Num rect_median 
         (
                    Fonc_Num  f,
                    Pt2di  side,
                    INT   max_vals,
                    bool CatInit = false,
                    bool aModePond = false,
                    Histo_Kieme::mode_h=Histo_Kieme::undef
         );
Fonc_Num rect_median 
         (
                    Fonc_Num  f,
                    INT    side,
                    INT   max_vals,
                    bool CatInit = false,
                    bool aModePond = false,
                    Histo_Kieme::mode_h=Histo_Kieme::undef
         );
Fonc_Num MedianBySort(Fonc_Num f,INT NbMed);
Fonc_Num OmbrageKL(Fonc_Num Prof,Fonc_Num Masq,double aPente,int aSzV);
Fonc_Num NFoncDilatCond(Fonc_Num f2Dil,Fonc_Num fCond,bool aV4,int aNb);



Fonc_Num RectKth_Pondere
          (
               Fonc_Num    f1,
               Fonc_Num    f2,
               REAL         kth,
               INT          side,
               INT         max_vals
          );  // Non benche


// Fonc_Num  rect_kth (Fonc_Num f,INT kth,Box2di side,INT max_vals);


Fonc_Num  rect_rank(Fonc_Num f,Box2di side,INT max_vals,bool CatInit = false);
Fonc_Num  rect_egal_histo(Fonc_Num f,Box2di side,INT max_vals);


         // assoc

Fonc_Num rect_red(const OperAssocMixte &,Fonc_Num f,Box2di,bool aCatFinit = false);

Fonc_Num rect_var_som(Fonc_Num f,Fonc_Num fside,Box2di);
Fonc_Num rect_som(Fonc_Num f,Box2di,bool aCatFinit = false);
Fonc_Num rect_som(Fonc_Num f,Pt2di,bool aCatFinit = false);
Fonc_Num rect_som(Fonc_Num f,INT,bool aCatFinit = false);

Fonc_Num Moy(Fonc_Num,int);
Fonc_Num Moy1(Fonc_Num aF);
Fonc_Num MoyIter(Fonc_Num aF,int aSz,int aNbIter);


Fonc_Num rect_max(Fonc_Num f,Box2di,bool aCatFinit = false);
Fonc_Num rect_max(Fonc_Num f,Pt2di,bool aCatFinit = false);
Fonc_Num rect_max(Fonc_Num f,INT,bool aCatFinit = false);


Fonc_Num rect_min(Fonc_Num f,Box2di,bool aCatFinit = false);
Fonc_Num rect_min(Fonc_Num f,Pt2di,bool aCatFinit = false);
Fonc_Num rect_min(Fonc_Num f,INT,bool aCatFinit = false);

     // Iteration de sommes
Fonc_Num FoncMoy(Fonc_Num aF,int aK);
std::vector<int>  DecompSigmaEnInt(double aSigma,int aNb);
Fonc_Num MoyByIterSquare(Fonc_Num aF,double aSigmaCible,int aNbIter);




    /*  -  -  -  -  -  -  -  -  -  -  */
    /*       "Jump" operator          */
    /*  -  -  -  -  -  -  -  -  -  -  */

Fonc_Num  clip_def(Fonc_Num,REAL,Pt2di,Pt2di);
Fonc_Num  clip_def(Fonc_Num,REAL,INT* p0,INT* p1,INT dim);


    /*  -  -  -  -  -  -  -  -  -  -  */
    /*            BINARY              */
    /*  -  -  -  -  -  -  -  -  -  -  */


    // binary associative mixte operators

extern   Fonc_Num operator + (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator * (Fonc_Num,Fonc_Num);
            //  ( Max and Min, because, max and min create some tricky confusion with
            //     the functions on integral types.)

extern   Fonc_Num Max        (Fonc_Num,Fonc_Num);
extern   Fonc_Num Min        (Fonc_Num,Fonc_Num);

extern   Fonc_Num Max        (Fonc_Num,Fonc_Num,Fonc_Num);
extern   Fonc_Num Min        (Fonc_Num,Fonc_Num,Fonc_Num);
extern   Fonc_Num Max        (Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num);
extern   Fonc_Num Min        (Fonc_Num,Fonc_Num,Fonc_Num,Fonc_Num);

    // binary mixte  operators

extern   Fonc_Num operator / (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator - (Fonc_Num,Fonc_Num);
extern   Fonc_Num pow(Fonc_Num,Fonc_Num);


    // binary associative integer operators

extern   Fonc_Num operator &  (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator && (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator |  (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator || (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator ^  (Fonc_Num,Fonc_Num);
extern   Fonc_Num ElXor       (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator %  (Fonc_Num,Fonc_Num);
extern   Fonc_Num mod         (Fonc_Num,Fonc_Num); // as %, but the result is always >= 0
extern   Fonc_Num operator << (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator >> (Fonc_Num,Fonc_Num);

    //  comparison operators

extern   Fonc_Num operator == (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator != (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator < (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator <= (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator > (Fonc_Num,Fonc_Num);
extern   Fonc_Num operator >= (Fonc_Num,Fonc_Num);

       // version without oveload (to avoid incompatibility with STL)

extern   Fonc_Num SuperiorStrict(Fonc_Num,Fonc_Num);
extern   Fonc_Num NotEqual(Fonc_Num,Fonc_Num);


// HYPER LENT POU MISE AU POINT
double GETVALPT(Fonc_Num,Pt2di);


    /*  -  -  -  -  -  -  -  -  -  -  */
    /*            UNARY               */
    /*  -  -  -  -  -  -  -  -  -  -  */


    // unary mixte  operators

extern   Fonc_Num operator -(Fonc_Num);
extern   Fonc_Num Abs(Fonc_Num);
extern   Fonc_Num Square(Fonc_Num);
extern   Fonc_Num Cube(Fonc_Num);
Fonc_Num PowI(Fonc_Num f,INT aDegre);


extern   double Square(double);

    // Operateur utile au devt en four des fonction radiale

double CosRx(double);  // cos sqrt x
Fonc_Num CosRx(Fonc_Num);  // cos sqrt x
double SinCardRx(double);  // cos sqrt x
Fonc_Num SinCardRx(Fonc_Num);  // cos sqrt x

double IsInf(double);  // boolean is it inf ?
Fonc_Num IsInf(Fonc_Num);  
double IsNan(double);  // boolean is it nan ?
Fonc_Num IsNan(Fonc_Num);  
double IsBadNum(double);  // boolean is it inf or nan ?
Fonc_Num IsBadNum(Fonc_Num);  


    // Operateur utile a la fonction de conversion des Fish Eye Linear

double AtRxSRx(double x);      // Atan(sqrt(x)) / sqrt(x)
Fonc_Num AtRxSRx(Fonc_Num x);     
double DerAtRxSRx(double x);   // Derivee de  AtRxSRx
Fonc_Num DerAtRxSRx(Fonc_Num x); 
double At2Rx(double x);        // Atan ^2 (sqrt(x)) 
Fonc_Num At2Rx(Fonc_Num x);       
double DerAt2Rx(double x);     // Derivee At2Rx
Fonc_Num DerAt2Rx(Fonc_Num x);    

// Predicteur d'inversion
double TgRxSRx(double x);      // Tang(sqrt(x)) / sqrt(x)
Fonc_Num TgRxSRx(Fonc_Num x);      // Tang(sqrt(x)) / sqrt(x)

    // Operateur utile a la fonction de conversion des Fish Eye equisolid
    //
    //  2 sin (At(X)/2)
    //
double Dl_f2SAtRxS2SRx(double x);
double Std_f2SAtRxS2SRx(double x);
double  f2SAtRxS2SRx(double); // 2 sin(Atan(sqrt(x))/2) / sqrt(x)
Fonc_Num  f2SAtRxS2SRx(Fonc_Num); 

double Dl_Der2SAtRxS2SRx(double x);   
double Std_Der2SAtRxS2SRx(double x);   
double Der2SAtRxS2SRx(double x);   // Der f2SAtRxS2SRx


double PrecStereographique(double x);
double Der_PrecStereographique(double x);
double SqM2CRx_StereoG(double x);
double Der_SqM2CRx_StereoG(double x);
double Inv_PrecStereographique(double x);
Fonc_Num Der_PrecStereographique(Fonc_Num f);
Fonc_Num Der_SqM2CRx_StereoG(Fonc_Num f);


    // Operateur utile a la fonction de conversion des Fish Eye Linear

double AtRxSRx(double x);      // Atan(sqrt(x)) / sqrt(x)
Fonc_Num AtRxSRx(Fonc_Num x);     
double DerAtRxSRx(double x);   // Derivee de  AtRxSRx
Fonc_Num DerAtRxSRx(Fonc_Num x); 
double At2Rx(double x);        // Atan ^2 (sqrt(x)) 
Fonc_Num At2Rx(Fonc_Num x);       
double DerAt2Rx(double x);     // Derivee At2Rx
Fonc_Num DerAt2Rx(Fonc_Num x);    

// Predicteur d'inversion
double TgRxSRx(double x);      // Tang(sqrt(x)) / sqrt(x)
Fonc_Num TgRxSRx(Fonc_Num x);      // Tang(sqrt(x)) / sqrt(x)

    // Operateur utile a la fonction de conversion des Fish Eye equisolid
    //
    //  2 sin (At(X)/2)
    //
double Dl_f2SAtRxS2SRx(double x);
double Std_f2SAtRxS2SRx(double x);
double  f2SAtRxS2SRx(double); // 2 sin(Atan(sqrt(x))/2) / sqrt(x)
Fonc_Num  f2SAtRxS2SRx(Fonc_Num); 

double Dl_Der2SAtRxS2SRx(double x);   
double Std_Der2SAtRxS2SRx(double x);   
double Der2SAtRxS2SRx(double x);   // Der f2SAtRxS2SRx
Fonc_Num Der2SAtRxS2SRx(Fonc_Num x);
double f4S2AtRxS2(double);    ;  // 4 sin^2(Atan(sqrt(x))/2))
Fonc_Num f4S2AtRxS2(Fonc_Num);    ;  

double  Dl_Der4S2AtRxS2(double);
double  Std_Der4S2AtRxS2(double);
double Der4S2AtRxS2(double);    ;  // Der f4S2AtRxS2
Fonc_Num Der4S2AtRxS2(Fonc_Num);    ;  // Der f4S2AtRxS2

// Predicteur d'inversion

double Dl_Tg2AsRxS2SRx(double x);      // 2 (Tang (Asin (sqrt(x))/2) / sqrt(x)
double Std_Tg2AsRxS2SRx(double x);      // 2 (Tang (Asin (sqrt(x))/2) / sqrt(x)
double Tg2AsRxS2SRx(double x);      //  (Tang 2 (Asin (sqrt(x)/2))) / sqrt(x)
Fonc_Num Tg2AsRxS2SRx(Fonc_Num x);      //  (Tang 2 (Asin (sqrt(x)/2))) / sqrt(x)




     // mathematical operator 

extern   Fonc_Num sqrt(Fonc_Num);
extern   Fonc_Num cos (Fonc_Num);
extern   Fonc_Num sin (Fonc_Num);
extern   Fonc_Num tan (Fonc_Num);
extern   Fonc_Num atan (Fonc_Num);
extern   Fonc_Num erfcc (Fonc_Num);
extern   Fonc_Num log (Fonc_Num);
extern   Fonc_Num log2 (Fonc_Num);
extern   Fonc_Num exp (Fonc_Num);

extern   Fonc_Num  AtanXY(Fonc_Num,Fonc_Num);
extern   double    AtanXY(double,double);

// Renvoie 255 * erfc ((f-s1)/sqrt(s2))
extern Fonc_Num FoncNormalisee_S1S2 (Flux_Pts aFl,Fonc_Num aFPds,Fonc_Num aF);
extern Fonc_Num FoncNormalisee_S1S2 (Flux_Pts aFl,Fonc_Num aF); // FPds=1.0

#if ( ELISE_windows & !ELISE_MinGW & _MSC_VER < 1800 )
inline double log2(const double & aD)
{
       return log(aD) / log(2.0);
}
#endif


     // convertion functions 

                // convert f to a real function

extern Fonc_Num Rconv(Fonc_Num f);

                 // convert f to an integer function
                 // Iconv : faster but use  C standard-convertion
                 // round_up : smallest integer >= to
                 // round_down : highest integer <= to
                 // round_ni : closest  integer; i+0.5 => i+1
                 // round_ni_inf : closest  integer; i+0.5 => i

                 // you can use Iconv if you accept a rudimentary
                 // convertion with machine-dependance results (the only
                 // waranty is that |r-i| < 1.0; if you have precise
                 // requirement use others

extern Fonc_Num Iconv(Fonc_Num f);   
extern Fonc_Num round_up(Fonc_Num f);   
extern Fonc_Num round_down(Fonc_Num f);   
extern Fonc_Num round_ni(Fonc_Num f);   
extern Fonc_Num round_ni_inf(Fonc_Num f);   
extern Fonc_Num signed_frac(Fonc_Num f);   
extern Fonc_Num ecart_frac(Fonc_Num f);   
          

     // integer  

extern   Fonc_Num operator ! (Fonc_Num);
extern   Fonc_Num operator ~ (Fonc_Num);


    /*  -  -  -  -  -  -  -  -  -  -  */
    /*            COMPLEX             */
    /*  -  -  -  -  -  -  -  -  -  -  */

         // binary 

Fonc_Num mulc(Fonc_Num f1,Fonc_Num f2);  // multiplication
extern   Fonc_Num divc(Fonc_Num,Fonc_Num);   // 1/f

         // unary 

extern   Fonc_Num squarec(Fonc_Num);      // f * f
extern   Fonc_Num squarec_divN(Fonc_Num);      // f * f / |f|
extern   Fonc_Num polar(Fonc_Num);       // x,y => rho, teta
// Implementee  a partir d'Elise
extern    Fonc_Num  Hypot(Fonc_Num Fx,Fonc_Num Fy);
extern    Fonc_Num  Hypot(Fonc_Num Fxyz);  // Autant de dim qu'on veut
extern    Fonc_Num TronkUC(Fonc_Num);  // Tronque 0 - 256
extern    Fonc_Num Scal(Fonc_Num F1,Fonc_Num F2);
         //  specifies the value of teta when x,y = 0
extern   Fonc_Num polar(Fonc_Num,REAL rh0); 
extern   Fonc_Num divc(Fonc_Num);   // 1/f


    /*  -  -  -  -  -  -  -  -  -  -  -  -  -  - */
    /*  N-DIMENBTIONALL OPERATOR                 */
    /*  -  -  -  -  -  -  -  -  -  - -  -  -  -  */

         //  UNARY

Fonc_Num diag_m2_sym(Fonc_Num f);



    /*  -  -  -  -  -  -  -  -  -  -  -  -  -  - */
    /*  COMPOSITION/ CHANG OF COORDINATE         */
    /*  -  -  -  -  -  -  -  -  -  - -  -  -  -  */


// Fonc_Num operator [] (Fonc_Num);  => member function of class Fonc_Num

        // translation 

extern Fonc_Num trans(Fonc_Num,const INT*,INT);
extern Fonc_Num trans(Fonc_Num,Pt2di);
extern Fonc_Num trans(Fonc_Num,Pt3di);
extern Fonc_Num trans(Fonc_Num,INT);


    /*  -  -  -  -  -  -  -  -  -  -  -  -  -  - */
    /*  SPECIAL PURPOSE - AD HOC operator        */
    /*  -  -  -  -  -  -  -  -  -  - -  -  -  -  */



    /*----------------------------------------*/
    /*                                        */
    /*          Output                        */
    /*                                        */
    /*----------------------------------------*/

class Output : public  PRC0
{
     friend class Pipe_Out_Not_Comp;
     friend class CatCoord_Out_Not_Comp;
     public :
        ~Output();
        Output(class Output_Not_Comp *);
        Output(class Liste_Pts_Gen);
        class Output_Computed * compute (const class Arg_Output_Comp &);
        static Output onul(INT dim = 1);
        Output chc(Fonc_Num);
     private :
};

         // operator on output

#ifndef NoOperatorVirgule
extern   Output operator , (Output,Output);
#endif
extern   Output Virgule(Output,Output);
extern   Output Virgule(Output,Output,Output);
extern   Output Virgule(Output,Output,Output,Output);
extern   Output Virgule(Output,Output,Output,Output,Output);
extern   Output Virgule(Output,Output,Output,Output,Output,Output);

extern   Output operator | (Output,Output);

         // operator mixing output and function

extern   Output operator << (Output,Fonc_Num);

extern Output Filtre_Out_RedBin   (Output anOut);
extern Output Filtre_Out_RedBin_X (Output anOut);
extern Output Filtre_Out_RedBin_Y (Output anOut);
extern Output Filtre_Out_RedBin_Gen(Output anOut,bool aRedX,bool aRedY);

         //  REDUCTION 



template <class Type>    Output  reduc(const OperAssocMixte &,Type &);
template <class Type>    Output  reduc(const OperAssocMixte &,Type *,INT nb);

//  The sigma function is just a syntactic shugar that is rewriten as follow :
//     sigma(v) ==> reduc(OpSum,v)
// 

template <class Type>    Output  sigma(Type &);
template <class Type>    Output  sigma(Type *,INT nb);

template <class Type>    Output  VMax(Type &);
template <class Type>    Output  VMax(Type *,INT nb);

template <class Type>    Output  VMin(Type &);
template <class Type>    Output  VMin(Type *,INT nb);


template <class Type>    Output  WhichMax(Type &);
template <class Type>    Output  WhichMax(Type *,INT nb);

template <class Type>    Output  WhichMin(Type &);
template <class Type>    Output  WhichMin(Type *,INT nb);


    /*----------------------------------------*/
    /*                                        */
    /*      Neighbooring relations            */
    /*                                        */
    /*----------------------------------------*/

template <class Type,class TyBase> class Im2D;

class Neighbourhood : public PRC0
{
     friend class Neigh_Rel;
     friend class B2d_Spec_Neigh_Not_Comp;

     public :
         Neighbourhood (Im2D<INT4,INT>);
         Neighbourhood (Pt2di *,INT nb);

         static Neighbourhood   v4();
         static Neighbourhood   v8();

     private :
          class Data_Neighbourood * data_n();
};


class Neigh_Rel : public PRC0
{
     public :
        Neigh_Rel(class Neigh_Rel_Not_Comp *);
        Neigh_Rel(Neighbourhood);
        class Neigh_Rel_Compute * compute (const class Arg_Neigh_Rel_Comp &);

        Fonc_Num red_op(const OperAssocMixte &  op,Fonc_Num f);
        Fonc_Num red_sum(Fonc_Num f);
        Fonc_Num red_max(Fonc_Num f);
        Fonc_Num red_min(Fonc_Num f);
        Fonc_Num red_mul(Fonc_Num f);
};

         // 

Neigh_Rel sel_func(Neigh_Rel,Fonc_Num);
Neigh_Rel sel_flag(Neigh_Rel n,Fonc_Num f);


        //  operaror with Flux of point

          //  see upper : extern Flux_Pts dilate(Flux_Pts flx,Neigh_Rel fonc); 
          //  see upper : extern Flux_Pts conc(Flux_Pts flx,Neigh_Rel fonc); 


    /*----------------------------------------*/
    /*                                        */
    /*          Operator (arithmetic)         */
    /*                                        */
    /*----------------------------------------*/

extern const OperAssocMixte & OpSum;
extern const OperAssocMixte & OpMax;
extern const OperAssocMixte & OpMin;
extern const OperAssocMixte & OpMul;

    /*----------------------------------------*/
    /*                                        */
    /*          Symb_FNum                     */
    /*                                        */
    /*----------------------------------------*/

class Symb_FNum : public  Fonc_Num
{
   friend class Fonc_Num;
   public :
     Symb_FNum(Fonc_Num);

};

    /*----------------------------------------*/
    /*                                        */
    /*          top level                     */
    /*                                        */
    /*----------------------------------------*/

extern bool DebugCopy;
extern bool DebugInProj;

void ElCopy(Flux_Pts,Fonc_Num,Output,INT,const char *);
#define  ELISE_COPY(a,b,c) ElCopy((a),(b),(c),__LINE__,__FILE__)

    /*----------------------------------------*/
    /*                                        */
    /*          Rectang_Object                 */
    /* (== window, bitmaps, file image ...)   */
    /*                                        */
    /*----------------------------------------*/

class Elise_Rect
{
   public :
     INT _p0[Elise_Std_Max_Dim];
     INT _p1[Elise_Std_Max_Dim];
     INT _dim;
     Elise_Rect (Pt2di,Pt2di);
     Elise_Rect (const INT *,const INT*,INT);
     Box2dr ToBoxR() const;
};

class Rectang_Object
{
   public :
     virtual Elise_Rect box() const = 0;
     virtual ~Rectang_Object () ;

     Fonc_Num  inside() const ;
     Flux_Pts all_pts() const ;
     Flux_Pts interior(INT) const ;
     Flux_Pts border(INT) const ;

     // only for objects known to be of dimention 1
     Flux_Pts lmr_all_pts(INT) const ;   // lmr = linear map rect
     INT  x0() const ;
     INT  x1() const ;

     // only for objects known to be of dimention 2
     Pt2di  p0() const ;
     Pt2di  p1() const ;
     Flux_Pts lmr_all_pts(Pt2di) const ;  

};



class PtsKD
{
  public :
    PtsKD(INT dim);
    PtsKD(REAL * v,INT dim);
    PtsKD(const PtsKD &);
    void operator = (const PtsKD &);
    
    
    ~PtsKD();
    const REAL  & operator ()(INT k) const
    {
      ELISE_ASSERT((k>=0)&&(k<_dim),"Out of Range in PtsKD(int)");
      return _x[k];
    }
    
    REAL  & operator ()(INT k)
    {
      ELISE_ASSERT((k>=0)&&(k<_dim),"Out of Range in PtsKD(int)");
      return _x[k];
    }
    REAL * AdrX0() {return _x;}
    
    
    INT Dim() const {return _dim;}
  private :
      void init(INT Dim);
      REAL * _x;
      INT    _dim;
};

class cSimpleFoncNum
{
     public :
       Fonc_Num ToF();
                                                                                                      
       virtual REAL SFN_Calc(const REAL *) const;
       virtual REAL SFN_Calc(const INT *) const;
       virtual void AcceptDim(INT )const ; // Defaut : non
       virtual ~cSimpleFoncNum();
                                                                                                      
                                                                                                      
     protected :
         cSimpleFoncNum(INT aDim);
     private :
       cSimpleFoncNum(const cSimpleFoncNum &) ; // Non Impl
                                                                                                      
       INT   mDim;
};

class cSFN2D : public cSimpleFoncNum
{
     public :
        cSFN2D();
     private :
       virtual REAL SFN2_Calc(const Pt2dr &) const;
       virtual REAL SFN2_Calc(const Pt2di &) const;
                                                                                                      
       virtual REAL SFN_Calc(const REAL *) const;
       virtual REAL SFN_Calc(const INT *) const;
};

/*
    Voir aussi (liste totalement non exhaustive) :


    BoxedConc  : Pour analyser des conpmosantes connexe incluse dans une boxe

    fonc_a_trou : definir explicitement des valeurs en des points particuliers

    gauss_noise_4 unif_noise_4  ....
        
*/




#endif  /* !_ELISE_ABSTRACT_TYPE_H */







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
