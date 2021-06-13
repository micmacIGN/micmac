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



#ifndef _ELISE_GENERAL_GEOM_VECTEUR_H  // general
#define _ELISE_GENERAL_GEOM_VECTEUR_H

#include "ext_stl/fifo.h"

class Seg2d
{
   public :

     Seg2d(REAL x1,REAL y1,REAL x2,REAL y2);
     Seg2d(Pt2dr,Pt2dr);
     Seg2d();  // create an empty seg

    Seg2d reverse() const {assert_non_empty(); return Seg2d(p1(),p0());}

     Seg2d clip(Box2dr) const;
     Seg2d clip(Box2dr,REAL,REAL,bool IsSeg) const;
     Seg2d clipDroite(Box2dr) const;
     Seg2d clip(Box2di) const;

     inline bool  empty() const {return _empty;}
     inline Pt2dr p0() const {assert_non_empty();  return _pts[0];}
     inline Pt2dr p1() const {assert_non_empty();  return _pts[1];}

     inline REAL x0() const {assert_non_empty();  return _pts[0].x;}
     inline REAL x1() const {assert_non_empty();  return _pts[1].x;}
     inline REAL y0() const {assert_non_empty();  return _pts[0].y;}
     inline REAL y1() const {assert_non_empty();  return _pts[1].y;}

     inline Pt2dr milieu() const {assert_non_empty();  return (_pts[1]+_pts[0])/2.0;}

     inline Pt2dr v01() const {assert_non_empty();  return _pts[1]-_pts[0];}

     inline Seg2d trans(Pt2dr tr) const {assert_non_empty();  return Seg2d(_pts[0]+tr,_pts[1]+tr);}

      REAL AbsiceInterDroiteHoriz(REAL anOrdonnee) const;

   protected :

    inline Pt2dr kpts(INT k) const {return _pts[k];}

     void assert_non_empty() const
     {
          ELISE_ASSERT
          (    (! _empty)   ,
               "try to access to value of empty segment"
          );
     }



     Pt2dr  _pts[2];
     bool   _empty;
};


class SegComp : public Seg2d
{
    // equation normale de la droite
    //  (p1 p). _normale = 0;
    //   p._normale  - _p1. _normale = 0
    //  c = - _p1. _normale


//  Le repere de la droite (quand on parle abscisse et ordonnees)
//  est le repere Ortho Normee direct d'origine p1 et de 1ere dir p1p2

     public :

        Pt2dr ToRhoTeta() const;
        static SegComp FromRhoTeta(const Pt2dr &aP);

        typedef enum
        {
           droite =0,
           demi_droite =1,
           seg =2
        }  ModePrim;


        // Constructeur par defaut, necessaire pour creer des vector,
        //  ElFilo etc... A ne jamais utiliser sinon (etat douteux)
        SegComp();
        SegComp(Pt2dr p1,Pt2dr p2);
        SegComp(const Seg2d &);


        // tel que Cx X + Cy Y + C0 est la fonction ordonnee
    // qui prend une valeur 1 en P, utile pour calculer
    // une "matrice" de coeeficent bary dans un triangle
        void CoeffFoncOrdonnee
         (
              const Pt2dr& aP,
          double & Cx,
          double & Cy,
          double & aC0
             ) const;


        REAL ordonnee(Pt2dr pt) const;
        REAL ordonnee(Pt3dr pt) const;  // Point Projectif
        Fonc_Num ordonnee(Pt3d<Fonc_Num> pt) const;  // Point Projectif Formel
        REAL  abscisse(Pt2dr pt) const;

        REAL  abscisse_proj_seg(Pt2dr pt) const;  // clippee dans [0 lenght]
        REAL  recouvrement_seg(const Seg2d &) const;  // clippee dans [0 lenght]

        Pt2dr to_rep_loc(Pt2dr) const;
        Pt2dr from_rep_loc(Pt2dr) const;

        REAL  length()   const  {return _length;}
        Pt2dr tangente() const  {return _tangente;}
        Pt2dr normale()  const  {return _normale;}
        REAL  c()        const  {return _c;}

        bool in_bande(Pt2dr pt,ModePrim) const;

        bool BoxContains(Pt2dr pt,REAL DLong,REAL DLarg) const;
        bool BoxContains(const Seg2d & ,REAL DLong,REAL DLarg) const;

   //=========   DISTANCES   ==============

        REAL square_dist_droite(Pt2dr pt) const;
        REAL square_dist_demi_droite(Pt2dr pt) const;
        REAL square_dist_seg(Pt2dr pt) const;
        REAL square_dist(ModePrim  mode,Pt2dr   pt) const;

        REAL dist_droite(Pt2dr pt) const;
        REAL dist_demi_droite(Pt2dr pt) const;
        REAL dist_seg(Pt2dr pt) const;
        REAL dist(ModePrim  mode,Pt2dr   pt) const;

        REAL square_dist(ModePrim,const SegComp &,ModePrim) const;
        REAL dist(ModePrim,const SegComp &,ModePrim) const;

   //=========   DISTANCES DE HAUSSDORF   ==============

        // la 1ere primitive est tjs consideree comme un segment
        // (sinon Haussdorf = infini, hors paralellisme notoire)
        // dans les version assym, il s'agit de la dist du point de la premiere
        // le pus loin de la deuxieme.

        REAL  square_haussdorf_seg_assym(const SegComp &) const;
        REAL  square_haussdorf_seg_sym(const SegComp &) const;
        REAL  square_haussdorf_droite_assym(const SegComp &) const;

              // max de "square_haussdorf_droite_assym" dans
              // les 2 sens, donc pas vraiment une distance de haussdorf
              //  au sens mathematique du terme

        REAL  square_haussdorf_droite_sym(const SegComp &) const;

   //=========   PROJECTIONS   ==============

        Pt2dr  proj_ortho_droite(Pt2dr pt) const;
        Pt2dr  proj_ortho_demi_droite(Pt2dr pt) const;
        Pt2dr  proj_ortho_seg(Pt2dr pt) const;
        Pt2dr  proj_ortho(ModePrim,Pt2dr pt) const;
        Seg2d  proj_ortho(ModePrim,const SegComp &,ModePrim) const;

   //=========   INTERSECTION   ==============

        Pt2dr   inter(const SegComp &,bool &) const;  // droite
        Pt2dr   inter(ModePrim,const SegComp &,ModePrim,bool &) const;

        void inter_polyline
             (
                 ModePrim,
                 const ElFifo<Pt2dr> &,
                 ElFifo<INT>  &,   // index
                 ElFifo<Pt2dr> &   // resultats
             );

        static const Pt2dr NoPoint;
        std::vector<Seg2d> Clip(const std::vector<Pt2dr> &);

     protected :

        Pt2dr _tangente;
        REAL  _length;
        Pt2dr _normale;
        REAL  _c;
        REAL  _a1;   // abscisse p1, dans le repere de la droite

        REAL   _square_dist(ModePrim m1,const SegComp & s2,ModePrim m2) const;
        void   proj_ortho
               (
                   ModePrim,
                   const SegComp &,
                   ModePrim,
                   REAL & dmin,
                   Pt2dr & p0min,
                   Pt2dr & p1min
               ) const;

};


class cElTriangleComp
{
    public :
        cElTriangleComp(Pt2dr aP0,Pt2dr aP1,Pt2dr aP2);
            REAL square_dist(Pt2dr pt) const;

        bool Inside(const Pt2dr &) const;

        Pt2dr P0() const;
        Pt2dr P1() const;
        Pt2dr P2() const;

           // Renvoie une matrice telle que pour
       //  un point (x,y) on trouve ses trois
       //  coordonnees bary a partir de
       //
       //
       //    | X |   |  Coeff P1
       //  M | Y | = |  Coeff P2
       //    | 1 |   |  Coeff P3
       //

        ElMatrix<double>  MatCoeffBarry() const;


        Pt3dr  CoordBarry(const Pt2dr &) const;
        Pt2dr  FromCoordBarry(REAL,REAL,REAL) const;

        static void Test();

        const SegComp & S01() const;
        const SegComp & S12() const;
        const SegComp & S20() const;

            // Est ce que ordre trigo
        static bool ToSwap(const Pt2dr & aP0,const  Pt2dr & aP1,const Pt2dr & aP2);
    private :
        static SegComp ReorderDirect(Pt2dr & aP0, Pt2dr & aP1,Pt2dr & aP2);
        SegComp mS01;
        SegComp mS12;
        SegComp mS20;


};


template <class Type> class Mat_Inertie
{
     public  :

       Mat_Inertie();
       Mat_Inertie
       (
              ElTyName Type::TypeScal S,
              ElTyName Type::TypeEff  S1,
              ElTyName Type::TypeEff  S2,
              ElTyName Type::TypeScal S11,
              ElTyName Type::TypeScal S12,
              ElTyName Type::TypeScal S22
       )
          :
               _s     (S),
               _s1    (S1),
               _s2    (S2),
               _s11   (S11),
               _s12   (S12),
               _s22   (S22)
       {
       }

       void add_pt_en_place(ElTyName Type::TypeEff v1,ElTyName  Type::TypeEff v2)
       {
            _s   += 1;
            _s1  += v1;
            _s2  += v2;
            _s11 += scal(v1,v1); // scal = v1*v1
            _s12 += scal(v1,v2);
            _s22 += scal(v2,v2);
       }

       void add_pt_en_place
           (
               ElTyName Type::TypeEff v1,
               ElTyName  Type::TypeEff v2,
               ElTyName Type::TypeScal pds
            )
       {
            _s   += pds;
            _s1  += v1 *pds;
            _s2  += v2 *pds;
            _s11 += scal(v1,v1) *pds;
            _s12 += scal(v1,v2) *pds;
            _s22 += scal(v2,v2) *pds;
       }


       ElTyName Type::TypeScal    s()    const {return _s;}
       ElTyName Type::TypeEff     s1()   const {return _s1;}
       ElTyName Type::TypeEff     s2()   const {return _s2;}
       ElTyName Type::TypeScal    s11()  const {return _s11;}
       ElTyName Type::TypeScal    s12()  const {return _s12;}
       ElTyName Type::TypeScal    s22()  const {return _s22;}


       Mat_Inertie  plus_cple
                    (
                      ElTyName Type::TypeEff v1,
                      ElTyName Type::TypeEff v2,
                      ElTyName Type::TypeScal pds =1
                    ) const
       {
            return Mat_Inertie
                   (
                       _s   + pds,
                       _s1  +  v1 * pds,
                       _s2  +  v2 * pds,
                       _s11 +  scal(v1,v1) * pds,
                       _s12 +  scal(v1,v2) * pds,
                       _s22 +  scal(v2,v2) * pds
                   );
       }

       Mat_Inertie operator - (const Mat_Inertie &) const;
       void operator += (const Mat_Inertie &);


            // renvoie la droite au moinde carre, point initial = cdg;
            // second point (indertermine a pi pres) situe a dun distance norm


       Mat_Inertie<ElTyName Type::TypeReel>  normalize() const
       {
             ELISE_ASSERT
             (
                  _s != 0,
                  "som pds = 0 in Mat_Inertie::normalize"
             );

             ElTyName Type::TypeReel::TypeEff  S1 =  _s1 / (REAL) _s;  // _s1 = sigma(v1) (sum of all v1 value)
             ElTyName Type::TypeReel::TypeEff  S2 =  _s2 / (REAL) _s;  // _s2 = sigma(v2)


#if ( ELISE_windows & ELISE_MinGW )
    return Mat_Inertie<typename Type::TypeReel>
#else
    return Mat_Inertie<ElTypeName_NotMSW Type::TypeReel>
#endif
                    (
                         _s,
                         S1,
                         S2,
                         _s11/(REAL)_s  -scal(S1,S1),   // _s = number of added element
                         _s12/(REAL)_s  -scal(S1,S2),
                         _s22/(REAL)_s  -scal(S2,S2)
                    );
       }

       REAL  correlation(REAL epsilon = 1e-14) const
       {
           #if ( ELISE_windows & ELISE_MinGW )
             Mat_Inertie<typename  Type::TypeReel> m =  normalize();
           #else
             Mat_Inertie<ElTypeName_NotMSW  Type::TypeReel> m =  normalize();
           #endif
             return m.s12() / sqrt(ElMax(epsilon,m.s11()*m.s22()));
       }

       REAL  correlation_with_def(REAL aDef) const
       {
            #if ( ELISE_windows & ELISE_MinGW )
              Mat_Inertie<typename  Type::TypeReel> m =  normalize();
            #else
              Mat_Inertie<ElTypeName_NotMSW  Type::TypeReel> m =  normalize();
            #endif
         if ((m.s11()<=0) || (m.s22() <=0)) return aDef;
             return m.s12() / sqrt(m.s11()*m.s22());
       }



       ElTyName   Type::TypeScal S0() const {return _s;}
       typename Type::TypeReel::TypeEff  V2toV1(const typename Type::TypeReel::TypeEff & aV2,REAL epsilon = 1e-14);
       typename Type::TypeReel::TypeEff  V1toV2(const typename Type::TypeReel::TypeEff & aV2,REAL epsilon = 1e-14);


    private :


        ElTyName   Type::TypeScal         _s;
        ElTyName   Type::TypeEff          _s1;
        ElTyName   Type::TypeEff          _s2;
        ElTyName   Type::TypeScal         _s11;
        ElTyName   Type::TypeScal         _s12;
        ElTyName   Type::TypeScal         _s22;
};

REAL   surf_or_poly(const ElFifo<Pt2dr> &);
REAL   surf_or_poly(const std::vector<Pt2dr> &);
REAL   perimetre_poly(const std::vector<Pt2dr> &);
Pt2dr  barrycentre(const ElFifo<Pt2dr> &);
Pt2dr  barrycentre(const std::vector<Pt2dr> &);
REAL   SquareDistPointPoly(const ElFifo<Pt2dr> & f,Pt2dr pt);
bool   PointInPoly(const ElFifo<Pt2dr> & f,Pt2dr pt);
bool   PointInPoly(const std::vector<Pt2dr> & f,Pt2dr pt);
bool   PointInterieurPoly(const ElFifo<Pt2dr> & f,Pt2dr pt,REAL d);
void   BoxPts(ElFifo<Pt2dr> & pts,Pt2dr & p0,Pt2dr & p1);
Box2dr BoxPts(const std::vector<Pt2dr> & pts);

bool HasInter(const std::vector<Pt2dr> & f1,const std::vector<Pt2dr> & f2);

std::vector<Pt2dr> DilateHomotetik
                   (const std::vector<Pt2dr> &,double,const Pt2dr & aCentre);
std::vector<Pt2dr> DilateHomotetikCdg(const std::vector<Pt2dr> &,double);

void PtsOfSquare(ElFifo<Pt2dr> & pts,Pt2dr p0,Pt2dr p1);



#define IMat_Inertie  Mat_Inertie<ElStdTypeScal<INT> >
#define RMat_Inertie  Mat_Inertie<ElStdTypeScal<REAL> >


Pt2dr  LSQSolDroite(const  RMat_Inertie & aMatr,double & aDelta);
Pt2dr  LSQSolDroite(const  RMat_Inertie & aMatr);
double   LSQResiduDroite(const  RMat_Inertie & aMatr);
double   LSQMoyResiduDroite(const  RMat_Inertie & aMatr);


inline Pt2dr MatCdg(const RMat_Inertie& aMat)
{
   return Pt2dr(aMat.s1(),aMat.s2());
}
inline double ValQuad(const RMat_Inertie& aMat,const Pt2dr aP)
{
   return     aMat.s11() * ElSquare(aP.x)
         +  2*aMat.s12() * aP.x* aP.y
         +    aMat.s22() * ElSquare(aP.y);
}

template <class Type> REAL square_dist_droite(const SegComp &, const Mat_Inertie<Type> &);
template <class Type> Seg2d seg_mean_square(const Mat_Inertie<Type> &,REAL norm = 1.0);


void env_conv
     (
         ElFifo<INT> &          ind,
         const ElFilo<Pt2di> &  pts,
         bool                   env_min
     );
void env_conv
     (
         ElFifo<INT> &          ind,
         const ElFilo<Pt2dr> &  pts,
         bool                   env_min
     );



class EventInterv
{
     public :
       EventInterv(REAL absc,bool entr);
       REAL absc() const;
       bool entr() const;
       EventInterv();

     private :
       REAL    _absc;
       bool    _entr;
};

class PileEvInterv
{
      public :

        void add_ev(EventInterv);
        void clear();
        void sort_ev();

        ElFilo<EventInterv> & events();

      private :

        ElFilo<EventInterv> _events;
};


class  IntervDisjoint
{
      public :

            const ElFilo<Interval>  & intervs() {return _intervs;};

            void init(PileEvInterv &);
            IntervDisjoint(PileEvInterv &);
            IntervDisjoint();


      private :

            ElFilo<Interval>  _intervs;
};



/*****************************************************************/
/*                                                               */
/*        Classes pour QTree                                     */
/*                                                               */
/*****************************************************************/

class ElQTRegionPlan
{
      public :

         virtual REAL D2(const Box2dr &) const = 0;
         virtual REAL D2(const Pt2dr &)  const = 0;
         virtual REAL D2(const SegComp &)  const = 0;
         virtual REAL D2(const cElTriangleComp &)  const ; // Def = err fatale
     virtual ~ElQTRegionPlan() {}
};


class ElQTRegPt : public ElQTRegionPlan
{
      public :

         virtual REAL D2(const Box2dr &)  const ;
         virtual REAL D2(const Pt2dr & )  const ;
         virtual REAL D2(const SegComp &)  const;
         virtual REAL D2(const cElTriangleComp &)  const ; // Implantee

         ElQTRegPt (Pt2dr);
     virtual ~ElQTRegPt() {}

      private :

          Pt2dr _pt;
};

class ElQTRegSeg : public ElQTRegionPlan
{
      public :

         virtual REAL D2(const Box2dr &)  const ;
         virtual REAL D2(const Pt2dr & )  const ;
         virtual REAL D2(const SegComp &)  const;

         ElQTRegSeg (Seg2d);
     virtual ~ElQTRegSeg() {}
      private :

          SegComp _seg;
};

class ElQTRegBox : public ElQTRegionPlan
{
      public :

         virtual REAL D2(const Box2dr &)  const ;
         virtual REAL D2(const Pt2dr & )  const ;
         virtual REAL D2(const SegComp &)  const;
         ElQTRegBox (const Box2dr &box);
     virtual ~ElQTRegBox() {}
      private :

          Box2dr _box;
};






class NewElQdtGen
{
      public  :

          INT NbObjMax() const { return _NbObjMax;}
          double SzMin()    const { return _SzMin;}

      protected :

          NewElQdtGen
          (
                     Box2dr        box,
                     INT           NbObjMax,
                     REAL          SzMin
          );


          Box2dr                        _box;
          INT                           _NbObjMax;
          double                        _SzMin;

     private :
          static Box2dr BoxQdt(const Box2dr &);

};


/*******************************************************/
/*                                                     */
/*    3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D 3D     */
/*                                                     */
/*******************************************************/

class cRapOnZ;


Pt3dr   norm_or_poly(const ElFifo<Pt3dr> &,REAL * surf =0);

struct cResOptInterFaisceaux
{
  public :
     void Init(const ElMatrix<double> &);
     double  mVal1;
     double  mVal2;
     double  mVal3;
     Pt3dr   mVec1;
     Pt3dr   mVec2;
     Pt3dr   mVec3;
};


/*
TIME :
     3.03199   ElSeg3D::L2InterFaisceaux 
     0.473224   InterSeg(const Pt3dr & aP0,...
     1.23799    InterSeg(const std::vector<Pt3r> 
*/

Pt3dr InterSeg(const std::vector<Pt3dr> & aVP0, const std::vector<Pt3dr> & aVP1,bool &Ok);
Pt3dr InterSeg(const Pt3dr & aP0,const Pt3dr & aP1,const Pt3dr & aQ0,const Pt3dr & aQ1,bool & Ok,double * aSquareD=0);
Pt3dr InterSeg(const ElRotation3D & aR2to1 ,const Pt3dr & aQ1,const Pt3dr & aQ2,bool & Ok,double * aSquareD=0);
Pt3dr InterSeg(const ElRotation3D & aR2to1 ,const Pt2dr & aP1,const Pt2dr & aP2,bool & Ok,double * aSquareD=0);
Pt3dr InterSeg(const std::vector<ElSeg3D> & aVP0, bool &Ok);





class ElSeg3D
{
     public :
         ElSeg3D(Pt3dr aP0,Pt3dr aP1);

          Pt3dr  Tgt() const;
          Pt3dr  TgNormee() const;
          Pt3dr  ProjOrtho(Pt3dr aP0) const;  // A la droite
          REAL   DistDoite(Pt3dr aP0) const;
          Pt3dr  PtOfAbsc(REAL anAbsc) const;
          REAL   AbscOfProj(Pt3dr aP) const;


          static ElSeg3D  CombinL1(const std::vector<Pt3dr> & aV);
          static ElSeg3D  CreateL2(const std::vector<Pt3dr> & aV);
          double  SomDistDroite(const std::vector<Pt3dr> & aV) const;

          void   AbscissesPseudoInter(REAL &anAbsc1,REAL & anAbsc2,const ElSeg3D & aS2);

          Pt3dr  PseudoInter(const ElSeg3D & aS2);



          void   Projections(Pt3dr & Proj2On1,const ElSeg3D & aS2,Pt3dr & Proj1On2);

          Pt3dr P0() const;
          Pt3dr P1() const;
          Pt3dr Mil() const;

      static Pt3dr L2InterFaisceaux
                   (
                           const std::vector<double> *,
                           const std::vector<ElSeg3D> &aVS,
                           bool * aOK=0,
                           const cRapOnZ *      aRAZ = 0,
                           cResOptInterFaisceaux * = 0,
                           const std::vector<Pt3dr> *  aVPts =0// Si existe doit etre pair et c'est une alternance pts/inc
               );
     private :

         Pt3dr mP0;
         Pt3dr mP1;
         Pt3dr mTN;
};

class cElPlan3D
{
      public :
          // Peu importe P0,P1,P2 du moment
          // qu'ils definissent le meme plan
          cElPlan3D(Pt3dr aP0,Pt3dr aP1,Pt3dr aP2);

              // Plan au moindre carres; si Pds=0 -> Pds[aK] = 1
              cElPlan3D(const std::vector<Pt3dr> &,const std::vector<double>*,ElSeg3D * aBestSeg=0);

          Pt3dr Inter(const cElPlan3D&,const cElPlan3D &,bool &OK) const;
          Pt3dr Inter(const ElSeg3D &,bool *OK=0) const;
              ElSeg3D Inter(const cElPlan3D&,bool &OK) const;

          // Plante si Plan Vertical
          REAL   ZOfXY(Pt2dr aP) const;
          Pt3dr  AddZ(Pt2dr aP) const;

             // void L1Ameliore(const std::vector<Pt3dr> & aVP,int aNbMax=-1);
             ElRotation3D CoordPlan2Euclid();
             const Pt3dr & Norm() const;
             const Pt3dr & U() const;
             const Pt3dr & V() const;
             const Pt3dr & P0() const;

             Pt3dr Proj(const Pt3dr &) const;

             void NoOp();

             void Revert() ;   // Z => -Z
      private :
          // Le plan est donne par son equation normale
          // mScal + mNorm.P = 0
          Pt3dr mNorm;
          REAL mScal;
          Pt3dr mP0;
          Pt3dr mU;
          Pt3dr mV;
};


cElPlan3D RobustePlan3D
          (
             const std::vector<Pt3dr> & aVPts,
             const std::vector<double> * aVPondInit,
             double anEffort,
             double aRatioTirage = 1.0,
             int    aNbStepLin = 7 
          );



/*
class cInterfSystemeCoordonne3D
{
     public :

         virtual Pt3dr ToEuclid(const Pt3dr & aP) const = 0;
         virtual Pt3dr FromEuclid(const Pt3dr & aP) const = 0;
     public :
};
*/

void TestInterPolyCercle();

REAL SurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerASurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerBSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerCSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerCElXSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerCElYSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerP0XSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerP0YSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerP1XSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
REAL DerP1YSurfIER (Pt2dr,REAL,REAL,REAL,Pt2dr,Pt2dr);
Fonc_Num    FN_SurfIER
            (
                Pt2d<Fonc_Num> aCel,
                Fonc_Num aA,Fonc_Num aB,Fonc_Num aC,
                Pt2d<Fonc_Num> aP0,Pt2d<Fonc_Num> aP1
            );

// A partir de l'image d'un repere orthonorme V0,V1
// calcule les parametres A,B,C d'une ellipse
// passant par V0 et V1

void ImRON2ParmEllipse
     (
         REAL & A,
         REAL & B,
         REAL & C,
         const Pt2dr & aV0,
         const Pt2dr & aV1
     );

Box2dr BoxEllipse(Pt2dr aCenter,REAL A,REAL B,REAL C);

     // return faux si pas Ellispe physique (ie pas deux VP pos)
bool EllipseEq2ParamPhys
     (
        REAL  & V1,
        REAL  & V2,
        REAL  & teta,
        REAL  A,
        REAL  B,
        REAL  C
     );

void InvertParamEllipse
     (
        REAL & A,  REAL & B,  REAL & C ,
        REAL  A0,  REAL  B0,  REAL   C0
     );


REAL  SimilariteEllipse(REAL A1,REAL B1,REAL C1,REAL A2,REAL B2,REAL C2);


/*************************************************************/
/*                                                           */
/*               Images de distances                         */
/*                                                           */
/*************************************************************/

void TestImDist(int,char **);

class cMailageSphere
{
    public :
        cMailageSphere(Pt2dr,Pt2dr,Pt2dr,bool Inv);
        void SetStep(Pt2dr);
        void SetMax(Pt2dr);
        void SetMin(Pt2dr);

        Pt2dr Pix2Spherik(Pt2dr aIndTP);
        Pt2di Spherik2PixI(Pt2dr  aTetaPhi);
        Pt2dr Spherik2PixR(Pt2dr  aTetaPhi);

        Pt2di SZEnglob();
        void WriteFile(const std::string & aNameFile);
        static cMailageSphere FromFile(const std::string & aNameFile);

        Pt3dr DirMoy();
        Pt2dr DirMoyH();

    private :
        Pt2dr mStep;  // Teta ,Phi
        Pt2dr mMin;
        Pt2dr mMax;
        int   mInv;
};

class cGridNuageP3D
{
    public :
            cGridNuageP3D
        (
             const std::string &,
         Pt2di aSz = Pt2di(-1,-1), // Def => Tiff.sz()
         Pt2di aP0 = Pt2di(0,0)
            );
        Pt2di Sz() const;
        INT   Cpt(Pt2di) const;
        Pt3dr P3D(Pt2di) const;

        std::string NameShade() const;
        Im2D_U_INT1   ImShade();
        Im2D_INT1     ImCpt();

        // Profondeur dans la direction moyenne
        Fonc_Num FProfDMoyH();

        Tiff_Im   TifFile(const std::string & aShortName);
    private :
        static const std::string theNameShade;
        void Init(Im2DGen,const std::string &);

        std::string   mName;
        Pt2di         mSz;
        Pt2di         mP0;

        Im2D_REAL4    mImX;
        REAL4 **      mDX;
        Im2D_REAL4    mImY;
        REAL4 **      mDY;
        Im2D_REAL4    mImZ;
        REAL4 **      mDZ;
        Im2D_INT1     mImCpt;
        Im2D_U_INT1   mImShade;
        cMailageSphere  mMSph;
};


class cQtcElNuageLaser;
class cResReqNuageLaser
{
   public :
     virtual void cResReqNuageLaser_Add(const Pt3dr & aP) = 0;
   virtual ~cResReqNuageLaser() {}
};

class cElNuageLaser
{
     public  :

       typedef enum
       {
            eConvId,
            eConvCarto2Terr,
            eConvCarto2TerIm
       } eModeConvGeom;
       cElNuageLaser
       (
              const std::string & aNameFile,
              const char *  aNameOri = NULL,
              const char *  aNameGeomCible = NULL,  // GeomCarto GeomTerrain GeomTerIm1
              const char *  aNameGeomInit = "GeomCarto"
       );
       const std::vector<Pt3dr> & VPts() const;
       void SauvCur(const std::string &);
       void Debug(const std::string & aName);


       REAL   ZMin () const;
       REAL   ZMax () const;
       Box2dr Box() const;

       void  AddQt(INT aNbObjMaxParFeuille,REAL aDistPave);

       void ParseNuage(cResReqNuageLaser & aResParse,Box2dr aBox);

     private :
       cElNuageLaser(const cElNuageLaser &);  // Non Implemente

       std::vector<Pt3dr>  mVPts;

       REAL                mZMax;
       REAL                mZMin;
       Pt2dr               mPInf;
       Pt2dr               mPSup;
       cQtcElNuageLaser *  mQt;
};


struct gpc_polygon;

class cElPolygone
{
    public :
       typedef std::vector<Pt2dr> tContour;
       typedef const std::list<tContour>  tConstLC;
       typedef std::list<tContour>::const_iterator  tItConstLC;

       void AddContour(const tContour &,bool isHole);
       cElPolygone();
       cElPolygone (const gpc_polygon &);
       struct gpc_polygon ToGPC() const;

       const std::list<tContour> & Contours() const;
       const std::list<bool> &     IsHole();
       tContour  ContSMax() const;


       cElPolygone operator * (const cElPolygone & aPol)  const;
       cElPolygone operator + (const cElPolygone & aPol)  const;
       cElPolygone operator - (const cElPolygone & aPol)  const;
       cElPolygone operator ^ (const cElPolygone & aPol)  const;

       double Surf() const;
       double DiamSimple() const;  // Suppose que existe surf englob

       static cElPolygone FromBox(const Box2dr & aBox);

    private  :
       cElPolygone GenOp(const cElPolygone & aPol,INT)const;

       std::list<tContour>   mContours;
       std::list<bool>       mIsHole;
};


// Representation maillee d'un nuage de points 3D



/*
*/

#endif // _ELISE_GENERAL_GEOM_VECTEUR_H


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
