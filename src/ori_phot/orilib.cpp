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
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"


/* define ENSEA pour enlever les routines inutiles pour l'ENSEA
#define ENSEA
*/
/* define UNIX pour enlever les specificites VMS/MATIS
#ifndef UNIX
#define UNIX
#endif
*/

//#ifndef __USE_ORILIB_ELISE__
//#else

//#include "StdAfx.h"

#if defined(UNIX)

#define max_string_descr 16
#define _min(v1,v2)	( (v2) < (v1) ? (v2) : (v1) )
#define _max(v1,v2)	( (v2) > (v1) ? (v2) : (v1) )
#define _i2(ii,jj,dim)    ((jj * dim) + ii)
#define _abs( a ) 	( a < 0 ? -(a) : (a) )
#define _egaux(a,b) 	( a == b ? 1 : 0 )
#define _superieur(a,b) ( a >= b ? 1 : 0 )
#define malloc$ malloc
#define free$ free

#else

/*#include "d$jamet:[jamet.these.outils]memlib.h"*/
#include "memlib.h"
#endif

#define _RayT 6366000.0
#define _Pi 3.14159265359
#define _refra1 1.28E-8
#define _refra2 3.91E-5
#define _pscalaire( a , b ) (a[0]*b[0]+a[1]*b[1]+a[2]*b[2])
#define _vecdiv( vv , den ) vv[0]=vv[0]/den;vv[1]=vv[1]/den;vv[2]=vv[2]/den
#ifndef orkMeridienDeParis
#define orkMeridienDeParis 	2.0+20.0/60.0+16.0/3600.0	/* 220'16'' Est */
#endif


// extern cDistorBilin GlobFromXmlGridStuct(const cCalibrationInterneGridDef &  aCIG);



void AdaptDist2PPaEqPPs(NS_ParamChantierPhotogram::cCalibDistortion & aCD)
{
     if (aCD.ModRad().IsInit())
     {
        const cCalibrationInterneRadiale & aCIR = aCD.ModRad().Val();
        if (aCIR.PPaEqPPs().ValWithDef(false))
        {
           cCalibrationInterneUnif aCIU;
           aCIU.TypeModele() = eModele_DRad_PPaEqPPs;
           aCIU.Params() = aCIR.CoeffDist();
           aCD.ModRad().SetNoInit();
           aCD.ModUnif().SetVal(aCIU);
        }
     }
     else if (aCD.ModPhgrStd().IsInit())
     {
        const cCalibrationInternePghrStd aCIPS =  aCD.ModPhgrStd().Val();
        const cCalibrationInterneRadiale & aCIR = aCIPS.RadialePart();

        if (aCIR.PPaEqPPs().ValWithDef(false))
        {

            cCalibrationInterneUnif aCIU;
            aCIU.TypeModele() = eModele_Fraser_PPaEqPPs;

        std::vector<double> & aVParam = aCIU.Params();
        aVParam = aCIR.CoeffDist();
            while (aVParam.size() < 5)
                  aVParam.push_back(0);

            aVParam.push_back(aCIPS.P1().ValWithDef(0));
            aVParam.push_back(aCIPS.P2().ValWithDef(0));
            aVParam.push_back(aCIPS.b1().ValWithDef(0));
            aVParam.push_back(aCIPS.b2().ValWithDef(0));

            aCD.ModPhgrStd().SetNoInit();
            aCD.ModUnif().SetVal(aCIU);
        }
     }
}

ElAffin2D AfGC2M(const cCalibrationInternConique & aCIC)
{
   if (aCIC.OrIntGlob().IsInit())
   {
        cOrIntGlob anOIG = aCIC.OrIntGlob().Val();
        ElAffin2D anAff = Xml2EL(anOIG.Affinite());
        return anOIG.C2M() ? anAff  : anAff.inv();
   }
   return ElAffin2D::Id();
}

double DMaxCoins(Pt2dr aP0,Pt2dr aP1,Pt2dr aC)
{
    Box2dr aB(aP0,aP1);
    Pt2dr Coins[4];
    aB.Corners(Coins);

    double aDMax = 0;
    for (int aK=0; aK<4 ; aK++)
    {
        ElSetMax(aDMax,euclid(Coins[aK],aC));
    }

    return aDMax;
}


double DMaxCoins(ElAffin2D AfC2M,Pt2dr aSzIm,Pt2dr aC)
{
    return DMaxCoins(AfC2M(Pt2dr(0,0)),AfC2M(aSzIm),aC);
}
double DMaxCoins(Pt2dr aSzIm,Pt2dr aC)
{
    return DMaxCoins(Pt2dr(0,0),aSzIm,aC);
}

void DebugOri(Ori3D_Std anOri)
{
   Data_Ori3D_Std * aDOR = anOri.dos();
   ELISE_ORILIB::or_orientation *  ori = aDOR->Ori();

   std::cout<< "Adr = " << ori  << "\n";
   std::cout<< "T2P = " << ori->gt2p.ns  << " " << ori->gt2p.nl << "\n";
   std::cout<< "P2T = " << ori->gp2t.ns  << " " << ori->gp2t.nl << "\n";
}

int XML_orlit_fictexte_orientation (const char *fic, or_orientation *ori,bool QuickGrid );


typedef char (*Orilib_Interp)( unsigned char /*huge*/ *, int*, int*, double*, double* );

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*				ROUTINES GENERALES			      */
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
int orsommet_de_pdv_terrain ( 	void* *phot,
            double *xterrain, double *yterrain, double *zterrain )
{
int		status ;
or_orientation	*ori ;

ori = (or_orientation *) *phot ;
if ( ori != 0 )
  {
  status = 1 ;
  *xterrain = (*ori).sommet[0] ;
  *yterrain = (*ori).sommet[1] ;
  *zterrain = (*ori).sommet[2] ;
  }
else
  {
  status = 0 ;
  }
return status ;
}
/*----------------------------------------------------------------------------*/
int orsommet_de_pdv_carto ( 	void * *phot,
            double *xcarto, double *ycarto, double *zcarto )
{
int		status ;
or_orientation	*ori ;

ori = (or_orientation *) *phot ;
if ( ori != 0 )
  {
  status = orterrain_to_carte ( phot,
        &((*ori).sommet[0]), &((*ori).sommet[1]), &((*ori).sommet[2]),
        xcarto, ycarto, zcarto ) ;
  }
else
  {
  status = 0 ;
  }
return status ;
}
/*----------------------------------------------------------------------------*/
 int orvecteur_solaire ( void* *phot, double soleil[3] )
/* vecteur soleil-terre dans le repere terrain */
{
or_orientation		*ori ;
double			latitude, longitude ;
int			status ;

ori = (or_orientation *) *phot ;
if ( ori != 0 )
  {
  status = 1 ;
  Lamb_geo( (*ori).origine[0], (*ori).origine[1], (*ori).lambert,
            &longitude , &latitude ) ;
    longitude = longitude + orkMeridienDeParis ;

    orvecteur_soleil_terre ( &((*ori).annee), &((*ori).jour), &((*ori).mois),
                    &((*ori).heure), &((*ori).minute), &latitude, &longitude,
                soleil ) ;
    }
else
  {
  status = 0 ;
  }
return status ;
}
/*----------------------------------------------------------------------------*/
 int orpoint_fuite_soleil ( void* *phot, double fuite[2] )
/* point de fuite des directions objets->ombre portees dans l'image */
{
double			soleil[3] ;
int			status ;

if ( *phot != 0 )
  {
  status = orvecteur_solaire ( phot, soleil ) ;
  if ( status == 1 )
    {
    orinters_SM_photo ( phot, soleil, &fuite[0], &fuite[1] ) ;
    }
  }
else
  {
  status = 0 ;
  }
return status ;
}
/*----------------------------------------------------------------------------*/
 int orpoint_fuite_verticale ( void* *phot, double fuite[2] )
/* point de fuite des directions objets->ombre portees dans l'image */
{
or_orientation		*ori ;
double			vert[3] ;
int			status ;

if ( *phot != 0 )
  {
  status = 1 ;
  ori = (or_orientation *) *phot ;
  /* centre de la terre */
  vert[0] = 0.0 ;
  vert[1] = 0.0 ;
  vert[2] = -_RayT ;
  /* verticale sur le sommet de prise de vue */
  vert[0] = vert[0] - (*ori).sommet[0] ;
  vert[1] = vert[1] - (*ori).sommet[1] ;
  vert[2] = vert[2] - (*ori).sommet[2] ;
  /* on retablit une norme de vecteur raisonnable */
  _vecdiv ( vert , _RayT ) ;
  /* point de fuite photo */
  orinters_SM_photo ( phot, vert, &fuite[0], &fuite[1] ) ;
  }
else
  {
  status = 0 ;
  }
return status ;
}
/*----------------------------------------------------------------------------*/
 int orlit_orientation (const  char *fichier, void* *phot )
/* La meme chose que image_orientation, mais sans gestion via la structure
    image ; cette routine est donc principalement a usage externe au labo ;
    Elle effectue une lecture in extenso (i.e. sans gestion de fenetrage)
    du fichier passe en parametre (nom complet avec extension) */
{
 int			status ;
 or_orientation 	*ori ;

 ori = (or_orientation *) *phot ;
 if ( ori != 0 )
     {
     status = 1 ;
     *phot = ori ;
    /* lecture du fichier d'orientation */
     status = NEW_orlit_fic_orientation ( fichier, ori ) ;
    }
 else
    {
    status = 0 ;
    }

 return status ;
}
/*----------------------------------------------------------------------------*/
 int orlit_orientation_texte (const char *fichier, void* *phot ,bool QuickGrid)
/* La meme chose que image_orientation, mais sans gestion via la structure
    image ; cette routine est donc principalement a usage externe au labo ;
    Elle effectue une lecture in extenso (i.e. sans gestion de fenetrage)
    du fichier passe en parametre (nom complet avec extension) */
{
 int			status ;
 or_orientation 	*ori ;

 ori = (or_orientation *) *phot ;
 if ( ori != 0 )
     {
     status = 1 ;
     *phot = (void*) ori ;
    /* lecture du fichier d'orientation */
     status = orlit_fictexte_orientation ( fichier, ori,QuickGrid ) ;
     }
 else
     {
     status = 0 ;
     }

 return status ;
}
/*----------------------------------------------------------------------------*/
 int orcopy_orientation ( void* *phot1, void* *phot2 )
{
 int			status ;
 or_orientation 	*ori1, *ori2 ;

 if ( (phot1 == 0) || (phot2 == 0) ) { return 0 ; }
 ori1 = (or_orientation *) *phot1 ;
 ori2 = (or_orientation *) *phot2 ;

 status = 1 ;
 *phot2 = (void*) ori2 ;
 (*ori2) = (*ori1) ;

 return status ;
}
/*----------------------------------------------------------------------------*/
 int orecrit_orientation ( void* *photo,const char *fichier )
{
 int			status ;
 or_orientation 	*ori ;

 status = 1 ;
 ori = (or_orientation *) *photo ;
 status = orecrit_fic_orientation ( fichier, ori ) ;
 return status ;
}
/*----------------------------------------------------------------------------*/
 int orecrit_orientation_texte ( void* *photo,const char *fichier )
{
 int			status ;
 or_orientation 	*ori ;

 status = 1 ;
 ori = (or_orientation *) *photo ;
 status = orecrit_fictexte_orientation ( fichier, ori ) ;
 return status ;
}
/*----------------------------------------------------------------------------*/
 int orfenetrage_orientation ( void* *phot,
                    double *left, double *top,
                double *right, double *bottom,
                double *hstep, double *vstep,
                void* *photout )
 /* gestion d'un fenetrage pour ajustement des parametres photogrammetriques */
 /* convention == (0,0) est le premier pixel */
{
 int			status ;
 void*			phot_s ;	/*sauvegarde*/
 or_orientation		*ori0_s ;
 or_orientation		*ori0 ;
 or_orientation		*ori ;
 int				ns,nl ;

 if ( ((*phot)==0) || ((*photout)==0) ) return 0 ;

    /* transformations liees au fenetrage de l'image :
    seuls la position du point principal,le pas et les grilles changent */
 status = oralloc_orientation ( &phot_s ) ;
 if ( status != 1 ) return status ;
 ori0 = (or_orientation *) *phot ;
 ori = (or_orientation *) *photout ;
 ori0_s = (or_orientation *) phot_s ;
 (*ori) = (*ori0) ;
 (*ori0_s) = (*ori0) ;

 /* on repasse en metres */
 (*ori).ipp[0] = (*ori).ipp[0] * (*ori).pix[0] ;
 (*ori).ipp[1] = (*ori).ipp[1] * (*ori).pix[1] ;

 (*ori).ipp[0] = (*ori).ipp[0] -
         (*ori).pix[0] * (*left) ;
 (*ori).ipp[1] = (*ori).ipp[1] -
         (*ori).pix[1] * (*top) ;
 (*ori).pix[0] = (*ori).pix[0] * (*hstep) ;
 (*ori).pix[1] = (*ori).pix[1] * (*vstep) ;

 /* on repasse en pixels */
 (*ori).ipp[0] = (*ori).ipp[0] / (*ori).pix[0] ;
 (*ori).ipp[1] = (*ori).ipp[1] / (*ori).pix[1] ;

 /* taille de l'image */
 ns = 1 + (int) ( (*right - *left)/ (*hstep) ) ;
 nl = 1 + (int) ( (*bottom - *top)/ (*vstep) ) ;
 (*ori).ins = ns ;
 (*ori).inl = nl ;

 /* fenetrage de la grille : on passe par une reprojection standard */
 orfenetrage_grille (  &((*ori0_s).gt2p), left, top, hstep, vstep,
            &ns, &nl, &((*ori).gt2p)  ) ;
 orfenetrage_grille (  &((*ori0_s).gp2t), left, top, hstep, vstep,
            &ns, &nl, &((*ori).gp2t)  ) ;
 status = orfree_orientation ( &phot_s ) ;
 return status ;
}
/*----------------------------------------------------------------------------*/
void orfenetrage_grille ( or_grille *gr1 ,
               double *left, double *top,
               double *hstep, double *vstep,
               int *ins, int *inl,
               or_grille *gr2 )
/* les deformation grille sont en pixels */
{
double			gpas ;
double			x0, y0, dx, dy, xx, yy, xx1, yy1 ;
int			ii, jj, ip ;
int			istat ;


// std::cout << "orfenetrage_grille " << (*gr1).ns << " " << (*gr1).nl << "\n";

if ( (*gr1).pas > 0 )
    {
    /* dimensions de la nouvelle grille */
    ordim_grille ( ins, inl, &((*gr2).ns), &((*gr2).nl), &((*gr2).pas) ) ;
    gpas = _min (  (double)(*gr1).pas/(*hstep) , (double)(*gr1).pas/(*vstep)  ) ;
    (*gr2).pas = _max ( (*gr2).pas , (int)(gpas+0.5) ) ;
    (*gr2).ns = 1 + (*ins - 1 + (*gr2).pas - 1)/ (*gr2).pas ;
    (*gr2).nl = 1 + (*inl - 1 + (*gr2).pas - 1)/ (*gr2).pas ;

    /* re-echantillonnage (calcul en coordonnees images) */
    x0 = *left ;
    y0 = *top ;
    dx = (double)(*gr2).pas * (*hstep) ;
    dy = (double)(*gr2).pas * (*vstep) ;

    yy = y0 ;
    ip = 0 ;
    for ( jj = 0 ; jj < (*gr2).nl ; jj++ )
        {
        xx = x0 ;
        for ( ii = 0 ; ii < (*gr2).ns ; ii++ )
        {
        xx1 = xx ; yy1 = yy ;
        istat = orcorrige_distortion ( &xx1, &yy1, gr1 ) ;
        if ( istat == 1 )
            {
            (*gr2).dx[ip] = (xx1-xx)/(*hstep) ;
            (*gr2).dy[ip] = (yy1-yy)/(*vstep) ;
            }
        else
            {
            (*gr2).dx[ip] = 0.0 ;
            (*gr2).dy[ip] = 0.0 ;
            }
        ip++ ;
        xx = xx + dx ;
        }
        yy = yy + dy ;
        }
    }
else
     {
     }
}
/*----------------------------------------------------------------------------*/
 int oralloc_orientation ( void* *phot )
{
 int		status ;
 or_orientation	*ori ;

 ori = (or_orientation *) malloc( sizeof(or_orientation) ) ;
status = ( ori != 0 ) ;

*phot = (void*) ori ;
return status ;
}
/*----------------------------------------------------------------------------*/
 int orfree_orientation ( void* *phot )
{
 int	status ;
 or_orientation *ori ;

 status = 1 ;
 ori = (or_orientation *) *phot ;
 free ( (char*) ori ) ;

 return status ;
}
/*----------------------------------------------------------------------------*/
#ifndef ENSEA
void orAddFilmDistortions ( or_grille *gr, or_grille *igr,
                             double IdealMarks[16], double RealMarks[16],
                             int *NMarks )
/*
ATTENTION :
    - la grille en entree est deja initialisee (on incremente les deformations) ;
    - les reperes de fond de chambre sont situes dans une image d'origine (1,1)
    - leurs coordonnees sont en pixels

    gr =  grille pour passage Terrain -> Photo
    igr = grille pour passage Photo -> Terrain
*/
{
double Coefs[8] ;
double teta ;
int ig, is, il ;
double xyTheorique[2] ;
double xyFilm[2] ;
int istat ;
double xx,yy,xx1,yy1,dd,ddmax,emq ;
double dx, dy ;
double mdx, mdy ;

/* Calcul de la grille t->p */
teta = FDRotCorrection ( IdealMarks ) ;
FDBilinearTransform ( teta, IdealMarks, RealMarks, *NMarks, Coefs ) ;
ig = 0 ;
mdx = 0.0 ;
mdy = 0.0 ;
for ( il = 0 ; il < (*gr).pas*(*gr).nl ; il = il + (*gr).pas )
    {
    for ( is = 0 ; is < (*gr).pas*(*gr).ns ; is = is + (*gr).pas )
        {
        xyTheorique[0] =  (double)(is+1) ;
        xyTheorique[1] =  (double)(il+1) ;
        FDTransformPoint ( xyTheorique, teta, Coefs, xyFilm ) ;
        dx = (xyFilm[0] - xyTheorique[0]) ;
        if ( _abs(dx) > mdx ) mdx = _abs(dx) ;
        dy = (xyFilm[1] - xyTheorique[1]) ;
        if ( _abs(dy) > mdy ) mdy = _abs(dy) ;
        (*gr).dx[ig] = (*gr).dx[ig] + dx ;
        (*gr).dy[ig] = (*gr).dy[ig] + dy ;
        ig++ ;
        }
    }




/* Calcul de la grille p->t */
FDBilinearTransform ( teta, RealMarks, IdealMarks, *NMarks, Coefs ) ;
ig = 0 ;
for ( il = 0 ; il < (*igr).pas*(*igr).nl ; il = il + (*igr).pas )
    {
    for ( is = 0 ; is < (*igr).pas*(*igr).ns ; is = is + (*igr).pas )
        {
        xyFilm[0] = (double) (is+1) ;
        xyFilm[1] = (double) (il+1) ;
        FDTransformPoint ( xyFilm, teta, Coefs, xyTheorique ) ;
        (*igr).dx[ig] = (*igr).dx[ig] + (xyTheorique[0] - xyFilm[0]) ;
        (*igr).dy[ig] = (*igr).dy[ig] + (xyTheorique[1] - xyFilm[1]) ;
        ig++ ;
        }
    }

/* Controle de la precision de la grille inverse */
ddmax = 0.0 ;
emq = 0.0 ;
ig = 0 ;
for ( il = 0 ; il < (*igr).pas*((*igr).nl-1) ; il = il + 50 )
    {
    yy = ((double) il) ;
    for ( is = 0 ; is < (*igr).pas*((*igr).ns-1) ; is = is + 50 )
        {
        xx = ((double) is) ;
        xx1 = xx ;
        yy1 = yy ;
        istat = orcorrige_distortion ( &xx1, &yy1, igr ) ;
        if ( istat == 1 )
            {
            xyFilm[0] = xx + 1.0 ;
            xyFilm[1] = yy + 1.0 ;
            FDTransformPoint ( xyFilm, teta, Coefs, xyTheorique ) ;
            dd = (xyTheorique[0]-1.0-xx1)*(xyTheorique[0]-1.0-xx1) +
                (xyTheorique[1]-1.0-yy1)*(xyTheorique[1]-1.0-yy1) ;
            if ( dd > ddmax )
                {
                ddmax = dd ;
                }
            emq = emq + dd ;
            ig++ ;
            }
        }
    }
emq = sqrt(emq/(double)ig) ;
ddmax = sqrt(ddmax) ;


/*
Nota Bene :
Etant calculee par moindres carres, la grille igr n'est pas strictement l'inverse de
la grille gr; on verifie donc ici que l'erreur d'inversion est acceptable
*/
ddmax = 0.0 ;
emq = 0.0 ;
ig = 0 ;
for ( il = 0 ; il < (*gr).pas*(*gr).nl ; il = il + (*gr).pas )
    {
    yy = (double) il ;
    for ( is = 0 ; is < (*gr).pas*(*gr).ns ; is = is + (*gr).pas )
        {
        xx = (double) is ;
        xx1 = xx ;
        yy1 = yy ;
        istat = orcorrige_distortion ( &xx1, &yy1, gr ) ;
        istat = orcorrige_distortion ( &xx1, &yy1, igr ) ;
        dd = (xx-xx1)*(xx-xx1) + (yy-yy1)*(yy-yy1) ;
        if ( dd > ddmax ) ddmax = dd ;
        emq = emq + dd ;
        ig++ ;
        }
    }
emq = sqrt(emq/(double)ig) ;
ddmax = sqrt(ddmax) ;

}
#endif
/*----------------------------------------------------------------------------*/
 void orgrille_refraction ( or_orientation *ori, or_grille *gr )
{
int			ig ;
int			is, il ;
double			col , lig ;
double			rr, dr ;
double			pp, HH ;
double			tt ;
double			coef ;



if ( (*gr).pas <= 0.0 ) return ;

/*
 * On uitlise la formule de Divelec donnee par le cours de
 * Christophe Dekeyne (avec des constantes pour un calcul en metres
 */
HH = (*ori).sommet[2] - (*ori).altisol ;
pp = (*ori).focale ;
coef = _refra1*HH*pp*(1.0 - _refra2*HH) ;		/* pour un resultat
                               en m */

ig = 0 ;
for ( il = 0 ; il < (*gr).pas*(*gr).nl ; il = il + (*gr).pas )
    {
     for ( is = 0 ; is < (*gr).pas*(*gr).ns ; is = is + (*gr).pas )
    {
    col = (*ori).pix[0]* ( (double)is - (*ori).ipp[0] ) ;
    lig = (*ori).pix[1]* ( (double)il - (*ori).ipp[1] ) ;
    rr = sqrt( col*col  + lig*lig ) ;
    if ( rr > 1E-12 )
        {
        tt = rr / (*ori).focale ;
         dr = coef * tt * (1 + tt*tt ) ;
        (*gr).dx[ig] = (*gr).dx[ig] + dr * (col / rr) / (*ori).pix[0] ;
        (*gr).dy[ig] = (*gr).dy[ig] + dr * (lig / rr) / (*ori).pix[1] ;
        }
    ig++ ;
    }
    }
}
/*----------------------------------------------------------------------------*/
 void orinverse_grille ( or_grille *gin, or_grille *gout )
{

int			ig ;
int			is, il ;
double			col  , lig ;
double			col0 , lig0 ;
double			dx, dy ;

(*gout).pas = (*gin).pas ;
(*gout).ns = (*gin).ns ;
(*gout).nl = (*gin).nl ;

for ( ig = 0 ; ig < ((*gin).nl * (*gin).ns) ; ig++ )
    { (*gout).dx[ig] = - (*gin).dx[ig] ;
      (*gout).dy[ig] = - (*gin).dy[ig] ; }

if ( (*gin).pas <= 0.0 ) return ;

ig = 0 ;
for ( il = 0 ; il < (*gin).pas*(*gin).nl ; il = il + (*gin).pas )
    {
    for ( is = 0 ; is < (*gin).pas*(*gin).ns ; is = is + (*gin).pas )
    {
    dx = 0.0 ;
    dy = 0.0 ;
    do  {
        (*gout).dx[ig] = (*gout).dx[ig] + dx ;
        (*gout).dy[ig] = (*gout).dy[ig] + dy ;
        col0 = (double)is + (*gout).dx[ig] ;
        lig0 = (double)il + (*gout).dy[ig] ;
        col = col0 ; lig = lig0 ;
        orcorrige_distortion( &col, &lig, gin ) ;
            dx = (double)is - col ;
        dy = (double)il - lig ;
        } while ( 	(col != col0) && (lig != lig0) &&
            ( _abs( dx ) > 0.01 ) && ( _abs( dy ) > 1E-8 )	) ;
    ig++ ;
    }
    }
}
/*----------------------------------------------------------------------------*/
void ordim_grille ( int *ins, int *inl, int *gns, int *gnl, int *gpas )
{
*gpas = _max ( (*ins -1 + _NS_GRILLE-2)/(_NS_GRILLE-1) ,
           (*inl -1 + _NS_GRILLE-2)/(_NS_GRILLE-1) ) ;
*gns = 1 + (*ins - 1 + *gpas - 1)/ *gpas ;
*gnl = 1 + (*inl - 1 + *gpas - 1)/ *gpas ;
if ( *gns > _NS_GRILLE ) Tjs_El_User.ElAssert (0,EEM0<< "BUG-Grille  (in orilib ?)" );
if ( *gnl > _NS_GRILLE ) Tjs_El_User.ElAssert (0,EEM0<< "BUG-Grille  (in orilib ?)" );
}
/*----------------------------------------------------------------------------*/
 int orinit_distortions ( char *chambre, int *refrac,
                            double IdealMarks[8], double RealMarks[16],
                            int *NMarks,
                            void* *phot )
{
/*
 * Les distortions de chambre ne sont pas traitees :
 * on en rentrera les modeles au fur et a mesure des besoins
 */
or_orientation 		*ori ;
or_grille		*gr, *igr ;
int			ipg ;

ori = (or_orientation *) *phot ;
gr = &((*ori).gt2p) ;
igr = &((*ori).gp2t) ;

(*ori).refraction = *refrac ;
(*ori).distor = (strlen(chambre) > 0) || (*refrac != 0) || ((IdealMarks!=0)&&(RealMarks!=0)) ;
sprintf( (*ori).chambre, "%s", chambre ) ;

ordim_grille ( &((*ori).ins), &((*ori).inl),
         &((*gr).ns), &((*gr).nl), &((*gr).pas) ) ;
for ( ipg = 0 ; ipg < _NS_GRILLE*_NS_GRILLE ; ipg++ )
     { (*gr).dx[ipg] = 0.0 ; (*gr).dy[ipg] = 0.0 ; }


    /* on traite d'abord la refraction */
if ( *refrac != 0 ) orgrille_refraction ( ori, gr ) ;

    /* ensuite on ajoute les deformations dues a la chambre */

    /* on calcule la grille inverse */
orinverse_grille ( gr, igr ) ;

    /* ensuite on ajoute les deformations Film (sur les deux grilles :
        on ne passe pas par inverse_grille pour les deformations film sur gp2t
        dans la mesure ou la formulation inverse est calculable  */
if ( (IdealMarks != 0) && (RealMarks != 0) )
    {
#ifndef ENSEA
    orAddFilmDistortions ( gr, igr, IdealMarks, RealMarks, NMarks ) ;
#endif
    }


return 1 ;
}
/*----------------------------------------------------------------------------*/
void orinit_grille_0 ( void* *phot )
/* initialise une grille sans distortions
   ( utilie pour les grilles d'epipolaires) */
{
or_orientation 		*ori ;
or_grille		*gr, *igr ;
int			ipg ;

ori = (or_orientation *) *phot ;
gr = &((*ori).gt2p) ;
igr = &((*ori).gp2t) ;

(*gr).ns = 2 ;
(*gr).nl = 2 ;
(*gr).pas = 0 ;
for ( ipg = 0 ; ipg < _NS_GRILLE*_NS_GRILLE ; ipg++ )
    { (*gr).dx[ipg] = 0.0 ; (*gr).dy[ipg] = 0.0 ; }
(*igr).ns = 2 ;
(*igr).nl = 2 ;
(*igr).pas = 0 ;
for ( ipg = 0 ; ipg < _NS_GRILLE*_NS_GRILLE ; ipg++ )
    { (*igr).dx[ipg] = 0.0 ; (*igr).dy[ipg] = 0.0 ; }
}
/*----------------------------------------------------------------------------*/
 int orterrain_to_photo (  void* *phot,
            const double *xx, const double *yy, const double *zz,
            double *colonne, double *ligne )
{
 or_orientation		*ori ;
 double 		SM[3] ;
 int			status ;

 status = 1 ;

 ori = (or_orientation *) *phot ;

 /* SM */
 SM[0] = *xx - (*ori).sommet[0] ;
 SM[1] = *yy - (*ori).sommet[1] ;
 SM[2] = *zz - (*ori).sommet[2] ;


 orinters_SM_photo ( phot, SM, colonne, ligne ) ;

 return status ;
}
/*----------------------------------------------------------------------------*/
 double orphotos_to_terrain(void* *phot1, double *col1, double *lig1,
                 void* *phot2, double *col2, double *lig2,
                 double *xx, double *yy, double *zz )
{
 or_orientation		*ori1, *ori2 ;
 double			m1[3], m2[3], S1S2[3] ;
 double			aa,bb,cc,dd,ee,det ;
 double 		lambda1, lambda2 ;
 double			status ;	/*on retourne l'erreur d'intersection*/

 status = 1 ;
 ori1 = (or_orientation *) *phot1 ;
 ori2 = (or_orientation *) *phot2 ;

    /* on resoud : lambda1 * m1S1 + S1S2 + lambda2 * S2m2 orthogonal
       aux deux vecteurs S1m1 et S2m2 */

 /* Coordonnees des points image dans le repere terrain */
 orSM ( phot1, col1, lig1, m1 ) ;			/* S1m1 */
 orSM ( phot2, col2, lig2, m2 ) ;			/* S2m2 */

 S1S2[0] = (*ori2).sommet[0] - (*ori1).sommet[0] ;	/* S1S2 */
 S1S2[1] = (*ori2).sommet[1] - (*ori1).sommet[1] ;
 S1S2[2] = (*ori2).sommet[2] - (*ori1).sommet[2] ;

 /* resolution du systeme :
    aa lambda1 + bb lambda2 = dd
    bb lambda1 + cc lambda2 = ee 		*/
 aa = _pscalaire( m1 , m1 ) ;
 bb = - _pscalaire( m1 , m2 ) ;
 cc = _pscalaire( m2 , m2 ) ;
 dd = _pscalaire( m1 , S1S2 ) ;
 ee = - _pscalaire( m2 , S1S2 ) ;
 det = (cc*aa) - (bb*bb) ;
 if ( det <= 0 )
    {
    /* on ne peut pas intersecter deux rayons paralleles !! */
    return (-1.0) ;
    }
 lambda1 = ( (cc * dd) - (bb * ee) ) / det ;
 lambda2 = ( (aa * ee) - (bb * dd) ) / det ;

 /* coordonnes en metres dans le repere terrestre */
 m1[0] = (*ori1).sommet[0] + lambda1 * m1[0] ;
 m1[1] = (*ori1).sommet[1] + lambda1 * m1[1] ;
 m1[2] = (*ori1).sommet[2] + lambda1 * m1[2] ;

 m2[0] = (*ori2).sommet[0] + lambda2 * m2[0] ;
 m2[1] = (*ori2).sommet[1] + lambda2 * m2[1] ;
 m2[2] = (*ori2).sommet[2] + lambda2 * m2[2] ;


 *xx = ( m1[0] + m2[0] ) / 2.0 ;
 *yy = ( m1[1] + m2[1] ) / 2.0 ;
 *zz = ( m1[2] + m2[2] ) / 2.0 ;

 /* vecteur m1m2 */
 m2[0] = m2[0] - m1[0] ;
 m2[1] = m2[1] - m1[1] ;
 m2[2] = m2[2] - m1[2] ;
 status = sqrt( _pscalaire( m2 , m2 ) ) ;

 return status ;
}
/*----------------------------------------------------------------------------*/
 int ororigine_terrain ( void* *phot, double origine[2] )
{
 or_orientation		*ori ;
 int			status ;
 status = 1 ;
 ori = (or_orientation *) *phot ;
 origine[0] = (*ori).origine[0] ;
 origine[1] = (*ori).origine[1] ;
 return status ;
}
/*----------------------------------------------------------------------------*/
 int orcarte_to_terrain ( void* *phot,
                double *cx, double *cy, double *cz,
               double *tx, double *ty, double *tz )
/* D'apres les formules des routines de P.Julien (SUBPHOT) */
{
 or_orientation		*ori ;
 double 		origine[2] ;
 double			UU[2] ;
 double			TT, CC ;
 int			status ;

 status = 1 ;
 ori = (or_orientation *) *phot ;
 origine[0] = (*ori).origine[0] ;
 origine[1] = (*ori).origine[1] ;

 UU[0] = *cx - origine[0] ;
 UU[1] = *cy - origine[1] ;
 TT = ( UU[0]*UU[0] + UU[1]*UU[1] ) / ( 4.0 * _RayT ) ;
 CC = ( _RayT + *cz ) / ( TT + _RayT ) ;
 *tx = UU[0] * CC ;
 *ty = UU[1] * CC ;
 *tz = ( (_RayT - TT) * CC ) - _RayT ;
 return status ;
}
/*----------------------------------------------------------------------------*/
 int orterrain_to_carte ( void* *phot,
               double *tx, double *ty, double *tz,
                double *cx, double *cy, double *cz )
/* D'apres les formules des routines de P.Julien (SUBPHOT) */
{
 or_orientation		*ori ;
 double 		origine[2] ;
 double			RZ, RH ;
 int			status ;

 status = 1 ;
 ori = (or_orientation *) *phot ;

 origine[0] = (*ori).origine[0] ;
 origine[1] = (*ori).origine[1] ;

 RZ = _RayT + *tz ;
 RH = sqrt ( (RZ * RZ) + (*tx * *tx) + (*ty * *ty) ) ;
 *cx = origine[0] + ( *tx * 2.0 * _RayT ) / ( RZ + RH ) ;
 *cy = origine[1] + ( *ty * 2.0 * _RayT ) / ( RZ + RH ) ;
 *cz = RH - _RayT ;

 return status ;
}
/*----------------------------------------------------------------------------*/
 int orlit_fic_orientation ( char *fic, or_orientation *ori )
{
 FILE 			*fp ;
 char			*fori ;
 or_grille		*gr ;
 int 			gtaille ;
 int			status ;
 int			nlu ;

 status = 0 ;
 fp = ElFopen ( fic, "rb" ) ;
 if ( fp != 0 )
     {
     fori = (char *) ori ;
     nlu = (int) fread ( fori, sizeof(or_file_orientation), 1, fp ) ;
     status = _egaux( nlu , 1 ) ;
     nlu = (int) fread ( (char*) &gtaille, sizeof(int), 1, fp ) ;
     status = status & _egaux( nlu , 1 ) ;

     if ( status == 1 )
    {
    /* controle de taille */
    status = ( gtaille <= (_NS_GRILLE*_NS_GRILLE) ) ;
    if ( status != 0 )
         {
         status = 1 ;
         gr = &((*ori).gt2p) ;
             nlu = (int) fread ( (char*) &((*gr).ns), sizeof(int), 1, fp ) ;
             status = _egaux( nlu , 1 ) ;
             nlu = (int) fread ( (char*) &((*gr).nl), sizeof(int), 1, fp ) ;
             status = status & _egaux( nlu , 1 ) ;
         status = status & _superieur( gtaille , ((*gr).ns*(*gr).nl) ) ;
             nlu = (int) fread ( (char*) &((*gr).pas), sizeof(int), 1, fp ) ;
             status = status & _egaux( nlu , 1 ) ;

            nlu = (int) fread ( (char*) (*gr).dx, sizeof(double), gtaille, fp ) ;
             status = status & _egaux( nlu , gtaille ) ;
            nlu = (int) fread ( (char*) (*gr).dy, sizeof(double), gtaille, fp ) ;
            status = status & _egaux( nlu , gtaille ) ;

        gr = &((*ori).gp2t) ;
            nlu = (int) fread ( (char*) &((*gr).ns), sizeof(int), 1, fp ) ;
            status = status & _egaux( nlu , 1 ) ;
            nlu = (int) fread ( (char*) &((*gr).nl), sizeof(int), 1, fp ) ;
            status = status & _egaux( nlu , 1 ) ;
        status = status & _superieur( gtaille , ((*gr).ns*(*gr).nl) ) ;
            nlu = (int) fread ( (char*) &((*gr).pas), sizeof(int), 1, fp ) ;
            status = status & _egaux( nlu , 1 ) ;

            nlu = (int) fread ( (char*) (*gr).dx, sizeof(double), gtaille, fp ) ;
            status = status & _egaux( nlu , gtaille ) ;
            nlu = (int) fread ( (char*) (*gr).dy, sizeof(double), gtaille, fp ) ;
            status = status & _egaux( nlu , gtaille ) ;
        }
    }
    }
 ElFclose ( fp ) ;
 return status ;
}

int NEW_orlit_fic_orientation (const char *fic, or_orientation *ori )
{
    ELISE_fp    fp(fic,ELISE_fp::READ);
    or_grille            *gr ;
    int                   gtaille ;

    fp.read(ori,sizeof(or_file_orientation),1);
    fp.read(&gtaille,sizeof(int),1);

    /* controle de taille */
    Tjs_El_User.ElAssert
    (
        gtaille <= (_NS_GRILLE*_NS_GRILLE),
        EEM0 << "Inconsistent Binary orientation file"
    );
    gr = &((*ori).gt2p) ;
    fp.read(&((*gr).ns),sizeof(int), 1);
    fp.read(&((*gr).nl),sizeof(int), 1);

    Tjs_El_User.ElAssert
    (
        gtaille >=  ((*gr).ns*(*gr).nl),
        EEM0 << "Inconsistent Binary orientation file"
    );
    fp.read(&((*gr).pas),sizeof(int),1) ;

    fp.read((*gr).dx,sizeof(double),gtaille);
    fp.read((*gr).dy,sizeof(double),gtaille);

    gr = &((*ori).gp2t) ;
    fp.read(&((*gr).ns),sizeof(int),1);
    fp.read(&((*gr).nl),sizeof(int),1);
    Tjs_El_User.ElAssert
    (
        gtaille >= ((*gr).ns*(*gr).nl),
        EEM0 << "Inconsistent Binary orientation file"
    );
    fp.read(&((*gr).pas),sizeof(int),1) ;

    fp.read((*gr).dx,sizeof(double),gtaille);
    fp.read((*gr).dy,sizeof(double),gtaille);
    fp.close();

     make_std_inversion_grille(ori);

    return 1;
}
/*----------------------------------------------------------------------------*/

void InitGrilleFromCam(or_orientation * ori,or_grille * aGr,const ElCamera & aCam,bool M2C)
{
   INT aNb = _NS_GRILLE-1;
   aGr->ns = aGr->nl = aNb+1;
   aGr->pas = ElMax(ori->ins,ori->inl) / aNb;

   for (INT anX=0 ; anX<=aNb ; anX++)
   {
       for (INT anY=0 ; anY<=aNb  ; anY++)
       {
            Pt2dr aP  = Pt2dr(anX,anY) * aGr->pas;
        Pt2dr aQ = M2C ?  aCam.DistDirecte(aP): aCam.DistInverse(aP) ;
            aP =aQ-aP;
            INT ind = anX + anY * aGr->ns;
            aGr->dx[ind] = aP.x;
            aGr->dy[ind] = aP.y;
       }
   }
}




void InitGrilleFromPol(or_orientation * ori,or_grille * aGr,ElDistRadiale_PolynImpair & aPol)
{
   INT aNb = _NS_GRILLE-1;
   aGr->ns = aGr->nl = aNb+1;
   aGr->pas = ElMax(ori->ins,ori->inl) / aNb;

   for (INT anX=0 ; anX<=aNb ; anX++)
   {
       for (INT anY=0 ; anY<=aNb  ; anY++)
       {
            Pt2dr aP  = Pt2dr(anX,anY) * aGr->pas;
            aP =aPol.Direct(aP)-aP;
            INT ind = anX + anY * aGr->ns;
            aGr->dx[ind] = aP.x;
            aGr->dy[ind] = aP.y;
       }
   }
}

void or_orientation::InitNewParam()
{
   mDontUseDist = false;
   mOC = 0;
   mCorrDistM2C = 0;
}

int orlit_fictexte_orientation (const char *fic, or_orientation *ori,bool QuikcGrid )
{
   ori->mName = std::string(fic);

   if (StdPostfix(fic) == "xml")
      return XML_orlit_fictexte_orientation(fic,ori,QuikcGrid);

 FILE 			*fp ;
 or_grille		*gr ;
 int 			gtaille ;
 int			ii ;
 //int			buf[3] ;
 int			status ;

 status = 0 ;
 fp = ElFopen ( fic, "r" ) ;
 if (fp==0)
     cout << "FILE =" << fic << "\n";
 ELISE_ASSERT(fp!=0,"Cannot Open file in  orlit_fictexte_orientation");
 if ( fp != 0 )
     {

         char a[200];
         VoidFscanf ( fp, "%s", a);
         if (std::string("TEXT")!=a)
         {
std::cout << "??????????????????????? \n";
              fseek(fp,0,SEEK_SET);
         }
         // ELISE_ASSERT(std::string("TEXTE")==a,"Unexpected header in orlit_fictexte_orientation ");


#ifdef __16BITS__
 INTByte8               Lbuf[3] ;
         ori->InitNewParam();
     VoidFscanf ( fp, "%ld", &((*ori).distor) ) ;

     VoidFscanf ( fp, "%ld", &((*ori).refraction) ) ;

     VoidFscanf ( fp, "%d %d %d %d %d %d %d %d",
            &((*ori).chambre[0]), &((*ori).chambre[1]), &((*ori).chambre[2]),
            &((*ori).chambre[3]), &((*ori).chambre[4]), &((*ori).chambre[5]),
            &((*ori).chambre[6]), &((*ori).chambre[7])  ) ;

     VoidFscanf ( fp, "%ld %ld %ld %ld %ld %ld", &((*ori).jour), &((*ori).mois),
          &((*ori).annee),
          &((*ori).heure), &((*ori).minute), &((*ori).seconde) ) ;

     VoidFscanf ( fp,"%lld", &(Lbuf[0])  ) ;
     (*ori).altisol = ( (double) Lbuf[0] ) / 1000.0 ;
     (*ori).mProf =-1;


     VoidFscanf ( fp,"%lld %lld", &(Lbuf[0]), &(Lbuf[1]) ) ;
    (*ori).origine[0] = ((double) Lbuf[0]) / 1000.0 ;
    (*ori).origine[1] = ((double) Lbuf[1]) / 1000.0 ;

    VoidFscanf ( fp,"%lld", &((*ori).lambert) ) ;

    VoidFscanf ( fp,"%lld %lld %lld", &(Lbuf[0]), &(Lbuf[1]), &(Lbuf[2]) ) ;
    (*ori).sommet[0] = ((double) Lbuf[0]) / 1000.0 ;
     (*ori).sommet[1] = ((double) Lbuf[1]) / 1000.0 ;
     (*ori).sommet[2] = ((double) Lbuf[2]) / 1000.0 ;

     VoidFscanf ( fp, "%lld", &(Lbuf[0]) ) ;
    (*ori).focale = ((double) Lbuf[0]) / 1000.0 ;

    VoidFscanf ( fp, "%lld %lld %lld", &(Lbuf[0]), &(Lbuf[1]), &(Lbuf[2]) ) ;
    (*ori).vi[0] = ((double) Lbuf[0]) / 1000000000.0 ;
    (*ori).vi[1] = ((double) Lbuf[1]) / 1000000000.0 ;
    (*ori).vi[2] = ((double) Lbuf[2]) / 1000000000.0 ;


    VoidFscanf ( fp, "%lld %lld %lld", &(Lbuf[0]), &(Lbuf[1]), &(Lbuf[2]) ) ;
    (*ori).vj[0] = ((double) Lbuf[0]) / 1000000000.0 ;
    (*ori).vj[1] = ((double) Lbuf[1]) / 1000000000.0 ;
     (*ori).vj[2] = ((double) Lbuf[2]) / 1000000000.0 ;

    VoidFscanf ( fp, "%lld %lld %lld", &(Lbuf[0]), &(Lbuf[1]), &(Lbuf[2]) ) ;
    (*ori).vk[0] = ((double) Lbuf[0]) / 1000000000.0 ;
    (*ori).vk[1] = ((double) Lbuf[1]) / 1000000000.0 ;
     (*ori).vk[2] = ((double) Lbuf[2]) / 1000000000.0 ;

    VoidFscanf ( fp, "%lld %lld", &(Lbuf[0]), &(Lbuf[1]) ) ;
    (*ori).pix[0] = ((double) Lbuf[0]) / 1000000.0 ;
    (*ori).pix[1] = ((double) Lbuf[1]) / 1000000.0 ;
    (*ori).focale = (*ori).focale * (*ori).pix[0] ;

    VoidFscanf ( fp, "%ld %ld", &((*ori).ins), &((*ori).inl) ) ;

     VoidFscanf ( fp, "%lld %lld", &(Lbuf[0]), &(Lbuf[1]) ) ;
    (*ori).ipp[0] = ((double) Lbuf[0]) / 1000.0 ;
     (*ori).ipp[1] = ((double) Lbuf[1]) / 1000.0 ;

     VoidFscanf ( fp, "%ld", &gtaille ) ;

     status = _superieur( (_NS_GRILLE*_NS_GRILLE) , gtaille ) ;
     if ( status == 1 )
    {
    gr = &((*ori).gt2p) ;
    VoidFscanf ( fp, "%ld %ld %ld", &((*gr).ns), &((*gr).nl), &((*gr).pas) ) ;
    status = status & _superieur( gtaille , ((*gr).ns*(*gr).nl) ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
         {
             VoidFscanf ( fp, "%lld %lld", &(Lbuf[0]), &(Lbuf[1]) ) ;
             (*gr).dx[ii] = ((double) Lbuf[0]) / 1000000.0 ;
             (*gr).dy[ii] = ((double) Lbuf[1]) / 1000000.0 ;
         }

    gr = &((*ori).gp2t) ;
    VoidFscanf ( fp, "%ld %ld %ld", &((*gr).ns), &((*gr).nl), &((*gr).pas) ) ;
    status = status & _superieur( gtaille , ((*gr).ns*(*gr).nl) ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
         {
             VoidFscanf ( fp, "%lld %lld", &(Lbuf[0]), &(Lbuf[1]) ) ;
             (*gr).dx[ii] = ((double) Lbuf[0]) / 1000000.0 ;
             (*gr).dy[ii] = ((double) Lbuf[1]) / 1000000.0 ;
         }
    }
#else
 double                 DLbuf[3] ;
         ori->InitNewParam();
     VoidFscanf ( fp, "%d", &((*ori).distor) ) ;

     VoidFscanf ( fp, "%d", &((*ori).refraction) ) ;

{
     /* MPD-MODIF */
         INT c[8],k;
     VoidFscanf ( fp, "%d %d %d %d %d %d %d %d",c,c+1,c+2,c+3,c+4,c+5,c+6,c+7);
         for (k=0; k<8 ; k++) ori->chambre[k] = c[k];

}

     VoidFscanf ( fp, "%d %d %d %d %d %d", &((*ori).jour), &((*ori).mois),
          &((*ori).annee),
          &((*ori).heure), &((*ori).minute), &((*ori).seconde) ) ;

     VoidFscanf ( fp,"%lf", &(DLbuf[0])  ) ;
     (*ori).altisol = ( (double) DLbuf[0] ) / 1000.0 ;
     (*ori).mProf =-1;


     VoidFscanf ( fp,"%lf  %lf ", &(DLbuf[0]), &(DLbuf[1]) ) ;
    (*ori).origine[0] = ((double) DLbuf[0]) / 1000.0 ;
    (*ori).origine[1] = ((double) DLbuf[1]) / 1000.0 ;

    VoidFscanf ( fp,"%d", &((*ori).lambert) ) ;

    VoidFscanf ( fp,"%lf %lf %lf", &(DLbuf[0]), &(DLbuf[1]), &(DLbuf[2]) ) ;
    (*ori).sommet[0] = ((double) DLbuf[0]) / 1000.0 +  (*ori).origine[0] ;
    (*ori).sommet[1] = ((double) DLbuf[1]) / 1000.0 + (*ori).origine[1];
     (*ori).sommet[2] = ((double) DLbuf[2]) / 1000.0 ;

     VoidFscanf ( fp, "%lf", &(DLbuf[0]) ) ;
    (*ori).focale = ((double) DLbuf[0]) / 1000.0 ;

    VoidFscanf ( fp, "%lf %lf %lf", &(DLbuf[0]), &(DLbuf[1]), &(DLbuf[2]) ) ;
    (*ori).vi[0] = ((double) DLbuf[0]) / 1000000000.0 ;
    (*ori).vi[1] = ((double) DLbuf[1]) / 1000000000.0 ;
    (*ori).vi[2] = ((double) DLbuf[2]) / 1000000000.0 ;

    VoidFscanf ( fp, "%lf %lf %lf", &(DLbuf[0]), &(DLbuf[1]), &(DLbuf[2]) ) ;
    (*ori).vj[0] = ((double) DLbuf[0]) / 1000000000.0 ;
    (*ori).vj[1] = ((double) DLbuf[1]) / 1000000000.0 ;
     (*ori).vj[2] = ((double) DLbuf[2]) / 1000000000.0 ;

    VoidFscanf ( fp, "%lf %lf %lf", &(DLbuf[0]), &(DLbuf[1]), &(DLbuf[2]) ) ;
    (*ori).vk[0] = ((double) DLbuf[0]) / 1000000000.0 ;
    (*ori).vk[1] = ((double) DLbuf[1]) / 1000000000.0 ;
     (*ori).vk[2] = ((double) DLbuf[2]) / 1000000000.0 ;


    VoidFscanf ( fp, "%lf %lf", &(DLbuf[0]), &(DLbuf[1]) ) ;
    (*ori).pix[0] = ((double) DLbuf[0]) / 1000000.0 ;
    (*ori).pix[1] = ((double) DLbuf[1]) / 1000000.0 ;

    ori->pix[0] = (ori->pix[0] == 0) ? 1.0 : ori->pix[0];
    ori->pix[1] = (ori->pix[1] == 0) ? 1.0 : ori->pix[1];

    (*ori).focale = (*ori).focale * (*ori).pix[0] ;



    VoidFscanf ( fp, "%d %d", &((*ori).ins), &((*ori).inl) ) ;

    VoidFscanf ( fp, "%lf %lf ", &(DLbuf[0]), &(DLbuf[1]) ) ;
     (*ori).ipp[0] = ((double) DLbuf[0]) / 1000.0 ;
     (*ori).ipp[1] = ((double) DLbuf[1]) / 1000.0 ;


    if ((ori->distor == 2) || (ori->distor == 0))
    {
        double  R[3];
        Pt2dr   aC;

        Pt2dr aPMil = Pt2dr((*ori).ins,(*ori).inl) / 2.0;
        if (ori->distor == 2)
        {
           VoidFscanf(fp,"%lf %lf %lf",&(R[0]),&(R[1]),&(R[2]));
           VoidFscanf(fp,"%lf %lf",&(aC.x),&(aC.y));
        }
        else
        {
            aC = aPMil;
            R[0] = R[1] = R[2] = 0;
        }

        REAL aRay = euclid(aPMil);

        ElDistRadiale_PolynImpair aDistM2C(aRay,aC);

        aDistM2C.PushCoeff(R[0]);
        aDistM2C.PushCoeff(R[1]);
        aDistM2C.PushCoeff(R[2]);

       ElDistRadiale_PolynImpair aDistC2M = aDistM2C.DistRadialeInverse(aRay*1.05,2);

       InitGrilleFromPol(ori,&((*ori).gt2p),aDistM2C);
       InitGrilleFromPol(ori,&((*ori).gp2t),aDistC2M);

       status = 1;
       ori->distor = 1;
    }
    else if (ori->distor ==1)
    {

     VoidFscanf ( fp, "%d", &gtaille ) ;

     status = _superieur( (_NS_GRILLE*_NS_GRILLE) , gtaille ) ;
     if ( status == 1 )
    {
    gr = &((*ori).gt2p) ;
    VoidFscanf ( fp, "%d %d %d", &((*gr).ns), &((*gr).nl), &((*gr).pas) ) ;
    status = status & _superieur( gtaille , ((*gr).ns*(*gr).nl) ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
         {
             VoidFscanf ( fp, "%lf %lf", &(DLbuf[0]), &(DLbuf[1]) ) ;
             (*gr).dx[ii] = ((double) DLbuf[0]) / 1000000.0 ;
             (*gr).dy[ii] = ((double) DLbuf[1]) / 1000000.0 ;
         }

    gr = &((*ori).gp2t) ;
    VoidFscanf ( fp, "%d %d %d", &((*gr).ns), &((*gr).nl), &((*gr).pas) ) ;
    status = status & _superieur( gtaille , ((*gr).ns*(*gr).nl) ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
         {
             VoidFscanf ( fp, "%lf %lf", &(DLbuf[0]), &(DLbuf[1]) ) ;
             (*gr).dx[ii] = ((double) DLbuf[0]) / 1000000.0 ;
             (*gr).dy[ii] = ((double) DLbuf[1]) / 1000000.0 ;
         }
    }
    }
#endif
 }
 make_std_inversion_grille(ori);
 ElFclose ( fp ) ;
 return status ;
}







void make_std_inversion_grille(or_orientation *ori)
{
    orinverse_grille(&((*ori).gt2p),&((*ori).gp2t));
}


/*----------------------------------------------------------------------------*/
int orecrit_fictexte_orientation (const  char *fic, or_orientation *ori )
{
  FILE 			*fp ;
  or_grille		*gr ;
  int 			gtaille ;
  int			ii ;
  // int    		buf[3] ;
  INTByte8              Lbuf[3] ;
  int			status ;

  status = 1 ;
  fp = ElFopen ( fic, "w" ) ;
  if (fp==0) Tjs_El_User.ElAssert(0,EEM0<< "can't open file " << fic << " in orecrit_fictexte_orientation") ;
  if ( fp != 0 )  {
#ifdef __16BITS__

         ori->InitNewParam();
    fprintf ( fp, "TEXT\n") ;
    fprintf ( fp, "%ld\n", (*ori).distor ) ;
    fprintf ( fp, "%ld\n", (*ori).refraction ) ;
    fprintf ( fp, "%d %d %d %d %d %d %d %d\n",
         (*ori).chambre[0], (*ori).chambre[1], (*ori).chambre[2],
         (*ori).chambre[3], (*ori).chambre[4], (*ori).chambre[5],
         (*ori).chambre[6], (*ori).chambre[7]  ) ;
    fprintf ( fp, "%ld %ld %ld %ld %ld %ld\n", (*ori).jour, (*ori).mois,
         (*ori).annee,
         (*ori).heure, (*ori).minute, (*ori).seconde ) ;

    Lbuf[0] =  (INTByte8) ( (*ori).altisol * 1000.0 + 0.5 ) ;
    fprintf ( fp,"%lld\n", Lbuf[0] ) ;

    Lbuf[0] = (INTByte8) ( (*ori).origine[0] * 1000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).origine[1] * 1000.0 + 0.5 ) ;
    fprintf ( fp,"%lld %lld\n", Lbuf[0], Lbuf[1] ) ;

    fprintf ( fp,"%ld\n", (*ori).lambert ) ;

    buf[0] = (INTByte8) ( (*ori).sommet[0] * 1000.0 + 0.5 ) ;
    buf[1] = (INTByte8) ( (*ori).sommet[1] * 1000.0 + 0.5 ) ;
    buf[2] = (INTByte8) ( (*ori).sommet[2] * 1000.0 + 0.5 ) ;
    fprintf ( fp,"%ld %ld %ld\n", buf[0], buf[1], buf[2] ) ;

    buf[0] =  (INTByte8) ( ((*ori).focale/(*ori).pix[0]) * 1000.0 + 0.5 ) ;
    fprintf ( fp, "%ld\n", buf[0] ) ;

    buf[0] = (INTByte8) ( (*ori).vi[0] * 1000000000.0 + 0.5 ) ;
    buf[1] = (INTByte8) ( (*ori).vi[1] * 1000000000.0 + 0.5 ) ;
    buf[2] = (INTByte8) ( (*ori).vi[2] * 1000000000.0 + 0.5 ) ;
    fprintf ( fp,"%ld %ld %ld\n", buf[0], buf[1], buf[2] ) ;

    buf[0] = (INTByte8) ( (*ori).vj[0] * 1000000000.0 + 0.5 ) ;
    buf[1] = (INTByte8) ( (*ori).vj[1] * 1000000000.0 + 0.5 ) ;
    buf[2] = (INTByte8) ( (*ori).vj[2] * 1000000000.0 + 0.5 ) ;
    fprintf ( fp,"%ld %ld %ld\n", buf[0], buf[1], buf[2] ) ;

    buf[0] = (INTByte8) ( (*ori).vk[0] * 1000000000.0 + 0.5 ) ;
    buf[1] = (INTByte8) ( (*ori).vk[1] * 1000000000.0 + 0.5 ) ;
    buf[2] = (INTByte8) ( (*ori).vk[2] * 1000000000.0 + 0.5 ) ;
    fprintf ( fp,"%ld %ld %ld\n", buf[0], buf[1], buf[2] ) ;


    buf[0] = (INTByte8) ( (*ori).pix[0] * 1000000.0 + 0.5 ) ;
    buf[1] = (INTByte8) ( (*ori).pix[1] * 1000000.0 + 0.5 ) ;
    fprintf ( fp,"%ld %ld\n", buf[0], buf[1] ) ;

    fprintf ( fp, "%ld %ld\n", (*ori).ins, (*ori).inl ) ;

    buf[0] = (INTByte8) ( (*ori).ipp[0] * 1000.0 + 0.5 ) ;
    buf[1] = (INTByte8) ( (*ori).ipp[1] * 1000.0 + 0.5 ) ;
    fprintf ( fp,"%ld %ld\n", buf[0], buf[1] ) ;

    gr = &((*ori).gt2p) ;
    gtaille = (*gr).ns * (*gr).nl ;
    gr = &((*ori).gp2t) ;
    gtaille = _max ( gtaille , (*gr).ns*(*gr).nl ) ;
    fprintf ( fp, "%ld\n", gtaille ) ;

    gr = &((*ori).gt2p) ;
    fprintf ( fp, "%ld %ld %ld\n", (*gr).ns, (*gr).nl, (*gr).pas ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
    {
      buf[0] = (INTByte8) ((*gr).dx[ii]*1000000.0 + 0.5) ;
      buf[1] = (INTByte8) ((*gr).dy[ii]*1000000.0 + 0.5) ;
      fprintf ( fp, "%ld %ld\n", buf[0], buf[1] ) ;
    }

    gr = &((*ori).gp2t) ;
    fprintf ( fp, "%ld %ld %ld\n", (*gr).ns, (*gr).nl, (*gr).pas ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
    {
      buf[0] = (INTByte8) ((*gr).dx[ii]*1000000.0 + 0.5) ;
      buf[1] = (INTByte8) ((*gr).dy[ii]*1000000.0 + 0.5) ;
      fprintf ( fp, "%ld %ld\n", buf[0], buf[1] ) ;
    }
#else
         ori->InitNewParam();
    fprintf ( fp, "TEXT\n") ;
    fprintf ( fp, "%d\n", (*ori).distor ) ;
    fprintf ( fp, "%d\n", (*ori).refraction ) ;
    fprintf ( fp, "%d %d %d %d %d %d %d %d\n",
         (*ori).chambre[0], (*ori).chambre[1], (*ori).chambre[2],
         (*ori).chambre[3], (*ori).chambre[4], (*ori).chambre[5],
         (*ori).chambre[6], (*ori).chambre[7]  ) ;
    fprintf ( fp, " %d %d %d %d %d %d\n", (*ori).jour, (*ori).mois,
         (*ori).annee,
         (*ori).heure, (*ori).minute, (*ori).seconde ) ;

    Lbuf[0] =  (INTByte8) ( (*ori).altisol * 1000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d\n", Lbuf[0] ) ;
    #else
        fprintf ( fp,"%lld\n", Lbuf[0] ) ;
    #endif

    Lbuf[0] = (INTByte8) ( (*ori).origine[0] * 1000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).origine[1] * 1000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d\n", Lbuf[0], Lbuf[1] ) ;
    #else
        fprintf ( fp,"%lld %lld\n", Lbuf[0], Lbuf[1] ) ;
    #endif

    fprintf ( fp,"%d\n", (*ori).lambert ) ;

    Lbuf[0] = (INTByte8) ( (*ori).sommet[0] * 1000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).sommet[1] * 1000.0 + 0.5 ) ;
    Lbuf[2] = (INTByte8) ( (*ori).sommet[2] * 1000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d %I64d\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #else
        fprintf ( fp,"%lld %lld %lld\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #endif

    Lbuf[0] =  (INTByte8) ( ((*ori).focale/(*ori).pix[0]) * 1000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp, "%I64d\n", Lbuf[0] ) ;
    #else
        fprintf ( fp, "%lld\n", Lbuf[0] ) ;
    #endif

    Lbuf[0] = (INTByte8) ( (*ori).vi[0] * 1000000000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).vi[1] * 1000000000.0 + 0.5 ) ;
    Lbuf[2] = (INTByte8) ( (*ori).vi[2] * 1000000000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d %I64d\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #else
        fprintf ( fp,"%lld %lld %lld\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #endif

    Lbuf[0] = (INTByte8) ( (*ori).vj[0] * 1000000000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).vj[1] * 1000000000.0 + 0.5 ) ;
    Lbuf[2] = (INTByte8) ( (*ori).vj[2] * 1000000000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d %I64d\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #else
        fprintf ( fp,"%lld %lld %lld\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #endif

    Lbuf[0] = (INTByte8) ( (*ori).vk[0] * 1000000000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).vk[1] * 1000000000.0 + 0.5 ) ;
    Lbuf[2] = (INTByte8) ( (*ori).vk[2] * 1000000000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d %I64d\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #else
        fprintf ( fp,"%lld %lld %lld\n", Lbuf[0], Lbuf[1], Lbuf[2] ) ;
    #endif


    Lbuf[0] = (INTByte8) ( (*ori).pix[0] * 1000000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).pix[1] * 1000000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d\n", Lbuf[0], Lbuf[1] ) ;
    #else
        fprintf ( fp,"%lld %lld\n", Lbuf[0], Lbuf[1] ) ;
    #endif

    fprintf ( fp, "%d %d\n", (*ori).ins, (*ori).inl ) ;

    Lbuf[0] = (INTByte8) ( (*ori).ipp[0] * 1000.0 + 0.5 ) ;
    Lbuf[1] = (INTByte8) ( (*ori).ipp[1] * 1000.0 + 0.5 ) ;
    #if (ELISE_MinGW)
        fprintf ( fp,"%I64d %I64d\n", Lbuf[0], Lbuf[1] ) ;
    #else
        fprintf ( fp,"%lld %lld\n", Lbuf[0], Lbuf[1] ) ;
    #endif

    gr = &((*ori).gt2p) ;
    gtaille = (*gr).ns * (*gr).nl ;
    gr = &((*ori).gp2t) ;
    gtaille = _max ( gtaille , (*gr).ns*(*gr).nl ) ;
    fprintf ( fp, "%d\n", gtaille ) ;

    gr = &((*ori).gt2p) ;
    fprintf ( fp, "%d %d %d\n", (*gr).ns, (*gr).nl, (*gr).pas ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
    {
      Lbuf[0] = (INTByte8) ((*gr).dx[ii]*1000000.0 + 0.5) ;
      Lbuf[1] = (INTByte8) ((*gr).dy[ii]*1000000.0 + 0.5) ;
      #if (ELISE_MinGW)
        fprintf ( fp, "%I64d %I64d\n", Lbuf[0], Lbuf[1] ) ;
      #else
        fprintf ( fp, "%lld %lld\n", Lbuf[0], Lbuf[1] ) ;
      #endif
    }

    gr = &((*ori).gp2t) ;
    fprintf ( fp, "%d %d %d\n", (*gr).ns, (*gr).nl, (*gr).pas ) ;
    for ( ii = 0 ; ii < gtaille ; ii++ )
    {
      Lbuf[0] = (INTByte8) ((*gr).dx[ii]*1000000.0 + 0.5) ;
      Lbuf[1] = (INTByte8) ((*gr).dy[ii]*1000000.0 + 0.5) ;
      #if (ELISE_MinGW)
        fprintf ( fp, "%I64d %I64d\n", Lbuf[0], Lbuf[1] ) ;
      #else
        fprintf ( fp, "%lld %lld\n", Lbuf[0], Lbuf[1] ) ;
      #endif
    }
#endif

  } else {
          Tjs_El_User.ElAssert(0,EEM0<< "can't open file " << fic << " in orecrit_fictexte_orientation") ;
  }
  ElFclose ( fp ) ;
  return status ;
}
/*----------------------------------------------------------------------------*/
 int orecrit_fic_orientation (const char *fic, or_orientation *ori )
{
 FILE 			*fp ;
 or_grille		*gr ;
 char			*fori ;
 int			gtaille ;
 int			necr ;
 int			status ;

 status = 0 ;
 fp = ElFopen ( fic, "wb" ) ;
 if ( fp != 0 )
    {
    fori = (char *) ori ;
    necr = (int) fwrite ( fori, sizeof(or_file_orientation), 1, fp ) ;
    status = _egaux( necr , 1 ) ;

    gr = &((*ori).gt2p) ;
    gtaille = (*gr).ns * (*gr).nl ;
    gr = &((*ori).gp2t) ;
    gtaille = _max ( gtaille , (*gr).ns*(*gr).nl ) ;

    necr = (int) fwrite ( (const char*) &gtaille, sizeof(int), 1, fp ) ;
    status = status & _egaux( necr , 1 ) ;

     if ( status == 1 )
    {
        gr = &((*ori).gt2p) ;
            necr = (int) fwrite ( (const char*) &((*gr).ns), sizeof(int), 1, fp ) ;
            status = _egaux( necr , 1 ) ;
             necr = (int) fwrite ( (const char*) &((*gr).nl), sizeof(int), 1, fp ) ;
            status = status & _egaux( necr , 1 ) ;
            necr = (int) fwrite ( (const char*) &((*gr).pas), sizeof(int), 1, fp ) ;
            status = status & _egaux( necr , 1 ) ;

            necr = (int) fwrite ( (const char*) (*gr).dx, sizeof(double),
                gtaille, fp ) ;
            status = status & _egaux( necr , gtaille ) ;
            necr = (int) fwrite ( (const char*) (*gr).dy, sizeof(double),
                gtaille, fp ) ;
            status = status & _egaux( necr , gtaille ) ;

        gr = &((*ori).gp2t) ;
            necr = (int) fwrite ( (const char*) &((*gr).ns), sizeof(int), 1, fp ) ;
            status = status & _egaux( necr , 1 ) ;
            necr = (int) fwrite ( (const char*) &((*gr).nl), sizeof(int), 1, fp ) ;
            status = status & _egaux( necr , 1 ) ;
            necr = (int) fwrite ( (const char*) &((*gr)).pas, sizeof(int), 1, fp ) ;
            status = status & _egaux( necr , 1 ) ;

            necr = (int) fwrite ( (const char*) (*gr).dx, sizeof(double),
                gtaille, fp ) ;
             status = status & _egaux( necr , gtaille ) ;
            necr = (int) fwrite ( (const char*) (*gr).dy, sizeof(double),
                gtaille, fp ) ;
            status = status & _egaux( necr , gtaille ) ;
    }
    }
 ElFclose ( fp ) ;
 return status ;
}
/*----------------------------------------------------------------------------*/

void orSetAltiSol ( void* *phot, double *AltiSol)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 (*ori).altisol  = *AltiSol;
}

void orSetSommet ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 (*ori).sommet[0] = *x;
 (*ori).sommet[1] = *y;
 (*ori).sommet[2] = *z;
}


void orDirI ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 *x = (*ori).vi[0] ;
 *y = (*ori).vi[1] ;
 *z = (*ori).vi[2] ;
}
void orSetDirI ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 (*ori).vi[0] = *x;
 (*ori).vi[1] = *y;
 (*ori).vi[2] = *z;
}




void orDirJ ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 *x = (*ori).vj[0] ;
 *y = (*ori).vj[1] ;
 *z = (*ori).vj[2] ;
}
void orSetDirJ ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 (*ori).vj[0] = *x;
 (*ori).vj[1] = *y;
 (*ori).vj[2] = *z;
}


void orDirK ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 *x = (*ori).vk[0] ;
 *y = (*ori).vk[1] ;
 *z = (*ori).vk[2] ;
}
void orSetDirK ( void* *phot, double *x, double *y,double * z)
{
 or_orientation		*ori ;
 ori = (or_orientation *) *phot ;
 (*ori).vk[0] = *x;
 (*ori).vk[1] = *y;
 (*ori).vk[2] = *z;
}



 void orSM ( void* *phot, double *colonne, double *ligne, double SM[3] )
{
 double			col, lig ;
 double			aa, bb;
 or_orientation		*ori ;
 int			status  ; GccUse(status);

 /* Coordonnees des points image dans le repere terrain */
 ori = (or_orientation *) *phot ;


 col = *colonne ;
 lig = *ligne ;
 if ( (*ori).distor != 0 )
    {
    //status = orcorrige_distortion( &col, &lig, &((*ori).gp2t) ) ;
    status  = ori->CorrigeDist_P2T(&col, &lig);
    }
 aa = ( col - (*ori).ipp[0] ) * (*ori).pix[0] ;
 bb = ( lig - (*ori).ipp[1] ) * (*ori).pix[1] ;

 SM[0] = (*ori).focale * (*ori).vk[0] +
     aa * (*ori).vi[0] + bb * (*ori).vj[0] ;
 SM[1] = (*ori).focale * (*ori).vk[1] +
     aa * (*ori).vi[1] + bb * (*ori).vj[1] ;
 SM[2] = (*ori).focale * (*ori).vk[2] +
     aa * (*ori).vi[2] + bb * (*ori).vj[2] ;
}

 void orPhoto_to_DirLoc ( void* *phot, double *colonne, double *ligne, double SM[3] )
{
 double			col, lig ;
 double			aa, bb;
 or_orientation		*ori ;
 int			status  ; GccUse(status);

 /* Coordonnees des points image dans le repere terrain */
 ori = (or_orientation *) *phot ;


 col = *colonne ;
 lig = *ligne ;
 if ( (*ori).distor != 0 )
    {
    //status = orcorrige_distortion( &col, &lig, &((*ori).gp2t) ) ;
    status = ori->CorrigeDist_P2T(&col, &lig);
    }
 aa = ( col - (*ori).ipp[0] ) * (*ori).pix[0] ;
 bb = ( lig - (*ori).ipp[1] ) * (*ori).pix[1] ;

 SM[0] = aa   ;
 SM[1] =  bb  ;
 SM[2] = (*ori).focale  ;
}




/*----------------------------------------------------------------------------*/
 void orinters_SM_photo ( void* *phot, double SM[3],
               double *colonne, double *ligne )
{
or_orientation		*ori ;
double			factx, facty ;
int			status ; GccUse(status);

ori = (or_orientation *) *phot ;

 /* coordonnees en metres du vecteur point principal -> inters rayon dans
    le plan image */

facty = (*ori).focale / _pscalaire ( SM , (*ori).vk ) ;

//  Si facty <0 on est dans un cas de singularite ou le rayon est passe par
//  l'infini, a priori il faut faire facty = beaucoup,  a voir

if (facty<=0)
{
   *colonne = -1e6;
   return;
/*
    std::cout << "Name " << ori->mName << "\n";
ELISE_ASSERT
(
    facty >0,
    "orinters_SM_photo : singularite a traiter"
);
*/
}

factx = facty / (*ori).pix[0] ;
facty = facty / (*ori).pix[1] ;

*colonne = (*ori).ipp[0] + factx * _pscalaire ( SM , (*ori).vi ) ;
*ligne   = (*ori).ipp[1] + facty * _pscalaire ( SM , (*ori).vj ) ;

    /* prise en compte des distortions */
if ( (*ori).distor != 0 )
    {
    // status = orcorrige_distortion( colonne, ligne, &((*ori).gt2p) ) ;
    status = ori->CorrigeDist_T2P(colonne, ligne);
    }
}

void orDirLoc_to_photo( void* *phot, double SM[3],
            double *colonne, double *ligne )
{
or_orientation		*ori ;
double			factx, facty ;
int			status ; GccUse(status);

ori = (or_orientation *) *phot ;

 /* coordonnees en metres du vecteur point principal -> inters rayon dans
    le plan image */
facty = (*ori).focale /SM[2];
factx = facty / (*ori).pix[0] ;
facty = facty / (*ori).pix[1] ;

*colonne = (*ori).ipp[0] + factx * SM[0];
*ligne   = (*ori).ipp[1] + facty * SM[1];


    /* prise en compte des distortions */
if ( (*ori).distor != 0 )
    {
    // status = orcorrige_distortion( colonne, ligne, &((*ori).gt2p) ) ;
    status = ori->CorrigeDist_T2P(colonne, ligne);
    }
}





/*----------------------------------------------------------------------------*/
 int orphoto_et_z_to_terrain ( void* *phot,
                 const double *col0, const double *lig0, const double *zz,
                 double *xterre, double *yterre )
{
or_orientation		*ori ;
double 			SM[3] ;
double 			lambda ;

if ( *phot == 0 ) { return 0 ; }
ori = (or_orientation *) *phot ;

orSM ( phot, const_cast<double *>(col0),  const_cast<double *>(lig0), SM ) ;
lambda = (*zz - (*ori).sommet[2]) / SM[2] ;
*xterre = (*ori).sommet[0] + lambda * SM[0] ;
*yterre = (*ori).sommet[1] + lambda * SM[1] ;
return 1 ;
}


/*----------------------------------------------------------------------------*/

static inline double Ori_Focale(void* *phot)
{
  return ((or_orientation *)(*phot))->focale;
}
static inline Pt3dr Ori_RayonPerspectifUnitaire(void* * ori,Pt2dr aP)
{
   double  SM[3];
   orSM(ori,&aP.x,&aP.y,SM);
   return Pt3dr(SM[0],SM[1],SM[2]) / Ori_Focale(ori);
}

static inline Pt3dr Ori_CPdv(void* *phot)
{

   or_orientation * ori = (or_orientation *) *phot ;
   return Pt3dr
          (
               ori->sommet[0],
               ori->sommet[1],
               ori->sommet[2]
          );
}
static inline Pt2dr PP(void* *phot)
{
   or_orientation * ori = (or_orientation *) *phot ;
   return Pt2dr((*ori).ipp[0],(*ori).ipp[1]);
}

int orphoto1_et_prof2_to_terrain ( void* *phot1,
                 const double *col1,
                                 const double *lig1,
                                 void* *phot2,
                                 const double *prof2,
                 double *xterre,
                                 double *yterre ,
                                 double *zterre
                               )
{
   Pt3dr aR1 = Ori_RayonPerspectifUnitaire(phot1,Pt2dr(*col1,*lig1));
   Pt3dr aC1 = Ori_CPdv(phot1);
   Pt3dr aR2 = Ori_RayonPerspectifUnitaire(phot2,PP(phot2));
   Pt3dr aC2 = Ori_CPdv(phot2);
   double aLambda = (*prof2 -scal(aC1-aC2,aR2))/scal(aR1,aR2);

   Pt3dr aRes = aC1 + aR1*aLambda;
   *xterre = aRes.x;
   *yterre = aRes.y;
   *zterre = aRes.z;
   return 1;
}

int orphoto_et_prof_to_terrain ( void* *phot,
                 const double *col,
                                 const double *lig,
                                 const double *prof,
                 double *xterre,
                                 double *yterre ,
                                 double *zterre
                               )
{

   Pt3dr aR = Ori_RayonPerspectifUnitaire(phot,Pt2dr(*col,*lig));
   Pt3dr aC = Ori_CPdv(phot);
   Pt3dr aRes = aC + aR* *prof;

   *xterre = aRes.x;
   *yterre = aRes.y;
   *zterre = aRes.z;
   return 1;
}

int  or_prof(  void* *phot,
                 const double *xterre,
                 const double *yterre ,
                 const double *zterre,
                 double       *prof
            )
{
   Pt3dr aC = Ori_CPdv(phot);
   Pt3dr aR = Ori_RayonPerspectifUnitaire(phot,PP(phot));
   Pt3dr aPT(*xterre,*yterre,*zterre);
   *prof = scal(aR,aPT-aC);
   return 1;
}





/*----------------------------------------------------------------------------*/
 int orphoto_et_zCarte_to_terrain ( void* *phot,
                 double *col0, double *lig0, double *zz,
                 double *xterre, double *yterre, double *zterre )
{
/*
Si Z est donnee en carto, on fait une resolution approchee
en considerant localement (au sommet de pdv) le repere carte
comme euclidien .
Cela revient a faire l'intersection avec un plan tangeant a la sphere
terrestre
*/
or_orientation		*ori ;
double 			SM[3] ;
double 			lambda ;
double 			xcarto, ycarto, zcarto ;
/* double			resol ;*/
double			xt, yt, zt ;

if ( *phot == 0 ) { return 0 ; }
ori = (or_orientation *) *phot ;

/* "vecteur" SM en cooord carte */
orSM ( phot, col0, lig0, SM ) ;
xt = (*ori).sommet[0] + SM[0] ;
yt = (*ori).sommet[1] + SM[1] ;
zt = (*ori).sommet[2] + SM[2] ;
orterrain_to_carte ( phot, &xt, &yt, &zt, &SM[0], &SM[1], &SM[2] ) ;
orsommet_de_pdv_carto ( phot, &xcarto, &ycarto, &zcarto ) ;
SM[0] = SM[0] - xcarto ;
SM[1] = SM[1] - ycarto ;
SM[2] = SM[2] - zcarto ;


/* zterrain approche correspondant */
lambda = (*zz - zcarto) / SM[2] ;
xcarto = xcarto + lambda * SM[0] ;
ycarto = ycarto + lambda * SM[1] ;
zcarto = zcarto + lambda * SM[2] ;

/* on renvoie les coord terrain */
orcarte_to_terrain ( phot, &xcarto, &ycarto, &zcarto,
                      xterre, yterre, zterre ) ;
return 1 ;
}
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
/*				GEOMETRIE EPIPOLAIRE			      */
/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/
 int orminmax_paral ( 	void* *epi1, void* *epi2,
                double *zmin, double *zmax,
                double *paralmin, double *paralmax )
{
/* renvoie les valeurs de paralaxes min et max en fonction de zmin et zmax ;
    valable uniquement pour les images epipolaires */

/* on cherche les quatre coins de epi1 et on regarde la paralaxe */
or_orientation 		*or1 ;
double			x0, y0, x1, y1 ;
double			MM[3] ;
double			cc, ll ;

int			istat ; GccUse(istat);

or1 = (or_orientation *) *epi1 ;
x0 = 0.0 ;
y0 = 0.0 ;
x1 = (double) ((*or1).ins-1) ;
y1 = (double) ((*or1).inl-1) ;

        /* paralmax sur zmin */
orphoto_et_zCarte_to_terrain ( epi1, &x0, &y0, zmin, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmax = cc-x0 ;

orphoto_et_zCarte_to_terrain ( epi1, &x1, &y0, zmin, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmax = _max( *paralmax , cc-x1 ) ;

orphoto_et_zCarte_to_terrain ( epi1, &x1, &y1, zmin, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmax = _max( *paralmax , cc-x1 ) ;

orphoto_et_zCarte_to_terrain ( epi1, &x0, &y1, zmin, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmax = _max( *paralmax , cc-x0 ) ;


        /* paralmin sur zmax */
orphoto_et_zCarte_to_terrain ( epi1, &x0, &y0, zmax, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmin = cc-x0 ;

orphoto_et_zCarte_to_terrain ( epi1, &x1, &y0, zmax, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmin = _min( *paralmin , cc-x1 ) ;

orphoto_et_zCarte_to_terrain ( epi1, &x1, &y1, zmax, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmin = _min( *paralmin , cc-x1 ) ;

orphoto_et_zCarte_to_terrain ( epi1, &x0, &y1, zmax, &MM[0], &MM[1], &MM[2] ) ;
istat = orterrain_to_photo ( epi2, &MM[0], &MM[1], &MM[2], &cc, &ll ) ;
*paralmin = _min( *paralmin , cc-x0 ) ;

return 1 ;
}
/*----------------------------------------------------------------------------*/
 int orepipolaire ( void* *epi1, void* *epi2 )
{
/* 1 si les deux photos sont epipolaires ; 0 sinon */
/*
   le critere de precision est la coplanarite entre la ligne 0 et S1S2 a
   ns pixels de distance de l'origine (taille moyenne des couples) ;
   puis entre la ligne nl et S1S2 ;

   ceci doit garantir la geometrie epipolaire et la meme resolution en y ;
   on verifie ensuite que les resolutions en x sont proportionnelles aux
   resolutions en y ;
*/
or_orientation		*ori1, *ori2 ;
double			SM1[3], SM2[3], ortho[3] ;
double			norme, erreur ;
double			x0,y0,x1,y1 ;

ori1 = (or_orientation *) *epi1 ;
ori2 = (or_orientation *) *epi2 ;

x0 = 0.0 ;
y0 = 0.0 ;
x1 = (double) ( _min ( (*ori1).ins , (*ori2).ins ) -1) ;
y1 = (double) ( _min ( (*ori1).inl , (*ori2).inl ) -1) ;

/* verification du plan de la premiere ligne */
orSM ( epi1, &x0, &y0, SM1 ) ;
orSM ( epi2, &x0, &y0, SM2 ) ;
ortho[0] = SM1[1]*SM2[2] - SM1[2]*SM2[1] ;
ortho[1] = SM1[2]*SM2[0] - SM1[0]*SM2[2] ;
ortho[2] = SM1[0]*SM2[1] - SM1[1]*SM2[0] ;
norme = _pscalaire ( ortho , ortho ) ;
if ( norme <= 0.0 )
    {
    x0 = 1.0 ;
    orSM ( epi2, &x0, &y0, SM2 ) ;
    ortho[0] = SM1[1]*SM2[2] - SM1[2]*SM2[1] ;
    ortho[1] = SM1[2]*SM2[0] - SM1[0]*SM2[2] ;
    ortho[2] = SM1[0]*SM2[1] - SM1[1]*SM2[0] ;
    norme = _pscalaire ( ortho , ortho ) ;
    x0 = 0.0 ;
    }
norme = sqrt(norme) ;
ortho[0] = ortho[0] / norme ;
ortho[1] = ortho[1] / norme ;
ortho[2] = ortho[2] / norme ;

orSM ( epi1, &x1, &y0, SM1 ) ;
erreur = _pscalaire ( SM1 , ortho ) ;
erreur = _abs ( erreur ) ;
if ( erreur > (*ori1).pix[1]/10.0 ) { return 0 ; }

orSM ( epi2, &x1, &y0, SM2 ) ;
erreur = _pscalaire ( SM2 , ortho ) ;
erreur = _abs ( erreur ) ;
if ( erreur > (*ori2).pix[1]/10.0 ) { return 0 ; }

/* verification du plan de la ligne nl */
orSM ( epi1, &x0, &y1, SM1 ) ;
orSM ( epi2, &x0, &y1, SM2 ) ;
ortho[0] = SM1[1]*SM2[2] - SM1[2]*SM2[1] ;
ortho[1] = SM1[2]*SM2[0] - SM1[0]*SM2[2] ;
ortho[2] = SM1[0]*SM2[1] - SM1[1]*SM2[0] ;
norme = _pscalaire ( ortho , ortho ) ;
norme = sqrt(norme) ;
ortho[0] = ortho[0] / norme ;
ortho[1] = ortho[1] / norme ;
ortho[2] = ortho[2] / norme ;

orSM ( epi1, &x1, &y1, SM1 ) ;
erreur = _pscalaire ( SM1 , ortho ) ;
erreur = _abs ( erreur ) ;
if ( erreur > (*ori1).pix[1]/10.0 ) { return 0 ; }

orSM ( epi2, &x1, &y1, SM2 ) ;
erreur = _pscalaire ( SM2 , ortho ) ;
erreur = _abs ( erreur ) ;
if ( erreur > (*ori2).pix[1]/10.0 ) { return 0 ; }

/* resolution en x : erreur inferieure au 1/10 pixel au bout de ns pixels */
erreur = (*ori1).pix[0] - (*ori2).pix[0]*(*ori1).pix[1]/(*ori2).pix[1] ;
erreur = _abs(erreur) ;
if ( x1 * erreur > (*ori1).pix[0]/10.0 ) { return 0 ; }

return 1 ;
}
/*----------------------------------------------------------------------------*/
 int orrepere_3D_image ( void* *phot, double coin[3],
                double uu[3], double vv[3] )
{
or_orientation 		*ph ;
double			SM[3] ;
double			x0, y0 ;

if ( *phot == 0 ) return 0 ;
ph = (or_orientation *) *phot ;

x0 = 0.0 ; y0 = 0.0 ;
orSM ( phot, &x0, &y0, SM ) ;

coin[0] = SM[0] + (*ph).sommet[0] ;
coin[1] = SM[1] + (*ph).sommet[1] ;
coin[2] = SM[2] + (*ph).sommet[2] ;

uu[0] = (*ph).pix[0] * (*ph).vi[0] ;
uu[1] = (*ph).pix[0] * (*ph).vi[1] ;
uu[2] = (*ph).pix[0] * (*ph).vi[2] ;

vv[0] = (*ph).pix[1] * (*ph).vj[0] ;
vv[1] = (*ph).pix[1] * (*ph).vj[1] ;
vv[2] = (*ph).pix[1] * (*ph).vj[2] ;

return 1 ;
}
/*----------------------------------------------------------------------------*/


int orprojette_image (
    void* *phot, unsigned char /*huge*/ *idata,
    int *ins, int *inl,
    void* *epipo,
        Orilib_Interp,
    unsigned char /*huge*/ *odata, int *ons, int *onl )
{
or_orientation 		*ph, *epi ;
int			istat ;
double			xx, yy, zz ;
double 			phcc, phll ;
int			ip ;
int			is, il ;

istat = 0 ;
ph = (or_orientation *) *phot ;
epi = (or_orientation *) *epipo ;

if ( (*ins > (*ph).ins) || (*inl > (*ph).inl) ||
     (*ons > (*epi).ins) || (*onl > (*epi).inl) ) return istat ;

/* re-echantillonnage */
zz = ( (*ph).altisol + (*epi).altisol ) / 2.0 ;
ip = 0 ;
for ( il = 0 ; il < *onl ; il++ )
    {
    for ( is = 0 ; is < *ons ; is++ )
    {
    phcc = (double) is ;
    phll = (double) il ;
    istat = orphoto_et_z_to_terrain ( epipo , &phcc, &phll, &zz,
                       &xx, &yy ) ;
    if ( istat == 1 ) istat = orterrain_to_photo ( phot, &xx, &yy, &zz,
                            &phcc, &phll ) ;
    if ( istat != 1 ) return istat ;
/*	odata[ip] = interpolation ( idata, ins, inl, &phcc, &phll ) ;*/
/*	if (il % 10 == 0 && is % 10 == 0) cout << phcc << " " << phll << endl ;*/
    if (ip<40000) odata[ip] = idata[ (int)(phcc) + (int)(phll) * (* ins)] ;
    ip++ ;
    }
    }
return istat ;
}
/*----------------------------------------------------------------------------*/
 int orprojette_epipolaire (
    void* *phot,
    unsigned char *idata,
    int *ins, int *inl,
    void* *epipo,
    unsigned char (*interpolation)( unsigned char*, int*, int*,
    double*, double* ),
    unsigned char *odata, int *ons, int *onl )
{
or_orientation 		*ph, *epi ;
int			istat ;
double			xyz0[3], xx0, yy0, zz0 ;
double			xx, yy, zz ;
double 			phcc, phll ;
int			ip ;
int			is, il ;
double			dlig[3], dcol[3] ;

istat = 0 ;
ph = (or_orientation *) *phot ;
epi = (or_orientation *) *epipo ;

if ( (*ins > (*ph).ins) || (*inl > (*ph).inl) ||
     (*ons > (*epi).ins) || (*onl > (*epi).inl) ) return istat ;

/* Coordonnees terrain de l'origine de l'image epipolaire */
istat = orrepere_3D_image ( epipo, xyz0, dcol, dlig ) ;
xx0 = xyz0[0] ;
yy0 = xyz0[1] ;
zz0 = xyz0[2] ;

/* re-echantillonnage */
ip = 0 ;
for ( il = 0 ; il < *onl ; il++ )
     {
    xx = xx0 ;
    yy = yy0 ;
    zz = zz0 ;
     for ( is = 0 ; is < *ons ; is++ )
    {
    istat = orterrain_to_photo ( phot, &xx, &yy, &zz, &phcc, &phll ) ;
    if ( istat != 1 ) return istat ;
    odata[ip] = interpolation ( idata, ins, inl, &phcc, &phll ) ;
    ip++ ;
    xx = xx + dcol[0] ;
    yy = yy + dcol[1] ;
    zz = zz + dcol[2] ;
    }
    xx0 = xx0 + dlig[0] ;
    yy0 = yy0 + dlig[1] ;
    zz0 = zz0 + dlig[2] ;
    }
return istat ;
}
/*----------------------------------------------------------------------------*/
 int ororient_epipolaires ( 	void* *phot1, void* *phot2,
                double *col0, double *lig0,
                double *col1, double *lig1,
                double *zmin, double *zmax,
                void* *epiphot1, void* *epiphot2,
                int *ns, int *nl )
/* > on positionne le plan epipolaire a l'altitude (zmin+zmax)/2 pour le point
  central de la zone designee sur l'image gauche
   > remplit_orient_epipolaire definit le plan epipolaire (focale, orientation)
   > emprise_epipo definit l'image sur le plan epipolaire

   > NB: la resolution des epipos est fixee a la resolution des images
   d'entree */

{
or_orientation		*ori1, *ori2 ;
or_orientation		*epi1, *epi2 ;
double 			ccentre, lcentre, hcentre ;
double			MM[3], vi[3], vj[3], vk[3]  ;
double			pas ;
int			status ;


ori1 = (or_orientation *) *phot1 ;
ori2 = (or_orientation *) *phot2 ;
epi1 = (or_orientation *) *epiphot1 ;
epi2 = (or_orientation *) *epiphot2 ;

if ( (epi1 != 0) && (epi2 != 0) && (ori1 != 0) && (ori2 != 0) )
    {
    status = 1 ;
    /* repere epipolaire */
    orrepere_epipolaire ( (*ori1).sommet, (*ori2).sommet,
                      vi, vj, vk ) ;

    /* point de passage */
    ccentre = (*col0 + *col1) / 2.0 ;
     lcentre = (*lig0 + *lig1) / 2.0 ;
    hcentre = (*zmin + *zmax) / 2.0 ;
    orphoto_et_zCarte_to_terrain ( phot1, &ccentre, &lcentre, &hcentre,
                &MM[0], &MM[1], &MM[2] ) ;

    /*resolution*/
    pas = orbest_resol ( phot1, phot2, zmin ) ;

        /*===============================*/
            /* gauche */
    /* definition du plan epipolaire */
    *epiphot1 = (void*) epi1 ;
    orremplit_orient_epipolaire ( phot1, MM, vi, vj, vk, &pas, epiphot1 ) ;

    /* position de l'origine de l'image */
    oremprise_epipo_gauche (  phot1, col0, lig0, col1, lig1, zmin, zmax,
                   epiphot1 ) ;
    *ns = (*epi1).ins ;
    *nl = (*epi1).inl ;

        /*===============================*/
            /* droite */

     *epiphot2 = (void*) epi2 ;
    orremplit_orient_epipolaire ( phot2, MM, vi, vj, vk, &pas, epiphot2 ) ;

    /* position de l'origine de l'image */
     oremprise_epipo_droite ( epiphot1, epiphot2 ) ;

    }

else

    {
    status = 0 ;
    }
return status ;
}
/*----------------------------------------------------------------------------*/
int orcorrige_distortion( double *col, double *lig, or_grille *gr )
{
/*
static int aCpt=0; aCpt++;
bool aBug = (aCpt==2017);
if (aBug)
   std::cout << "Cpt " << aCpt << "\n";
   */


double			xx, yy ;
double			fx, fy ;
double			dx, dy ;
int			i0, j0 ;
int			p0, p1, p2, p3 ;
int			istat ;


   if ( (*gr).pas <= 0 ) return 0 ;

   xx = *col / (double)((*gr).pas) ;
   yy = *lig / (double)((*gr).pas) ;
   i0 = (int) ( xx ) ;
   j0 = (int) ( yy ) ;

// if (aBug) std::cout << "CCCC\n";
   if ( i0 < 0 ) { i0 = 0 ; xx = 0.0 ; }
   if ( i0 > ((*gr).ns-1) ) { i0 = ((*gr).ns-1) ; xx = (double)((*gr).ns-1) ; }
   if ( j0 < 0 ) { j0 = 0 ; yy = 0.0 ; }
   if ( j0 > ((*gr).nl-1) ) { j0 = ((*gr).nl-1) ; yy = (double)((*gr).nl-1) ; }

// if (aBug) std::cout << "DDD\n";
   p0 = j0 * (*gr).ns + i0 ;
   if ( i0 < ((*gr).ns-1) ) { p1 = p0 + 1 ; } else { p1 = p0 ; }
   if ( j0 < ((*gr).nl-1) ) { p2 = p0 + (*gr).ns ; } else { p2 = p0 ; }
   if ( i0 < ((*gr).ns-1) ) { p3 = p2 + 1 ; } else { p3 = p2 ; }

// if (aBug) std::cout << "EEEE\n";
/* interpolation lineaire */
   fx = (double)(i0+1) - xx ;
   fy = (double)(j0+1) - yy ;

/*
if (aBug)
{
   std::cout << p0 << " " << p1 << " " << p2 << " " << p3 << "\n";
   std::cout << (*gr).ns  << " " << (*gr).nl << "\n";
}
*/
   dx = fy * 		( fx*((*gr).dx[p0]) + (1.0-fx)*((*gr).dx[p1]) ) +
        (1.0-fy) *	( fx*((*gr).dx[p2]) + (1.0-fx)*((*gr).dx[p3]) ) ;
   dy = fy * 		( fx*((*gr).dy[p0]) + (1.0-fx)*((*gr).dy[p1]) ) +
        (1.0-fy) *	( fx*((*gr).dy[p2]) + (1.0-fx)*((*gr).dy[p3]) ) ;
   istat = 1 ;

// if (aBug) std::cout << "GGGGG\n";
   *col = *col + dx ;
   *lig = *lig + dy ;
// if (aBug) std::cout << "HHHH\n";
   return istat ;
}

int or_orientation::CorrigeDist_T2P(double *col, double *lig)
{
    if (mDontUseDist)
        return 1;
    if (mCorrDistM2C)
    {
       Pt2dr aQ = mCorrDistM2C->Direct(Pt2dr(*col,*lig));
       *col = aQ.x;
       *lig = aQ.y;
       return 1;;
    }
    else
    {
       return orcorrige_distortion(col,lig,&gt2p);
    }
}

int or_orientation::CorrigeDist_P2T(double *col, double *lig)
{
    if (mDontUseDist)
        return 1;
    if (mCorrDistM2C)
    {
       Pt2dr aQ = mCorrDistM2C->Inverse(Pt2dr(*col,*lig));
       *col = aQ.x;
       *lig = aQ.y;
       return 1;;
    }
    else
    {
       return orcorrige_distortion(col,lig,&gp2t);
    }
}


/*----------------------------------------------------------------------------*/
double orbest_resol ( void* *phot1, void* *phot2, double *zmin )
{
or_orientation 		*ori1, *ori2 ;
double			hh ;
double			pas, pas1, pas2 ;

ori1 = (or_orientation *) *phot1 ;
ori2 = (or_orientation *) *phot2 ;

hh = (*ori1).sommet[2] - *zmin ;
pas1 = hh * (*ori1).pix[0] / (*ori1).focale ;
pas  = hh * (*ori1).pix[1] / (*ori1).focale ;
pas1 = _min ( pas , pas1 ) ;

hh = (*ori2).sommet[2] - *zmin ;
pas2 = hh * (*ori2).pix[0] / (*ori2).focale ;
pas  = hh * (*ori2).pix[1] / (*ori2).focale ;
pas2 = _min ( pas , pas2 ) ;

pas = _min ( pas1 , pas2 ) ;
return pas ;
}
/*----------------------------------------------------------------------------*/
void orremplit_orient_epipolaire ( void* *phot,
                     double MM[3], double vi[3],
                     double vj[3], double vk[3],
                     double *pas,
                     void* *epiphot )
{
or_orientation		*ori, *epi ;
double			SM[3] ;

ori = (or_orientation *) *phot ;
epi = (or_orientation *) *epiphot ;

/* les epipolaires sont corrigees des distortions lors de la projection */
(*epi).distor = 0 ;
orinit_grille_0 ( epiphot ) ;

/* copie des valeurs inchangees */
(*epi).jour = (*ori).jour ; (*epi).mois = (*ori).mois ;
(*epi).annee = (*ori).annee ;
(*epi).heure = (*ori).heure ;	(*epi).minute = (*ori).minute ;
(*epi).seconde = (*ori).seconde ;
sprintf( (*epi).chambre, "%s", (*ori).chambre ) ;
(*epi).altisol = (*ori).altisol ;
(*epi).origine[0] = (*ori).origine[0] ;
(*epi).origine[1] = (*ori).origine[1] ;
(*epi).lambert = (*ori).lambert ;
(*epi).sommet[0] = (*ori).sommet[0] ;
(*epi).sommet[1] = (*ori).sommet[1] ;
(*epi).sommet[2] = (*ori).sommet[2] ;

/* repere */
(*epi).vi[0] = vi[0] ; (*epi).vi[1] = vi[1] ; (*epi).vi[2] = vi[2] ;
(*epi).vj[0] = vj[0] ; (*epi).vj[1] = vj[1] ; (*epi).vj[2] = vj[2] ;
(*epi).vk[0] = vk[0] ; (*epi).vk[1] = vk[1] ; (*epi).vk[2] = vk[2] ;

/* focale */
SM[0] = MM[0] - (*epi).sommet[0] ;
SM[1] = MM[1] - (*epi).sommet[1] ;
SM[2] = MM[2] - (*epi).sommet[2] ;
(*epi).focale = _pscalaire ( SM , (*epi).vk ) ;

/* pas d'echantillonnage */
(*epi).pix[0] = *pas ;
(*epi).pix[1] = *pas ;

/* temporairement, on situe l'origine du plan au Point principal */
(*epi).ipp[0] = 0.0 ;
(*epi).ipp[1] = 0.0 ;

/* donnees images */
}
/*----------------------------------------------------------------------------*/
void oremprise_epipo_droite ( void* *epiphot1, void* *epiphot2 )
/* dans le plan epipolaire choisi, les deux images sont superposables
   Comme S1S2 est parallele a l'axe des x, ipp[1] a la meme valeur pour
   les deux images; ne reste qu'a calculer ipp[0] */
{
or_orientation		*epi1 ;
or_orientation		*epi2 ;
double 			S1S2[3] ;
double			xx ;

epi1 = (or_orientation *) *epiphot1 ;
epi2 = (or_orientation *) *epiphot2 ;

/* projection de S2 dans le repere de epiphot1 */
S1S2[0] = (*epi2).sommet[0] - (*epi1).sommet[0] ;
S1S2[1] = (*epi2).sommet[1] - (*epi1).sommet[1] ;
S1S2[2] = (*epi2).sommet[2] - (*epi1).sommet[2] ;

xx = (*epi1).ipp[0] + ( _pscalaire ( S1S2 , (*epi1).vi ) / (*epi1).pix[0] ) ;

(*epi2).ipp[0] = xx ;
(*epi2).ipp[1] = (*epi1).ipp[1] ;

(*epi2).ins = (*epi1).ins ;
(*epi2).inl = (*epi1).inl ;

}
/*----------------------------------------------------------------------------*/
void oremprise_epipo_gauche ( void* *phot, double *ph_c0, double *ph_l0,
                  double *ph_c1, double *ph_l1,
                  double *zmin, double *zmax,
                  void* *epiphot )
{
or_orientation		*epi ;
double			cc[8],ll[8] ;
double 			epi_c0, epi_l0, epi_c1, epi_l1 ;
double			SM[3] ;
double			fmin, fmax ;
double			focale ;
int			ii ;

epi = (or_orientation *) *epiphot ;
focale = (*epi).focale ;
fmin = focale - (*zmax - *zmin)/2.0 ;
fmax = focale + (*zmax - *zmin)/2.0 ;

/* origine de l'image */
orSM ( phot, ph_c0, ph_l0, SM ) ;
(*epi).focale = fmin ;
orinters_SM_photo ( epiphot , SM, &cc[0], &ll[0] ) ;
(*epi).focale = fmax ;
orinters_SM_photo ( epiphot , SM, &cc[1], &ll[1] ) ;

orSM ( phot, ph_c1, ph_l0, SM ) ;
(*epi).focale = fmin ;
orinters_SM_photo ( epiphot , SM, &cc[2], &ll[2] ) ;
(*epi).focale = fmax ;
orinters_SM_photo ( epiphot , SM, &cc[3], &ll[3] ) ;

orSM ( phot, ph_c1, ph_l1, SM ) ;
(*epi).focale = fmin ;
orinters_SM_photo ( epiphot , SM, &cc[4], &ll[4] ) ;
(*epi).focale = fmax ;
orinters_SM_photo ( epiphot , SM, &cc[5], &ll[5] ) ;

orSM ( phot, ph_c0, ph_l1, SM ) ;
(*epi).focale = fmin ;
orinters_SM_photo ( epiphot , SM, &cc[6], &ll[6] ) ;
(*epi).focale = fmax ;
orinters_SM_photo ( epiphot , SM, &cc[7], &ll[7] ) ;


epi_c0 = cc[0] ;
epi_l0 = ll[0] ;
epi_c1 = cc[0] ;
epi_l1 = ll[0] ;
for ( ii = 1 ; ii < 8 ; ii++ )
    {
    epi_c0 = _min ( epi_c0 , cc[ii] ) ;
    epi_c1 = _max ( epi_c1 , cc[ii] ) ;
    epi_l0 = _min ( epi_l0 , ll[ii] ) ;
    epi_l1 = _max ( epi_l1 , ll[ii] ) ;
    }

(*epi).ipp[0] = - epi_c0 ;
(*epi).ipp[1] = - epi_l0 ;
(*epi).focale = focale ;
(*epi).ins = (int) (epi_c1 - epi_c0) + 1 ;
(*epi).inl = (int) (epi_l1 - epi_l0) + 1 ;
}
/*----------------------------------------------------------------------------*/
 void orrepere_epipolaire ( double S1[3], double S2[3],
                 double vi[3], double vj[3], double vk[3] )
{
double			inorme, jnorme ;

vi[0] = S2[0] - S1[0] ;
vi[1] = S2[1] - S1[1] ;
vi[2] = S2[2] - S1[2] ;

vj[0] =  vi[1] ;		/* verticale descendante ^ vi */
vj[1] = -vi[0] ;
vj[2] =  0.0 ;
jnorme = _pscalaire ( vj , vj ) ;
inorme = jnorme + vi[2]*vi[2] ;

if ( (inorme > 0.0) && (jnorme > 0.0) )
    {
    inorme = sqrt ( inorme ) ;
    jnorme = sqrt ( jnorme ) ;
    _vecdiv ( vi , inorme ) ;
    _vecdiv ( vj , jnorme ) ;

    vk[0] = - vi[2]*vj[1] ;
    vk[1] =   vi[2]*vj[0] ;
    vk[2] =   vi[0]*vj[1] - vi[1]*vj[0] ;
    }
else
    {
    vi[0] = 0.0 ;
    vi[1] = 0.0 ;
    vi[2] = 0.0 ;
    vj[0] = 0.0 ;
    vj[1] = 0.0 ;
    vj[2] = 0.0 ;
    vk[0] = 0.0 ;
    vk[1] = 0.0 ;
    vk[2] = 0.0 ;
    }
}
/*----------------------------------------------------------------------------*/
double orresolution_sol ( void* *photo )
{
or_orientation 		*ori ;
if ( *photo == 0 ) return 0 ;
ori= (or_orientation *) *photo ;
return ( (*ori).pix[0] * ((*ori).sommet[2]-(*ori).altisol) / (*ori).focale ) ;
}
/*----------------------------------------------------------------------------*/
double oraltitude_sol ( void* *photo )
{
/*or_orientation 		*ori ;*/
if ( *photo == 0 ) return 0 ;

Tjs_El_User.ElAssert(0,EEM0<< "Utilisation de oraltitude_sol sans initialisation...");
/*return ( (*ori).altisol ) ;*/
    return 0;
}
/*----------------------------------------------------------------------------*/
int oremprise_carte ( void* *photo, double *zmin, double* zmax,
            int* marge,
            double *xmin, double *ymin,
            double *xmax, double *ymax )
{
or_orientation 		*ori ;
double			zz ;
int			ns,nl ;
double			col, lig;
double			cx,cy,cz ;
double			MM[3] ;

int			istat ;

if ( *photo == 0 ) return 0 ;
ori= (or_orientation *) *photo ;
ns = (*ori).ins ;
nl = (*ori).inl ;

col = 0.0 - *marge ;
lig = 0.0 - *marge ;
zz = *zmin ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = cx ; *ymin = cy ;
*xmax = cx ; *ymax = cy ;
zz = *zmax ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;

col = (double)(ns-1) + *marge ;
lig = 0.0 - *marge ;
zz = *zmin ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;
zz = *zmax ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;

col = (double)(ns-1) + *marge ;
lig = (double)(nl-1) + *marge ;
zz = *zmin ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;
zz = *zmax ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;

col = 0.0 - *marge ;
lig = (double)(nl-1) + *marge;
zz = *zmin ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;
zz = *zmax ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, &zz,
                                        &MM[0], &MM[1], &MM[2] ) ;
if ( istat != 1 ) return istat ;
istat = orterrain_to_carte ( photo,&MM[0],&MM[1],&MM[2],&cx,&cy,&cz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , cx ) ; *ymin = _min ( *ymin , cy ) ;
*xmax = _max ( *xmax , cx ) ; *ymax = _max ( *ymax , cy ) ;

/* arrondi a 10m pres */
*xmin = (double) ( (int)(*xmin/10.0) * 10 ) ;
*xmax = (double) ( (int)((*xmax+9.99)/10.0) * 10 ) ;
*ymin = (double) ( (int)(*ymin/10.0) * 10 ) ;
*ymax = (double) ( (int)((*ymax+9.99)/10.0) * 10 ) ;
return 1 ;
}
/*----------------------------------------------------------------------------*/
int oremprise_terrain ( void* *photo, double *zmin, double* zmax,
            int* marge,
            double *xmin, double *ymin,
            double *xmax, double *ymax )
{
or_orientation 		*ori ;
double			zz ;
int			ns,nl ;
double			col, lig;
double			tx,ty ;

int			istat ;

if ( *photo == 0 ) return 0 ;
ori= (or_orientation *) *photo ;
ns = (*ori).ins ;
nl = (*ori).inl ;

col = 0.0 - *marge ;
lig = 0.0 - *marge ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmin,
                                        &tx, &ty, &zz ) ;
if ( istat != 1 ) return istat ;
*xmin = tx ; *ymin = ty ;
*xmax = tx ; *ymax = ty ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmax,
                                        &tx, &ty, &zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;

col = (double)(ns-1) + *marge ;
lig = 0.0 - *marge ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmin,
                                        &tx, &ty, &zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmax,
                                        &tx, &ty, &zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;

col = (double)(ns-1) + *marge ;
lig = (double)(nl-1) + *marge ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmin,
                                        &tx, &ty, &zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmax,
                                        &tx, &ty,&zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;

col = 0.0 - *marge ;
lig = (double)(nl-1) + *marge;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmin,
                                        &tx, &ty, &zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;
istat = orphoto_et_zCarte_to_terrain ( photo, &col, &lig, zmax,
                                        &tx, &ty,&zz ) ;
if ( istat != 1 ) return istat ;
*xmin = _min ( *xmin , tx ) ; *ymin = _min ( *ymin , ty ) ;
*xmax = _max ( *xmax , tx ) ; *ymax = _max ( *ymax , ty ) ;

/* arrondi a 10m pres */
*xmin = (double) ( (int)(*xmin/10.0) * 10 ) ;
*xmax = (double) ( (int)((*xmax+9.99)/10.0) * 10 ) ;
*ymin = (double) ( (int)(*ymin/10.0) * 10 ) ;
*ymax = (double) ( (int)((*ymax+9.99)/10.0) * 10 ) ;
return 1 ;
}
/*----------------------------------------------------------------------------*/
int oremprise_photo_carte ( void* *photo,
                             double *xmin, double *ymin,
                             double *xmax, double *ymax,
                             double *zmin, double *zmax,
                             double *cmin, double *lmin,
                             double *cmax, double *lmax )
{
double		xt, yt, zt ;
double		col, lig ;
int			status ;

status = orcarte_to_terrain ( photo, xmin, ymin, zmin, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, cmin, lmin ) ;
if ( status != 1 ) return 0 ;
*cmax = *cmin ;
*lmax = *lmin ;

status = orcarte_to_terrain ( photo, xmin, ymin, zmax, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

status = orcarte_to_terrain ( photo, xmax, ymin, zmin, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

status = orcarte_to_terrain ( photo, xmax, ymin, zmax, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

status = orcarte_to_terrain ( photo, xmin, ymax, zmin, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

status = orcarte_to_terrain ( photo, xmin, ymax, zmax, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

status = orcarte_to_terrain ( photo, xmax, ymax, zmin, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

status = orcarte_to_terrain ( photo, xmax, ymax, zmax, &xt, &yt, &zt ) ;
if ( status != 1 ) return 0 ;
status = orterrain_to_photo ( photo, &xt, &yt, &zt, &col, &lig ) ;
if ( status != 1 ) return 0 ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;
return (0) ;
}
/*----------------------------------------------------------------------------*/
int oremprise_photo_epipolaire ( void* *epi, void* *photo,
                                 double *cmin, double *lmin,
                                 double *cmax, double *lmax )
{
or_orientation		*orip, *orie ;
double		SM[3] ;
double		col, lig ;

if ( (epi ==0)||(photo==0) ) return 0 ;
orip= (or_orientation *) *photo ;
orie= (or_orientation *) *epi ;
if ( ( _abs ( (*orip).sommet[0] - (*orie).sommet[0] ) > 0.00001 ) ||
     ( _abs ( (*orip).sommet[1] - (*orie).sommet[1] ) > 0.00001 ) ||
     ( _abs ( (*orip).sommet[2] - (*orie).sommet[2] ) > 0.00001 )  ) return 0 ;

col = 0.0 ;
lig = 0.0 ;
orSM ( epi, &col, &lig, SM ) ;
orinters_SM_photo ( photo, SM, cmin, lmin ) ;
*cmax = *cmin ;
*lmax = *lmin ;

col = (double)(*orie).ins ;
lig = 0.0 ;
orSM ( epi, &col, &lig, SM ) ;
orinters_SM_photo ( photo, SM, &col, &lig ) ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

col = 0.0 ;
lig = (double)(*orie).inl ;
orSM ( epi, &col, &lig, SM ) ;
orinters_SM_photo ( photo, SM, &col, &lig ) ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;

col = (double)(*orie).ins ;
lig = (double)(*orie).inl ;
orSM ( epi, &col, &lig, SM ) ;
orinters_SM_photo ( photo, SM, &col, &lig ) ;
*cmin = _min ( *cmin , col ) ;
*cmax = _max ( *cmax , col ) ;
*lmin = _min ( *lmin , lig ) ;
*lmax = _max ( *lmax , lig ) ;
return 1 ;
}




//
//       Conventions d'Orientation
//


cConvExplicite MakeExplicite(eConventionsOrientation aConv)
{
// std::cout << "CONNVvv " << aConv << " " <<  eConvAngPhotoMDegre << "\n";
   cConvExplicite aRes;
   {
      switch (aConv)
      {
          case eConvApero_DistM2C :
          case eConvApero_DistC2M :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = (aConv==eConvApero_DistC2M);
             aRes.MatrSenC2M() = true;
         aRes.ColMul() = Pt3dr(1,1,1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() = eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(2,1,0);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(aConv);
          }
          break;
          case eConvOriLib :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = true;
         aRes.ColMul() = Pt3dr(1,1,1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() = eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(0,1,2);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(eConvOriLib);
          }
          break;

          case eConvMatrPoivillier_E :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = false;
         aRes.ColMul() = Pt3dr(1,-1,-1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() = eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(0,1,2);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(eConvMatrPoivillier_E);
          }
          break;

          case eConvAngErdas :
          case eConvAngErdas_Grade :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = false;
         aRes.ColMul() = Pt3dr(1,-1,-1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() = (aConv==eConvAngErdas_Grade) ? eUniteAngleGrade : eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(0,1,2);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(aConv);
          }
          break;

          case eConvAngPhotoMDegre :
          case eConvAngPhotoMGrade :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = true;
         aRes.ColMul() = Pt3dr(1,-1,-1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() =  (aConv==eConvAngPhotoMGrade) ? eUniteAngleGrade : eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(0,1,2);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(aConv);
          }
          break;

          case eConvMatrixInpho :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = false;
         aRes.ColMul() = Pt3dr(1,-1,-1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() =  eUniteAngleUnknown;
         aRes.NumAxe() = Pt3di(0,1,2);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(aConv);
          }
          break;





          case eConvAngLPSDegre :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = true;
         aRes.ColMul() = Pt3dr(1,-1,-1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() =  eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(1,0,2);
         aRes.SensCardan() = true;
         aRes.Convention().SetVal(aConv);
          }
          break;





          default :
               ELISE_ASSERT(false,"Unknown eConventionsOrientation");
          break;
      }
   }
   return aRes;
}


/*
          case eConvAngErdas :
          case eConvAngErdas_Grade :
          {
         aRes.SensYVideo() = ConvIsSensVideo(aConv);
         aRes.DistSenC2M() = false;
             aRes.MatrSenC2M() = true;
  std::cout << "AAAAAAAAAAAA ERDAS \n";
             aRes.MatrSenC2M() = false;
         aRes.ColMul() = Pt3dr(1,-1,-1);
         aRes.LigMul() = Pt3dr(1,1,1);
             aRes.UniteAngles() = (aConv==eConvAngErdas_Grade) ? eUniteAngleGrade : eUniteAngleDegre;
         aRes.NumAxe() = Pt3di(0,1,2);
         aRes.SensCardan() = true;
         // aRes.SensCardan() = false;
         aRes.Convention().SetVal(aConv);
          }
          break;

          default :
               ELISE_ASSERT(false,"Unknown eConventionsOrientation");
          break;

*/



cConvExplicite MakeExplicite(const cConvOri & aConv)
{
   cConvExplicite aRes;

   if (aConv.ConvExplicite().IsInit())
   {
      return aConv.ConvExplicite().Val();
   }
   aRes = MakeExplicite(aConv.KnownConv().Val());
   return aRes;
}


class cDistFromCIC
{
       public :
          cDistFromCIC
      (
                 const cCalibrationInternConique & aCIC,
                 const cConvExplicite *  aArgConv,
         bool  RequireC2M
          );

           Pt2dr CorrY(Pt2dr aP)
           {
                 return  mConv.SensYVideo().Val() ? aP  : Pt2dr(aP.x,mCIC.SzIm().y-aP.y);
           }
       CamStenope * Cam();
       cCamStenopeBilin * CamBilin();

       cCamStenopeDistRadPol *  CamDRad();
       cCamStenopeModStdPhpgr * CamPhgrStd();
       cCam_Ebner *             CamEbner();
           cCam_DRad_PPaEqPPs *     CamDRad_PPaEqPPs();
           cCam_Fraser_PPaEqPPs *   CamFraser_PPaEqPPs();
       cCam_DCBrown *           CamDCBrown();

       cCam_Polyn0 *            CamPolyn0();
       cCam_Polyn1 *            CamPolyn1();
       cCam_Polyn2 *            CamPolyn2();
       cCam_Polyn3 *            CamPolyn3();
       cCam_Polyn4 *            CamPolyn4();
       cCam_Polyn5 *            CamPolyn5();
       cCamera_Param_Unif_Gen * CamUnif();

       cCam_RadFour7x2 *             Cam_RadFour7x2();
       cCam_RadFour11x2 *            Cam_RadFour11x2();
       cCam_RadFour15x2 *            Cam_RadFour15x2();
       cCam_RadFour19x2 *            Cam_RadFour19x2();

       CamStenopeIdeale *             CamSI();

       private :
       ElDistRadiale_PolynImpair  DRP(const cCalibrationInternConique & aCIC,const cCalibrationInterneRadiale &,bool C2M);

       CamStenope              * mCam;
       cCamStenopeBilin        * mCamBilin;
       cCamStenopeModStdPhpgr  * mCamPS;
       cCamStenopeDistRadPol   * mCamDR;
           CamStenopeIdeale        * mCamSI;
       cCam_Ebner *              mCamEb;
       cCam_RadFour7x2 *         mCam_RadFour7x2;
       cCam_RadFour11x2 *        mCam_RadFour11x2;
       cCam_RadFour15x2 *        mCam_RadFour15x2;
       cCam_RadFour19x2 *        mCam_RadFour19x2;
           cCam_DRad_PPaEqPPs *      mCamDR_PPas;
           cCam_Fraser_PPaEqPPs *    mCamFras_PPas;
       cCam_DCBrown *            mCamDCB;
       cCam_Polyn0 *             mCamPolyn0;
       cCam_Polyn1 *             mCamPolyn1;
       cCam_Polyn2 *             mCamPolyn2;
       cCam_Polyn3 *             mCamPolyn3;
       cCam_Polyn4 *             mCamPolyn4;
       cCam_Polyn5 *             mCamPolyn5;
       cCam_Polyn6 *             mCamPolyn6;
       cCam_Polyn7 *             mCamPolyn7;
       cCamLin_FishEye_10_5_5 *     mCamLinFE_10_5_5;
       cCamEquiSol_FishEye_10_5_5 *     mCamEquiSolFE_10_5_5;
       cCamStereoGraphique_FishEye_10_5_5 *     mCamStereographique_FE_10_5_5;
           cCamStenopeGrid *         mCamGrid;

       cCalibrationInternConique   mCIC;
           cConvExplicite              mConv;
};

CamStenope * cDistFromCIC::Cam()
{
   ELISE_ASSERT(mCam!=0,"cDistFromCIC::Cam");
   return mCam;
}

cCam_RadFour7x2 * cDistFromCIC::Cam_RadFour7x2()
{
   ELISE_ASSERT(mCam_RadFour7x2!=0,"cDistFromCIC::Cam");
   return mCam_RadFour7x2;
}
cCam_RadFour11x2 * cDistFromCIC::Cam_RadFour11x2()
{
   ELISE_ASSERT(mCam_RadFour11x2!=0,"cDistFromCIC::Cam");
   return mCam_RadFour11x2;
}
cCam_RadFour15x2 * cDistFromCIC::Cam_RadFour15x2()
{
   ELISE_ASSERT(mCam_RadFour15x2!=0,"cDistFromCIC::Cam");
   return mCam_RadFour15x2;
}
cCam_RadFour19x2 * cDistFromCIC::Cam_RadFour19x2()
{
   ELISE_ASSERT(mCam_RadFour19x2!=0,"cDistFromCIC::Cam");
   return mCam_RadFour19x2;
}

cCamStenopeBilin * cDistFromCIC::CamBilin()
{
   ELISE_ASSERT(mCamBilin!=0,"cDistFromCIC::Cam");
   return mCamBilin;
}

cCam_Ebner * cDistFromCIC::CamEbner()
{
   ELISE_ASSERT(mCamEb!=0,"cDistFromCIC::Cam");
   return mCamEb;
}

cCam_DRad_PPaEqPPs * cDistFromCIC::CamDRad_PPaEqPPs()
{
   ELISE_ASSERT(mCamDR_PPas!=0,"cDistFromCIC::Cam");
   return mCamDR_PPas;
}
cCam_Fraser_PPaEqPPs * cDistFromCIC::CamFraser_PPaEqPPs()
{
   ELISE_ASSERT(mCamFras_PPas!=0,"cDistFromCIC::Cam");
   return mCamFras_PPas;
}


cCam_DCBrown *  cDistFromCIC::CamDCBrown()
{
   ELISE_ASSERT(mCamDCB!=0,"cDistFromCIC::Cam");
   return mCamDCB;
}

cCam_Polyn0 *  cDistFromCIC::CamPolyn0()
{
   ELISE_ASSERT(mCamPolyn0!=0,"cDistFromCIC::Cam");
   return mCamPolyn0;
}

cCam_Polyn1 *  cDistFromCIC::CamPolyn1()
{
   ELISE_ASSERT(mCamPolyn1!=0,"cDistFromCIC::Cam");
   return mCamPolyn1;
}
cCam_Polyn2 *  cDistFromCIC::CamPolyn2()
{
   ELISE_ASSERT(mCamPolyn2!=0,"cDistFromCIC::Cam");
   return mCamPolyn2;
}



cCam_Polyn3 *  cDistFromCIC::CamPolyn3()
{
   ELISE_ASSERT(mCamPolyn3!=0,"cDistFromCIC::Cam");
   return mCamPolyn3;
}
cCam_Polyn4 *  cDistFromCIC::CamPolyn4()
{
   ELISE_ASSERT(mCamPolyn4!=0,"cDistFromCIC::Cam");
   return mCamPolyn4;
}
cCam_Polyn5 *  cDistFromCIC::CamPolyn5()
{
   ELISE_ASSERT(mCamPolyn5!=0,"cDistFromCIC::Cam");
   return mCamPolyn5;
}




cCamera_Param_Unif_Gen *  cDistFromCIC::CamUnif()
{
    if (mCam_RadFour7x2) return mCam_RadFour7x2;
    if (mCam_RadFour11x2) return mCam_RadFour11x2;
    if (mCam_RadFour15x2) return mCam_RadFour15x2;
    if (mCam_RadFour19x2) return mCam_RadFour19x2;

    if (mCamEb) return mCamEb;
    if (mCamDCB) return mCamDCB;

    if (mCamPolyn0) return mCamPolyn0;
    if (mCamPolyn1) return mCamPolyn1;
    if (mCamPolyn2) return mCamPolyn2;
    if (mCamPolyn3) return mCamPolyn3;
    if (mCamPolyn4) return mCamPolyn4;
    if (mCamPolyn5) return mCamPolyn5;
    if (mCamPolyn6) return mCamPolyn6;
    if (mCamPolyn7) return mCamPolyn7;
    if (mCamLinFE_10_5_5) return mCamLinFE_10_5_5;
    if (mCamEquiSolFE_10_5_5) return mCamEquiSolFE_10_5_5;
    if (mCamStereographique_FE_10_5_5) return mCamStereographique_FE_10_5_5;


    if (mCamDR_PPas)   return mCamDR_PPas;
    if (mCamFras_PPas)
    {
       return mCamFras_PPas;
    }


    ELISE_ASSERT(false,"cDistFromCIC::CamUnif");
    return 0;
}

CamStenopeIdeale * cDistFromCIC::CamSI()
{
   ELISE_ASSERT(mCamSI!=0,"cDistFromCIC::Cam");
   return mCamSI;
}

cCamStenopeDistRadPol * cDistFromCIC::CamDRad()
{
   ELISE_ASSERT(mCamDR!=0,"cDistFromCIC::Cam");
   return mCamDR;
}

cCamStenopeModStdPhpgr * cDistFromCIC::CamPhgrStd()
{
   ELISE_ASSERT(mCamPS!=0,"cDistFromCIC::Cam");
   return mCamPS;
}




ElDistRadiale_PolynImpair  cDistFromCIC::DRP
                           (const cCalibrationInternConique & aCIC,const cCalibrationInterneRadiale & aCIR,bool C2M)
{
    Pt2dr aC = CorrY(aCIR.CDist());
    double aRay = DMaxCoins(AfGC2M(aCIC),Pt2dr(mCIC.SzIm()),aC);

    ElDistRadiale_PolynImpair aDist(aRay,aC);

    for (int aK=0;aK<int(aCIR.CoeffDist().size()) ; aK++)
    {
        aDist.PushCoeff(aCIR.CoeffDist()[aK]);
    }


    if (! C2M)
    {
       double aDI =  aDist.DistInverse(aRay);
       double aRI = aRay * (1+aDI);
       if (ElAbs(aDI)>0.5)
       {
           std::cout << "DI " << aDI << "Ray " << aRay << "\n";
           ELISE_ASSERT
           (
                 (ElAbs(aDI)<0.5),
                 "Radiale distorsion abnormaly high"
           );
       }
       aDist.SetRMax(aRI);
    }

/*
 std::cout << C2M << " RAY = " << aRay << "\n";
    double aRI = aRay * (1+aDist.DistInverse(aRay));
    double aRR = aRI *  (1+aDist.DistDirecte(aRI));
std::cout << " R-INV " << aRI << " " << aRR  << "\n";
 std::cout <<"\n";
*/

    return aDist;
}


std::vector<double> StdEtat_F_PP(const cCalibrationInterneUnif & aCIU,const cCalibrationInternConique & aCIC)
{
    std::vector<double> aVE = aCIU.Etats();
    if (aVE.empty())
       aVE.push_back(aCIC.F());
    if (aVE.size()<=1)
       aVE.push_back(aCIC.PP().x);
    if (aVE.size()<=2)
       aVE.push_back(aCIC.PP().y);

   return aVE;
}







cDistFromCIC::cDistFromCIC
(
    const cCalibrationInternConique & aCIC,
    const cConvExplicite *  anArgConv,
    bool  RequireC2M
)
{

    mCIC = aCIC;
    if (anArgConv)
    {
        mConv = *  anArgConv;
    if (aCIC.KnownConv().IsInit())
    {
            ELISE_ASSERT
        (
             aCIC.KnownConv().Val() == anArgConv->Convention().Val(),
         "Incoherernce between conventions"
        );
    }
    }
    else
    {
          ELISE_ASSERT
          (
              aCIC.KnownConv().IsInit(),
              "Aucune convention specifiee"
          );

      mConv = MakeExplicite(aCIC.KnownConv().Val());
    }
    bool aC2M = mConv.DistSenC2M().Val();


    const std::vector<cCalibDistortion> &aVCD = aCIC.CalibDistortion();
    int aNbD = (int)aVCD.size();


    std::vector<ElDistortion22_Gen *> aV2D;
    std::vector<bool>                 aVDisDirect;
    for (int aKD = 0 ; aKD<aNbD ; aKD++)
    {
       mCam = 0;
       mCamPS =0;
       mCamDR =0;
       mCamBilin = 0;
       mCamEb = 0;
       mCam_RadFour7x2 = 0;
       mCam_RadFour11x2 = 0;
       mCam_RadFour15x2 = 0;
       mCam_RadFour19x2 = 0;
       mCamSI = 0;
       mCamDR_PPas = 0;
       mCamFras_PPas = 0;
       mCamDCB = 0;
       mCamPolyn0 = 0;
       mCamPolyn1 = 0;
       mCamPolyn2 = 0;
       mCamPolyn3 = 0;
       mCamPolyn4 = 0;
       mCamPolyn5 = 0;
       mCamPolyn6 = 0;
       mCamPolyn7 = 0;
       mCamLinFE_10_5_5 = 0;
       mCamEquiSolFE_10_5_5 = 0;
       mCamStereographique_FE_10_5_5 = 0;
       mCamGrid = 0;

       cCalibDistortion  aCD = aVCD[aKD];
       AdaptDist2PPaEqPPs(aCD);

       bool aKC2M = aC2M;
       if (aKD<int(aCIC.ComplIsC2M().size()) && (aKD!=aNbD-1))
          aKC2M = aCIC.ComplIsC2M()[aKD];

       if (aCD.ModNoDist().IsInit())
       {
            mCamSI =  new CamStenopeIdeale(aKC2M,aCIC.F(),aCIC.PP(),aCIC.ParamAF());

            mCam = mCamSI;
       }
       else if (aCD.ModGridDef().IsInit())
       {
            // mCamBilin = GlobFromXmlGridStuct(aCIC.F(),aCIC.PP(),aCD.ModGridDef().Val());
            mCamBilin = new cCamStenopeBilin(aCIC.F(),aCIC.PP(),cDistorBilin::FromXmlGridStuct(aCD.ModGridDef().Val()));
            mCam = mCamBilin;

       }
       else if (aCD.ModRad().IsInit())
       {
           const cCalibrationInterneRadiale & aCIR = aCD.ModRad().Val();
           mCamDR = new cCamStenopeDistRadPol
                    (
                 aKC2M,
                     aCIC.F(),
                     aCIC.PP(),
                             DRP(aCIC,aCIR,aC2M),
                             aCIC.ParamAF()
                );
       mCam = mCamDR;
       }
       else if (aCD.ModPhgrStd().IsInit())
       {
           const cCalibrationInternePghrStd aCIPS =  aCD.ModPhgrStd().Val();
           const cCalibrationInterneRadiale & aCIR = aCIPS.RadialePart();

               cDistModStdPhpgr aDPS(DRP(aCIC,aCIR,aC2M));
           double aSign = 1 ;
           aDPS.P1() = aSign * aCIPS.P1().Val();
           aDPS.P2() = aSign * aCIPS.P2().Val();
           aDPS.b1() = aSign * aCIPS.b1().Val();
           aDPS.b2() = aSign * aCIPS.b2().Val();

           mCamPS = new cCamStenopeModStdPhpgr(aKC2M,aCIC.F(),aCIC.PP(),aDPS,aCIC.ParamAF());
           mCam = mCamPS;
       }
       else if (aCD.ModUnif().IsInit())
       {
           const cCalibrationInterneUnif & aCIU = aCD.ModUnif().Val();
           eModelesCalibUnif aTypeModele = aCIU.TypeModele();
       switch (aTypeModele)
       {

           case eModeleRadFour7x2 :
           case eModeleRadFour11x2 :
           case eModeleRadFour15x2 :
           case eModeleRadFour19x2 :
               {
                std::vector<double> aVE = aCIU.Etats();
                    Pt2dr aSzIm  = Pt2dr(aCIC.SzIm())/2.0;
                    if (aVE.size()==0) aVE.push_back(euclid(aSzIm));
                    if (aVE.size()==1) aVE.push_back(aSzIm.x);
                    if (aVE.size()==2) aVE.push_back(aSzIm.y);

                    if (aTypeModele == eModeleRadFour7x2)
                    {
                        mCam_RadFour7x2  = new cCam_RadFour7x2
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aCIU.Params(),
                                                   &aVE
                                                );
                        mCam = mCam_RadFour7x2;
                    }
                    else if (aTypeModele == eModeleRadFour11x2)
                    {
                        mCam_RadFour11x2  = new cCam_RadFour11x2
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aCIU.Params(),
                                                   &aVE
                                                );
                        mCam = mCam_RadFour11x2;
                    }
                    else if (aTypeModele == eModeleRadFour15x2)
                    {
                        mCam_RadFour15x2  = new cCam_RadFour15x2
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aCIU.Params(),
                                                   &aVE
                                                );
                        mCam = mCam_RadFour15x2;
                    }
                    else if (aTypeModele == eModeleRadFour19x2)
                    {
                        mCam_RadFour19x2  = new cCam_RadFour19x2
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aCIU.Params(),
                                                   &aVE
                                                );
                        mCam = mCam_RadFour19x2;
                    }

               }
           break;

           case eModeleEbner :
           {
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
            {
               // Chez Ebner le param B est la base en repere image, on lui donne
               // pour valeur par defaut ce qu'elle vaut en rec 60%
               aVE.push_back(0.4 *ElMin(aCIC.SzIm().x,aCIC.SzIm().y));
            }
                    mCamEb = new cCam_Ebner
                                 (
                                     aKC2M,
                                     aCIC.F(),
                                     aCIC.PP(),
                                     Pt2dr(aCIC.SzIm()),
                                     aCIC.ParamAF(),
                                     &aCIU.Params(),
                                     &aVE
                                  );
            mCam = mCamEb;
               }
           break;

           case eModele_DRad_PPaEqPPs :
           {
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
            {
               aVE.push_back(aCIC.F());
            }
                    mCamDR_PPas = new cCam_DRad_PPaEqPPs
                                 (
                                     aKC2M,
                                     aCIC.F(),
                                     aCIC.PP(),
                                     Pt2dr(aCIC.SzIm()),
                                     aCIC.ParamAF(),
                                     &aCIU.Params(),
                                     &aVE
                                  );
            mCam = mCamDR_PPas;
               }
           break;

           case eModele_Fraser_PPaEqPPs :
           {
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
            {
               aVE.push_back(aCIC.F());
            }
                    mCamFras_PPas = new cCam_Fraser_PPaEqPPs
                                 (
                                     aKC2M,
                                     aCIC.F(),
                                     aCIC.PP(),
                                     Pt2dr(aCIC.SzIm()),
                                     aCIC.ParamAF(),
                                     &aCIU.Params(),
                                     &aVE
                                  );
            mCam = mCamFras_PPas;
               }
           break;



           // cCam_Fraser_PPaEqPPs *    mCamFras_PPas;
           // cCam_Fraser_PPaEqPPs *   CamFraser_PPaEqPPs();

           case eModeleDCBrown :
           {
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
               aVE.push_back(aCIC.F());
                    mCamDCB = new cCam_DCBrown
                                  (
                                      aKC2M,
                                      aCIC.F(),
                                      aCIC.PP(),
                                      Pt2dr(aCIC.SzIm()),
                                      aCIC.ParamAF(),
                                      &aCIU.Params(),
                                      &aVE
                                   );
            mCam = mCamDCB;
               }
           break;
		   
           case eModelePolyDeg0 :
           {
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn0 = new cCam_Polyn0
                                     (
                                           aKC2M,aCIC.F(),
                                           aCIC.PP(),
                                           Pt2dr(aCIC.SzIm()),
                                           aCIC.ParamAF(),
                                           &aCIU.Params(),
                                           &aVE
                                     );
               mCam = mCamPolyn0;
           };
           break;
		   
           case eModelePolyDeg1 :
           {
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn1 = new cCam_Polyn1
                                     (
                                           aKC2M,aCIC.F(),
                                           aCIC.PP(),
                                           Pt2dr(aCIC.SzIm()),
                                           aCIC.ParamAF(),
                                           &aCIU.Params(),
                                           &aVE
                                     );
               mCam = mCamPolyn1;
           };
           break;

           case eModelePolyDeg2 :
           {
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn2 = new cCam_Polyn2
                                     (
                                           aKC2M,aCIC.F(),
                                           aCIC.PP(),
                                           Pt2dr(aCIC.SzIm()),
                                           aCIC.ParamAF(),
                                           &aCIU.Params(),
                                           &aVE
                                     );
               mCam = mCamPolyn2;
           };
           break;


           case eModelePolyDeg3 :
               {
/*
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
               aVE.push_back(aCIC.F());
*/
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn3 = new cCam_Polyn3
                                     (
                                          aKC2M,
                                          aCIC.F(),
                                          aCIC.PP(),
                                          Pt2dr(aCIC.SzIm()),
                                          aCIC.ParamAF(),
                                          &aCIU.Params(),
                                          &aVE
                                     );
            mCam = mCamPolyn3;
               };
           break;

           case eModelePolyDeg4 :
               {
/*
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
               aVE.push_back(aCIC.F());
*/
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn4 = new cCam_Polyn4
                                     (
                                          aKC2M,aCIC.F(),
                                          aCIC.PP(),
                                          Pt2dr(aCIC.SzIm()),
                                          aCIC.ParamAF(),
                                          &aCIU.Params(),
                                          &aVE
                                     );
            mCam = mCamPolyn4;
               };
           break;

           case eModelePolyDeg5 :
               {
/*
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
               aVE.push_back(aCIC.F());
*/
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn5 = new cCam_Polyn5
                                     (
                                          aKC2M,
                                          aCIC.F(),
                                          aCIC.PP(),
                                          Pt2dr(aCIC.SzIm()),
                                          aCIC.ParamAF(),
                                          &aCIU.Params(),
                                          &aVE
                                     );
            mCam = mCamPolyn5;
               };
           break;

           case eModelePolyDeg6 :
               {
/*
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
               aVE.push_back(aCIC.F());
*/
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn6 = new cCam_Polyn6
                                     (
                                          aKC2M,
                                          aCIC.F(),
                                          aCIC.PP(),
                                          Pt2dr(aCIC.SzIm()),
                                          aCIC.ParamAF(),
                                          &aCIU.Params(),
                                          &aVE
                                     );
            mCam = mCamPolyn6;
               };
           break;

           case eModelePolyDeg7 :
               {
/*
                std::vector<double> aVE = aCIU.Etats();
            if (aVE.empty())
               aVE.push_back(aCIC.F());
*/
                    std::vector<double> aVE = StdEtat_F_PP(aCIU,aCIC);
                    mCamPolyn7 = new cCam_Polyn7
                                     (
                                          aKC2M,
                                          aCIC.F(),
                                          aCIC.PP(),
                                          Pt2dr(aCIC.SzIm()),
                                          aCIC.ParamAF(),
                                          &aCIU.Params(),
                                          &aVE
                                     );
            mCam = mCamPolyn7;
               };
           break;


           case eModele_FishEye_10_5_5 :
           case eModele_EquiSolid_FishEye_10_5_5 :
           case eModele_Stereographik_FishEye_10_5_5  :
               {
                    std::vector<double> aVE = aCIU.Etats();
if (0)
{
    std::cout << "FishEyCreate " << aVE.size()  << " " << aVE[0] << "\n";
    getchar();
}
                    if (aVE.empty())
                       aVE.push_back(aCIC.F());
                    std::vector<double> aPar = aCIU.Params();
                    if (aPar.empty())
                    {
                       aPar.push_back(aCIC.PP().x);  // MPD : je vois pas pourquoi on / par 2 ???
                       aPar.push_back(aCIC.PP().y);

                       // aPar.push_back(aCIC.PP().x/2.0);
                       // aPar.push_back(aCIC.PP().y/2.0);
                    }
                    if (aCIU.TypeModele()== eModele_FishEye_10_5_5)
                    {
                        mCamLinFE_10_5_5 = new cCamLin_FishEye_10_5_5
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aPar,
                                                   &aVE
                                               );
                         mCam = mCamLinFE_10_5_5;
                    }
                    else if (aCIU.TypeModele()==eModele_EquiSolid_FishEye_10_5_5)
                    {
                        mCamEquiSolFE_10_5_5 = new cCamEquiSol_FishEye_10_5_5
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aPar,
                                                   &aVE
                                               );
                        mCam = mCamEquiSolFE_10_5_5;
                    }
                    else  // eModele_Stereographik_FishEye_10_5_5
                    {
                        mCamStereographique_FE_10_5_5 = new cCamStereoGraphique_FishEye_10_5_5
                                               (
                                                   aKC2M,
                                                   aCIC.F(),
                                                   aCIC.PP(),
                                                   Pt2dr(aCIC.SzIm()),
                                                   aCIC.ParamAF(),
                                                   &aPar,
                                                   &aVE
                                               );
                        mCam = mCamStereographique_FE_10_5_5;
                    }
               };
           break;



               default :
           {
                    ELISE_ASSERT(false,"Do not handle Model Polynomiale");
           }

       }

       }
       else if (aCD.ModGrid().IsInit())
       {
          const cCalibrationInterneGrid  & aCIG = aCD.ModGrid().Val();
          cDbleGrid * a2G =  new cDbleGrid(aCIG.Grid());
          ElDistortion22_Gen * aDPC = 0;

          if (aCIG.PreCondGrid().IsInit())
          {
             aDPC = ElDistortion22_Gen::AllocPreC(aCIG.PreCondGrid().Val());
          }


          cDistCamStenopeGrid * aDCG = new cDistCamStenopeGrid(aDPC,a2G);


          mCamGrid = new cCamStenopeGrid
                         (
                             aCIC.F(),
                             aCIC.PP(),
                             aDCG,
                             aCIC.SzIm(),
                             aCIC.ParamAF()
                         );
          mCam = mCamGrid;
       }

       ELISE_ASSERT(mCam!= 0,"Incoherence in cDistFromCIC::cDistFromCIC");

       if (aKD != (aNbD-1))
       {
           aV2D.push_back(&(mCam->Dist()));
       aVDisDirect.push_back(!aKC2M);
       }
    }

/*
    if (aC2M)
       mCam->SetDistInverse();
    else
       mCam->SetDistDirecte();
*/
   mCam->AddDistCompl(aVDisDirect,aV2D);
   if (aCIC.CorrectionRefractionAPosteriori().IsInit())
   {
       mCam->AddCorrecRefrac(new cCorrRefracAPost(aCIC.CorrectionRefractionAPosteriori().Val()));
   }
   /*
    for (int aK2=0 ; aK2<int(aV2D.size()) ; aK2++)
    {
        mCam->AddDistCompl(aVDisDirect[aK2],aV2D[aK2]);
    }
    */

    mCam->SetSz(aCIC.SzIm());

    if (aCIC.PixelSzIm().IsInit())
    {
         mCam->SetSzPixel(aCIC.PixelSzIm().Val());
    }
// std::cout << "RRRRRRRRRuuuuu   " << aCIC.RayonUtile().IsInit() << "\n";
    if (aCIC.RayonUtile().IsInit())
    {
        mCam->SetRayonUtile(aCIC.RayonUtile().Val(),30);
    }

    if (aCIC.ParamForGrid().IsInit())
    {
        mCam->SetParamGrid(aCIC.ParamForGrid().Val());
    }
    mCam->SetScanned(aCIC.ScannedAnalogik().ValWithDef(false));
}





//  Calcule la rotation affine qui transforme un point de coordonnee
//  camera en point de coordonnee monde, les colonnes de la matrice
//  sont directement injectable dans orilib

ElMatrix<double>   Std_RAff_C2M
                   (
                       const cRotationVect             & aRVect,
                       const cConvExplicite            & aConv,
                       bool & TrueRot
                   )
{
   TrueRot = true;

   ElMatrix<double> aM(3,true); // Initialisee a l'identite


   if (aRVect.CodageMatr().IsInit())
   {
       SetLig(aM,0,aRVect.CodageMatr().Val().L1());
       SetLig(aM,1,aRVect.CodageMatr().Val().L2());
       SetLig(aM,2,aRVect.CodageMatr().Val().L3());


       TrueRot = aRVect.CodageMatr().Val().TrueRot().ValWithDef(true);

       if ((!TrueRot) && (false)) // false : valeur de ForceTrueRot avant supression
       {
           TrueRot = true;
           cElWarning::TrueRot.AddWarn("Force True Rot",__LINE__,__FILE__);

       }

   }
   else if(aRVect.CodageAngulaire().IsInit())
   {
      double aVTeta[3];
      aRVect.CodageAngulaire().Val().to_tab(aVTeta);
      int  aKTeta[3];
      aConv.NumAxe().Val().to_tab(aKTeta);
      for (int aK=0 ; aK<3 ; aK++)
      {
          double aTeta = ToRadian(aVTeta[aK],aConv.UniteAngles().Val());
          ElMatrix<double> aDM = ElMatrix<double>::Rotation3D(aTeta,aKTeta[aK]);
      if (aConv.SensCardan().Val())
             aM = aM * aDM;
          else
         aM = aDM * aM;
      }
   }
   else if (aRVect.CodageSymbolique().IsInit())
   {
        aM =  ElMatrix<double>::PermRot(aRVect.CodageSymbolique().Val());
   }



   if ( !aConv.MatrSenC2M().Val())
   {
      if (TrueRot)
         aM.self_transpose();
      else
         aM = gaussj(aM);
   }

  double aCMul[3],aLMul[3];

  aConv.ColMul().Val().to_tab(aCMul);
  aConv.LigMul().Val().to_tab(aLMul);


  // std::cout << "AAaaaAaaaaaaaaaaaaaaaaaaa\n";
  for (int aC=0 ; aC<3 ; aC++)
  {
      for (int aL=0 ; aL<3 ; aL++)
      {
          // std::cout << "ColllLiig " << aCMul[aC] << " " <<  aLMul[aL] << " " << aConv.MatrSenC2M() << "\n";
          aM(aC,aL) *= aCMul[aC] * aLMul[aL];

          // std::cout << " " <<  aM(aC,aL) ;
      }
      // std::cout << "\n";
  }




   return aM;
}


ElRotation3D  Std_RAff_C2M
              (
                 const cOrientationExterneRigide & aCE,
             const cConvExplicite            & aConv
         )
{
  bool TrueRot;
  ElMatrix<double> aM =  Std_RAff_C2M(aCE.ParamRotation(),aConv,TrueRot);

  return ElRotation3D(aCE.Centre(),aM,TrueRot);
}



int XML_orlit_fictexte_orientation (const char *fic, or_orientation *ori,bool QuickGrid )
{
  double aNbGrid = QuickGrid ? 20 : 150;
  double aDMinGrid = QuickGrid ? 300.0 : 20.0;
  double aSeuilVerifGrid = QuickGrid ? 2.0 : 5e-2 ;


   ori->InitNewParam();
   {
       const char * Unknown= "Unknown";
       for (int k=0; k<8 ; k++)
           ori->chambre[k] = Unknown[k];
       ori->jour = 0;
       ori->mois = 0;
       ori->annee = 0;
       ori->heure = 0;
       ori->minute = 0;
       ori->seconde = 0;

       ori->pix[0] = 1.0;
       ori->pix[1] = 1.0;

       // Origine du repere tangent local et zone lambert,
       // inutilise : on donne des valeurs "absurdes"
       ori->origine[0] = 0;
       ori->origine[1] = 0;
       ori->lambert = -1;
   }

   cOrientationConique anOC = StdGetObjFromFile<cOrientationConique>
                             (
                                 fic,
                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                 "OrientationConique",
                 "OrientationConique"
                 );
   AssertOrIntImaIsId(anOC);
   ori->mOC= new cOrientationConique(anOC);

   cOrientationExterneRigide anOER =  anOC.Externe();
   ELISE_ASSERT(anOER.AltiSol().IsInit(),"Altisol Non Init Pour un passage en ori");
   ori->altisol = anOER.AltiSol().Val();
   ori->mProf = anOER.Profondeur().ValWithDef(-1);


   cConvExplicite  aConvOri =MakeExplicite(anOC.ConvOri());

   ElRotation3D aRot = Std_RAff_C2M(anOC.Externe(),aConvOri);
   {


       aRot.ImAff(Pt3dr(0,0,0)).to_tab(ori->sommet);
       aRot.ImVect(Pt3dr(1,0,0)).to_tab(ori->vi);
       aRot.ImVect(Pt3dr(0,1,0)).to_tab(ori->vj);
       aRot.ImVect(Pt3dr(0,0,1)).to_tab(ori->vk);


   }

   // OOOO
   ELISE_ASSERT(anOC.Interne().IsInit(),"cCalibrationInternConique");
   const cCalibrationInternConique  & anOI =  anOC.Interne().Val();
   cDistFromCIC  aDF(anOI,&aConvOri,false);
   CamStenope * aCam = aDF.Cam();
   aCam->SetOrientation(aRot.inv()); // Pour la verif !
   bool  CanDelete = true;
   {

     ori->ins = round_ni(anOI.SzIm().x);
     ori->inl = round_ni(anOI.SzIm().y);

     ori->ipp[0] =  anOI.PP().x;
     ori->ipp[1] =  anOI.PP().y;
     ori->focale =  anOI.F();



     if (aCam->CanExportDistAsGrid())
     {

    //  Je suis pas sur que ca serve encore, je le laisse la ou
    //  il etait avant
          InitGrilleFromCam(ori,&((*ori).gt2p),*aCam,true); //true=M2C
          InitGrilleFromCam(ori,&((*ori).gp2t),*aCam,false);
          ori->distor = 1;




           cElDistFromCam aDFC(*aCam,false);
           Box2dr aBoxI(Pt2dr(0,0),Pt2dr(anOI.SzIm()));
           Box2dr aBoxMonde = aDFC.ImageRecOfBox(aBoxI);
           Pt2dr aRab(2,2);
           double aDDiscr = ElMin(aDMinGrid,euclid(anOI.SzIm())/aNbGrid);
          ori->mCorrDistM2C = new cDbleGrid
                         (
                                false,  // P0P1 DIrect : false car Monde 2 Cam
                             true,
                             aBoxMonde._p0 - aRab,
                              aBoxMonde._p1 + aRab,
                              Pt2dr(aDDiscr,aDDiscr),
                              aDFC  ,// aC2.mDistC2M,
                              "toto"
                 );
     }
     else
     {
         std::cout << "SUPRESS DIS ORILIB mmmmmmmmmmmm\n";
         CanDelete = false;
         ori->mCorrDistM2C = new cElDistFromCam(*aCam,false);
     }
   }

   if (anOC.Verif().IsInit())
   {

      Data_Ori3D_Std aDataOri3(ori);
      cCamera_Orilib aCamOri(&aDataOri3);
      std::list<Appar23> aLAp;

      Ori3D_Std anOri(ori);
      const cVerifOrient & aV = anOC.Verif().Val();

      int aKPts=0;
      for
      (
        std::list<cMesureAppuis>::const_iterator itA=aV.Appuis().begin();
        itA != aV.Appuis().end();
        itA++
      )
      {
         Pt2dr aPIm1 = aDF.CorrY(itA->Im());
         Pt3dr aPTer = itA->Ter();
         Pt2dr aPIm2 = aCam->R3toF2(aPTer);
         Pt2dr aPIm3 = aCamOri.R3toF2(aPTer);

             double aD12 = euclid(aPIm1,aPIm2);
             double aD23 = euclid(aPIm3,aPIm2);
         if (  (aD12>aV.Tol()) ||  (aD23 > aSeuilVerifGrid))
         {
            std::cout << "For File =" << fic  << " KPT = " << aKPts<< "\n";
            std::cout << "Dist =" << aD12 << " " << aD23 << "\n";
            std::cout << aPIm1 << aPIm2 << aPIm3 << "\n";
            std::cout << (aPIm1-Pt2dr(round_down(aPIm1)))
                          << (aPIm2-Pt2dr(round_down(aPIm2)))
              << (aPTer-Pt3dr(round_down(aPTer))) << "\n";
                Pt3dr A = aCam->R3toL3(aPTer);
                Pt2dr B = aCam->R3toC2(aPTer);
                Pt2dr C = aCam->DistDirecte(B);
                std::cout << "A " << A  << (A-Pt3dr(round_down(A))) << "\n";
                std::cout << "B " << B  << (B-Pt2dr(round_down(B))) << "\n";
                std::cout << "C " << C  << (C-Pt2dr(round_down(C))) << "\n";

                std::cout  << "Foc " << aCam->Focale() << "\n";
                std::cout  << "PP  " << aCam->PP() << "\n";

                if (0)
                {
                   cCamStenopeDistRadPol* aCDR =  aDF.CamDRad();
                   ElDistRadiale_PolynImpair & aPDR = aCDR->DRad();

                   std::cout  <<  "CCC " <<  aPDR.Direct(B) << "\n";
                   double aR = euclid(aPDR.Centre()-B);
                   std::cout << "RAY " << aR << " DIST " << aPDR.DistDirecte(aR) * aR << "\n";
                   std::cout << "R MAX " << aPDR.RMax() << "\n";

                   std::cout  << "CDIST " << aPDR.Centre() << "\n";
                   std::cout  << "NbCoeff " << aPDR.NbCoeff() << "\n";
                   std::cout  << " C2  " << aPDR.Coeff(0) << "\n";
                   std::cout  << " C4  " << aPDR.Coeff(1) << "\n";
                   std::cout  << " C6  " << aPDR.Coeff(2) << "\n";
                   std::cout  << " C6  " << aPDR.Coeff(3) << "\n";
                   std::cout  << " C6  " << aPDR.Coeff(4) << "\n";

                   //for (int aK=0 ; aK<1400 ; aK+=50)
                   //   std::cout  << aK  << "   " << aPDR.DistDirecte(aK) << "\n";
                }

                ELISE_ASSERT
            (
                false,
            "Pb in verif orient (XML_orlit_fictexte_orientation)"
            );
         }
             aKPts++;
      }

#if (0)
      if (aV.IsTest().Val())
      {
       double aDMin;
           ElRotation3D aRMin = aCam.CombinatoireOFPA(1000,aLAp,&aDMin);
       std::cout << "---------- DMIN = " << aDMin << "\n";
       aCam.SetOrientation(aRMin);
       std::cout << "CO : " <<  aCam.CentreOptique() << "\n";
           ElRotation3D anOR2 = aCam.Orient();
       std::cout << anOR2.IRecVect(Pt3dr(1,0,0)) << "\n";
       std::cout << anOR2.IRecVect(Pt3dr(0,1,0)) << "\n";
       std::cout << anOR2.IRecVect(Pt3dr(0,0,1)) << "\n";
       Pt3dr aC = anOR2.IRecAff(Pt3dr(0,0,0));
       std::cout.precision(10);
       std::cout <<"C = " <<  aC << "\n";
       std::cout.precision(6);
       double aF = 180.0/PI;
       std::cout << "Teta = "
                 <<  anOR2.teta01() * aF << " "
                 <<  anOR2.teta02() * aF << " "
                 <<  anOR2.teta12() * aF << "\n";

           for
           (
             std::list<cMesureAppuis>::const_iterator itA=aV.Appuis().begin();
             itA != aV.Appuis().end();
             itA++
           )
           {
         Pt2dr aPIm = aC2.CorrY(itA->Im());
             aPIm = aC2.mDistC2M.Direct(aPIm);

         Pt2dr  aPImV = aCam.R3toF2(itA->Ter());
         double aD = euclid(aPImV,aPIm);
         if (aV.ShowMes().Val())
         {
                 std::cout << "Dist[" << itA->Num().Val()  << "]=" << aD
                   << " dif=" << (aPImV-aPIm)<<  " P = " << aPImV << "\n";
             }
       }
       e-xit(-1);
      }
#endif
   }

   if (CanDelete)
      delete aCam;

   return 1 ;
}


bool ConvIsSensVideo(eConventionsOrientation aConv)
{
   return (aConv != eConvMatrPoivillier_E);
}


cCamera_Param_Unif_Gen *  Std_Cal_Unif
                          (
                             const cCalibrationInternConique & aCIC,
                     eConventionsOrientation            aKnownC
                          )
{
    cConvExplicite aConv = MakeExplicite(aKnownC);
    cDistFromCIC  aDF(aCIC,&aConv,false);

    cCamera_Param_Unif_Gen * aRes = aDF.CamUnif();
    if (aCIC.OrIntGlob().IsInit())
    {
        aRes->SetIntrImaC2M(AfGC2M(aCIC));
    }
    return aRes;
}


cCamStenopeBilin *  Std_Cal_Bilin
                          (
                             const cCalibrationInternConique & aCIC,
                     eConventionsOrientation            aKnownC
                          )
{
    cConvExplicite aConv = MakeExplicite(aKnownC);
    cDistFromCIC  aDF(aCIC,&aConv,false);

    cCamStenopeBilin * aRes = aDF.CamBilin();
    if (aCIC.OrIntGlob().IsInit())
    {
        aRes->SetIntrImaC2M(AfGC2M(aCIC));
    }
    return aRes;
}












cCamStenopeDistRadPol * Std_Cal_DRad_C2M
             (
                const cCalibrationInternConique & aCIC,
                    eConventionsOrientation            aKnownC
             )
{
    cConvExplicite aConv = MakeExplicite(aKnownC);
    cDistFromCIC  aDF(aCIC,&aConv,false);

    cCamStenopeDistRadPol * aRes = aDF.CamDRad();
    if (aCIC.OrIntGlob().IsInit())
    {
        aRes->SetIntrImaC2M(AfGC2M(aCIC));
    }
    return aRes;
}
/*
*/


cCamStenopeModStdPhpgr  *Std_Cal_PS_C2M
             (
                const cCalibrationInternConique & aCIC,
                    eConventionsOrientation            aKnownC
             )
{
    cConvExplicite aConv = MakeExplicite(aKnownC);
    cDistFromCIC  aDF(aCIC,&aConv,false);

    cCamStenopeModStdPhpgr * aRes = aDF.CamPhgrStd();
    if (aCIC.OrIntGlob().IsInit())
    {
        aRes->SetIntrImaC2M(AfGC2M(aCIC));
    }
    return aRes;
}


CamStenope * Std_Cal_From_CIC
             (
               const cCalibrationInternConique & aCIC,
               const std::string & aNameFile
             )
{
    eConventionsOrientation aKC = aCIC.KnownConv().ValWithDef(eConvApero_DistC2M);
    cConvExplicite aConv = MakeExplicite(aKC);
    cDistFromCIC  aDF(aCIC,&aConv,false);

    CamStenope *  aRes = aDF.Cam();
    if (aCIC.OrIntGlob().IsInit())
    {
        aRes->SetIntrImaC2M(AfGC2M(aCIC));
    }
    aRes->SetIdentCam(aNameFile);
    return aRes;
}

CamStenope * Std_Cal_From_CIC( const cCalibrationInternConique & aCIC)
{
   return Std_Cal_From_CIC(aCIC,"NoFile");
}

CamStenope * CamOrientGenFromFile(const std::string & aNameFile, cInterfChantierNameManipulateur * anICNM, bool throwAssert)
{
   std::string aFullFileName;
   if ( isUsingSeparateDirectories() )
      aFullFileName = MMOutputDirectory()+aNameFile;
   else
   {
      std::string aName0 = aNameFile;
      aFullFileName = (anICNM ? anICNM->Dir() : "") + aNameFile;
      if ( (!ELISE_fp::exist_file(aFullFileName)) && (ELISE_fp::exist_file(aName0)))
      {
         aFullFileName = aName0;
      }
   }


    cElXMLTree aTree(aFullFileName);
    cElXMLTree * aF1 = aTree.Get("CalibrationInternConique");
    if (aF1) return Std_Cal_From_File(aFullFileName,"CalibrationInternConique");

    cElXMLTree * aF2 = aTree.Get("OrientationConique");
    if (aF2) return Cam_Gen_From_File(aFullFileName,"OrientationConique",anICNM)->CS();

   if (throwAssert)
   {
        std::cout << "For name " << aNameFile << "\n";
        ELISE_ASSERT(false,"Cannot Get Orientation from File");
   }
   return 0;
}

CamStenope * BasicCamOrientGenFromFile(const std::string & aNameFile)
{
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(DirOfFile(aNameFile));
    return CamOrientGenFromFile(NameWithoutDir(aNameFile),anICNM);
}


CamStenope * Std_Cal_From_File
             (
               const std::string & aNameFile,
                   const std::string &  aNameTag
             )
{
   cCalibrationInternConique  aCIC =  StdGetObjFromFile<cCalibrationInternConique>
                  (
                        aNameFile,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        aNameTag,
                        "CalibrationInternConique"
                 );
   CamStenope * aRes=  Std_Cal_From_CIC(aCIC,aNameFile);
   return aRes;
}

static std::map<std::string,CamStenope *> theDic;

ElCamera * Gen_Cam_Gen_From_XML (bool CanUseGr,const cOrientationConique  & anOC,cInterfChantierNameManipulateur * anICNM,const std::string & aDir,const std::string & aNameFile)
{

   ElCamera * aRes = 0;
   cCalibrationInternConique  aCIC;

   eTypeProjectionCam aTPC = anOC.TypeProj().ValWithDef(eProjStenope);

   ELISE_ASSERT
   (
          (aTPC==eProjGrid) == (anOC.ModuleOrientationFile().IsInit()),
          "ModuleOrientationFile must be init IF and ONLY IF TypeProj equals ProjGrid"
   );

   if (aTPC == eProjGrid)
   {
           std::cout << "Chargement de : "<<anOC.ModuleOrientationFile().Val().NameFileOri()<<std::endl;
           cAffinitePlane orIntImaM2C = anOC.OrIntImaM2C().Val();
           aRes = new cCameraModuleOrientation(new OrientationGrille(anOC.ModuleOrientationFile().Val().NameFileOri()),anOC.Interne().Val().SzIm(),Xml2EL(orIntImaM2C));
           std::cout << "Fin du chargement de la grille"<<std::endl;
       return aRes;
   }
   else if (anOC.TypeProj().ValWithDef(eProjStenope) == eProjStenope)
   {
      if (anOC.Interne().IsInit())
      {
         double aRayonInv=-1;
         aCIC = anOC.Interne().Val();
         if (aCIC.RayonUtile().IsInit())
         {
               aRayonInv=aCIC.RayonUtile().Val();
         }
         CamStenope * aCS = Std_Cal_From_CIC(aCIC,aNameFile);
         aCS->SetIdentCam(aNameFile);





         if (CanUseGr && (! aCS->IsGrid()))
         {
             Pt2dr aStepGr (20,20);
             if (aCIC.ParamForGrid().IsInit())
             {
                  aRayonInv=aCIC.ParamForGrid().Val().RayonInv();
                  aStepGr =aCIC.ParamForGrid().Val().StepGrid();
             }
             aCS = cCamStenopeGrid::Alloc(aRayonInv,*aCS,aStepGr);
             aCS->SetIdentCam(aNameFile);
         }
         aRes = aCS;
      }
      else
      {
          ELISE_ASSERT(anOC.FileInterne().IsInit(),"Cam_Gen_From_XML, Interne :  ni Val ni File");
          std::string  aName = anOC.FileInterne().Val();



          if (anICNM)
          {
             string outputDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():anICNM->Dir() );
             if (anOC.RelativeNameFI().Val())
             {

                std::string aNewName = outputDirectory+aName;
                if (ELISE_fp::exist_file(aNewName))
                {
                   aName = aNewName;
                }
                else
                {
                   aNewName = outputDirectory + NameWithoutDir(aName);
                   if (ELISE_fp::exist_file(aNewName))
                   {
                         aName = aNewName;
                   }
                   else
                   {
                       aNewName = outputDirectory + aDir + NameWithoutDir(aName);
                       if (ELISE_fp::exist_file(aNewName))
                       {
                             aName = aNewName;
                       }
                       else
                       {
                           std::cout << "With dir = " <<  outputDirectory  << " and file = " << aName  << " and Dir Subst " << aDir << "\n";
                           ELISE_ASSERT(false,"Cannot get internal file");
                       }
                   }
                }
             }
             else
             {
                anICNM->StdTransfoNameFile(aName);
                if ((aDir !="") && (!ELISE_fp::exist_file(aName)) && (ELISE_fp::exist_file(aDir+ aName)))
                   aName = aDir+aName;
             }
          }
          if (theDic[aName]==0)
          {
              std::string aNameCalib = aName;
              if (!ELISE_fp::exist_file(aNameCalib))
              {
                    std::string aNameTested = aDir + aNameCalib;
                    if (ELISE_fp::exist_file(aNameTested))
                       aNameCalib = aNameTested;
              }

               aCIC =
                                   StdGetObjFromFile<cCalibrationInternConique>
                                   (
                                         aNameCalib,
                                         StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                         "CalibrationInternConique",
                                         "CalibrationInternConique"
                                   );
                cOrientationConique anOC2;
                anOC2.Interne().SetVal(aCIC);
                anOC2.TypeProj().SetVal(eProjStenope);
                anOC2.ConvOri().KnownConv().SetVal(aCIC.KnownConv().ValWithDef(eConvApero_DistM2C));
                ElCamera * aCam2  = Gen_Cam_Gen_From_XML(CanUseGr,anOC2,anICNM,"",aNameCalib);
                theDic[aName] = aCam2->CS();
          }
          if (CanUseGr)
             aRes = new cCamStenopeGen(*theDic[aName]);
          else
             aRes = theDic[aName]->Dupl();
      }
   }
   else if (anOC.TypeProj().ValWithDef(eProjStenope) == eProjOrthographique)
   {
        if (anOC.Interne().IsInit())
        {
            aCIC = anOC.Interne().Val();
            if (aCIC.CalibDistortion()[0].ModNoDist().IsInit())
            {
               aRes = cCameraOrtho::Alloc(aCIC.SzIm());
            }
            else
            {
                ELISE_ASSERT(false,"Cam ortho, dist non geree");
            }
        }
        else
        {
            ELISE_ASSERT(false,"Cam ortho, file interne non geree");
        }
   }
   else
   {
       ELISE_ASSERT(false,"Cam_Gen_From_File, proj non geree");
   }

/*
   if (aCIC.OrIntGlob().IsInit())
   {
        aRes->SetIntrImaC2M(AfGC2M(aCIC));
   }
*/

   aRes->SetScanImaM2C(AffCur(anOC));

// mCam->IsScanned() 
//    if (anOC.ZoneUtileInPixel().ValWithDef(true))
  // aRes->SetZoneUtilInPixel(anOC.ZoneUtileInPixel().ValWithDef(aRes->IsScanned()));


   eConventionsOrientation aConvEnum = eConvApero_DistM2C;
   aConvEnum = anOC.ConvOri().KnownConv().ValWithDef(aConvEnum);
   aConvEnum = anOC.Externe().KnownConv().ValWithDef(aConvEnum);
/*
*/
   cConvExplicite aConv = MakeExplicite(anOC.ConvOri());


   // inv car les camera elise ont une (putain de) convention M2C
   aRes->SetOrientation(::Std_RAff_C2M(anOC.Externe(),aConv).inv());



   aRes->SetTime(anOC.Externe().Time().ValWithDef(TIME_UNDEF()));


   if (anOC.Externe().AltiSol().IsInit())
      aRes->SetAltiSol(anOC.Externe().AltiSol().Val());

   if (anOC.Externe().Profondeur().IsInit())
      aRes->SetProfondeur(anOC.Externe().Profondeur().Val());


   if (anOC.Externe().Vitesse().IsInit())
      aRes->SetVitesse(anOC.Externe().Vitesse().Val());

   aRes->SetIncCentre(anOC.Externe().IncCentre().ValWithDef(Pt3dr(1,1,1)));

   return aRes;
}


ElCamera * Cam_Gen_From_XML (const cOrientationConique  & anOC,cInterfChantierNameManipulateur * anICNM,const std::string& aNameFile)
{
   return Gen_Cam_Gen_From_XML(false,anOC,anICNM,"",aNameFile);
}

ElCamera * Gen_Cam_Gen_From_File
           (
                  bool CanUseGr,
                  const std::string & aNameFileOri,
                  const std::string &  aNameTag,
                  cInterfChantierNameManipulateur * anICNM
           )
{
   std::string aNameFile = aNameFileOri;
   if ((! ELISE_fp::exist_file(aNameFile)) &&(anICNM!=0))  aNameFile = anICNM->Dir()  + aNameFile;

   if (StdPostfix(aNameFile)=="xml")
   {
       cOrientationConique  anOC =  StdGetObjFromFile<cOrientationConique>
                  (
                        aNameFile,
                        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                        aNameTag,
                        "OrientationConique"
                 );


       ElCamera * aRes = Gen_Cam_Gen_From_XML(CanUseGr,anOC,anICNM,DirOfFile(aNameFile),aNameFile);


       aRes->SetIdentCam(aNameFileOri);
       return aRes;
   }
   if ((StdPostfix(aNameFile)=="ori") || (StdPostfix(aNameFile)=="ORI") )
   {
       Ori3D_Std * anOri = new Ori3D_Std (aNameFile.c_str(),false,false,true);
       return new cCamera_Orilib(anOri->dos());
   }
   std::cout << "For File =[" << aNameFile << "]\n";
   ELISE_ASSERT(false,"Do not understand file in Cam_Gen_From_File");
   return 0;
}

ElCamera * Cam_Gen_From_File
           (
                  const std::string & aNameFile,
                  const std::string &  aNameTag,
                  cInterfChantierNameManipulateur * anICNM
           )
{
   return Gen_Cam_Gen_From_File(false,aNameFile,aNameTag,anICNM);
}

std::map<std::string,ElCamera *> theDicoName2Ori;

ElCamera * Cam_Gen_From_File
           (
                  const std::string & aNameFile,
                  const std::string &  aNameTag,
                  bool Memo,
                  bool CanUseGr,
                  cInterfChantierNameManipulateur * anICNM
           )
{
   if ( ! Memo)
      return Gen_Cam_Gen_From_File(CanUseGr,aNameFile,aNameTag,anICNM);

   ElCamera ** aRes = & theDicoName2Ori[aNameFile];
   if (*aRes ==0)
   {
      *aRes =  Gen_Cam_Gen_From_File(CanUseGr,aNameFile,aNameTag,anICNM);
   }

   return *aRes;
}






ElRotation3D  Std_RAff_C2M
              (
                 const cOrientationExterneRigide & aCE,
         eConventionsOrientation aConv
          )
{
   cConvExplicite  aConvOri =MakeExplicite(aConv);

   return  ELISE_ORILIB::Std_RAff_C2M(aCE,aConvOri);
}

cOrientationExterneRigide From_Std_RAff_C2M
                          (
                               const ElRotation3D & aRC2M,
                   bool aModeMatr
                          )
{
   // std::cout << "  TruuuuueRot = " << aRC2M.IsTrueRot() << "\n";
   cOrientationExterneRigide aRes;
   aRes.Centre() = aRC2M.ImAff(Pt3dr(0,0,0));
   if (aModeMatr)
   {
      cTypeCodageMatr  aCM;
/*    NE MARCHE PLUS AVEC MATRICE POSSIBLEMENT NON ORTHO
      // L1 L2 .. sont des ligne, donc transpose, donc image reciproques ...
      aCM.L1() = aRC2M.IRecVect(Pt3dr(1,0,0));
      aCM.L2() = aRC2M.IRecVect(Pt3dr(0,1,0));
      aCM.L3() = aRC2M.IRecVect(Pt3dr(0,0,1));
*/

       ElMatrix<double> aMat = aRC2M.Mat();
       aMat.GetLig(0, aCM.L1() );
       aMat.GetLig(1, aCM.L2() );
       aMat.GetLig(2, aCM.L3() );

      if (aRC2M.IsTrueRot())
      {
          aCM.TrueRot().SetNoInit();
      }
      else
      {
          aCM.TrueRot().SetVal(false);
      }
      aRes.ParamRotation().CodageMatr().SetVal(aCM);
   }
   else
   {
      double unRadInDegre = 180.0/PI;
      aRes.ParamRotation().CodageAngulaire().SetVal
      (
          Pt3dr
      (
         aRC2M.teta01() * unRadInDegre,
         -aRC2M.teta02() * unRadInDegre,
         aRC2M.teta12() * unRadInDegre
      )
      );
   }
   aRes.KnownConv().SetVal(eConvApero_DistM2C);


/*
{
   ElRotation3D aBis = Std_RAff_C2M(aRes,eConvOrilib);
   std::cout << aRC2M.ImAff(Pt3dr(0,0,0)) << aBis.ImAff(Pt3dr(0,0,0)) << "\n";
   std::cout << aRC2M.ImAff(Pt3dr(10,0,0)) << aBis.ImAff(Pt3dr(10,0,0)) << "\n";
getchar();
}
*/


   return aRes;
}

ElMatrix<double>   Std_RAff_C2M
                   (
                       const cRotationVect             & aRVect,
                       eConventionsOrientation aConv
                   )
{
    bool TrueRot;
    return ::Std_RAff_C2M(aRVect,MakeExplicite(aConv),TrueRot);
}





double TIME_UNDEF() {return -1e30;}

double  ALTISOL_UNDEF() {return 1e30;}
bool  ALTISOL_IS_DEF(double aZ) {return aZ< 1e29;}


cConvExplicite GlobMakeExplicite(eConventionsOrientation aConv)
{
   return  ELISE_ORILIB::MakeExplicite(aConv);
}
ElRotation3D  GlobStd_RAff_C2M
              (
                 const cOrientationExterneRigide & aCE,
             const cConvExplicite            & aConv
         )
{
   return ELISE_ORILIB::Std_RAff_C2M(aCE,aConv);
}
cConvExplicite GlobMakeExplicite(const cConvOri & aConv)
{
   return ELISE_ORILIB::MakeExplicite(aConv);
}
//#endif



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
