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
#include "all_phot.h"

#define EL_ORILIB_NSP ELISE_ORILIB::



//#ifndef __USE_ORILIB_ELISE_
//#else

/********************************************************/
/********************************************************/
/***                                                  ***/
/***              Data_Ori3D_Gen                      ***/
/***                                                  ***/
/********************************************************/
/********************************************************/

Data_Ori3D_Gen::Data_Ori3D_Gen() {}


/********************************************************/
/********************************************************/
/***                                                  ***/
/***              Data_Ori3D_Std                      ***/
/***                                                  ***/
/********************************************************/
/********************************************************/

void Data_Ori3D_Std::init_commun(ELISE_ORILIB::or_orientation * anOri)
{
   if (anOri==0)
   {
      _ori =  NEW_ONE(ELISE_ORILIB::or_orientation);
      mOri2Del = true;
   }
   else
   {
      _ori =  anOri;
      mOri2Del = false;
   }
   _photo = (void*) _ori;
   _inv_y = false;
}

Data_Ori3D_Std::Data_Ori3D_Std() :
       Data_Ori3D_Gen()
{
   init_commun();
}

Data_Ori3D_Std::Data_Ori3D_Std(ELISE_ORILIB::or_orientation     * anOri)  :
       Data_Ori3D_Gen()
{
   init_commun(anOri);
}

Data_Ori3D_Std::Data_Ori3D_Std(const char *name,bool inv_y,bool binary,bool QuickGrid) :
       Data_Ori3D_Gen()
{
   init_commun();
   if (binary)
      Tjs_El_User.ElAssert
      (
          ELISE_ORILIB::orlit_orientation(name,&_photo)!=0,
          EEM0 << "Cannot read orientation-file " << name
      );
   else
      Tjs_El_User.ElAssert
      (
          ELISE_ORILIB::orlit_orientation_texte(name,&_photo,QuickGrid)!=0,
          EEM0 << "Cannot read orientation-file " << name
      );

   _inv_y = inv_y;
}

Pt2dr Data_Ori3D_Std::OrigineTgtLoc() const
{
   return Pt2dr(_ori->origine[0],_ori->origine[1]);
}

Pt2dr Ori3D_Std::OrigineTgtLoc() const
{
   return dos()->OrigineTgtLoc();
}

Data_Ori3D_Std::Data_Ori3D_Std(Data_Ori3D_Std * orin,REAL zoom,Box2dr aBox)
{
   init_commun();
   if (aBox.surf() ==0)
   {
      aBox._p0 = Pt2dr(0,0);
      aBox._p1 = Pt2dr( orin->_ori->ins,orin->_ori->inl);
   }
   REAL step = 1/zoom;
   Tjs_El_User.ElAssert
   (
       ELISE_ORILIB::orfenetrage_orientation
       (
            &(orin->_photo),
            &aBox._p0.x,
            &aBox._p0.y,
            &aBox._p1.x,
            &aBox._p1.y,
            &step,
            &step,
            &(_photo)
       ) !=0,
       EEM0 << "incosistent call to orfenetrage_orientation"
   );
}



Pt2dr Data_Ori3D_Std::to_photo(Pt3dr p,bool DontUseDist)
{
   Pt2dr r;
   _ori->mDontUseDist = DontUseDist;
   if (!ELISE_ORILIB::orterrain_to_photo(&_photo,&(p.x),&(p.y),&(p.z),&(r.x),&(r.y)))
   {
       Tjs_El_User.ElAssert(0,EEM0<<"Bad status on  to_photo conversion");
       
   }
   _ori->mDontUseDist = false;
   correct(r);
   return r;
}

Pt3dr Data_Ori3D_Std::to_terrain
     (
            Pt2dr pp1,
            Data_Ori3D_Std & ph2,
            Pt2dr pp2,
            REAL & dist
     )
{
      correct(pp1);
      correct(pp2);

      Pt3dr  pt;
      dist =  ELISE_ORILIB::orphotos_to_terrain
              (   
                      &_photo,&(pp1.x),&(pp1.y),
                 &ph2._photo,&(pp2.x),&(pp2.y),
                  &(pt.x),&(pt.y),&(pt.z)
              );
      return pt;
}

Pt3dr  Data_Ori3D_Std::ImEtProf_To_Terrain(Pt2dr pp,REAL z)
{
   Pt3dr aRes;
   correct(pp);
   if (! ELISE_ORILIB::orphoto_et_prof_to_terrain
           (
               &(_photo),
               &(pp.x),&(pp.y),&z,
               &(aRes.x),&(aRes.y),&(aRes.z)
           )
        )
   {
        Tjs_El_User.ElAssert
        (
            0,
            EEM0 << "Bad status on ImEtProf_To_Terrain"
        );
   }

   return aRes;
}

Pt3dr Data_Ori3D_Std::Im1etProf2_To_Terrain
      (
          Pt2dr pp1,
          Data_Ori3D_Std * ph2,
          double prof2
       )
{
  Pt3dr aRes;
  correct(pp1);
  if ( !ELISE_ORILIB::orphoto1_et_prof2_to_terrain
        (
              &(_photo),
              &(pp1.x),&(pp1.y),
              &(ph2->_photo),
              &prof2,
              &(aRes.x),&(aRes.y),&(aRes.z)
        )
     )
  {
        Tjs_El_User.ElAssert
        (
            0,
            EEM0 << "Bad status on Im1etProf2_To_Terrain"
        );
  }
  return aRes;
}

double Data_Ori3D_Std::Prof(const Pt3dr & aP)
{
    double aRes;
    
  if ( !ELISE_ORILIB::or_prof(&_photo,&aP.x,&aP.y,&aP.z,&aRes))
  {
        Tjs_El_User.ElAssert
        (
            0,
            EEM0 << "Bad status on Data_Ori3D_Std::Prof"
        );
  }
  return aRes;
}

double Data_Ori3D_Std::ProfInDir(const Pt3dr & aP,const Pt3dr & aDir) 
{
    return scal(aP-orsommet_de_pdv_terrain (),aDir);
}


Pt3dr Data_Ori3D_Std::to_terrain(Pt2dr pp,REAL z)
{
     Pt3dr r;

     correct(pp);
     if (! ELISE_ORILIB::orphoto_et_z_to_terrain
           (
               &(_photo),
               &(pp.x),&(pp.y),&z,
               &(r.x),&(r.y)
           )
        )
     {
        Tjs_El_User.ElAssert
        (
            0,
            EEM0 << "Bad status on photo_et_z_to_terrain"
        );
     }

      r.z = z;
      return r;
}


/*
Facette_2D Data_Ori3D_Std::carte_et_z_to_terrain(Facette_2D fcarte,REAL z_lamb,REAL & z_ter)
{
   {
        Pt3dr p0t =  carte_to_terr(Pt3dr(fcarte[0],z_lamb));
        z_ter = p0t.z;
   }

   INT nb = fcarte.nb();

   Facette_2D fter (nb);
   for(INT i=0; i<nb; i++)
   {
      Pt3dr pt = carte_to_terr(Pt3dr(fcarte[i],z_lamb));
      fter.push(Pt2dr(pt.x,pt.y));
   }
   return fter;
}
*/

Pt3dr Data_Ori3D_Std::carte_to_terr(Pt3dr pc)
{
      Pt3dr pt;

      if (! ELISE_ORILIB::orcarte_to_terrain
           (
               &(_photo),
               &(pc.x),&(pc.y),&(pc.z),
               &(pt.x),&(pt.y),&(pt.z)
           )
        )
     {
        Tjs_El_User.ElAssert
        (
            0,
            EEM0 << "Bad status on orcarte_to_terrain"
        );
     }

     return pt;
}


Data_Ori3D_Std::~Data_Ori3D_Std()
{
    if (mOri2Del)
        DELETE_ONE(_ori);
}


void Data_Ori3D_Std::correct
     (
          REAL * xp,
          REAL * yp,
          INT    nb
     )
{
    if (_inv_y)
    {
        INT tx = _ori->ins-1;
        INT ty = _ori->inl-1;

        for (  ;  nb>0  ;  nb--,xp++,yp++)
        {
             *xp = tx - *xp;
             *yp = ty - *yp;
        }
    }
}

void   Data_Ori3D_Std::to_photo
       (
                REAL *x_ph,
                REAL *y_ph,
                const REAL *x_ter,
                const REAL *y_ter,
                const REAL *z_ter,
                INT nb
      )
{
       INT nb0 = nb;

       for (; nb; nb--)
       {
           if (!  ELISE_ORILIB::orterrain_to_photo
                  (
                      &_photo,
                      x_ter++,
                      y_ter++,
                      z_ter++,
                      x_ph++,
                      y_ph++
                  )
              )
              {
                 Tjs_El_User.ElAssert
                 (
                    0,
                    EEM0 << "Bad status in fterrain_to_photo"
                 );
              }
       }
       correct(x_ph-nb0,y_ph-nb0,nb0);
}

void   Data_Ori3D_Std::photo_et_z_to_terrain
       (
                     REAL *xt,
                     REAL *yt,
                     REAL *x_tmp,
                     REAL *y_tmp,
                     const REAL *xp,
                     const REAL *yp,
                     const REAL * zt,
                     INT nb
      )
{
       convert(x_tmp,xp,nb);
       convert(y_tmp,yp,nb);

       correct(x_tmp,y_tmp,nb);

       for (; nb; nb--)
       {
           if (!  ELISE_ORILIB::orphoto_et_z_to_terrain
                  (
                      &_photo,
                      x_tmp++,
                      y_tmp++,
                      zt++,
                      xt++,
                      yt++
                  )
              )
              {
                 Tjs_El_User.ElAssert
                 (
                    0,
                    EEM0 << "Bad status in photo_et_z_to_terrain"
                 );
              }
       }
}


void  Data_Ori3D_Std::ororient_epipolaires
      (
                   Data_Ori3D_Std * & epi1, Data_Ori3D_Std & phot1,Pt2dr aP1,
                   Data_Ori3D_Std * & epi2, Data_Ori3D_Std & phot2,Pt2dr aP2
      )
{

    REAL x0 = 0.0;
    REAL y0 = 0.0;
    REAL x1 = phot1._ori->ins;
    REAL y1 = phot1._ori->inl;

    REAL aDist;

    Pt3dr aPTer   = phot1.to_terrain(aP1,phot2,aP2,aDist);
    Pt3dr aPCarte = phot1.carte_to_terr(aPTer)  ;



    REAL aZMin = aPCarte.z-1.0;
    REAL aZMax = aPCarte.z+1.0;


    epi1 = new Data_Ori3D_Std;
    epi2 = new Data_Ori3D_Std;

    INT aTx,aTy;

    ELISE_ORILIB::ororient_epipolaires
    (
             &(phot1._photo), &(phot2._photo),
             &x0,&y0,&x1,&y1,
             &aZMin,&aZMax,
             &(epi1->_photo), &(epi2->_photo),
             &aTx,&aTy
    );

}
REAL Data_Ori3D_Std::altisol() 
{
    return _ori->altisol;
}

REAL Data_Ori3D_Std::profondeur() 
{
    return _ori->mProf;
}

NS_ParamChantierPhotogram::cOrientationConique * Data_Ori3D_Std::OC() const
{
   return _ori->mOC;
}

void  Data_Ori3D_Std::SetAltiSol(REAL aZ)
{
   _ori->altisol = aZ;
}

inline void FromTab(Pt3dr & aP,double * aTab)
{
    aP.x = aTab[0];
    aP.y = aTab[1];
    aP.z = aTab[2];
}

void Data_Ori3D_Std::orrepere_3D_image(Pt3dr & aP0,Pt3dr & anU,Pt3dr & aV,Pt3dr * aW)
{
   double           aTabP0[3],aTabU[3],aTabV[3];

   ELISE_ORILIB::orrepere_3D_image (&(_photo), aTabP0, aTabU, aTabV ) ;

   FromTab(aP0,aTabP0);
   FromTab(anU,aTabU);
   FromTab(aV,aTabV);
 
   if (aW)
   {
      *aW = anU ^ aV;
      *aW = *aW / euclid(*aW);
   }
}

void Data_Ori3D_Std::SetOrientation(const ElRotation3D & aR)
{
    Pt3dr aTr = aR.tr();
    Pt3dr aImI = aR.ImVect(Pt3dr(1,0,0));
    Pt3dr aImJ = aR.ImVect(Pt3dr(0,1,0));
    Pt3dr aImK = aR.ImVect(Pt3dr(0,0,1));
    ELISE_ORILIB::orSetSommet(&(_photo),& aTr.x,& aTr.y,& aTr.z);
    ELISE_ORILIB::orSetDirI  (&(_photo),&aImI.x,&aImI.y,&aImI.z);
    ELISE_ORILIB::orSetDirJ  (&(_photo),&aImJ.x,&aImJ.y,&aImJ.z);
    ELISE_ORILIB::orSetDirK  (&(_photo),&aImK.x,&aImK.y,&aImK.z);
}

ElRotation3D  Data_Ori3D_Std::GetOrientation() const
{
    Data_Ori3D_Std * aDos = const_cast<Data_Ori3D_Std *>(this);
    return  ElRotation3D 
            (
                aDos->orsommet_de_pdv_terrain(),
                MatFromCol( aDos->orDirI(),aDos->orDirJ(),aDos->orDirK()),
                true
            );
}







Pt3dr Data_Ori3D_Std::orsommet_de_pdv_terrain ()
{
   Pt3dr aP;
   ELISE_ORILIB::orsommet_de_pdv_terrain (&(_photo),&(aP.x),&(aP.y),&(aP.z));
   return aP;
}

Pt3dr Data_Ori3D_Std::orDirI()
{
     Pt3dr aI;
     ELISE_ORILIB::orDirI(&(_photo),&(aI.x),&(aI.y),&(aI.z));
     return aI;
}

Pt3dr Data_Ori3D_Std::orDirJ()
{
     Pt3dr aJ;
     ELISE_ORILIB::orDirJ(&(_photo),&(aJ.x),&(aJ.y),&(aJ.z));
     return aJ;
}

Pt3dr Data_Ori3D_Std::orDirK()
{
     Pt3dr aK;
     ELISE_ORILIB::orDirK(&(_photo),&(aK.x),&(aK.y),&(aK.z));
     return aK;
}


Pt2dr Data_Ori3D_Std::orDirLoc_to_photo(Pt3dr aDirRay)
{
  Pt2dr aPh;
  double TDR[3];

  TDR[0]= aDirRay.x;
  TDR[1]= aDirRay.y;
  TDR[2]= aDirRay.z;
  ELISE_ORILIB::orDirLoc_to_photo(&(_photo),TDR,&aPh.x,&aPh.y);

  return aPh;
}

Pt3dr  Data_Ori3D_Std::orPhoto_to_DirLoc(Pt2dr aPh)
{
    double TDR[3];

    ELISE_ORILIB::orPhoto_to_DirLoc(&(_photo),&aPh.x,&aPh.y,TDR);
    Pt3dr aRay;
    FromTab(aRay,TDR);

    return aRay;
}


Pt3dr  Data_Ori3D_Std::orSM(Pt2dr aP)
{
    double aTabRay[3];

    ELISE_ORILIB::orSM(&(_photo),&(aP.x),&(aP.y),aTabRay);
    return Pt3dr(aTabRay[0],aTabRay[1],aTabRay[2]);
}

INT Data_Ori3D_Std::ZoneLambert () const
{
   return _ori->lambert;
}

Pt2di Data_Ori3D_Std::SzIm() const
{
   return Pt2di(_ori->ins,_ori->inl);
}

Pt3dr  Data_Ori3D_Std::ImDirEtProf2Terrain(const Pt2dr & aPIm,const REAL & aProf,const Pt3dr & aNormPl) 
{
    Pt3dr  aRay =  orSM(aPIm);

    double aLamda = aProf / scal(aRay,aNormPl);

    return    orsommet_de_pdv_terrain () +  aRay * aLamda;
}


Pt3dr Data_Ori3D_Std::Im1DirEtProf2_To_Terrain
                  (Pt2dr aPIm,Data_Ori3D_Std *  ph2,double prof2,const Pt3dr & aDir) 
{

   Pt3dr  aRay =  orSM(aPIm);
   Pt3dr  aC1 = orsommet_de_pdv_terrain ();
   Pt3dr  aC2 = ph2->orsommet_de_pdv_terrain ();

   double aLamda =  (prof2+scal(aC2-aC1,aDir))/scal(aRay,aDir);

   return aC1 + aRay * aLamda;
}

/********************************************************/
/********************************************************/
/***                                                  ***/
/***              Ori3D_Gen                           ***/
/***                                                  ***/
/********************************************************/
/********************************************************/

Ori3D_Gen::Ori3D_Gen(class Data_Ori3D_Gen * o3g) :
    PRC0(o3g)
{
}

/********************************************************/
/********************************************************/
/***                                                  ***/
/***              Ori3D_Std                           ***/
/***                                                  ***/
/********************************************************/
/********************************************************/

Box2dr Ori3D_Std::TheNoBox(Pt2dr(0,0),Pt2dr(0,0));

Ori3D_Std::Ori3D_Std(const char * name,bool inv_y,bool binary,bool QuickGrid) :
     Ori3D_Gen ( new Data_Ori3D_Std(name,inv_y,binary,QuickGrid))
{
// // // // std::cout << "KKKKKKKKKKKKKKKKKKKKK\n";
}

Ori3D_Std::Ori3D_Std(Ori3D_Std orin,REAL zoom,Box2dr aBox)  :
     Ori3D_Gen ( new Data_Ori3D_Std(orin.dos(),zoom,aBox))
{
// std::cout << "LLLLLLLLLLLLLLLLLLLLL\n";
}

Ori3D_Std::Ori3D_Std(ELISE_ORILIB::or_orientation * anOri) :
   Ori3D_Gen (new Data_Ori3D_Std(anOri))
{
}

/*
Facette_2D Ori3D_Std::carte_et_z_to_terrain(Facette_2D fcarte,REAL z_lamb,REAL & z_ter)
{
    return dos()->carte_et_z_to_terrain(fcarte,z_lamb,z_ter);
}
*/

Pt3dr Ori3D_Std::orDirI() const
{
   return dos()->orDirI();
}
Pt3dr Ori3D_Std::orDirJ() const
{
   return dos()->orDirJ();
}
Pt3dr Ori3D_Std::orDirK() const
{
   return dos()->orDirK();
}


Pt3dr Ori3D_Std::Im1DirEtProf2_To_Terrain
                  (Pt2dr aPIm,Ori3D_Std   ph2,double prof2,const Pt3dr & aDir)  const
{
   return dos()->Im1DirEtProf2_To_Terrain(aPIm,ph2.dos(),prof2,aDir);
}


Pt2dr Ori3D_Std::to_photo(Pt3dr p,bool DontUseDist) const
{
      return dos()->to_photo(p,DontUseDist);
}

Pt3dr Ori3D_Std::to_terrain
      ( Pt2dr p1,Ori3D_Std ph2,Pt2dr p2,REAL & dist)
{
      return dos()->to_terrain(p1,*(ph2.dos()),p2,dist);
}

Pt3dr  Ori3D_Std::ImDirEtProf2Terrain(const Pt2dr & aPIm,const REAL & aProf,const Pt3dr & aNormPl)  const
{
   return dos()->ImDirEtProf2Terrain(aPIm,aProf,aNormPl);
}

Pt3dr Ori3D_Std::to_terrain(Pt2dr p1,REAL z) const
{
      return dos()->to_terrain(p1,z);
}

Pt3dr Ori3D_Std::ImEtProf_To_Terrain(Pt2dr pp,REAL z) const
{
   return dos()->ImEtProf_To_Terrain(pp,z);
}

Pt3dr  Ori3D_Std::Im1etProf2_To_Terrain
               (Pt2dr p1,Ori3D_Std  ph2,double prof2) const
{
   return dos()->Im1etProf2_To_Terrain(p1,ph2.dos(),prof2);
}

Pt3dr Ori3D_Std::carte_to_terr(Pt3dr pc) const
{
      return dos()->carte_to_terr(pc);
}

void Ori3D_Std::write_txt(const char * name)
{
      Tjs_El_User.ElAssert
      (
          ELISE_ORILIB::orecrit_orientation_texte(&(dos()->_photo),name) !=0,
          EEM0 << "Cannot write orientation-file " << name
      );

}

void Ori3D_Std::write_bin(const char * name)
{
      Tjs_El_User.ElAssert
      (
          ELISE_ORILIB::orecrit_orientation(&(dos()->_photo),name) !=0,
          EEM0 << "Cannot write orientation-file " << name
      );

}

void  Ori3D_Std::SetAltiSol(REAL aZ)
{
   dos()->SetAltiSol(aZ);
}

NS_ParamChantierPhotogram::cOrientationConique * Ori3D_Std::OC() const
{
   return dos()->OC();
}


REAL Ori3D_Std::FocaleAngulaire() const    //HJ
{
      return dos()->_ori->focale / dos()->_ori->pix[0];
}

REAL Ori3D_Std::resolution() const    //HJ
{
      REAL focale = dos()->_ori->focale;
      REAL altisol = dos()->_ori->altisol;
      REAL altisommet = dos()->_ori->sommet[2];
      REAL taille_pix = dos()->_ori->pix[0];

      return (altisommet - altisol) / focale * taille_pix;
}

REAL Ori3D_Std::resolution_angulaire() const    //HJ
{
      REAL focale = dos()->_ori->focale;
      REAL taille_pix = dos()->_ori->pix[0];

      return taille_pix / focale ;
}

double Ori3D_Std::ProfInDir(const Pt3dr & aP,const Pt3dr & aDir)  const
{
   return dos()->ProfInDir(aP,aDir);
}




REAL Ori3D_Std::altisol() const
{
    return dos()->_ori->altisol;
}
REAL Ori3D_Std::profondeur() const
{
    return dos()->profondeur();
}


Pt3dr Ori3D_Std::orsommet_de_pdv_terrain() const
{
	return dos()->orsommet_de_pdv_terrain();
}

double Ori3D_Std::Prof(const Pt3dr & aP) const
{
    return dos()->Prof(aP);
}

REAL Ori3D_Std::BSurH(Ori3D_Std Ori2)
{
     REAL H = ElAbs(orsommet_de_pdv_terrain().z-altisol());
     REAL B = euclid(orsommet_de_pdv_terrain()-Ori2.orsommet_de_pdv_terrain());

     return B/H;
}

void Ori3D_Std::UnCpleHom(Ori3D_Std Ori2,Pt2dr & pH1,Pt2dr & pH2)
{
      Pt3dr P1 = orsommet_de_pdv_terrain();
      Pt3dr P2 = Ori2.orsommet_de_pdv_terrain();

      Pt3dr P = (P1+P2)/2.0;
      P.z = altisol();
      pH1  = to_photo(P);
      pH2  = Ori2.to_photo(P);
}

INT  Ori3D_Std::ZoneLambert () const
{
   return dos()->ZoneLambert();
}



/*************************/
/*       AJOUT  HJMPD    */
/*************************/

Pt3dr Data_Ori3D_Std::terr_to_carte(Pt3dr pc)  
{
      Pt3dr pt;

      if (! ELISE_ORILIB::orterrain_to_carte
           (
               &(_photo),
               &(pc.x),&(pc.y),&(pc.z),
               &(pt.x),&(pt.y),&(pt.z)
           )
        )
     {
        Tjs_El_User.ElAssert
        (
            0,
            EEM0 << "Bad status on orterrain_to_carte"
        );
     }

     return pt;
}

Pt3dr Ori3D_Std::terr_to_carte(Pt3dr pc) const
{
      return dos()->terr_to_carte(pc);
}


void  Data_Ori3D_Std::SetOrigineTgtLoc(const Pt2dr & aP)
{
     Pt3dr aPT1 = orsommet_de_pdv_terrain ();
// std::cout << "AAAAAA " << aP << "PT1 " << aPT1 << "\n";
     Pt3dr aPC = terr_to_carte(aPT1);
     _ori->origine[0] = aP.x;
     _ori->origine[1] = aP.y;
     Pt3dr aPT2 = carte_to_terr(aPC);
// std::cout << "PT2 " << aPT2 << "\n";
     _ori->sommet[0] = aPT2.x;
     _ori->sommet[1] = aPT2.y;
     _ori->sommet[2] = aPT2.z;
}

Pt2dr  Data_Ori3D_Std::PP() const
{
   // return  Pt2dr(_ori->ipp[0]/_ori->pix[0],_ori->ipp[1]/_ori->pix[1]);
   return  Pt2dr(_ori->ipp[0],_ori->ipp[1]);
}
double Data_Ori3D_Std::FocaleAngulaire() const
{
   return _ori->focale / _ori->pix[0];
}
/*
Pt3dr Ori3D_Std::carte_to_terr(Pt3dr pc)  
{
      return dos()->carte_to_terr(pc);
}
*/

/**************************************************/
/*                                                */
/*       cOri3D_OneEpip                           */
/*                                                */
/**************************************************/


cOri3D_OneEpip::cOri3D_OneEpip ( Data_Ori3D_Std  * aPh, Data_Ori3D_Std  *anEpi) :
    mPhot   (aPh),
    mEpi    (anEpi),
    mTer2Epi (3,3)
{
   mEpi->orrepere_3D_image(mP0Epi,mUEpi,mVEpi,&mWEpi);

   SetCol(mTer2Epi,0,mUEpi);
   SetCol(mTer2Epi,1,mVEpi);
   SetCol(mTer2Epi,2,mWEpi);
   self_gaussj(mTer2Epi);

   mCPDV = mTer2Epi * (mEpi->orsommet_de_pdv_terrain()-mP0Epi);
}

cOri3D_OneEpip::~cOri3D_OneEpip()
{
   delete mPhot;
   delete mEpi;
}


bool  cOri3D_OneEpip::OwnInverse(Pt2dr & aP) const
{

   aP = mPhot->to_photo( mP0Epi + mUEpi *aP.x + mVEpi * aP.y);
   return true;
}

Pt2dr cOri3D_OneEpip::Direct(Pt2dr aP) const 
{
    Pt3dr aRay = mTer2Epi * mPhot->orSM(aP);
    REAL aLambda = - (mCPDV.z / aRay.z);

    return Pt2dr(mCPDV.x+aLambda * aRay.x,mCPDV.y+aLambda * aRay.y);
}

void  cOri3D_OneEpip::Diff(ElMatrix<REAL> &,Pt2dr) const 
{
   ELISE_ASSERT(false,"No  cOri3D_OneEpip");
}

cOri3D_OneEpip cOri3D_OneEpip::MapingChScale(REAL aChSacle) const
{
   return cOri3D_OneEpip
          (
               new Data_Ori3D_Std(mPhot,aChSacle),
               new Data_Ori3D_Std(mEpi,aChSacle)
          );
}

ElDistortion22_Gen  * cOri3D_OneEpip::D22G_ChScale(REAL aS) const
{
    return new cOri3D_OneEpip(MapingChScale(aS));
}


void cOri3D_OneEpip::SetPhot0() { mPhot=0; }
void cOri3D_OneEpip::SetEpi0() { mEpi=0; }

Data_Ori3D_Std  * cOri3D_OneEpip::Phot() {return mPhot;}
Data_Ori3D_Std  * cOri3D_OneEpip::Epip() {return mEpi;}



/********************************************************/
/********************************************************/
/***                                                  ***/
/***              cDistorsionOrilib                   ***/
/***                                                  ***/
/********************************************************/
/********************************************************/


class cDistorsionOrilib : public ElDistortion22_Gen
{
    public :
       cDistorsionOrilib(Data_Ori3D_Std * anOri) : 
             mOri(anOri) 
       {
       }



    private :
        Data_Ori3D_Std * mOri;

        void  Diff(ElMatrix<REAL> &,Pt2dr) const
        {
              ELISE_ASSERT(false,"No cDistorsionOrilib::Diff");
        }
        
        Pt2dr Direct(Pt2dr aP) const
        {
            mOri->Ori()->CorrigeDist_T2P(&aP.x,&aP.y);
            return aP;
            // return mOri->orDirLoc_to_photo(Pt3dr(aP.x,aP.y,1.0));
        }

        
        bool OwnInverse(Pt2dr & aP) const
        {
            mOri->Ori()->CorrigeDist_P2T(&aP.x,&aP.y);
            return true;
/*
            Pt3dr aRay =  mOri->orPhoto_to_DirLoc(aP);
            aP.x = aRay.x / aRay.z;
            aP.y = aRay.y / aRay.z;
            return true;
*/
        }
};

/********************************************************/
/********************************************************/
/***                                                  ***/
/***              cCamera_Orilib                      ***/
/***                                                  ***/
/********************************************************/
/********************************************************/

tParamAFocal aNoPAF;

cCamera_Orilib::cCamera_Orilib
(
    Data_Ori3D_Std * anOri
) :
    // OO CamStenope(false,1.0,Pt2dr(0,0)),
    CamStenope(false,anOri->FocaleAngulaire(),anOri->PP(),aNoPAF),
    mOri       (anOri->Ori()),
    mDist (new cDistorsionOrilib (anOri))
{
    ElRotation3D aRot
                 (
                     anOri->orsommet_de_pdv_terrain(),
                     MatFromCol
                     ( 
                           anOri->orDirI(),
                           anOri->orDirJ(),
                           anOri->orDirK()
                      ),
                      true
                 );
    SetSz(anOri->SzIm());
    SetOrientation(aRot.inv());
    SetAltiSol(anOri->altisol());

    // std::cout << "ProfORI " << mProfondeur <<  I << "\n";
}

Ori3D_Std * cCamera_Orilib::CastOliLib()
{
   return & mOri;
}

ElDistortion22_Gen   &  cCamera_Orilib::Dist()
{
    return *mDist;
}

const ElDistortion22_Gen   &  cCamera_Orilib::Dist() const
{
    return *mDist;
}



double TestOrientationOri(Pt2dr aP,Ori3D_Std anOri)
{
   Pt2dr aQ0 = anOri.to_photo(Pt3dr(aP.x,aP.y,0));
   Pt2dr aQX = anOri.to_photo(Pt3dr(aP.x+1,aP.y,0));
   Pt2dr aQY = anOri.to_photo(Pt3dr(aP.x,aP.y+1,0));

   return (aQX-aQ0) ^ (aQY-aQ0);

}


Pt2dr Ori3D_Std::DistT2P(const Pt2dr & aP) const
{
    Pt2dr aRes = aP;
    if (dos()->Ori()->distor != 0)
       orcorrige_distortion(&aRes.x,&aRes.y,&(dos()->Ori()->gt2p));
    return aRes;
}

Pt2dr Ori3D_Std::DistP2T(const Pt2dr & aP) const
{
    Pt2dr aRes = aP;
    if (dos()->Ori()->distor != 0)
       orcorrige_distortion(&aRes.x,&aRes.y,&(dos()->Ori()->gp2t));
    return aRes;
}

ElRotation3D   Ori3D_Std::GetOrientation() const
{
   return  dos()->GetOrientation();
}

void Ori3D_Std::SetOrientation(const ElRotation3D & aR)
{
    dos()->SetOrientation(aR);
}

void  Ori3D_Std::SetOrigineTgtLoc(const Pt2dr & aPt)
{
    dos()->SetOrigineTgtLoc(aPt);
}

Pt2di Ori3D_Std::SzIm() const
{
   return dos()->SzIm();
}

Pt2di Ori3D_Std::P0() const
{
   return Pt2di(0,0);
}

std::vector<Pt2dr> Ori3D_Std::EmpriseSol(double aZ) const
{
   std::vector<Pt2dr> aRes;

   Box2di aBox(P0(),P0()+SzIm());
   Pt2di aC[4];
   aBox.Corners(aC);
   for (int aK=0 ; aK<4 ; aK++)
   {
        Pt3dr aP = to_terrain(Pt2dr(aC[aK]),aZ);
        aRes.push_back(Pt2dr(aP.x,aP.y));
   }
   return aRes;
}

std::vector<Pt2dr> Ori3D_Std::EmpriseSol() const
{
   return EmpriseSol(altisol());
}


std::vector<Pt2dr> Ori3D_Std::Emprise2InVue1(Ori3D_Std Ori2,double aZP,bool ByAlti)
{
   std::vector<Pt2dr> aRes;

   Box2di aBox(Ori2.P0(),Ori2.P0()+Ori2.SzIm());
   Pt2di aC[4];
   aBox.Corners(aC);
   for (int aK=0 ; aK<4 ; aK++)
   {
        Pt3dr aP =  ByAlti  ?
	            Ori2.to_terrain(Pt2dr(aC[aK]),aZP) :
	            Ori2.ImEtProf_To_Terrain(Pt2dr(aC[aK]),aZP) ;
        aRes.push_back(to_photo(aP));
   }
   return aRes;
}


typedef  const std::list<std::vector<Pt2dr> > tLVP;

std::vector<Pt2dr> Ori3D_Std::Inter(Ori3D_Std Ori2,double aZP,bool ByAlti)
{
   cElPolygone aP1; 
   cElPolygone aP2;

   aP1.AddContour(Emprise2InVue1(*this,aZP,ByAlti),false);
   aP2.AddContour(Emprise2InVue1(Ori2 ,aZP,ByAlti),false);

   cElPolygone anInter = aP1 * aP2;

   tLVP & aCont = anInter.Contours() ;

   std::vector<Pt2dr> aRes;
   double aSurfMax = 0;

   for (tLVP::const_iterator itV=aCont.begin(); itV!=aCont.end() ; itV++)
   {
       double aS = ElAbs(surf_or_poly(*itV));
       if (aS > aSurfMax)
       {
           aSurfMax = aS;
	   aRes = *itV;
       }
   }
   
   return aRes;
}
  

std::vector<Pt2dr>  Ori3D_Std::Inter(Ori3D_Std Ori2)
{
   return Inter(Ori2,Ori2.altisol(),true);
}


double     Ori3D_Std::PropInter(Ori3D_Std Ori2,double aZP,bool ByAlti)
{
     std::vector<Pt2dr>  anInter = Inter(Ori2,aZP,ByAlti);

     return ElAbs(surf_or_poly(anInter)) / (SzIm().x*SzIm().y);
}


double     Ori3D_Std::PropInter(Ori3D_Std Ori2)
{
   return PropInter(Ori2,Ori2.altisol(),true);
}


void Ori3D_Std::SetUseDist(bool UseDist) const
{
   dos()->_ori->mDontUseDist = !UseDist;
}

//#endif











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
