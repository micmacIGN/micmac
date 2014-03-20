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



#ifndef _ELISE_PRIVATE_ORILIB_H
#define _ELISE_PRIVATE_ORILIB_H


class cOri3D_OneEpip;

class Data_Ori3D_Gen : public RC_Object
{
      public :
         Data_Ori3D_Gen ();
         virtual Pt2dr to_photo(Pt3dr,bool DontUseDist=false) = 0;


         virtual void  to_photo
                 (
                     REAL *xp,
                     REAL *yp,
                     const REAL *xt,
                     const REAL *yt,
                     const REAL * zt,
                     INT nb
                 )=0;
      private :
};


struct or_orientation;


class Data_Ori3D_Std : public Data_Ori3D_Gen
{
      friend class Ori3D_Std;
      friend class cOri3D_OneEpip ;
      friend class Fnum_O3d_phot_et_z_to_terrain;
      public :

         void SetOrientation(const ElRotation3D &);
         ElRotation3D  GetOrientation() const;

         typedef Data_Ori3D_Std * tPtr;


         virtual ~Data_Ori3D_Std ();
         void  to_photo
               (
                     REAL *xp,
                     REAL *yp,
                     const REAL *xt,
                     const REAL *yt,
                     const REAL * zt,
                     INT nb
               );

         void  photo_et_z_to_terrain
               (
                     REAL *xt,
                     REAL *yt,
                     REAL *x_tmp,
                     REAL *y_tmp,
                     const REAL *xp,
                     const REAL *yp,
                     const REAL * zt,
                     INT nb
               );

         Pt2dr  PP() const;
         double  FocaleAngulaire() const;

         Pt3dr orsommet_de_pdv_terrain ();
         // Direction du triedre camera
         Pt3dr orDirI();
         Pt3dr orDirJ();
         Pt3dr orDirK();
         Pt2dr OrigineTgtLoc() const;

         //  Transforme une direction de rayon dans le repere
         // locale IJK en un point photo
         Pt2dr orDirLoc_to_photo(Pt3dr);

         //  Transforme   un point photo en 
         // une direction de rayon dans le repere locale IJK 
         Pt3dr orPhoto_to_DirLoc(Pt2dr);


	 REAL altisol();
	 REAL profondeur();
         void SetAltiSol(REAL aZ);

         Pt3dr orSM(Pt2dr aP); // renvoie la dierction du rayon perspectif dans le repere terrain
         // W est optionnel, calcule comme produit vectoriel
         void orrepere_3D_image(Pt3dr & aP0,Pt3dr & anU,Pt3dr & aV,Pt3dr * aW);

         void  correct(REAL *xp,REAL *yp,INT nb);

	 Data_Ori3D_Std(or_orientation    * _ori);

         Data_Ori3D_Std (const char *,bool inv_y,bool binary,bool QuickGrid );
         Data_Ori3D_Std (Data_Ori3D_Std *,REAL zoom,Box2dr = Ori3D_Std::TheNoBox);
         void init_commun( or_orientation * = 0);

         Pt2dr to_photo(Pt3dr p,bool DontUseDist=false);
         Pt3dr to_terrain
               (
                   Pt2dr p1,
                   Data_Ori3D_Std & ph2,Pt2dr p2,
                   REAL & dist
               );
         Pt3dr to_terrain(Pt2dr p1,REAL z);
         Pt3dr carte_to_terr(Pt3dr);
         Pt3dr terr_to_carte(Pt3dr);    //   HJMPD

         // Renvoie un point se projetant en PIm et siuter a une profondeur de aProf
	 // dans la direction aNormPl
	 Pt3dr  ImDirEtProf2Terrain(const Pt2dr & aPIm,const REAL & aProf,const Pt3dr & aNormPl);

         Pt3dr ImEtProf_To_Terrain(Pt2dr p1,REAL prof);
         Pt3dr Im1etProf2_To_Terrain
               (Pt2dr p1,Data_Ori3D_Std * ph2,double prof2) ;
          double Prof(const Pt3dr &) ;
	  double ProfInDir(const Pt3dr & aP,const Pt3dr & aDir) ;

          Pt3dr Im1DirEtProf2_To_Terrain
                 (Pt2dr p1,Data_Ori3D_Std *  ph2,double prof2,const Pt3dr & aDir);


         Fonc_Num petp_to_carto
                  (
                       Pt2d<Fonc_Num> aPtIm1,
                       Data_Ori3D_Std  *  ph2,
                       Pt2d<Fonc_Num> aPtIm2
                  );
         Fonc_Num petp_to_3D
                  (
                       Pt2d<Fonc_Num> aPtIm1,
                       Data_Ori3D_Std  *  ph2,
                       Pt2d<Fonc_Num> aPtIm2,
                       bool           ToCarto
                  );




          Fonc_Num petp_to_terrain
                  (
                       Pt2d<Fonc_Num> aPtIm1,
                       Data_Ori3D_Std  *  ph2,
                       Pt2d<Fonc_Num> aPtIm2
                  );




         Tab_CPT_REF<Pt2dr> carte_et_z_to_terrain(Tab_CPT_REF<Pt2dr>,REAL z_lamb,REAL & z_ter);

         // Met dans epi1 epi2 les orientation  epipolaire de phot1, phot2
         // aP1 , aP2 est un couple de point approximatvement homologue,
         // il permet que les paralaxes soit approximativement centres et
         // (il sert a regler le zmin, zmax de l'appel a orilib)
         static void  ororient_epipolaires 
                (
                   tPtr & epi1, Data_Ori3D_Std & phot1,Pt2dr aP1,
                   tPtr & epi2, Data_Ori3D_Std  & phot2,Pt2dr aP2
               );

          INT  ZoneLambert() const;

         or_orientation * Ori() const {return _ori;}
         void  SetOrigineTgtLoc(const Pt2dr & aPt);

         Pt2di SzIm() const;
	 cOrientationConique * OC() const;


      private :
         // void  rev_tab_pt_photo(REAL *xt,REAL *yt,INT nb);

         inline void  correct(Pt2dr &); // enventually reverse
         void*                    _photo;
         or_orientation    * _ori;
         bool                    _inv_y;
	 bool                     mOri2Del;


         Data_Ori3D_Std ();


};

class cOri3D_OneEpip : public  ElDistortion22_Gen
{
      public :
           cOri3D_OneEpip
           (
                 Data_Ori3D_Std  * aPh,
                 Data_Ori3D_Std  *anEpi
           );

           virtual bool OwnInverse(Pt2dr &) const ;    //  Epi vers Phot
           virtual Pt2dr Direct(Pt2dr) const  ;    // Photo vers Epi
           cOri3D_OneEpip MapingChScale(REAL aChSacle) const;
           ~cOri3D_OneEpip();

          Data_Ori3D_Std  * Phot();
          Data_Ori3D_Std  * Epip();

       // Pour eventuellement eviter un delete dans le ~
          void SetPhot0();
          void SetEpi0();

      private :

            virtual ElDistortion22_Gen  * D22G_ChScale(REAL aS) const; // 
            void  Diff(ElMatrix<REAL> &,Pt2dr) const ;  //  Erreur Fatale




            Data_Ori3D_Std  * mPhot;
            Data_Ori3D_Std  * mEpi;


            Pt3dr             mCPDV;  // Normalement commun a epi et photo, en coordonnees epi
            Pt3dr             mP0Epi;
            Pt3dr             mUEpi;
            Pt3dr             mVEpi;
            Pt3dr             mWEpi;
            ElMatrix<REAL>    mTer2Epi;
};








#endif // !  _ELISE_PRIVATE_ORILIB_H

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
