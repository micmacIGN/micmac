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



#ifndef _ELISE_GENERAL_ORILIB_H
#define _ELISE_GENERAL_ORILIB_H

class Ori3D_Gen : public PRC0 
{
      friend class To_Phot_3Std_Not_Comp;
      public :

      protected  :
         Ori3D_Gen(class Data_Ori3D_Gen *);
      private :
         class Data_Ori3D_Gen * dog() {return (Data_Ori3D_Gen *)_ptr;}

};

struct or_orientation;

extern double  ALTISOL_UNDEF(); 
extern bool  ALTISOL_IS_DEF(double aZ);

extern double TIME_UNDEF();



class Ori3D_Std : public Ori3D_Gen
{
      friend class Fnum_O3d_phot_et_z_to_terrain;


      public :

   // Renvoie un polygone contenant la partie de Ori2 visible sur this ,
   // si true aZP est une altitude, sinon une profondeur de champs;
   //
   // liste vide si 
   //
         std::vector<Pt2dr> Emprise2InVue1(Ori3D_Std Ori2,double aZP,bool ByAlti);

         std::vector<Pt2dr> Inter(Ori3D_Std Ori2,double aZP,bool ByAlti);
         std::vector<Pt2dr> Inter(Ori3D_Std Ori2); // Avec altisol
         double    PropInter(Ori3D_Std Ori2); // entre 0 et 1
         double    PropInter(Ori3D_Std Ori2,double aZP,bool ByAlti); // entre 0 et 1

         static Box2dr TheNoBox;
         Ori3D_Std(const char *,bool inv_y = false,bool binarie = false,bool QuikGrid=false);
         Ori3D_Std(or_orientation    * _ori);

         // Pour creer une orientation correspondand a une image homthetique facteur zoom
         // zoom = 0.5 pour une orientation correspondant a une image plus petite
         Ori3D_Std(Ori3D_Std ,REAL zoom,Box2dr = TheNoBox);

         Fonc_Num  to_photo(Fonc_Num);
         Fonc_Num  photo_et_z_to_terrain(Fonc_Num);
         Pt2dr to_photo(Pt3dr p,bool DontUseDist=false) const;

	 void SetUseDist(bool UseDist) const;
         INT  ZoneLambert() const;


         REAL  FocaleAngulaire() const;         //HJ
         REAL  resolution_angulaire() const;         //HJ
         REAL  resolution() const;         //HJ
	 Pt3dr orsommet_de_pdv_terrain() const;
	 REAL altisol() const;
	 REAL profondeur() const;
         void SetAltiSol(REAL aZ);
	 REAL BSurH(Ori3D_Std Ori2);
	 void UnCpleHom(Ori3D_Std Ori2,Pt2dr & pH1,Pt2dr & pH2);

         Pt2dr DistT2P(const Pt2dr & aP) const;
         Pt2dr DistP2T(const Pt2dr & aP) const;

	 Pt3dr orDirI() const;
	 Pt3dr orDirJ() const;
         Pt3dr orDirK() const;


         Pt3dr to_terrain
               (
                   Pt2dr p1,
                   Ori3D_Std  ph2,Pt2dr p2,
                   REAL & dist
               );
         Pt3dr to_terrain(Pt2dr p1,REAL z) const;
         Pt3dr ImEtProf_To_Terrain(Pt2dr pp,REAL z) const;
         Pt3dr Im1etProf2_To_Terrain
               (Pt2dr p1,Ori3D_Std  ph2,double prof2) const;
         double Prof(const Pt3dr &) const ;
	 double ProfInDir(const Pt3dr & aP,const Pt3dr & aDir) const;

         Pt3dr carte_to_terr(Pt3dr) const;
         Pt3dr terr_to_carte(Pt3dr) const; // HJMPD
         Tab_CPT_REF<Pt2dr> carte_et_z_to_terrain(Tab_CPT_REF<Pt2dr>,REAL z_lamb,REAL & z_ter);
         
         void write_txt(const char *);
         void write_bin(const char *);

          class Data_Ori3D_Std * dos() {return (Data_Ori3D_Std *)_ptr;}
          class Data_Ori3D_Std * dos() const {return (Data_Ori3D_Std *)_ptr;}
          Pt2dr OrigineTgtLoc() const;
          void  SetOrigineTgtLoc(const Pt2dr & aPt);

          void SetOrientation(const ElRotation3D &);
          ElRotation3D  GetOrientation() const;


          Pt2di SzIm() const;
          Pt2di P0() const;
          std::vector<Pt2dr> EmpriseSol(double aZ) const;
          std::vector<Pt2dr> EmpriseSol() const;
	   
	   cOrientationConique * OC() const;

	  Pt3dr  ImDirEtProf2Terrain(const Pt2dr & aPIm,const REAL & aProf,const Pt3dr & aNormPl) const;

         Pt3dr Im1DirEtProf2_To_Terrain
               (Pt2dr p1,Ori3D_Std  ph2,double prof2,const Pt3dr & aDir) const;
      protected  :
      private :


};

// Regarde comment se transforme un repere direct  a z nulle
double TestOrientationOri(Pt2dr aP,Ori3D_Std anOri);

class cOriMntCarto
{
    public :

       // Dans les fichier ori, les grandeurs sont exprimes en unites
       // milimetrique, pour eviter tout erreur d'arrondi ulterieur
       // on peut avoir interet a ce que les grandeur tel que resol
       // soit des multiples exacts de ces unites

       static REAL ToUniteOri(REAL);

       cOriMntCarto(const std::string &);
       void ToFile(const std::string &);

       Pt2dr TerrainToPix(Pt2dr) const;
       Pt2dr ToPix(Pt2dr) const;
       Pt3dr PixToTerrain(Pt3dr) const;
       REAL ResolZ() const;
       REAL ResolPlani() const;

       cOriMntCarto
       (
           Pt2dr Ori,
           INT   mZoneLambert,
           Pt2di aSz,
           Pt2dr aResol,
           REAL  aZ0,
           REAL  aResolZ
       );
       
    private :


       static REAL StdLireIntAsReal(FILE * aFp);
       static long long  INT ToStdInt(REAL);
       static const REAL  UniteFile;
       Pt2dr mOrigine;
       INT   mZoneLambert; 
       Pt2di mSz;
       Pt2dr mResol;
       REAL  mZ0;
       REAL  mResolZ;
   
};

void DebugOri(Ori3D_Std anOri);

#endif // !  _ELISE_GENERAL_ORILIB_H

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
