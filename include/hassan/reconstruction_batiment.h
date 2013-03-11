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
#ifndef _ELISE_HASSAN_RECONSTRUCTION_BATIMENT_H
#define _ELISE_HASSAN_RECONSTRUCTION_BATIMENT_H



///////////////////////////////////////////////////////////////////////////
class HJ_Chantier
{ 
   public : 

     HJ_Chantier();
     HJ_Chantier(char* fichier_parametre);

     virtual ~HJ_Chantier();

     virtual void enregistrer_des_resultats();
};

///////////////////////////////////////////////////////////////////////////

class HJ_Reconstruction_Batiment : public HJ_Chantier
{
   public : 

     Facette_2d _f;

     REAL       _surf_inf_min;
     REAL       _surf_inf_moy;
     REAL       _surf_inf_max;


     HJ_Reconstruction_Batiment();
     HJ_Reconstruction_Batiment(char* fichier_parametre);
     virtual ~HJ_Reconstruction_Batiment();


             void fabrication_batiment(Facette_2d f);


   protected :

     virtual void creation_boite();

     virtual void detection_plans();

             void filtrage_plans();
     virtual void filtrage_plans_morpho(REAL seuil1, REAL seuil2);
     virtual void filtrage_plans_normale();

             void creation_graphe();
     virtual void ponderation_graphe_heur();

             void recherche_heuristique();
             void recherche_exhaustive();
     virtual void recherche_meil_solution();
     virtual void recherche_meil_solution_correl();
             void recherche_heuristique_iterative();
             void recherche_plans_meil_solution();

   public : 

     virtual void enregistrer_des_resultats();

};

///////////////////////////////////////////////////////////////////////////

class HJ_Reconstruction_Batiment_Couple : public HJ_Reconstruction_Batiment
{
   public :

     HJ_Reconstruction_Batiment_Couple();
     HJ_Reconstruction_Batiment_Couple(char* fichier_parametre);
     virtual ~HJ_Reconstruction_Batiment_Couple();

  protected :

     virtual void creation_boite();

     virtual void detection_plans();

     virtual void filtrage_plans_morpho(REAL seuil1, REAL seuil2);
     virtual void filtrage_plans_normale();
     virtual void ponderation_graphe_heur();
     virtual void recherche_meil_solution();
     virtual void recherche_meil_solution_correl();

   public :

     virtual void enregistrer_des_resultats();

};

///////////////////////////////////////////////////////////////////////////

class HJ_Reconstruction_Batiment_Cube : public HJ_Reconstruction_Batiment
{
   public :

     HJ_Reconstruction_Batiment_Cube();
     HJ_Reconstruction_Batiment_Cube(char* fichier_parametre);
     virtual ~HJ_Reconstruction_Batiment_Cube();

   protected :

     virtual void creation_boite();

     virtual void detection_plans();

     virtual void filtrage_plans_morpho(REAL seuil1, REAL seuil2);
     virtual void filtrage_plans_normale();
     virtual void ponderation_graphe_heur();
     virtual void recherche_meil_solution();
     virtual void recherche_meil_solution_correl();
   
   public :

     virtual void enregistrer_des_resultats();


};

///////////////////////////////////////////////////////////////////////////

class HJ_Reconstruction_Batiment_MNS : public HJ_Reconstruction_Batiment
{
   public :

     HJ_Reconstruction_Batiment_MNS();
     HJ_Reconstruction_Batiment_MNS(char* fichier_parametre);
     virtual ~HJ_Reconstruction_Batiment_MNS();

   protected :

     virtual void creation_boite();

     virtual void detection_plans();

     virtual void filtrage_plans_morpho(REAL seuil1, REAL seuil2);
     virtual void filtrage_plans_normale();
     virtual void ponderation_graphe_heur();
     virtual void recherche_meil_solution();
     virtual void recherche_meil_solution_correl();
   
   public :

     virtual void enregistrer_des_resultats();


};


#endif // _ELISE_HASSAN_RECONSTRUCTION_BATIMENT_H

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
