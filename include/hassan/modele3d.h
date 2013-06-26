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

#ifndef INCL_MODELE3D
#define INCL_MODELE3D

//------------------------- Fichiers inclus --------------------------

#include "struct_base/chaine.h"
#include "struct_base/bbox.h"
#include "struct_base/contrs.h"
#include "struct_base/point.h"
#include "struct_base/pile.h"
#include "ptypes.h"


//-------------------------- Definitions structures -------------------

typedef INT ADRESSE;
typedef ADRESSE *P_ADRESSE;	

	/*--------------------------*/
namespace modele
{
	class Objet3D;
	class Sommet3D;
	class Arete;
	class Face;
	class Triangle;
}

TChaine FHNomType (modele::Objet3D*) ;
TChaine FHNomType (modele::Sommet3D*) ;
TChaine FHNomType (modele::Arete*) ;
TChaine FHNomType (modele::Face*) ;
TChaine FHNomType (modele::Triangle*) ;


namespace modele
{

class Arete {

 public:

  ADRESSE num_face;  
	
  BOOLEAN fictive;						// signale si l'arete est reelle ou construction mentale
  BOOLEAN contour;						// signale si l'arete est de contour
  BOOLEAN nouvelle;						// signale si l'arete est nouvelle	
  BOOLEAN visible;						// signale si l'arete est visible en mode filaire
  
  TTableau1D <ADRESSE> num_sommets;	// n� des sommets de l'arete
	
  Arete() : num_sommets(2) {
    nouvelle = FALSE;
    visible = FALSE;
    contour = FALSE;
    fictive = FALSE;
  }
  
  int operator == (Arete const & theArete) const { 
    if ((theArete.num_sommets[0]==num_sommets[0])&&(theArete.num_sommets[1]==num_sommets[1])) return(TRUE);
    if ((theArete.num_sommets[0]==num_sommets[1])&&(theArete.num_sommets[1]==num_sommets[0])) return(TRUE);		
    return(FALSE);
  }
  /*
  friend ostream &operator <<(ostream &theStream,Arete const &theArete) {
    theStream << endl;
    theStream << "Arete fictive=" << theArete.fictive << endl;
    theStream << "Arete contour=" << theArete.contour << endl;
    theStream << "Arete reelle="  << theArete.nouvelle << endl;
    theStream << "Arete visible=" << theArete.visible << endl;
    theStream << "num_sommets=(" << theArete.num_sommets[0] << "," << theArete.num_sommets[1] << ")" << endl;

    return(theStream);
    }*/
};

   /*--------------------------*/

class Triangle {

 public:
	
  ADRESSE num_triangle;   			// n� general du triangle qui permet d'indexer la table d'appartenance aux faces et objets

  TTableau1D <ADRESSE> num_sommets; 	// n� du sommet dans la liste de sommets interne a la face

  Triangle() : num_sommets(3) {
    num_triangle = -1;
  }

  int operator == (Triangle const & theTriangle) const { 
    if (  (theTriangle.num_sommets[0]==num_sommets[0])
	  &&(theTriangle.num_sommets[1]==num_sommets[1])
	  &&(theTriangle.num_sommets[2]==num_sommets[2])) return(TRUE);
    
    if (  (theTriangle.num_sommets[0]==num_sommets[0])
	  &&(theTriangle.num_sommets[1]==num_sommets[2])
	  &&(theTriangle.num_sommets[2]==num_sommets[1])) return(TRUE);
    
    if (  (theTriangle.num_sommets[0]==num_sommets[1])
	  &&(theTriangle.num_sommets[1]==num_sommets[2])
	  &&(theTriangle.num_sommets[2]==num_sommets[0])) return(TRUE);
    
    if (  (theTriangle.num_sommets[0]==num_sommets[1])
	  &&(theTriangle.num_sommets[1]==num_sommets[0])
	  &&(theTriangle.num_sommets[2]==num_sommets[2])) return(TRUE);
    
    if (  (theTriangle.num_sommets[0]==num_sommets[2])
	  &&(theTriangle.num_sommets[1]==num_sommets[1])
	  &&(theTriangle.num_sommets[2]==num_sommets[0])) return(TRUE);
    
    if (  (theTriangle.num_sommets[0]==num_sommets[2])
	  &&(theTriangle.num_sommets[1]==num_sommets[0])
	  &&(theTriangle.num_sommets[2]==num_sommets[1])) return(TRUE);
    return(FALSE);
  }
  /*
  friend ostream &operator <<(ostream &theStream,Triangle const &theTriangle) {
    theStream << "num_triangle=" << theTriangle.num_triangle << endl;
    theStream << "num_sommets"  << theTriangle.num_sommets << endl;
    return(theStream);
    }*/
};

	/*--------------------------*/

class Face {
  
 public:

  ADRESSE num_face;  
  
  // a mettre dans infoFaces
  INT type_face;			// type de face
  INT matiere_face;  		// matiere de face
  INT code_face;			// code de face
  
  // a mettre dans normales face
  BOOLEAN existe_norm;	// boolen qui indique si la surface est orientee
  BOOLEAN filtrage_norm;	// boolen qui indique si la face doit etre filtree avant affichage
  BOOLEAN interpole_norm;	// boolen qui indique si la normale doit etre interpolee entre les normales aux sommets
  TPoint3D < float > vect_norm;
  TPileBase < TPoint3D <float> > *vect_norm_sommet; // normales aux sommets de la face
  
  TPileBase <ADRESSE> *num_sommets;
  TPileBase <Triangle> *triangle;
  
  Face() {
    existe_norm = FALSE;
    filtrage_norm = FALSE;
    interpole_norm = FALSE;
    type_face = -1;
    code_face = -1;
    matiere_face = -1;
  }
  
  /*
  friend ostream &operator <<(ostream &theStream,Face const &theFace) {
    theStream << "vect_norm" << theFace.vect_norm << endl;
    theStream << "vect_norm_sommet" << *(theFace.vect_norm_sommet) << endl;
    theStream << "num_sommets" << *(theFace.num_sommets) << endl;
    return(theStream);
    }*/
};
	
	/*--------------------------*/

class Sommet3D {

 public:

  TPoint3D < double > xyz;					
  TPileBase < TPoint2D < ADRESSE > > *lien_faces;
  
  Sommet3D() {
    xyz.x = 0.0;
    xyz.y = 0.0;
    xyz.z = 0.0;
  }
  
  int operator == (Sommet3D const & theSommet) const { return (theSommet.xyz==xyz) ; }
  /*
  friend ostream &operator <<(ostream &theStream,Sommet3D const &theSommet) {
    theStream << "xyz=" << theSommet.xyz << endl;
    //		theStream << "Nb liens avec face=" << (theSommet->lien_faces).iGetTaille() << endl;
    //		theStream << "liens avec face=" << theSommet->lien_faces << endl;
    return(theStream);
    }*/
  
 private:

};

	/*--------------------------*/

class Box3D {

 public:

  DOUBLE x_min;
  DOUBLE x_max;
  DOUBLE y_min;
  DOUBLE y_max;
  DOUBLE z_min;
  DOUBLE z_max;	
	
  Box3D() {
    x_min = MAXDOUBLE;
    x_max = MINDOUBLE;
    y_min = MAXDOUBLE;
    y_max = MINDOUBLE;
    z_min = MAXDOUBLE;
    z_max = MINDOUBLE;		
  }

  void Add(Sommet3D const &theSommet) {
    x_min = MIN2(theSommet.xyz.x,x_min);
    x_max = MAX2(theSommet.xyz.x,x_max);
    y_min = MIN2(theSommet.xyz.y,y_min);
    y_max = MAX2(theSommet.xyz.y,y_max);
    z_min = MIN2(theSommet.xyz.z,z_min);
    z_max = MAX2(theSommet.xyz.z,z_max);		
  }
  
  TPoint3D <double> Center() {
    TPoint3D <double> theBoxCenter;
    theBoxCenter.x = (x_max + x_min)/2.;
    theBoxCenter.y = (y_max + y_min)/2.;
    theBoxCenter.z = (z_max + z_min)/2.;
    return(theBoxCenter);
  }
  
  int operator == (Box3D const & theBox) const { 
    if (  (theBox.x_min==x_min)
	  &&(theBox.y_min==y_min)
	  &&(theBox.z_min==z_min)
	  &&(theBox.x_max==x_max)
	  &&(theBox.y_max==y_max)
	  &&(theBox.z_max==z_max)) 
    return(TRUE);
    return(FALSE);
  }
  /*
  friend ostream &operator <<(ostream &theStream,Box3D const &theBox) {
    theStream << "xmin=" << theBox.x_min << endl;
    theStream << "xmax=" << theBox.x_max << endl;
    theStream << "ymin=" << theBox.y_min << endl;
    theStream << "ymax=" << theBox.y_max << endl;
    theStream << "zmin=" << theBox.z_min << endl;
    theStream << "zmax=" << theBox.z_max << endl;
    return(theStream);
    }*/
	
 private:

};

 	/*--------------------------*/

class Objet3D {

 public:
	
  INT num_objet;      	 	  // numero d'identification de l'objet
  INT num_sous_objet;			// numero d'identification de sous-objet
  INT num_ilot_objet;			// numero d'identification de l'ilot
  INT existence;				// etat physique-semantique de l'objet
  INT num_groupe;				
  INT type_objet;				
  INT classe_objet;			
  BOOLEAN is_facette;
  BOOLEAN is_graphe;	

  Box3D box;					 		// boite englobante de l'objet
  TPileBase <ADRESSE> *num_sommets;	// liste d'adresses de sommets
  TPileBase <Arete> *arete;   		// liste d'aretes de l'objet
  TPileBase <Face> *face;	 			// liste de faces de l'objet

  /*
  friend ostream &operator <<(ostream &theStream,Objet3D const &theObjet) {
    theStream << "is_graphe=" << theObjet.is_graphe << endl;
    theStream << "is_facette=" << theObjet.is_facette << endl;
    theStream << "num_sommets" << *(theObjet.num_sommets) << endl;
    theStream << "arete" << *(theObjet.arete) << endl;
    theStream << "face" << *(theObjet.face) << endl;
    return(theStream);
    }*/
};
	 
 /*--------------------------*/

enum TYPE_MODELE3D {TRAPU,MNE,DXF,VRML,TIN,POIVILLIERS,HUMMEL,INCONNU};

class Modele3D {
	
 private:
	
 public:

  Orientation orientation;

  INT nb_sommets;		   		
  INT nb_aretes;		   		
  INT nb_triangles;			
  INT nb_faces;			  
  INT nb_objets;		

  Box3D box;	  										
  TPileBase < Objet3D > *objet;	  					
  
  TPileBase < INT > *infoSommetType;						
  INT flagSommetType;
  TPileBase < Sommet3D > *sommet;						
  TPileBase < TPoint3D < double > > *xyz;						
  TPileBase < TPoint2D < double > > *xy;
  TPileBase < double > *z;	
	
  TPileBase < TPoint3D < ADRESSE > > *lien_triangles;	

  TPileBase < TPoint2D < ADRESSE > > *lien_faces;
  TPileBase < TPoint2D < ADRESSE > > *infoFaces;
  TPileBase < ADRESSE > *normalesFaces;

  TPileBase < TPoint2D < ADRESSE > > *lien_aretes;
  TPileBase < TPoint2D < ADRESSE > > *infoAretes;
	
  TPileBase < TPoint2D < ADRESSE > > *lien_objets;
  TPileBase < TPoint2D < ADRESSE > > *infoObjets;

  Modele3D() {
    nb_sommets = 0;
    nb_aretes = 0;
    nb_triangles = 0;
    nb_faces = 0;
    nb_objets = 0;
  }

  // static Modele3D *Lit(P_CHAR nom,P_CHAR image,int minPoly=0 ,int maxPoly=-1);
  /** valeurs DX_TRAPU_LAMBERT et DY_TRAPU_LAMBERT par defaut sur Amiens */

  static Modele3D *Lit(P_CHAR nom,P_CHAR nomOut, const double DX_TRAPU_LAMBERT=590000.0, const double DY_TRAPU_LAMBERT=240000.0, int minPoly=0, int maxPoly=-1);
		
};

class ModeleTrapu: public Modele3D {

 public:
	
  ModeleTrapu() {

    nb_sommets = 0;
    nb_aretes = 0;
    nb_triangles = 0;
    nb_faces = 0;
    nb_objets = 0;
  }
		
  ModeleTrapu::ModeleTrapu(P_CHAR nom,P_CHAR nomOut,const double,const double,int,int);

};


class ModeleMne: public Modele3D {

 public:
	
  ModeleMne() {
    nb_sommets = 0;
    nb_aretes = 0;
    nb_triangles = 0;
    nb_faces = 0;
    nb_objets = 0;
  }
	
  ModeleMne(P_CHAR nom);
};

/*-------------------- Fontions amies -----------------------*/


void TrapuToTra(P_CHAR nom,P_CHAR nomOut,const double DX_TRAPU_LAMBERT = 590000.0, const double DY_TRAPU_LAMBERT = 1240000.0);

void TrapuToWx3d(P_CHAR nom,P_CHAR nomOut,const double DX_TRAPU_LAMBERT = 590000.0, const double DY_TRAPU_LAMBERT = 1240000.0,int minPoly = 0,int maxPoly=-1);

void TraToWx3d(P_CHAR nom,P_CHAR nomOut);


/*-------------------- Fontions privees -----------------------*/

TYPE_MODELE3D WhatTypeIsModele3D(P_CHAR nom);

VOID FaceCalculVecteurNormal(Face &face,TPileBase <Sommet3D> const &p_sommet);


}//fin namespace

#endif

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
