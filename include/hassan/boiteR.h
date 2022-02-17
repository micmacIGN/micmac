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
#ifndef _HASSAN_BOITER_H
#define _HASSAN_BOITER_H


/////////////////////////////////////////////////

class BoiteR : public PRC0
{
   public :

       Im3D_REAL8  _boite;
       Im2D_INT2  _masq;
       REAL8***   _boite_d;
       INT2**     _masq_d;
       Pt3dr      _p0;
       Pt3dr      _pas;
       Pt3dr      _d1;
       Pt3dr      _d2;
       Pt3dr      _d3;
       INT        _tx;
       INT        _ty;
       INT        _tz;
       
       BoiteR(); 
       BoiteR(Boite& b);
       
       Im3D_REAL8  boite()                          { return _boite;};
       REAL8       boite(INT x,INT y,INT z)         { return _boite.data()[z][y][x]; };
       REAL8       boite(Pt3di p)                   { return _boite.data()[p.z][p.y][p.x]; };
       void        boite(INT x,INT y,INT z,REAL8 v) { _boite.data()[z][y][x] = v; };
       void        boite(Pt3di p, REAL8 v)          { _boite.data()[p.z][p.y][p.x] = v; };
       
       Im2D_INT2   masq()                           { return _masq; };
       INT2        masq(INT x,INT y)                { return _masq.data()[y][x]; };
       INT2        masq(Pt2di p)                    { return _masq.data()[p.y][p.x]; };
       void        masq(INT x,INT y, INT2 v)        { _masq.data()[y][x]=v; };
       void        masq(Pt2di p, INT2 v)            { _masq.data()[p.y][p.x]=v; };
       void        masq(Im2D_INT2 masq)             { ELISE_COPY(_masq.all_pts(),masq.in(0),_masq.out()); };
       
       INT   tx()  { return _tx;  };
       INT   ty()  { return _ty;  };
       INT   tz()  { return _tz;  };
       Pt3dr p0()  { return _p0;  };
       Pt3dr pas() { return _pas; };
       Pt3dr d1()  { return _d1;  };
       Pt3dr d2()  { return _d2;  };
       Pt3dr d3()  { return _d3;  };

       void   initialiser(REAL8 v=0)                { ELISE_COPY(_boite.all_pts(), 0., _boite.out()); };

       void correl(  Boite& b_g, 
                     Boite& b_d, 
                     INT semi_fenet, 
                     INT dif_ng=40 
                  );

       void max_z();
};


#endif // _HASSAN_BOITER_H

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
